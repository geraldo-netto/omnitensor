"""Qualified Vulkan model adapters for speech and visual inference."""

from __future__ import annotations

import asyncio
import base64
import fcntl
import json
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from itertools import chain
from pathlib import Path

from omnitensor.plugins.media_transcription import (
    MediaTranscriptionError,
    SpeechSegment,
    SpeechTranscriber,
    SpeechTranscript,
    VisualFrame,
    VisualTranscriber,
    VisualTranscript,
)
from omnitensor.plugins.protocol import CancellationToken

from .errors import MediaGpuError
from .formats import AUDIO_SAMPLE_RATE, AvMediaAdapter, _visual_payload

_OFFLOAD = re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers\s+to\s+GPU", re.I)
_LLAMA_DEVICE = re.compile(r"using device Vulkan\d+ \((.+)\) \([0-9a-fA-F:.]+\)", re.I)
_WHISPER_DEVICE = re.compile(r"ggml_vulkan:\s*\d+\s*=\s*(.+?)\s*\(", re.I)
_WHISPER_BACKEND = re.compile(r"using Vulkan(\d+) backend", re.I)

# The same knob the vulkan generation runtime honours (OMNI-0586): how many
# model layers to place on the GPU, -1 meaning all of them. The sandbox
# forwards it to every worker; ignoring it here meant a vision model larger
# than VRAM could not ask this provider for a partial offload at all.
GPU_LAYERS_VARIABLE = "OMNITENSOR_GPU_LAYERS"


def _gpu_layer_budget() -> int:
    configured = os.environ.get(GPU_LAYERS_VARIABLE, "").strip()
    if not configured:
        return -1
    try:
        return int(configured)
    except ValueError:
        # "-1, meaning all layers" is the exact opposite of what somebody who
        # set a budget asked for; a typo must refuse, not maximise.
        raise MediaGpuError(
            f"{GPU_LAYERS_VARIABLE} must be an integer, not {configured!r}"
        ) from None


def _window_segments(spoken, offset_ms: int, window_ms: int):
    """One window's segments, moved to their place in the whole recording.

    `t0`/`t1` are centiseconds from the start of what the model was given,
    which is the window rather than the file, so every time needs the
    window's own offset added to it.
    """
    for item in spoken:
        if not isinstance(item.text, str) or not item.text.strip():
            continue
        start_ms = min(window_ms, max(0, int(item.t0) * 10))
        end_ms = min(window_ms, max(start_ms, int(item.t1) * 10))
        if end_ms > start_ms:
            yield SpeechSegment(offset_ms + start_ms, offset_ms + end_ms, item.text)


class VulkanLease:
    def __init__(self, path: Path) -> None:
        if not isinstance(path, Path) or not path.is_file():
            raise MediaGpuError("GPU accelerator lease is unavailable")
        self._path = path

    @contextmanager
    def hold(self) -> Iterator[None]:
        stream = self.acquire()
        try:
            yield
        finally:
            self.release(stream)

    def acquire(self):
        """Open the lease and hold it, or leave nothing open.

        `release` closes the stream, and the caller has no stream to release
        until this returns — so an interrupted `flock` used to leave the file
        open with nobody able to close it, and the descriptor held until the
        process exited. The Vulkan runtime carried the same leak and was fixed
        the same way.
        """
        stream = self._path.open("a+b")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        except BaseException:
            stream.close()
            raise
        return stream

    @staticmethod
    def release(stream) -> None:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


class WhisperVulkanTranscriber(SpeechTranscriber):
    def __init__(
        self,
        model_path: Path,
        decoder: AvMediaAdapter,
        lease: VulkanLease,
    ) -> None:
        self._model_path = model_path
        self._decoder = decoder
        # None means detect, which is what this did before the setting and
        # what it still does unless somebody says otherwise.
        self._language: str | None = None
        self._lease = lease
        self._device_proven = False

    async def preflight(self) -> None:
        await asyncio.to_thread(self._load_and_release)

    def _load_and_release(self) -> None:
        with self._lease.hold():
            model, _logs = self._load()
            _close_whisper(model)

    def prefer_language(self, language: str | None) -> None:
        """Transcribe as this language rather than detecting one.

        Held rather than passed per call: a worker reads its configuration
        once, at `start()`, and is replaced when it changes.
        """
        self._language = language or None

    async def transcribe(self, source: Path, cancellation: CancellationToken) -> SpeechTranscript:
        return await asyncio.to_thread(self._transcribe_sync, source, cancellation)

    def _transcribe_sync(self, source: Path, cancellation: CancellationToken) -> SpeechTranscript:
        """Transcribe a recording of any length, one decoded window at a time.

        The first window is decoded before the lease is taken, so a recording
        with no audio in it never loads a model or holds the card. After that
        the lease covers the whole loop: one model load serves every window,
        and the decode of window n+1 costs far less than the inference on
        window n that it overlaps with.
        """
        windows = self._decoder.decode_audio_windows(source, cancellation)
        first = next(iter(windows), None)
        if first is None:
            return SpeechTranscript(None, ())
        with self._lease.hold():
            model, _logs = self._load()
            try:
                return self._transcribe_windows(model, chain((first,), windows), cancellation)
            finally:
                _close_whisper(model)

    def _transcribe_windows(self, model, windows, cancellation) -> SpeechTranscript:
        language: str | None = self._language
        # Whether any window carried audio at all, tracked separately now that
        # `language` may be set before the loop: a recording with nothing in it
        # reports no language, and a configured one must not make it look
        # otherwise.
        heard = False
        segments: list[SpeechSegment] = []
        for start_sample, samples in windows:
            cancellation.raise_if_cancelled()
            if int(samples.size) == 0:
                continue
            heard = True
            if language is None:
                # Detected once, on the first window that carries audio: the
                # language of a recording does not change at minute five, and
                # re-detecting per window would let it appear to.
                detected, _probabilities = model.auto_detect_language(samples)
                language = detected[0]
            spoken = model.transcribe(
                samples,
                abort_callback=lambda: cancellation.cancelled,
                language=language,
                no_context=True,
                print_progress=False,
                print_realtime=False,
                suppress_blank=True,
                suppress_non_speech_tokens=True,
            )
            cancellation.raise_if_cancelled()
            offset_ms = int(start_sample) * 1000 // AUDIO_SAMPLE_RATE
            window_ms = max(1, int(samples.size) * 1000 // AUDIO_SAMPLE_RATE)
            segments.extend(_window_segments(spoken, offset_ms, window_ms))
        if language is None or not heard:
            return SpeechTranscript(None, ())
        return SpeechTranscript(language, tuple(segments))

    def _load(self):
        try:
            import _pywhispercpp as native  # noqa: PLC0415 - deferred: an optional or heavy dependency
            from pywhispercpp.model import Model  # noqa: PLC0415
        except ImportError as error:
            raise MediaGpuError(
                "install pywhispercpp 1.5.0 from source with GGML_VULKAN=1"
            ) from error
        logs = []
        native.whisper_log_set(lambda _level, message: logs.append(str(message)))
        try:
            model = Model(
                str(self._model_path),
                redirect_whispercpp_logs_to=False,
                context_params={"use_gpu": True, "flash_attn": True, "gpu_device": 0},
                n_threads=4,
                print_progress=False,
                print_realtime=False,
            )
        finally:
            native.whisper_log_set(None)
        joined = "".join(logs)
        match = _WHISPER_DEVICE.search(joined)
        backend = _WHISPER_BACKEND.search(joined)
        if match is not None:
            # A named device, whichever card it is. Pinning the name meant a
            # new card, or a driver that words the line differently, turned
            # transcription off — while what has to be true is only that this
            # did not quietly run on the CPU.
            self._device_proven = bool(match.group(1).strip())
        if backend is None or backend.group(1) != "0" or not self._device_proven:
            _close_whisper(model)
            raise MediaGpuError("Whisper did not prove it loaded onto a Vulkan device")
        return model, tuple(logs)


class QwenVulkanVisualTranscriber(VisualTranscriber):
    def __init__(
        self,
        model_path: Path,
        projector_path: Path,
        lease: VulkanLease,
    ) -> None:
        self._model_path = model_path
        self._projector_path = projector_path
        self._lease = lease
        self._lease_stream = None
        self._handler = None
        self._llama = None
        self._log_callback = None
        self._abort_callback = None

    async def preflight(self) -> None:
        await asyncio.to_thread(self._ensure_loaded)
        await self.release()

    async def transcribe(
        self, frame: VisualFrame, cancellation: CancellationToken
    ) -> VisualTranscript:
        return await asyncio.to_thread(self._transcribe_sync, frame, cancellation)

    def _transcribe_sync(
        self, frame: VisualFrame, cancellation: CancellationToken
    ) -> VisualTranscript:
        llama = self._ensure_loaded()
        cancellation.raise_if_cancelled()
        _llama_type, llama_cpp, _handler_type = self._runtime()

        @llama_cpp.ggml_abort_callback
        def abort(_data):
            return cancellation.cancelled

        self._abort_callback = abort
        llama_cpp.llama_set_abort_callback(llama.ctx, abort, None)
        encoded = base64.b64encode(_visual_payload(frame.path)).decode("ascii")
        cancellation.raise_if_cancelled()
        mime = "image/png"
        messages = [
            {
                "role": "system",
                "content": (
                    "Transcribe visual content faithfully. Preserve every legible script, "
                    "including Hebrew and mixed-direction text. Describe only visible facts."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    {
                        "type": "text",
                        "text": (
                            "Return JSON with visibleText containing exact legible text "
                            "(empty when none) and description containing a concise scene "
                            "transcription."
                        ),
                    },
                ],
            },
        ]
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["visibleText", "description"],
            "properties": {
                "visibleText": {"type": "string"},
                "description": {"type": "string", "minLength": 1},
            },
        }
        try:
            chunks = llama.create_chat_completion(
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                seed=0,
                # No token ceiling: a dense page of visible text needs what it
                # needs, and a truncated answer lands mid-object and surfaces
                # as visual-invalid — the ceiling-shaped wrong answer the
                # no-capping rule names. The model's stop token and the
                # context window are the only real bounds (OMNI-0600).
                max_tokens=None,
                # The 9B was observed looping a phrase to the token cap once
                # on strongly rotated text (its catalog entry says "serve
                # with a repeat penalty"); 0.3.34's default is 1.0, i.e.
                # none. Mild enough not to distort transcriptions, which
                # legitimately repeat words.
                repeat_penalty=1.15,
                response_format={"type": "json_object", "schema": schema},
                stream=True,
            )
            content = []
            for chunk in chunks:
                cancellation.raise_if_cancelled()
                choices = chunk.get("choices") if isinstance(chunk, Mapping) else None
                delta = choices[0].get("delta") if isinstance(choices, list) and choices else None
                text = delta.get("content") if isinstance(delta, Mapping) else None
                if isinstance(text, str):
                    content.append(text)
        except BaseException:
            cancellation.raise_if_cancelled()
            raise
        try:
            document = json.loads("".join(content))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise MediaTranscriptionError(
                "visual-invalid", "visual provider returned invalid JSON"
            ) from error
        if not isinstance(document, dict) or set(document) != {"visibleText", "description"}:
            raise MediaTranscriptionError(
                "visual-invalid", "visual provider returned invalid fields"
            )
        return VisualTranscript(
            frame.timestamp_ms,
            frame.structured_text or document["visibleText"],
            document["description"],
        )

    def _ensure_loaded(self):
        if self._llama is not None:
            return self._llama
        llama_type, llama_cpp, handler_type = self._runtime()
        self._lease_stream = self._lease.acquire()
        logs = []

        @llama_cpp.llama_log_callback
        def callback(_level, message, _data):
            logs.append(message.decode("utf-8", errors="replace"))

        self._log_callback = callback
        llama_cpp.llama_log_set(callback, None)
        try:
            self._handler = handler_type(str(self._projector_path), verbose=False)
            self._llama = llama_type(
                model_path=str(self._model_path),
                chat_handler=self._handler,
                n_ctx=8192,
                n_batch=1024,
                n_gpu_layers=_gpu_layer_budget(),
                main_gpu=0,
                offload_kqv=True,
                op_offload=True,
                flash_attn=True,
                verbose=False,
            )
        except Exception:
            self._release_sync()
            raise
        offload = _OFFLOAD.search("".join(logs))
        device = _LLAMA_DEVICE.search("".join(logs))
        if offload is None or device is None or int(offload.group(1)) < 1:
            # A load that names no Vulkan device, or put no layer on one, is
            # not a GPU load. A *partial* offload is (OMNI-0586): declared,
            # visible in the log, and real work — refusing it capped what
            # this machine may run, which is the owner's standing no-capping
            # rule said the other way round.
            self._release_sync()
            raise MediaGpuError("Qwen VL did not prove a Vulkan load")

        # The capture sink has done its job; leaving it installed appended
        # every later llama.cpp line to a list nothing reads for the life of
        # the worker (OMNI-0593) — the same leak the vulkan runtime already
        # fixed with its own discard callback.
        @llama_cpp.llama_log_callback
        def discard(_level, _message, _data):
            return None

        self._log_callback = discard
        llama_cpp.llama_log_set(discard, None)
        return self._llama

    @staticmethod
    def _runtime():
        try:
            from llama_cpp import (  # noqa: PLC0415 - deferred: an optional or heavy dependency
                Llama,
                llama_cpp,
            )
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler  # noqa: PLC0415
        except ImportError as error:
            raise MediaGpuError("Vulkan llama-cpp-python runtime is unavailable") from error
        return Llama, llama_cpp, Qwen25VLChatHandler

    async def release(self) -> None:
        await asyncio.to_thread(self._release_sync)

    def _release_sync(self) -> None:
        handler = self._handler
        self._handler = None
        if handler is not None:
            stack = getattr(handler, "_exit_stack", None)
            if stack is not None:
                stack.close()
        llama = self._llama
        self._llama = None
        self._abort_callback = None
        if llama is not None:
            llama.close()
        lease = self._lease_stream
        self._lease_stream = None
        if lease is not None:
            self._lease.release(lease)


def _close_whisper(model) -> None:
    if model is None:
        return
    context = getattr(model, "_ctx", None)
    if context is None:
        return
    model._ctx = None
    import _pywhispercpp as native  # noqa: PLC0415 - deferred: an optional or heavy dependency

    native.whisper_free(context)


__all__ = [
    "QwenVulkanVisualTranscriber",
    "VulkanLease",
    "WhisperVulkanTranscriber",
]
