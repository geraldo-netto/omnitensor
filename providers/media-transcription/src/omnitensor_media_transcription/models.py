"""Qualified Vulkan model adapters for speech and visual inference."""

from __future__ import annotations

import asyncio
import base64
import fcntl
import json
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

from .errors import QualifiedMediaError
from .formats import AUDIO_SAMPLE_RATE, AvMediaAdapter, _visual_payload

_OFFLOAD = re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers\s+to\s+GPU", re.I)
_LLAMA_DEVICE = re.compile(r"using device Vulkan\d+ \((.+)\) \([0-9a-fA-F:.]+\)", re.I)
_WHISPER_DEVICE = re.compile(r"ggml_vulkan:\s*\d+\s*=\s*(.+?)\s*\(", re.I)
_WHISPER_BACKEND = re.compile(r"using Vulkan(\d+) backend", re.I)


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
            raise QualifiedMediaError("GPU accelerator lease is unavailable")
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
        expected_device: str,
    ) -> None:
        self._model_path = model_path
        self._decoder = decoder
        self._lease = lease
        self._expected_device = expected_device
        self._device_proven = False

    async def preflight(self) -> None:
        await asyncio.to_thread(self._load_and_release)

    def _load_and_release(self) -> None:
        with self._lease.hold():
            model, _logs = self._load()
            _close_whisper(model)

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
        language: str | None = None
        segments: list[SpeechSegment] = []
        for start_sample, samples in windows:
            cancellation.raise_if_cancelled()
            if int(samples.size) == 0:
                continue
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
        if language is None:
            return SpeechTranscript(None, ())
        return SpeechTranscript(language, tuple(segments))

    def _load(self):
        try:
            import _pywhispercpp as native  # noqa: PLC0415 - deferred: an optional or heavy dependency
            from pywhispercpp.model import Model  # noqa: PLC0415
        except ImportError as error:
            raise QualifiedMediaError(
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
            self._device_proven = match.group(1).strip() == self._expected_device
        if backend is None or backend.group(1) != "0" or not self._device_proven:
            _close_whisper(model)
            raise QualifiedMediaError("Whisper did not prove its qualified Vulkan device")
        return model, tuple(logs)


class QwenVulkanVisualTranscriber(VisualTranscriber):
    def __init__(
        self,
        model_path: Path,
        projector_path: Path,
        lease: VulkanLease,
        expected_device: str,
    ) -> None:
        self._model_path = model_path
        self._projector_path = projector_path
        self._lease = lease
        self._expected_device = expected_device
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
                max_tokens=768,
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
                n_gpu_layers=-1,
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
        if (
            offload is None
            or device is None
            or offload.group(1) != offload.group(2)
            or device.group(1) != self._expected_device
        ):
            self._release_sync()
            raise QualifiedMediaError("Qwen VL did not prove full qualified Vulkan offload")
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
            raise QualifiedMediaError("Vulkan llama-cpp-python runtime is unavailable") from error
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
