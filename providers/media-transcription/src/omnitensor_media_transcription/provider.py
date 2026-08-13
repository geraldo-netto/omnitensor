"""Qualified in-process Vulkan provider for bounded multimedia transcription."""

from __future__ import annotations

import asyncio
import base64
import fcntl
import hashlib
import importlib.metadata
import importlib.resources
import io
import json
import math
import posixpath
import re
import shutil
import tempfile
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from omnitensor.plugins.media_transcription import (
    MAX_DOCUMENT_PAGES,
    MAX_DURATION_MS,
    MAX_PRESENTATION_SLIDES,
    MAX_VIDEO_DURATION_MS,
    MAX_VISUALS,
    DocumentTranscriber,
    FrameSampler,
    MediaInfo,
    MediaModality,
    MediaProbe,
    MediaProviderIdentity,
    MediaTranscriptionError,
    MediaTranscriptionPlugin,
    PresentationTranscriber,
    SpeechSegment,
    SpeechTranscriber,
    SpeechTranscript,
    VisualFrame,
    VisualTranscriber,
    VisualTranscript,
)
from omnitensor.plugins.protocol import CancellationToken
from omnitensor.sdk import current_plugin_bootstrap

PLUGIN_ID = "media-transcription"
PROVIDER_ID = "media-transcription-vulkan"
VISION_ARTIFACT_ID = "qwen2-5-vl-7b-instruct"
SPEECH_ARTIFACT_ID = "whisper-small-multilingual"
VISION_PROJECTOR = "mmproj.gguf"
MAX_DECODED_AUDIO_SAMPLES = MAX_DURATION_MS * 16
FRAME_INTERVAL_MS = 15_000
_IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".svg", ".webp"})
_AUDIO_SUFFIXES = frozenset({".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"})
_VIDEO_SUFFIXES = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})
_PRESENTATION_SUFFIXES = frozenset({".odp", ".pptx"})
_DOCUMENT_SUFFIXES = frozenset({".pdf", ".tif", ".tiff"})
_PRESENTATION_IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".webp"})
MAX_ARCHIVE_ENTRIES = 4096
MAX_ARCHIVE_EXPANDED_BYTES = 256 * 1024 * 1024
MAX_SLIDE_XML_BYTES = 4 * 1024 * 1024
MAX_PRESENTATION_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGES_PER_SLIDE = 4
MAX_SVG_BYTES = 8 * 1024 * 1024
MAX_RENDER_DIMENSION = 1600
MAX_INFERENCE_DIMENSION = 768
QUALIFICATION_EVIDENCE = "media-transcription-rx6600xt-report.json"
MAX_QUALIFICATION_EVIDENCE_BYTES = 256 * 1024
_OFFLOAD = re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers\s+to\s+GPU", re.I)
_LLAMA_DEVICE = re.compile(r"using device Vulkan\d+ \((.+)\) \([0-9a-fA-F:.]+\)", re.I)
_WHISPER_DEVICE = re.compile(r"ggml_vulkan:\s*\d+\s*=\s*(.+?)\s*\(", re.I)
_WHISPER_BACKEND = re.compile(r"using Vulkan(\d+) backend", re.I)


class QualifiedMediaError(RuntimeError):
    """Local provider does not match its frozen qualification."""


@dataclass(frozen=True, slots=True)
class PresentationSlide:
    number: int
    text: str
    images: tuple[Path, ...]


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
        stream = self._path.open("a+b")
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        return stream

    @staticmethod
    def release(stream) -> None:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


class AvMediaAdapter(MediaProbe, FrameSampler):
    """Decode metadata/audio/frames through bounded libav calls, never inference."""

    def __init__(self) -> None:
        self._owned_roots: set[Path] = set()

    async def inspect(self, source: Path) -> MediaInfo:
        return await asyncio.to_thread(self._inspect_sync, source)

    def _inspect_sync(self, source: Path) -> MediaInfo:
        suffix = source.suffix.lower()
        static = self._static_media_info(source, suffix)
        if static is not None:
            return static
        if suffix not in _AUDIO_SUFFIXES | _VIDEO_SUFFIXES:
            raise MediaTranscriptionError("media-unsupported", "selected media type is unsupported")
        try:
            import av

            with av.open(str(source), mode="r") as container:
                audio = next(iter(container.streams.audio), None)
                video = next(iter(container.streams.video), None)
                duration = _duration_ms(container, audio or video)
                if video is not None and suffix in _VIDEO_SUFFIXES:
                    return MediaInfo(
                        MediaModality.VIDEO,
                        duration,
                        int(video.codec_context.width),
                        int(video.codec_context.height),
                        audio is not None,
                    )
                if audio is not None and video is None:
                    return MediaInfo(MediaModality.AUDIO, duration, None, None, True)
        except MediaTranscriptionError:
            raise
        except Exception as error:
            raise MediaTranscriptionError(
                "media-invalid", "selected media could not be decoded"
            ) from error
        raise MediaTranscriptionError("media-invalid", "selected media has no supported stream")

    def _static_media_info(self, source: Path, suffix: str) -> MediaInfo | None:
        if suffix in _IMAGE_SUFFIXES:
            return self._image_info(source)
        if suffix in _PRESENTATION_SUFFIXES:
            return MediaInfo(
                MediaModality.PRESENTATION,
                None,
                None,
                None,
                False,
                _presentation_slide_count(source),
            )
        if suffix in _DOCUMENT_SUFFIXES:
            return MediaInfo(
                MediaModality.DOCUMENT,
                None,
                None,
                None,
                False,
                None,
                _document_page_count(source),
            )
        return None

    @staticmethod
    def _image_info(source: Path) -> MediaInfo:
        try:
            from PIL import Image

            content = _svg_png(source) if source.suffix.lower() == ".svg" else source
            with Image.open(content) as image:
                image.verify()
            with Image.open(content) as image:
                width, height = image.size
        except Exception as error:
            raise MediaTranscriptionError(
                "media-invalid", "selected image could not be decoded"
            ) from error
        return MediaInfo(MediaModality.IMAGE, None, int(width), int(height), False)

    async def decode_audio(self, source: Path, cancellation: CancellationToken):
        return await asyncio.to_thread(self._decode_audio_sync, source, cancellation)

    @staticmethod
    def _decode_audio_sync(source: Path, cancellation: CancellationToken):
        try:
            import av
            import numpy as np

            chunks = []
            samples = 0
            with av.open(str(source), mode="r") as container:
                stream = next(iter(container.streams.audio), None)
                if stream is None:
                    raise MediaTranscriptionError("audio-missing", "media has no audio stream")
                resampler = av.AudioResampler(format="fltp", layout="mono", rate=16_000)
                for decoded in container.decode(stream):
                    cancellation.raise_if_cancelled()
                    samples = _append_audio_frames(chunks, resampler.resample(decoded), samples, np)
                samples = _append_audio_frames(chunks, resampler.resample(None), samples, np)
            if not chunks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(chunks)
        except MediaTranscriptionError:
            raise
        except Exception as error:
            raise MediaTranscriptionError(
                "audio-invalid", "audio stream could not be decoded"
            ) from error

    async def sample(
        self,
        source: Path,
        media: MediaInfo,
        cancellation: CancellationToken,
    ) -> tuple[VisualFrame, ...]:
        if media.modality is MediaModality.IMAGE:
            if source.suffix.lower() == ".svg":
                cancellation.raise_if_cancelled()
                return await asyncio.to_thread(self._sample_svg_sync, source)
            return (VisualFrame(source, None),)
        return await asyncio.to_thread(self._sample_video_sync, source, media, cancellation)

    def _sample_svg_sync(self, source: Path) -> tuple[VisualFrame, ...]:
        root = Path(tempfile.mkdtemp(prefix="omnitensor-media-svg-"))
        self._owned_roots.add(root)
        try:
            path = root / "rendered-svg.png"
            path.write_bytes(_svg_png(source).getvalue())
            return (VisualFrame(path, None, _svg_text(source)),)
        except Exception:
            self._remove_root(root)
            raise

    def _sample_video_sync(
        self,
        source: Path,
        media: MediaInfo,
        cancellation: CancellationToken,
    ) -> tuple[VisualFrame, ...]:
        if media.modality is not MediaModality.VIDEO or media.duration_ms is None:
            raise MediaTranscriptionError("frames-invalid", "video metadata is unavailable")
        root = Path(tempfile.mkdtemp(prefix="omnitensor-media-frames-"))
        self._owned_roots.add(root)
        frames = []
        try:
            import av

            with av.open(str(source), mode="r") as container:
                stream = next(iter(container.streams.video), None)
                if stream is None:
                    raise MediaTranscriptionError("frames-invalid", "video stream is missing")
                for index, timestamp in enumerate(_sample_timestamps(media.duration_ms)):
                    cancellation.raise_if_cancelled()
                    container.seek(
                        int((timestamp / 1000) / float(stream.time_base)),
                        stream=stream,
                        any_frame=False,
                        backward=True,
                    )
                    frame = _frame_at_or_after(container.decode(stream), stream, timestamp)
                    path = root / f"frame-{index:02d}.png"
                    frame.to_image().save(path, format="PNG", optimize=False)
                    frames.append(VisualFrame(path, timestamp))
        except asyncio.CancelledError:
            self._remove_root(root)
            raise
        except Exception as error:
            self._remove_root(root)
            if isinstance(error, MediaTranscriptionError):
                raise
            raise MediaTranscriptionError(
                "frames-invalid", "video frames could not be decoded"
            ) from error
        if not frames:
            self._remove_root(root)
            raise MediaTranscriptionError("frames-invalid", "video produced no visual frames")
        return tuple(frames)

    async def discard(self, frames: Sequence[VisualFrame]) -> None:
        roots = {frame.path.parent for frame in frames if frame.path.parent in self._owned_roots}
        for root in roots:
            await asyncio.to_thread(self._remove_root, root)

    def _remove_root(self, root: Path) -> None:
        shutil.rmtree(root, ignore_errors=True)
        self._owned_roots.discard(root)


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
        audio = await self._decoder.decode_audio(source, cancellation)
        if audio.size == 0:
            return SpeechTranscript(None, ())
        return await asyncio.to_thread(self._transcribe_sync, audio, cancellation)

    def _transcribe_sync(self, audio, cancellation: CancellationToken) -> SpeechTranscript:
        with self._lease.hold():
            model, _logs = self._load()
            try:
                detected, _probabilities = model.auto_detect_language(audio)
                language = detected[0]
                segments = model.transcribe(
                    audio,
                    abort_callback=lambda: cancellation.cancelled,
                    language=language,
                    no_context=True,
                    print_progress=False,
                    print_realtime=False,
                    suppress_blank=True,
                    suppress_non_speech_tokens=True,
                )
                cancellation.raise_if_cancelled()
                duration_ms = max(1, int(audio.size) * 1000 // 16_000)
                bounded = []
                for item in segments:
                    if not isinstance(item.text, str) or not item.text.strip():
                        continue
                    start_ms = min(duration_ms, max(0, int(item.t0) * 10))
                    end_ms = min(duration_ms, max(start_ms, int(item.t1) * 10))
                    if end_ms > start_ms:
                        bounded.append(SpeechSegment(start_ms, end_ms, item.text))
                return SpeechTranscript(
                    language,
                    tuple(bounded),
                )
            finally:
                _close_whisper(model)

    def _load(self):
        try:
            import _pywhispercpp as native
            from pywhispercpp.model import Model
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
            from llama_cpp import Llama, llama_cpp
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
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


def _svg_png(source: Path) -> io.BytesIO:
    try:
        content = source.read_bytes()
        if not 0 < len(content) <= MAX_SVG_BYTES:
            raise ValueError("SVG size is invalid")
        from cairosvg.surface import PNGSurface

        width, height = _svg_output_size(content)
        rendered = PNGSurface.convert(
            bytestring=content,
            output_width=width,
            output_height=height,
            url_fetcher=_svg_url_fetcher,
        )
        return io.BytesIO(rendered)
    except Exception as error:
        raise MediaTranscriptionError(
            "media-invalid", "selected SVG could not be safely rasterized"
        ) from error


def _visual_payload(source: Path) -> bytes:
    try:
        from PIL import Image

        with Image.open(source) as image:
            image.load()
            image.thumbnail(
                (MAX_INFERENCE_DIMENSION, MAX_INFERENCE_DIMENSION),
                Image.Resampling.LANCZOS,
                reducing_gap=2,
            )
            stream = io.BytesIO()
            image.convert("RGB").save(stream, format="PNG", optimize=False)
        return stream.getvalue()
    except Exception as error:
        raise MediaTranscriptionError(
            "visual-invalid", "visual frame could not be normalized"
        ) from error


def _svg_output_size(content: bytes) -> tuple[int, int]:
    from cairosvg.helpers import size
    from cairosvg.parser import Tree

    tree = Tree(bytestring=content, unsafe=False, url_fetcher=_svg_url_fetcher)
    surface = SimpleNamespace(dpi=96, context_width=0, context_height=0, font_size=12)
    width = size(surface, tree.get("width", ""), "x")
    height = size(surface, tree.get("height", ""), "y")
    viewbox = tree.get("viewBox")
    if viewbox:
        parts = tuple(float(item) for item in re.split(r"[\s,]+", viewbox.strip()))
        if len(parts) != 4:
            raise ValueError("SVG viewBox is invalid")
        width = width or parts[2]
        height = height or parts[3]
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        raise ValueError("SVG dimensions are invalid")
    scale = min(MAX_RENDER_DIMENSION / width, MAX_RENDER_DIMENSION / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _svg_text(source: Path) -> str:
    from defusedxml import ElementTree

    root = ElementTree.fromstring(source.read_bytes())
    parts = (
        "".join(node.itertext())
        for node in root.iter()
        if node.tag.rsplit("}", 1)[-1] == "text"
    )
    return _joined_slide_text(parts)


def _svg_url_fetcher(url: str, resource_type: str):
    if not isinstance(url, str) or not url.startswith("data:image/"):
        raise ValueError("external SVG resources are forbidden")
    from cairosvg.url import fetch

    return fetch(url, resource_type)


class DocumentPageTranscriber(DocumentTranscriber):
    """Transcribe bounded PDF/TIFF pages as text plus whole-page visuals."""

    def __init__(self, vision: VisualTranscriber) -> None:
        self._vision = vision

    async def transcribe(
        self, source: Path, cancellation: CancellationToken
    ) -> tuple[VisualTranscript, ...]:
        page_count = await asyncio.to_thread(_document_page_count, source)
        root = Path(tempfile.mkdtemp(prefix="omnitensor-document-pages-"))
        try:
            results = []
            for page_number in range(1, page_count + 1):
                cancellation.raise_if_cancelled()
                text, path = await asyncio.to_thread(
                    _render_document_page, source, root, page_number
                )
                visual = await self._vision.transcribe(VisualFrame(path, None), cancellation)
                results.append(
                    VisualTranscript(
                        None,
                        _joined_slide_text((text, visual.visible_text)),
                        visual.description,
                        None,
                        page_number,
                    )
                )
            return tuple(results)
        finally:
            await asyncio.to_thread(shutil.rmtree, root, True)


def _document_page_count(source: Path) -> int:
    try:
        suffix = source.suffix.lower()
        if suffix == ".pdf":
            import pymupdf

            with pymupdf.open(source) as document:
                count = document.page_count
                if document.needs_pass:
                    raise ValueError("encrypted PDF")
        elif suffix in {".tif", ".tiff"}:
            from PIL import Image

            with Image.open(source) as image:
                count = image.n_frames
        else:  # pragma: no cover - caller suffix gate
            raise ValueError("unsupported document")
    except Exception as error:
        raise MediaTranscriptionError(
            "document-invalid", "selected document could not be decoded"
        ) from error
    if not 1 <= count <= MAX_DOCUMENT_PAGES:
        raise MediaTranscriptionError("document-invalid", "document page count is invalid")
    return count


def _render_document_page(source: Path, root: Path, page_number: int) -> tuple[str, Path]:
    path = root / f"page-{page_number:02d}.png"
    try:
        if source.suffix.lower() == ".pdf":
            text = _render_pdf_page(source, path, page_number)
        else:
            text = _render_tiff_page(source, path, page_number)
    except MediaTranscriptionError:
        raise
    except Exception as error:
        raise MediaTranscriptionError(
            "document-invalid", "document page could not be rendered"
        ) from error
    return text, path


def _render_pdf_page(source: Path, path: Path, page_number: int) -> str:
    import pymupdf

    with pymupdf.open(source) as document:
        page = document.load_page(page_number - 1)
        text = _joined_slide_text(page.get_text("text").splitlines())
        longest = max(float(page.rect.width), float(page.rect.height), 1.0)
        scale = min(2.0, MAX_RENDER_DIMENSION / longest)
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        if pixmap.width * pixmap.height > 50_000_000:
            raise MediaTranscriptionError(
                "document-invalid", "document page dimensions are invalid"
            )
        pixmap.save(path)
    return text


def _render_tiff_page(source: Path, path: Path, page_number: int) -> str:
    from PIL import Image

    with Image.open(source) as image:
        image.seek(page_number - 1)
        if image.width * image.height > 50_000_000:
            raise MediaTranscriptionError(
                "document-invalid", "document page dimensions are invalid"
            )
        rendered = image.convert("RGB")
        rendered.thumbnail((MAX_RENDER_DIMENSION, MAX_RENDER_DIMENSION))
        rendered.save(path, format="PNG", optimize=False)
    return ""


class PresentationArchiveTranscriber(PresentationTranscriber):
    """Transcribe bounded PPTX/ODP slide text and qualified embedded images."""

    def __init__(self, vision: VisualTranscriber) -> None:
        self._vision = vision

    async def transcribe(
        self, source: Path, cancellation: CancellationToken
    ) -> tuple[VisualTranscript, ...]:
        root = Path(tempfile.mkdtemp(prefix="omnitensor-presentation-"))
        try:
            slides = await asyncio.to_thread(_extract_presentation, source, root)
            results = []
            for slide in slides:
                cancellation.raise_if_cancelled()
                visible = [slide.text] if slide.text else []
                descriptions = []
                for image_path in slide.images:
                    visual = await self._vision.transcribe(
                        VisualFrame(image_path, None), cancellation
                    )
                    if visual.visible_text:
                        visible.append(visual.visible_text)
                    descriptions.append(visual.description)
                description = _slide_description(slide.text, descriptions)
                results.append(
                    VisualTranscript(
                        None,
                        _joined_slide_text(visible),
                        description,
                        slide.number,
                    )
                )
            return tuple(results)
        finally:
            await asyncio.to_thread(shutil.rmtree, root, True)


@dataclass(frozen=True, slots=True)
class _SlideBlueprint:
    number: int
    text: str
    image_entries: tuple[str, ...]


def _presentation_slide_count(source: Path) -> int:
    return len(_presentation_blueprints(source))


def _presentation_blueprints(source: Path) -> tuple[_SlideBlueprint, ...]:
    try:
        with zipfile.ZipFile(source) as archive:
            entries = _archive_entries(archive)
            suffix = source.suffix.lower()
            if suffix == ".pptx":
                slides = _pptx_blueprints(archive, entries)
            elif suffix == ".odp":
                slides = _odp_blueprints(archive, entries)
            else:  # pragma: no cover - caller suffix gate
                raise MediaTranscriptionError(
                    "presentation-unsupported", "presentation type is unsupported"
                )
    except MediaTranscriptionError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError, ValueError) as error:
        raise MediaTranscriptionError(
            "presentation-invalid", "presentation archive could not be decoded"
        ) from error
    if not 1 <= len(slides) <= MAX_PRESENTATION_SLIDES:
        raise MediaTranscriptionError("presentation-invalid", "presentation slide count is invalid")
    return slides


def _archive_entries(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if len(infos) > MAX_ARCHIVE_ENTRIES:
        raise MediaTranscriptionError(
            "presentation-invalid", "presentation archive has too many entries"
        )
    entries = {}
    expanded = 0
    for info in infos:
        path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in info.filename
            or info.flag_bits & 1
            or mode & 0o170000 == 0o120000
            or info.filename in entries
        ):
            raise MediaTranscriptionError(
                "presentation-invalid", "presentation archive entry is unsafe"
            )
        expanded += info.file_size
        if expanded > MAX_ARCHIVE_EXPANDED_BYTES:
            raise MediaTranscriptionError(
                "presentation-invalid", "presentation archive expands beyond its limit"
            )
        entries[info.filename] = info
    return entries


def _read_archive_entry(
    archive: zipfile.ZipFile,
    entries: Mapping[str, zipfile.ZipInfo],
    name: str,
    maximum: int,
) -> bytes:
    info = entries.get(name)
    if info is None or info.is_dir() or not 0 <= info.file_size <= maximum:
        raise MediaTranscriptionError(
            "presentation-invalid", "presentation archive entry is unavailable"
        )
    content = archive.read(info)
    if len(content) != info.file_size:
        raise MediaTranscriptionError(
            "presentation-invalid", "presentation archive entry was truncated"
        )
    return content


def _xml_root(content: bytes):
    try:
        from defusedxml import ElementTree

        return ElementTree.fromstring(content)
    except Exception as error:
        raise MediaTranscriptionError(
            "presentation-invalid", "presentation XML is invalid"
        ) from error


def _relationship_map(root, base_file: str) -> dict[str, str]:
    relationships = {}
    for item in root:
        identifier = item.attrib.get("Id")
        target = item.attrib.get("Target")
        if not identifier or not target or item.attrib.get("TargetMode") == "External":
            continue
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(base_file), target))
        if PurePosixPath(resolved).is_absolute() or ".." in PurePosixPath(resolved).parts:
            raise MediaTranscriptionError(
                "presentation-invalid", "presentation relationship is unsafe"
            )
        relationships[identifier] = resolved
    return relationships


def _pptx_blueprints(
    archive: zipfile.ZipFile, entries: Mapping[str, zipfile.ZipInfo]
) -> tuple[_SlideBlueprint, ...]:
    presentation_name = "ppt/presentation.xml"
    rels_name = "ppt/_rels/presentation.xml.rels"
    presentation = _xml_root(
        _read_archive_entry(archive, entries, presentation_name, MAX_SLIDE_XML_BYTES)
    )
    relationships = _relationship_map(
        _xml_root(_read_archive_entry(archive, entries, rels_name, MAX_SLIDE_XML_BYTES)),
        presentation_name,
    )
    relation_attribute = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    slide_tag = "{http://schemas.openxmlformats.org/presentationml/2006/main}sldId"
    slide_names = [
        relationships.get(item.attrib.get(relation_attribute, ""), "")
        for item in presentation.iter(slide_tag)
    ]
    if any(name not in entries for name in slide_names):
        raise MediaTranscriptionError(
            "presentation-invalid", "presentation slide relationship is unavailable"
        )
    return tuple(
        _pptx_slide_blueprint(archive, entries, name, index)
        for index, name in enumerate(slide_names, 1)
    )


def _pptx_slide_blueprint(
    archive: zipfile.ZipFile,
    entries: Mapping[str, zipfile.ZipInfo],
    slide_name: str,
    number: int,
) -> _SlideBlueprint:
    slide = _xml_root(_read_archive_entry(archive, entries, slide_name, MAX_SLIDE_XML_BYTES))
    text_tag = "{http://schemas.openxmlformats.org/drawingml/2006/main}t"
    text = _joined_slide_text(item.text or "" for item in slide.iter(text_tag))
    relation_name = posixpath.join(
        posixpath.dirname(slide_name),
        "_rels",
        f"{posixpath.basename(slide_name)}.rels",
    )
    relationships = {}
    if relation_name in entries:
        relationships = _relationship_map(
            _xml_root(_read_archive_entry(archive, entries, relation_name, MAX_SLIDE_XML_BYTES)),
            slide_name,
        )
    embed_attribute = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
    image_tag = "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"
    image_entries = tuple(
        relationships.get(item.attrib.get(embed_attribute, ""), "")
        for item in slide.iter(image_tag)
    )
    return _SlideBlueprint(number, text, _valid_image_entries(image_entries, entries))


def _odp_blueprints(
    archive: zipfile.ZipFile, entries: Mapping[str, zipfile.ZipInfo]
) -> tuple[_SlideBlueprint, ...]:
    content = _xml_root(_read_archive_entry(archive, entries, "content.xml", MAX_SLIDE_XML_BYTES))
    page_tag = "{urn:oasis:names:tc:opendocument:xmlns:drawing:1.0}page"
    paragraph_tag = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}p"
    image_tag = "{urn:oasis:names:tc:opendocument:xmlns:drawing:1.0}image"
    href_attribute = "{http://www.w3.org/1999/xlink}href"
    slides = []
    for number, page in enumerate(content.iter(page_tag), 1):
        text = _joined_slide_text("".join(item.itertext()) for item in page.iter(paragraph_tag))
        image_entries = tuple(item.attrib.get(href_attribute, "") for item in page.iter(image_tag))
        slides.append(_SlideBlueprint(number, text, _valid_image_entries(image_entries, entries)))
    return tuple(slides)


def _valid_image_entries(
    candidates: Sequence[str], entries: Mapping[str, zipfile.ZipInfo]
) -> tuple[str, ...]:
    selected = []
    for candidate in candidates:
        normalized = posixpath.normpath(candidate)
        suffix = PurePosixPath(normalized).suffix.lower()
        if normalized in entries and suffix in _PRESENTATION_IMAGE_SUFFIXES:
            selected.append(normalized)
        if len(selected) == MAX_IMAGES_PER_SLIDE:
            break
    return tuple(dict.fromkeys(selected))


def _joined_slide_text(parts) -> str:
    text = "\n".join(str(part).strip() for part in parts if str(part).strip())
    if len(text) > 16_384 or "\x00" in text:
        raise MediaTranscriptionError(
            "presentation-invalid", "presentation slide text exceeds its limit"
        )
    return text


def _extract_presentation(source: Path, root: Path) -> tuple[PresentationSlide, ...]:
    blueprints = _presentation_blueprints(source)
    slides = []
    with zipfile.ZipFile(source) as archive:
        entries = _archive_entries(archive)
        for blueprint in blueprints:
            paths = tuple(
                _write_presentation_image(archive, entries, entry, root, blueprint.number, index)
                for index, entry in enumerate(blueprint.image_entries, 1)
            )
            slides.append(PresentationSlide(blueprint.number, blueprint.text, paths))
    return tuple(slides)


def _write_presentation_image(
    archive: zipfile.ZipFile,
    entries: Mapping[str, zipfile.ZipInfo],
    entry: str,
    root: Path,
    slide_number: int,
    image_number: int,
) -> Path:
    content = _read_archive_entry(archive, entries, entry, MAX_PRESENTATION_IMAGE_BYTES)
    suffix = PurePosixPath(entry).suffix.lower()
    path = root / f"slide-{slide_number:02d}-image-{image_number:02d}{suffix}"
    path.write_bytes(content)
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.width * image.height > 50_000_000:
                raise ValueError("image is too large")
    except Exception as error:
        raise MediaTranscriptionError(
            "presentation-invalid", "presentation image is invalid"
        ) from error
    return path


def _slide_description(text: str, descriptions: Sequence[str]) -> str:
    if descriptions:
        description = "; ".join(dict.fromkeys(descriptions))
        if 0 < len(description) <= 16_384:
            return description
        raise MediaTranscriptionError(
            "presentation-invalid", "presentation image descriptions exceed their limit"
        )
    return "Presentation slide containing text." if text else "Blank presentation slide."


def _append_audio_frames(chunks, resampled, samples: int, np) -> int:
    frames = resampled if isinstance(resampled, list) else (resampled,)
    if resampled is None:
        frames = ()
    for frame in frames:
        chunk = frame.to_ndarray().reshape(-1).astype(np.float32, copy=False)
        samples += int(chunk.size)
        if samples > MAX_DECODED_AUDIO_SAMPLES:
            raise MediaTranscriptionError("media-too-long", "decoded audio exceeds ten minutes")
        chunks.append(chunk)
    return samples


def _duration_ms(container, stream) -> int:
    duration = container.duration
    if duration is not None:
        value = int(duration / 1000)
    elif stream is not None and stream.duration is not None:
        value = int(float(stream.duration * stream.time_base) * 1000)
    elif stream is not None:
        value = _demux_duration_ms(container, stream)
    else:
        raise MediaTranscriptionError("media-invalid", "media duration is unavailable")
    if not 0 < value <= MAX_DURATION_MS:
        raise MediaTranscriptionError("media-too-long", "media duration exceeds ten minutes")
    return value


def _demux_duration_ms(container, stream) -> int:
    if not callable(getattr(container, "demux", None)):
        raise MediaTranscriptionError("media-invalid", "media duration is unavailable")
    observed = 0
    for packet in container.demux(stream):
        position = packet.pts if packet.pts is not None else packet.dts
        if position is None:
            continue
        end = position + (packet.duration or 0)
        observed = max(observed, int(float(end * stream.time_base) * 1000))
        if observed > MAX_DURATION_MS:
            raise MediaTranscriptionError("media-too-long", "media duration exceeds ten minutes")
    if observed < 1:
        raise MediaTranscriptionError("media-invalid", "media duration is unavailable")
    return observed


def _sample_timestamps(duration_ms: int) -> tuple[int, ...]:
    if not 0 < duration_ms <= MAX_VIDEO_DURATION_MS:
        raise MediaTranscriptionError("frames-invalid", "video duration is invalid")
    count = min(MAX_VISUALS, max(1, math.ceil(duration_ms / FRAME_INTERVAL_MS)))
    if count == 1:
        return (0,)
    return tuple(round(index * (duration_ms - 1) / (count - 1)) for index in range(count))


def _frame_at_or_after(decoded, stream, timestamp_ms: int):
    fallback = None
    for frame in decoded:
        fallback = frame
        if frame.pts is None:
            return frame
        observed = int(float(frame.pts * stream.time_base) * 1000)
        if observed >= timestamp_ms:
            return frame
    if fallback is not None:
        return fallback
    raise MediaTranscriptionError("frames-invalid", "video produced no frame at sample time")


def _close_whisper(model) -> None:
    if model is None:
        return
    context = getattr(model, "_ctx", None)
    if context is None:
        return
    model._ctx = None
    import _pywhispercpp as native

    native.whisper_free(context)


def _qualification() -> dict:
    resource = importlib.resources.files("omnitensor_media_transcription").joinpath(
        "qualification.json"
    )
    try:
        document = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QualifiedMediaError("media qualification receipt is unreadable") from error
    required = {
        "version",
        "recordedAt",
        "qualified",
        "devices",
        "providerVersion",
        "runtimes",
        "artifacts",
        "evidenceSha256",
    }
    if not isinstance(document, dict) or set(document) != required or document.get("version") != 1:
        raise QualifiedMediaError("media qualification receipt is invalid")
    if document.get("recordedAt") != "2026-08-13":
        raise QualifiedMediaError("media qualification receipt date is invalid")
    if document.get("qualified") is not True:
        raise QualifiedMediaError("media provider has no accepted qualification")
    devices = document.get("devices")
    if (
        not isinstance(devices, dict)
        or set(devices) != {"speech", "vision"}
        or not all(isinstance(value, str) and value for value in devices.values())
    ):
        raise QualifiedMediaError("media qualification devices are invalid")
    if document.get("providerVersion") != importlib.metadata.version(
        "omnitensor-media-transcription"
    ):
        raise QualifiedMediaError("media provider differs from qualification")
    runtimes = document.get("runtimes")
    expected = {
        "llama-cpp-python": importlib.metadata.version("llama-cpp-python"),
        "pywhispercpp": importlib.metadata.version("pywhispercpp"),
    }
    if runtimes != expected:
        raise QualifiedMediaError("media runtimes differ from qualification")
    _validate_qualification_evidence(document)
    return document


def _validate_qualification_evidence(document: Mapping[str, object]) -> None:
    evidence = importlib.resources.files("omnitensor_media_transcription").joinpath(
        QUALIFICATION_EVIDENCE
    )
    try:
        raw = evidence.read_bytes()
    except OSError as error:
        raise QualifiedMediaError("media qualification evidence is unreadable") from error
    if (
        not 0 < len(raw) <= MAX_QUALIFICATION_EVIDENCE_BYTES
        or hashlib.sha256(raw).hexdigest() != document.get("evidenceSha256")
    ):
        raise QualifiedMediaError("media qualification evidence differs from receipt")


def create() -> MediaTranscriptionPlugin:
    qualification = _qualification()
    bootstrap = current_plugin_bootstrap(PLUGIN_ID)
    if bootstrap.accelerator_lease_path is None:
        raise QualifiedMediaError("GPU accelerator grant is unavailable")
    vision = bootstrap.require_artifact(VISION_ARTIFACT_ID)
    speech = bootstrap.require_artifact(SPEECH_ARTIFACT_ID)
    artifacts = qualification["artifacts"]
    observed = {
        vision.id: vision.sha256,
        f"{vision.id}/mmproj": dict(vision.companions).get(VISION_PROJECTOR),
        speech.id: speech.sha256,
    }
    if artifacts != observed:
        raise QualifiedMediaError("media artifacts differ from qualification")
    projector = vision.path.parent / VISION_PROJECTOR
    if not projector.is_file():
        raise QualifiedMediaError("qualified visual projector is unavailable")
    lease = VulkanLease(bootstrap.accelerator_lease_path)
    adapter = AvMediaAdapter()
    whisper = WhisperVulkanTranscriber(
        speech.path, adapter, lease, qualification["devices"]["speech"]
    )
    qwen = QwenVulkanVisualTranscriber(
        vision.path, projector, lease, qualification["devices"]["vision"]
    )

    async def preflight() -> None:
        await asyncio.to_thread(qwen._runtime)
        await whisper.preflight()
        await qwen.preflight()

    return MediaTranscriptionPlugin(
        identity=MediaProviderIdentity(PROVIDER_ID, "gpu"),
        probe=adapter,
        speech=whisper,
        frames=adapter,
        vision=qwen,
        presentations=PresentationArchiveTranscriber(qwen),
        documents=DocumentPageTranscriber(qwen),
        preflight=preflight,
    )


__all__ = [
    "AvMediaAdapter",
    "DocumentPageTranscriber",
    "PROVIDER_ID",
    "PresentationArchiveTranscriber",
    "QwenVulkanVisualTranscriber",
    "QualifiedMediaError",
    "VulkanLease",
    "WhisperVulkanTranscriber",
    "create",
]
