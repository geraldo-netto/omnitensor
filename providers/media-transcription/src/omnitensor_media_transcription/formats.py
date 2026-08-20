"""Bounded container, audio, image, and video format adapters."""

from __future__ import annotations

import asyncio
import io
import math
import re
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

from omnitensor.plugins.media_transcription import (
    FrameSampler,
    MediaInfo,
    MediaModality,
    MediaProbe,
    MediaTranscriptionError,
    VisualFrame,
)
from omnitensor.plugins.protocol import CancellationToken

from .documents import _document_page_count
from .presentations import _presentation_slide_count
from .text import joined_visible_text

# The one bound left on a recording's length, and it is about memory rather
# than policy: a decode pass currently holds every resampled sample at once.
# It disappears once transcription runs in windows (OMNI-0505); nothing else
# here refuses a recording for being long.
MAX_DECODED_AUDIO_SAMPLES = 600_000 * 16
FRAME_INTERVAL_MS = 15_000
_IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".svg", ".webp"})
_AUDIO_SUFFIXES = frozenset({".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"})
_VIDEO_SUFFIXES = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})
_PRESENTATION_SUFFIXES = frozenset({".odp", ".pptx"})
_DOCUMENT_SUFFIXES = frozenset({".pdf", ".tif", ".tiff"})
MAX_SVG_BYTES = 8 * 1024 * 1024
MAX_RENDER_DIMENSION = 1600
MAX_INFERENCE_DIMENSION = 768


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
        av = _require("av", "install the av media decoder")
        try:
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
        images = _require("PIL.Image", "install the Pillow image decoder")
        try:
            content = _svg_png(source) if source.suffix.lower() == ".svg" else source
            with images.open(content) as image:
                image.verify()
            with images.open(content) as image:
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
        av = _require("av", "install the av media decoder")
        np = _require("numpy", "install numpy")
        try:
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
        av = _require("av", "install the av media decoder")
        try:
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


def _svg_png(source: Path) -> io.BytesIO:
    surfaces = _require("cairosvg.surface", "install the CairoSVG rasterizer")
    try:
        content = source.read_bytes()
        if not 0 < len(content) <= MAX_SVG_BYTES:
            raise ValueError("SVG size is invalid")
        width, height = _svg_output_size(content)
        rendered = surfaces.PNGSurface.convert(
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
    images = _require("PIL.Image", "install the Pillow image decoder")
    try:
        with images.open(source) as image:
            image.load()
            image.thumbnail(
                (MAX_INFERENCE_DIMENSION, MAX_INFERENCE_DIMENSION),
                images.Resampling.LANCZOS,
                reducing_gap=2,
            )
            stream = io.BytesIO()
            image.convert("RGB").save(stream, format="PNG", optimize=False)
        return stream.getvalue()
    except Exception as error:
        raise MediaTranscriptionError(
            "visual-invalid", "visual frame could not be normalized"
        ) from error


def _require(module: str, detail: str):
    """Import a declared dependency, or refuse as a broken install.

    These imports used to sit inside the same broad `except Exception` as the
    decoding they enable, so a machine missing `av` or `cairosvg` — both
    declared dependencies of this distribution — told the person their media
    could not be decoded. That is a statement about their file, and it sent
    them to re-encode something that was never the problem.
    """
    from importlib import import_module

    try:
        return import_module(module)
    except ImportError as error:
        raise MediaTranscriptionError("media-runtime-unavailable", detail) from error


def _svg_output_size(content: bytes) -> tuple[int, int]:
    helpers = _require("cairosvg.helpers", "install the CairoSVG rasterizer")
    parser = _require("cairosvg.parser", "install the CairoSVG rasterizer")
    size = helpers.size

    tree = parser.Tree(bytestring=content, unsafe=False, url_fetcher=_svg_url_fetcher)
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
        "".join(node.itertext()) for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "text"
    )
    return joined_visible_text(parts)


def _svg_url_fetcher(url: str, resource_type: str):
    if not isinstance(url, str) or not url.startswith("data:image/"):
        raise ValueError("external SVG resources are forbidden")
    from cairosvg.url import fetch

    return fetch(url, resource_type)


def _append_audio_frames(chunks, resampled, samples: int, np) -> int:
    frames = resampled if isinstance(resampled, list) else (resampled,)
    if resampled is None:
        frames = ()
    for frame in frames:
        chunk = frame.to_ndarray().reshape(-1).astype(np.float32, copy=False)
        samples += int(chunk.size)
        if samples > MAX_DECODED_AUDIO_SAMPLES:
            raise MediaTranscriptionError(
                "media-too-long",
                "decoded audio does not fit in one pass; transcription in windows is not built yet",
            )
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
    if value < 1:
        raise MediaTranscriptionError("media-invalid", "media duration is unavailable")
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
    if observed < 1:
        raise MediaTranscriptionError("media-invalid", "media duration is unavailable")
    return observed


def _sample_timestamps(duration_ms: int) -> tuple[int, ...]:
    if duration_ms < 1:
        raise MediaTranscriptionError("frames-invalid", "video duration is invalid")
    # The cadence decides how many frames a video needs. Clamping the count to
    # twelve sampled the first 2m45s of anything longer and silently described
    # none of the rest.
    count = max(1, math.ceil(duration_ms / FRAME_INTERVAL_MS))
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


__all__ = ["AvMediaAdapter"]
