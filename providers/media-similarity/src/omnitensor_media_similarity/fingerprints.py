"""Streaming visual and acoustic fingerprints for one media file.

Every decoded video frame contributes to one fixed-time visual bucket.  The
bucketed representation makes two encodes with different frame rates line up
without retaining decoded frames or writing them to disk.  Audio follows the
same rule with one-second spectral windows.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from PIL import Image

VIDEO_BUCKET_MS = 500
AUDIO_SAMPLE_RATE = 16_000
AUDIO_WINDOW_SAMPLES = AUDIO_SAMPLE_RATE
HASH_WIDTH = 9
HASH_HEIGHT = 8
_MEDIA_SUFFIXES = frozenset(
    {
        ".avi",
        ".flac",
        ".m4a",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp3",
        ".mp4",
        ".ogg",
        ".opus",
        ".wav",
        ".webm",
    }
)


class MediaFingerprintError(ValueError):
    """Stable per-file failure that does not expose an absolute path."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class MediaQuality:
    """Comparable source properties read from container stream metadata."""

    width: int | None = None
    height: int | None = None
    video_bitrate: int | None = None
    audio_bitrate: int | None = None
    audio_sample_rate: int | None = None
    audio_channels: int | None = None
    lossless_audio: bool | None = None


@dataclass(frozen=True, slots=True)
class MediaFingerprint:
    relative_path: str
    file_name: str
    sha256: str
    modality: str
    duration_ms: int | None
    visual: tuple[int, ...]
    audio: tuple[int, ...]
    quality: MediaQuality = MediaQuality()


@dataclass(frozen=True, slots=True)
class FingerprintProgress:
    """One completed fingerprint work unit, without a source path."""

    stage: str
    completed: int
    total: int | None
    unit: str


FingerprintObserver = Callable[[FingerprintProgress], None]


def fingerprint_file(
    source: Path,
    relative_path: str,
    cancellation,
    observer: FingerprintObserver | None = None,
    *,
    digest: str | None = None,
) -> MediaFingerprint:
    """Decode one file once per stream and retain only compact descriptors."""
    if source.suffix.lower() not in _MEDIA_SUFFIXES:
        raise MediaFingerprintError("media-unsupported", "selected file type is unsupported")
    digest = digest or file_digest(source, cancellation, observer)
    try:
        with av.open(str(source), mode="r") as container:
            has_video = bool(container.streams.video)
            has_audio = bool(container.streams.audio)
            duration_ms = _container_duration_ms(container)
            quality = _media_quality(container)
    except Exception as error:
        raise MediaFingerprintError(
            "media-invalid", "selected media could not be opened"
        ) from error
    if not has_video and not has_audio:
        raise MediaFingerprintError("media-invalid", "selected media has no audio or video stream")
    visual, visual_duration = (
        _video_fingerprints(source, cancellation, observer) if has_video else ((), None)
    )
    audio, audio_duration = (
        _audio_fingerprints(source, cancellation, observer) if has_audio else ((), None)
    )
    observed_duration = duration_ms
    if observed_duration is None:
        observed_duration = max(
            (value for value in (visual_duration, audio_duration) if value is not None),
            default=None,
        )
    if not visual and not audio:
        raise MediaFingerprintError(
            "media-invalid", "selected media has no decodable audio or video content"
        )
    modality = "audio-video" if visual and audio else "video" if visual else "audio"
    return MediaFingerprint(
        relative_path,
        Path(relative_path).name,
        digest,
        modality,
        observed_duration,
        visual,
        audio,
        quality,
    )


def _media_quality(container) -> MediaQuality:
    video = next(iter(container.streams.video), None)
    audio = next(iter(container.streams.audio), None)
    audio_codec = "" if audio is None else str(audio.codec_context.name or "").lower()
    return MediaQuality(
        width=_positive_int(getattr(video, "width", None)),
        height=_positive_int(getattr(video, "height", None)),
        video_bitrate=_positive_int(getattr(video, "bit_rate", None)),
        audio_bitrate=_positive_int(getattr(audio, "bit_rate", None)),
        audio_sample_rate=_positive_int(getattr(audio, "sample_rate", None)),
        audio_channels=_positive_int(getattr(audio, "channels", None)),
        lossless_audio=(
            None
            if audio is None
            else audio_codec in {"alac", "flac", "wavpack"} or audio_codec.startswith("pcm_")
        ),
    )


def _positive_int(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def perceptual_hash(gray: np.ndarray) -> int:
    """Resolution-invariant 64-bit difference hash over one grayscale frame."""
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or 0 in gray.shape:
        raise ValueError("grayscale frame must be a non-empty two-dimensional array")
    image = Image.fromarray(np.asarray(gray, dtype=np.uint8), mode="L")
    values = np.asarray(_letterboxed(image, HASH_WIDTH, HASH_HEIGHT), dtype=np.uint8)
    differences = values[:, 1:] > values[:, :-1]
    result = 0
    for index, value in enumerate(differences.reshape(-1)):
        if value:
            result |= 1 << index
    return result


def spectral_hash(samples: np.ndarray) -> int:
    """Gain-resistant acoustic fingerprint for one mono PCM window."""
    values = np.asarray(samples).reshape(-1)
    if values.size < 128:
        raise ValueError("audio fingerprint window is too short")
    values = values - float(values.mean())
    peak = float(np.max(np.abs(values)))
    if peak <= 1e-8:
        return 0
    spectrum = np.abs(np.fft.rfft(values * np.hanning(values.size)))
    frequencies = np.fft.rfftfreq(values.size, 1.0 / AUDIO_SAMPLE_RATE)
    edges = np.geomspace(60.0, 7_600.0, 33)
    energies = np.array(
        [
            float(spectrum[(frequencies >= low) & (frequencies < high)].sum())
            for low, high in zip(edges[:-1], edges[1:], strict=True)
        ]
    )
    total = float(energies.sum())
    if total <= 1e-8:
        return 0
    energies = energies / total
    present = energies >= (float(energies.max()) * 0.02)
    result = 0
    for index, value in enumerate(present):
        if value:
            result |= 1 << index
    return result


def file_digest(
    source: Path, cancellation, observer: FingerprintObserver | None = None
) -> str:
    digest = hashlib.sha256()
    try:
        total = source.stat().st_size
        completed = 0
        _notify(observer, "digest", completed, total, "bytes")
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                cancellation.raise_if_cancelled()
                digest.update(chunk)
                completed += len(chunk)
                _notify(observer, "digest", completed, total, "bytes")
    except OSError as error:
        raise MediaFingerprintError(
            "media-unavailable", "selected media could not be read"
        ) from error
    return digest.hexdigest()


def _video_fingerprints(
    source: Path,
    cancellation,
    observer: FingerprintObserver | None = None,
) -> tuple[tuple[int, ...], int | None]:
    buckets: dict[int, list[int]] = {}
    last_timestamp = None
    try:
        _notify(observer, "video", 0, None, "frames")
        with av.open(str(source), mode="r") as container:
            stream = next(iter(container.streams.video), None)
            if stream is None:
                return (), None
            rotation = _rotation(stream)
            rate = float(stream.average_rate) if stream.average_rate else 0.0
            for index, frame in enumerate(container.decode(stream)):
                cancellation.raise_if_cancelled()
                timestamp = _frame_timestamp_ms(frame, stream, index, rate)
                gray = frame.to_ndarray(format="gray")
                if rotation:
                    # Container rotation is clockwise; ``rot90`` is
                    # counter-clockwise for a positive value.
                    gray = np.rot90(gray, -(rotation // 90))
                buckets.setdefault(timestamp // VIDEO_BUCKET_MS, []).append(perceptual_hash(gray))
                last_timestamp = timestamp
                _notify(observer, "video", index + 1, None, "frames")
    except MediaFingerprintError:
        raise
    except Exception as error:
        raise MediaFingerprintError("video-invalid", "video frames could not be decoded") from error
    return tuple(_majority_hash(buckets[index], 64) for index in sorted(buckets)), _ending_ms(
        last_timestamp, VIDEO_BUCKET_MS
    )


def _audio_fingerprints(
    source: Path,
    cancellation,
    observer: FingerprintObserver | None = None,
) -> tuple[tuple[int, ...], int | None]:
    windows: list[int] = []
    pending = np.zeros(0, dtype=np.float32)
    samples_seen = 0
    try:
        _notify(observer, "audio", 0, None, "samples")
        with av.open(str(source), mode="r") as container:
            stream = next(iter(container.streams.audio), None)
            if stream is None:
                return (), None
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=AUDIO_SAMPLE_RATE)
            for frame in container.decode(stream):
                cancellation.raise_if_cancelled()
                for converted in _resampled(resampler.resample(frame)):
                    chunk = np.asarray(converted.to_ndarray(), dtype=np.float32).reshape(-1)
                    pending = np.concatenate((pending, chunk))
                    pending, added = _consume_audio_windows(pending, windows, observer)
                    samples_seen += added
                    _notify(
                        observer,
                        "audio",
                        samples_seen + int(pending.size),
                        None,
                        "samples",
                    )
            for converted in _resampled(resampler.resample(None)):
                chunk = np.asarray(converted.to_ndarray(), dtype=np.float32).reshape(-1)
                pending = np.concatenate((pending, chunk))
                pending, added = _consume_audio_windows(pending, windows, observer)
                samples_seen += added
            if pending.size >= 128:
                windows.append(spectral_hash(pending))
                samples_seen += int(pending.size)
                _notify(observer, "audio", samples_seen, None, "samples")
    except MediaFingerprintError:
        raise
    except Exception as error:
        raise MediaFingerprintError("audio-invalid", "audio stream could not be decoded") from error
    duration = round(samples_seen * 1000 / AUDIO_SAMPLE_RATE) if samples_seen else None
    return tuple(windows), duration


def _consume_audio_windows(
    pending: np.ndarray,
    windows: list[int],
    observer: FingerprintObserver | None = None,
) -> tuple[np.ndarray, int]:
    consumed = 0
    while pending.size >= AUDIO_WINDOW_SAMPLES:
        windows.append(spectral_hash(pending[:AUDIO_WINDOW_SAMPLES]))
        pending = pending[AUDIO_WINDOW_SAMPLES:]
        consumed += AUDIO_WINDOW_SAMPLES
        _notify(observer, "audio", len(windows) * AUDIO_WINDOW_SAMPLES, None, "samples")
    return pending, consumed


def _notify(
    observer: FingerprintObserver | None,
    stage: str,
    completed: int,
    total: int | None,
    unit: str,
) -> None:
    if observer is not None:
        observer(FingerprintProgress(stage, completed, total, unit))


def _resampled(value) -> tuple:
    if value is None:
        return ()
    return tuple(value) if isinstance(value, (list, tuple)) else (value,)


def _letterboxed(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = min(width / image.width, height / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(size, Image.Resampling.LANCZOS)
    fill = int(np.median(np.asarray(resized)))
    canvas = Image.new("L", (width, height), color=fill)
    canvas.paste(resized, ((width - size[0]) // 2, (height - size[1]) // 2))
    return canvas


def _majority_hash(values: list[int], width: int) -> int:
    threshold = len(values) / 2
    result = 0
    for bit in range(width):
        if sum((value >> bit) & 1 for value in values) > threshold:
            result |= 1 << bit
    return result


def _frame_timestamp_ms(frame, stream, index: int, rate: float) -> int:
    if frame.pts is not None and stream.time_base is not None:
        return max(0, round(float(frame.pts * stream.time_base) * 1000))
    if rate > 0:
        return round(index * 1000 / rate)
    return index * VIDEO_BUCKET_MS


def _rotation(stream) -> int:
    try:
        value = int(stream.metadata.get("rotate", "0")) % 360
    except (TypeError, ValueError):
        return 0
    return value if value in {90, 180, 270} else 0


def _container_duration_ms(container) -> int | None:
    if container.duration is None:
        return None
    return max(1, round(container.duration / 1000))


def _ending_ms(last_timestamp: int | None, step: int) -> int | None:
    return None if last_timestamp is None else last_timestamp + step


__all__ = [
    "MediaFingerprint",
    "MediaFingerprintError",
    "MediaQuality",
    "FingerprintProgress",
    "file_digest",
    "fingerprint_file",
    "perceptual_hash",
    "spectral_hash",
]
