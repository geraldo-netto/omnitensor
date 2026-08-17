"""Bounded orchestration for accelerator-only media transcription."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..registry import validate_document
from ..sdk import (
    ManagedPlugin,
    PluginCancelledError,
    PluginHealth,
    PluginHealthStatus,
    PluginProgress,
    SDKContractError,
    cancelled_result,
    failed_result,
    succeeded_result,
)
from .protocol import CancellationToken, PluginRequest, PluginResult, ProgressReporter

PLUGIN_ID = "media-transcription"
READ_PERMISSION = "files:read-selected"
MAX_SOURCE_BYTES = 128 * 1024 * 1024
MAX_DURATION_MS = 600_000
MAX_VIDEO_DURATION_MS = 300_000
MAX_IMAGE_PIXELS = 50_000_000
MAX_PRESENTATION_SLIDES = 64
MAX_DOCUMENT_PAGES = 64
MAX_LANGUAGE_CHARACTERS = 35
_LANGUAGE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")


class MediaModality(StrEnum):
    AUDIO = "audio"
    DOCUMENT = "document"
    IMAGE = "image"
    VIDEO = "video"
    PRESENTATION = "presentation"


@dataclass(frozen=True, slots=True)
class MediaInfo:
    modality: MediaModality
    duration_ms: int | None
    width: int | None
    height: int | None
    has_audio: bool
    slide_count: int | None = None
    page_count: int | None = None


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True, slots=True)
class SpeechTranscript:
    language: str | None
    segments: tuple[SpeechSegment, ...]


@dataclass(frozen=True, slots=True)
class VisualFrame:
    path: Path
    timestamp_ms: int | None
    structured_text: str = ""


@dataclass(frozen=True, slots=True)
class VisualTranscript:
    timestamp_ms: int | None
    visible_text: str
    description: str
    slide_number: int | None = None
    page_number: int | None = None


@dataclass(frozen=True, slots=True)
class MediaProviderIdentity:
    provider_id: str
    accelerator: str


class MediaTranscriptionError(ValueError):
    """Stable refusal that contains no selected content or absolute path."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@runtime_checkable
class MediaProbe(Protocol):
    async def inspect(self, source: Path) -> MediaInfo: ...


@runtime_checkable
class SpeechTranscriber(Protocol):
    async def transcribe(
        self, source: Path, cancellation: CancellationToken
    ) -> SpeechTranscript: ...


@runtime_checkable
class FrameSampler(Protocol):
    async def sample(
        self, source: Path, media: MediaInfo, cancellation: CancellationToken
    ) -> tuple[VisualFrame, ...]: ...

    async def discard(self, frames: Sequence[VisualFrame]) -> None: ...


@runtime_checkable
class VisualTranscriber(Protocol):
    async def transcribe(
        self, frame: VisualFrame, cancellation: CancellationToken
    ) -> VisualTranscript: ...

    async def release(self) -> None: ...


@runtime_checkable
class PresentationTranscriber(Protocol):
    async def transcribe(
        self, source: Path, cancellation: CancellationToken
    ) -> tuple[VisualTranscript, ...]: ...


@runtime_checkable
class DocumentTranscriber(Protocol):
    async def transcribe(
        self, source: Path, cancellation: CancellationToken
    ) -> tuple[VisualTranscript, ...]: ...


def _require_port(candidate: object, protocol: type[Protocol], label: str):
    if not isinstance(candidate, protocol):
        raise TypeError(f"{label} port is required")
    return candidate


def _validate_identity(identity: MediaProviderIdentity) -> MediaProviderIdentity:
    if (
        not isinstance(identity, MediaProviderIdentity)
        or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", identity.provider_id) is None
        or identity.accelerator not in {"gpu", "npu"}
    ):
        raise MediaTranscriptionError("provider-invalid", "provider identity is invalid")
    return identity


class MediaTranscriptionPlugin(ManagedPlugin):
    """One selected media file in; one immutable text representation out."""

    plugin_id = PLUGIN_ID

    def __init__(
        self,
        *,
        identity: MediaProviderIdentity,
        probe: MediaProbe,
        speech: SpeechTranscriber,
        frames: FrameSampler,
        vision: VisualTranscriber,
        presentations: PresentationTranscriber,
        documents: DocumentTranscriber,
        preflight: Callable[[], object] | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        super().__init__()
        self._identity = _validate_identity(identity)
        self._probe = _require_port(probe, MediaProbe, "Media probe")
        self._speech = _require_port(speech, SpeechTranscriber, "Speech transcription")
        self._frames = _require_port(frames, FrameSampler, "Frame sampler")
        self._vision = _require_port(vision, VisualTranscriber, "Visual transcription")
        self._presentations = _require_port(
            presentations, PresentationTranscriber, "Presentation transcription"
        )
        self._documents = _require_port(documents, DocumentTranscriber, "Document transcription")
        if preflight is not None and not callable(preflight):
            raise TypeError("Media provider preflight must be callable")
        if clock_ms is not None and not callable(clock_ms):
            raise TypeError("Media transcription clock must be callable")
        self._preflight = preflight or (lambda: None)
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    async def on_start(self) -> None:
        self.permissions.require(READ_PERMISSION)
        outcome = self._preflight()
        if hasattr(outcome, "__await__"):
            await outcome

    async def on_health(self) -> PluginHealth:
        return PluginHealth(
            PluginHealthStatus.READY,
            "ready; audio, image, video, presentation, and page transcription",
            self._clock_ms(),
        )

    async def execute(
        self,
        request: PluginRequest,
        cancellation: CancellationToken,
        progress: ProgressReporter,
    ) -> PluginResult:
        frames: tuple[VisualFrame, ...] = ()
        try:
            source = self._selected_source(request)
            await self._report(request, progress, "inspect", 0.05)
            media = _validated_media(await self._probe.inspect(source))
            cancellation.raise_if_cancelled()
            speech = SpeechTranscript(None, ())
            if media.modality is MediaModality.AUDIO or media.has_audio:
                await self._report(request, progress, "speech", 0.15)
                speech = _validated_speech(
                    await self._speech.transcribe(source, cancellation), media
                )
            visuals: tuple[VisualTranscript, ...] = ()
            if media.modality is MediaModality.PRESENTATION:
                await self._report(request, progress, "slides", 0.55)
                visuals = tuple(
                    _validated_visual(item)
                    for item in await self._presentations.transcribe(source, cancellation)
                )
            elif media.modality is MediaModality.DOCUMENT:
                await self._report(request, progress, "pages", 0.55)
                visuals = tuple(
                    _validated_visual(item)
                    for item in await self._documents.transcribe(source, cancellation)
                )
            elif media.modality is not MediaModality.AUDIO:
                await self._report(request, progress, "frames", 0.55)
                frames = await self._frames.sample(source, media, cancellation)
                visuals = await self._transcribe_frames(request, frames, cancellation, progress)
            visuals = _validated_visuals(visuals, media)
            output = media_result(
                request.job_id,
                source,
                media,
                self._identity,
                speech,
                visuals,
            )
            await self._report(request, progress, "terminal", 1.0)
            return succeeded_result(
                request,
                output,
                completed_at_ms=self._clock_ms(),
                detail="media transcription ready",
            )
        except (PluginCancelledError, asyncio.CancelledError):
            return cancelled_result(
                request,
                "media transcription cancelled",
                completed_at_ms=self._clock_ms(),
            )
        except (MediaTranscriptionError, SDKContractError) as error:
            return failed_result(
                request,
                str(getattr(error, "code", "media-transcription-failed")),
                completed_at_ms=self._clock_ms(),
            )
        except Exception:
            return failed_result(
                request,
                "media-transcription-failed",
                completed_at_ms=self._clock_ms(),
            )
        finally:
            with contextlib.suppress(Exception):
                await self._frames.discard(frames)
            with contextlib.suppress(Exception):
                await self._vision.release()

    def _selected_source(self, request: PluginRequest) -> Path:
        if not isinstance(request, PluginRequest) or request.plugin_id != PLUGIN_ID:
            raise MediaTranscriptionError("request-invalid", "request names another plugin")
        if request.trigger != "manual":
            raise MediaTranscriptionError("request-invalid", "media transcription is manual only")
        self.permissions.require(READ_PERMISSION)
        sources = request.payload.get("sources")
        if not isinstance(sources, list) or len(sources) != 1 or not isinstance(sources[0], str):
            raise MediaTranscriptionError("source-invalid", "select exactly one media file")
        source = Path(sources[0])
        try:
            size = source.stat().st_size
        except OSError as error:
            raise MediaTranscriptionError(
                "source-unavailable", "selected media is unavailable"
            ) from error
        if not source.is_absolute() or not source.is_file() or not 0 < size <= MAX_SOURCE_BYTES:
            raise MediaTranscriptionError(
                "source-invalid", "selected media must be one bounded regular file"
            )
        return source

    async def _transcribe_frames(
        self,
        request: PluginRequest,
        frames: tuple[VisualFrame, ...],
        cancellation: CancellationToken,
        progress: ProgressReporter,
    ) -> tuple[VisualTranscript, ...]:
        # No ceiling on the sample count. The frame count is the sampling
        # cadence applied to the video's length; capping it at twelve dropped
        # every frame past 2m45s and described a long video from its opening.
        if not frames:
            raise MediaTranscriptionError("frames-invalid", "visual sample count is invalid")
        result = []
        for index, frame in enumerate(frames):
            cancellation.raise_if_cancelled()
            visual = await self._vision.transcribe(frame, cancellation)
            result.append(_validated_visual(visual))
            await self._report(
                request,
                progress,
                "vision",
                0.6 + (0.35 * (index + 1) / len(frames)),
            )
        return tuple(result)

    async def _report(
        self,
        request: PluginRequest,
        progress: ProgressReporter,
        stage: str,
        fraction: float,
    ) -> None:
        await progress.report(PluginProgress(request.job_id, stage, fraction, "", self._clock_ms()))


def _validated_media(media: object) -> MediaInfo:
    if not isinstance(media, MediaInfo):
        raise MediaTranscriptionError("media-invalid", "media probe returned an invalid result")
    _validate_media_duration(media)
    _validate_position_metadata(media)
    if media.modality is MediaModality.IMAGE:
        _validate_image_media(media)
    elif media.modality is MediaModality.PRESENTATION:
        _validate_presentation_media(media)
    elif media.modality is MediaModality.DOCUMENT:
        _validate_document_media(media)
    elif media.duration_ms is None:
        raise MediaTranscriptionError("media-invalid", "timed media requires a duration")
    if media.modality is MediaModality.AUDIO and not media.has_audio:
        raise MediaTranscriptionError("media-invalid", "audio stream is missing")
    return media


def _validate_position_metadata(media: MediaInfo) -> None:
    if media.modality is not MediaModality.PRESENTATION and media.slide_count is not None:
        raise MediaTranscriptionError("media-invalid", "slide metadata is inconsistent")
    if media.modality is not MediaModality.DOCUMENT and media.page_count is not None:
        raise MediaTranscriptionError("media-invalid", "page metadata is inconsistent")


def _validate_media_duration(media: MediaInfo) -> None:
    duration = media.duration_ms
    if duration is not None and (
        isinstance(duration, bool)
        or not isinstance(duration, int)
        or not 1 <= duration <= MAX_DURATION_MS
    ):
        raise MediaTranscriptionError("media-too-long", "media duration exceeds the bounded limit")
    if media.modality is MediaModality.VIDEO and (
        duration is None or duration > MAX_VIDEO_DURATION_MS
    ):
        raise MediaTranscriptionError("media-too-long", "video duration exceeds five minutes")


def _validate_image_media(media: MediaInfo) -> None:
    _validate_image_dimensions(media.width, media.height)
    if media.duration_ms is not None or media.has_audio:
        raise MediaTranscriptionError("media-invalid", "image media metadata is inconsistent")


def _validate_presentation_media(media: MediaInfo) -> None:
    slide_count = media.slide_count
    if (
        media.duration_ms is not None
        or media.has_audio
        or media.width is not None
        or media.height is not None
        or isinstance(slide_count, bool)
        or not isinstance(slide_count, int)
        or not 1 <= slide_count <= MAX_PRESENTATION_SLIDES
    ):
        raise MediaTranscriptionError(
            "presentation-invalid", "presentation metadata is inconsistent"
        )


def _validate_document_media(media: MediaInfo) -> None:
    page_count = media.page_count
    if (
        media.duration_ms is not None
        or media.has_audio
        or media.width is not None
        or media.height is not None
        or isinstance(page_count, bool)
        or not isinstance(page_count, int)
        or not 1 <= page_count <= MAX_DOCUMENT_PAGES
    ):
        raise MediaTranscriptionError("document-invalid", "document metadata is inconsistent")


def _validate_image_dimensions(width: object, height: object) -> None:
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width < 1
        or height < 1
        or width * height > MAX_IMAGE_PIXELS
    ):
        raise MediaTranscriptionError("image-invalid", "image dimensions are invalid")


def _bounded_content(value: object, *, required: bool) -> str:
    # Only the lower bound is ours to set. A dense slide carries more than
    # 16,384 characters of visible text and a monologue more than 4,096 in one
    # segment; ceilings there failed the whole job as "transcript-invalid"
    # rather than telling anybody what was said.
    if not isinstance(value, str):
        raise MediaTranscriptionError("transcript-invalid", "transcript content must be text")
    text = value.strip()
    if len(text) < (1 if required else 0) or "\x00" in text:
        raise MediaTranscriptionError("transcript-invalid", "transcript content is invalid")
    return text


def _validated_speech(value: object, media: MediaInfo) -> SpeechTranscript:
    # No segment ceiling. A three-hour recording has more than 2,048 segments,
    # and calling its transcript "invalid" discards everything said after about
    # forty minutes.
    if not isinstance(value, SpeechTranscript):
        raise MediaTranscriptionError("speech-invalid", "speech transcript is invalid")
    language = value.language
    if language is not None and (
        not isinstance(language, str)
        or len(language) > MAX_LANGUAGE_CHARACTERS
        or _LANGUAGE.fullmatch(language) is None
    ):
        raise MediaTranscriptionError("speech-invalid", "speech language is invalid")
    duration = media.duration_ms or 0
    segments = []
    previous_end = 0
    for segment in value.segments:
        if (
            not isinstance(segment, SpeechSegment)
            or isinstance(segment.start_ms, bool)
            or isinstance(segment.end_ms, bool)
            or not isinstance(segment.start_ms, int)
            or not isinstance(segment.end_ms, int)
            or segment.start_ms < previous_end
            or not segment.start_ms < segment.end_ms <= duration
        ):
            raise MediaTranscriptionError("speech-invalid", "speech segment timing is invalid")
        segments.append(
            SpeechSegment(
                segment.start_ms,
                segment.end_ms,
                _bounded_content(segment.text, required=True),
            )
        )
        previous_end = segment.end_ms
    return SpeechTranscript(language, tuple(segments))


def _validated_visual(value: object) -> VisualTranscript:
    if not isinstance(value, VisualTranscript) or (
        value.timestamp_ms is not None
        and (
            isinstance(value.timestamp_ms, bool)
            or not isinstance(value.timestamp_ms, int)
            or not 0 <= value.timestamp_ms <= MAX_VIDEO_DURATION_MS
        )
        or (
            value.slide_number is not None
            and (
                isinstance(value.slide_number, bool)
                or not isinstance(value.slide_number, int)
                or not 1 <= value.slide_number <= MAX_PRESENTATION_SLIDES
            )
        )
        or (
            value.page_number is not None
            and (
                isinstance(value.page_number, bool)
                or not isinstance(value.page_number, int)
                or not 1 <= value.page_number <= MAX_DOCUMENT_PAGES
            )
        )
    ):
        raise MediaTranscriptionError("visual-invalid", "visual transcript is invalid")
    return VisualTranscript(
        value.timestamp_ms,
        _bounded_content(value.visible_text, required=False),
        _bounded_content(value.description, required=True),
        value.slide_number,
        value.page_number,
    )


def _validated_visuals(
    visuals: Sequence[VisualTranscript], media: MediaInfo
) -> tuple[VisualTranscript, ...]:
    items = tuple(visuals)
    if media.modality is MediaModality.AUDIO and items:
        raise MediaTranscriptionError("visual-invalid", "audio cannot contain visual results")
    if media.modality is MediaModality.IMAGE and (
        len(items) != 1
        or items[0].timestamp_ms is not None
        or items[0].slide_number is not None
        or items[0].page_number is not None
    ):
        raise MediaTranscriptionError("visual-invalid", "image visual result is invalid")
    if media.modality is MediaModality.VIDEO and not _valid_video_visuals(items):
        raise MediaTranscriptionError("visual-invalid", "video visual results are invalid")
    if media.modality is MediaModality.PRESENTATION and not _valid_presentation_visuals(
        items, media.slide_count
    ):
        raise MediaTranscriptionError("visual-invalid", "presentation visual results are invalid")
    if media.modality is MediaModality.DOCUMENT and not _valid_document_visuals(
        items, media.page_count
    ):
        raise MediaTranscriptionError("visual-invalid", "document visual results are invalid")
    return items


def _valid_video_visuals(items: tuple[VisualTranscript, ...]) -> bool:
    timestamps = [item.timestamp_ms for item in items]
    return (
        len(items) >= 1
        and all(item.slide_number is None for item in items)
        and all(item.page_number is None for item in items)
        and all(timestamp is not None for timestamp in timestamps)
        and timestamps == sorted(timestamps)
    )


def _valid_presentation_visuals(
    items: tuple[VisualTranscript, ...], slide_count: int | None
) -> bool:
    return (
        len(items) == slide_count
        and 1 <= len(items) <= MAX_PRESENTATION_SLIDES
        and all(item.timestamp_ms is None for item in items)
        and all(item.page_number is None for item in items)
        and [item.slide_number for item in items] == list(range(1, len(items) + 1))
    )


def _valid_document_visuals(items: tuple[VisualTranscript, ...], page_count: int | None) -> bool:
    return (
        len(items) == page_count
        and 1 <= len(items) <= MAX_DOCUMENT_PAGES
        and all(item.timestamp_ms is None for item in items)
        and all(item.slide_number is None for item in items)
        and [item.page_number for item in items] == list(range(1, len(items) + 1))
    )


def media_result(
    request_id: str,
    source: Path,
    media: MediaInfo,
    identity: MediaProviderIdentity,
    speech: SpeechTranscript,
    visuals: Sequence[VisualTranscript],
) -> dict:
    """Build and schema-validate the only public media result shape."""
    try:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as error:
        raise MediaTranscriptionError(
            "source-unavailable", "selected media became unavailable"
        ) from error
    document = {
        "version": 1,
        "requestId": request_id,
        "providerId": identity.provider_id,
        "accelerator": identity.accelerator,
        "source": {
            "fileName": source.name,
            "sourceSha256": digest,
            "modality": str(media.modality),
            "durationMs": media.duration_ms,
        },
        "speech": {
            "language": speech.language,
            "segments": [
                {"startMs": item.start_ms, "endMs": item.end_ms, "text": item.text}
                for item in speech.segments
            ],
        },
        "visuals": [
            {
                "timestampMs": item.timestamp_ms,
                "visibleText": item.visible_text,
                "description": item.description,
                "slideNumber": item.slide_number,
                "pageNumber": item.page_number,
            }
            for item in visuals
        ],
    }
    violations = validate_document("media-transcription-result.schema.json", document)
    if violations:
        raise MediaTranscriptionError("result-invalid", violations[0])
    return document


__all__ = [
    "DocumentTranscriber",
    "FrameSampler",
    "MAX_DOCUMENT_PAGES",
    "MAX_DURATION_MS",
    "MAX_IMAGE_PIXELS",
    "MAX_SOURCE_BYTES",
    "MAX_PRESENTATION_SLIDES",
    "MAX_VIDEO_DURATION_MS",
    "MediaInfo",
    "MediaModality",
    "MediaProbe",
    "MediaProviderIdentity",
    "MediaTranscriptionError",
    "MediaTranscriptionPlugin",
    "PresentationTranscriber",
    "SpeechSegment",
    "SpeechTranscript",
    "SpeechTranscriber",
    "VisualFrame",
    "VisualTranscript",
    "VisualTranscriber",
    "media_result",
]
