"""Composition facade for qualified multimedia transcription adapters."""

# ruff: noqa: F401 -- compatibility facade intentionally re-exports legacy names.

from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.resources

from omnitensor.plugins.media_transcription import (
    MediaProviderIdentity,
    MediaTranscriptionPlugin,
    SpeechTranscript,
)
from omnitensor.sdk import current_plugin_bootstrap

from .documents import (
    DocumentPageTranscriber,
    _document_page_count,
    _render_document_page,
    _render_pdf_page,
)
from .errors import QualifiedMediaError
from .formats import (
    AUDIO_SAMPLE_RATE,
    AUDIO_WINDOW_SAMPLES,
    MAX_RENDER_DIMENSION,
    AvMediaAdapter,
    _demux_duration_ms,
    _duration_ms,
    _frame_at_or_after,
    _resampled_chunks,
    _sample_timestamps,
    _svg_output_size,
    _WindowBuffer,
)
from .models import (
    QwenVulkanVisualTranscriber,
    VulkanLease,
    WhisperVulkanTranscriber,
    _close_whisper,
)
from .presentations import (
    PresentationArchiveTranscriber,
    _joined_slide_text,
    _slide_description,
)
from .qualification import QUALIFICATION_EVIDENCE, load_qualification

PLUGIN_ID = "media-transcription"
PROVIDER_ID = "media-transcription-vulkan"
VISION_ARTIFACT_ID = "qwen2-5-vl-7b-instruct"
SPEECH_ARTIFACT_ID = "whisper-small-multilingual"
VISION_PROJECTOR = "mmproj.gguf"


def _qualification() -> dict:
    """Compatibility seam for tests and frozen composition wiring."""
    return load_qualification()


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
