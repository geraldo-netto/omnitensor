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

from .documents import DocumentPageTranscriber, _open_document, _read_page
from .errors import MediaGpuError
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

PLUGIN_ID = "media-transcription"
PROVIDER_ID = "media-transcription-vulkan"
VISION_ARTIFACT_ID = "qwen2-5-vl-7b-instruct"
SPEECH_ARTIFACT_ID = "whisper-small-multilingual"
VISION_PROJECTOR = "mmproj.gguf"


def create() -> MediaTranscriptionPlugin:
    """Wire the plugin from what is installed on this machine.

    This used to begin by reading a receipt frozen on one desk in August 2026
    and refusing the whole plugin unless the recorded date, this
    distribution's version, the installed `llama-cpp-python` and
    `pywhispercpp` versions, the three artifact digests and the sha256 of a
    benchmark report all still matched it. Every one of those is expected to
    change — the models above all — so the receipt turned an upgrade into a
    workload that stopped working, with a message about qualification rather
    than about anything a person did. What the runtime already guarantees is
    kept: the artifacts are resolved and verified by the service that
    installed them, and each model still has to prove it loaded onto a Vulkan
    device before it answers.
    """
    bootstrap = current_plugin_bootstrap(PLUGIN_ID)
    if bootstrap.accelerator_lease_path is None:
        raise MediaGpuError("GPU accelerator grant is unavailable")
    vision = bootstrap.require_artifact(VISION_ARTIFACT_ID)
    speech = bootstrap.require_artifact(SPEECH_ARTIFACT_ID)
    projector = vision.path.parent / VISION_PROJECTOR
    if not projector.is_file():
        raise MediaGpuError("the visual projector is unavailable")
    lease = VulkanLease(bootstrap.accelerator_lease_path)
    adapter = AvMediaAdapter()
    whisper = WhisperVulkanTranscriber(speech.path, adapter, lease)
    qwen = QwenVulkanVisualTranscriber(vision.path, projector, lease)

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
    "MediaGpuError",
    "VulkanLease",
    "WhisperVulkanTranscriber",
    "create",
]
