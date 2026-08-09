"""Stable contracts for workload plugins."""

from .artifacts import (
    DEFAULT_MAX_ARTIFACT_BYTES,
    ArtifactReference,
    ArtifactResolution,
    ArtifactResolver,
    artifact_filename,
    artifact_reference_error,
)
from .protocol import (
    CancellationToken,
    JsonObject,
    JsonValue,
    PluginContext,
    PluginHealth,
    PluginHealthStatus,
    PluginLifecycle,
    PluginProgress,
    PluginRequest,
    PluginResult,
    PluginResultStatus,
    ProgressReporter,
    WorkloadPlugin,
)

__all__ = [
    "DEFAULT_MAX_ARTIFACT_BYTES",
    "ArtifactReference",
    "ArtifactResolution",
    "ArtifactResolver",
    "CancellationToken",
    "JsonObject",
    "JsonValue",
    "PluginContext",
    "PluginHealth",
    "PluginHealthStatus",
    "PluginLifecycle",
    "PluginProgress",
    "PluginRequest",
    "PluginResult",
    "PluginResultStatus",
    "ProgressReporter",
    "WorkloadPlugin",
    "artifact_filename",
    "artifact_reference_error",
]
