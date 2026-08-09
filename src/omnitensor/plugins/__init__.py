"""Stable contracts for workload plugins."""

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
]
