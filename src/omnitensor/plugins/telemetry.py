"""Bounded per-plugin runtime telemetry without transition history."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import StrEnum
from threading import Lock

from .pipeline import PipelineStage

MAX_TELEMETRY_PLUGINS = 128
MAX_PLUGIN_QUEUE_DEPTH = 1_000_000
MAX_PLUGIN_ACTIVE_JOBS = 1024
MAX_TELEMETRY_COUNTER = 1_000_000_000
MAX_TELEMETRY_DETAIL_CHARS = 240
MAX_TELEMETRY_TIMESTAMP_MS = 9_007_199_254_740_991
PLUGIN_TELEMETRY_CONTRACT_VERSION = 1
_PLUGIN_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_ERROR_CODE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class PluginTelemetryHealth(StrEnum):
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    STOPPED = "stopped"


class ArtifactReadiness(StrEnum):
    UNKNOWN = "unknown"
    RESOLVING = "resolving"
    READY = "ready"
    MISSING = "missing"
    REJECTED = "rejected"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True, slots=True)
class PluginTelemetry:
    plugin_id: str
    health: PluginTelemetryHealth
    current_stage: PipelineStage | None
    artifact_readiness: ArtifactReadiness
    queued_jobs: int
    active_jobs: int
    last_success_at_ms: int | None
    last_error_code: str | None
    last_error_detail: str
    last_error_at_ms: int | None
    deadline_exceeded: int
    retries: int
    cancellations: int
    drops: int
    successes: int
    failures: int


def _initial(plugin_id: str) -> PluginTelemetry:
    return PluginTelemetry(
        plugin_id,
        PluginTelemetryHealth.INITIALIZING,
        None,
        ArtifactReadiness.UNKNOWN,
        0,
        0,
        None,
        None,
        "",
        None,
        0,
        0,
        0,
        0,
        0,
        0,
    )


class PluginTelemetryRegistry:
    """Maintain one immutable bounded snapshot per installed plugin."""

    def __init__(self, *, max_plugins: int = MAX_TELEMETRY_PLUGINS) -> None:
        if (
            isinstance(max_plugins, bool)
            or not isinstance(max_plugins, int)
            or not 1 <= max_plugins <= MAX_TELEMETRY_PLUGINS
        ):
            raise ValueError(f"max_plugins must be between 1 and {MAX_TELEMETRY_PLUGINS}")
        self._max_plugins = max_plugins
        self._items: dict[str, PluginTelemetry] = {}
        self._lock = Lock()

    def register(self, plugin_id: str) -> PluginTelemetry:
        _validate_plugin_id(plugin_id)
        with self._lock:
            existing = self._items.get(plugin_id)
            if existing is not None:
                return existing
            if len(self._items) >= self._max_plugins:
                raise ValueError("plugin telemetry capacity is exhausted")
            snapshot = _initial(plugin_id)
            self._items[plugin_id] = snapshot
            return snapshot

    def unregister(self, plugin_id: str) -> bool:
        _validate_plugin_id(plugin_id)
        with self._lock:
            return self._items.pop(plugin_id, None) is not None

    def snapshot(self, plugin_id: str) -> PluginTelemetry:
        _validate_plugin_id(plugin_id)
        with self._lock:
            return self._required(plugin_id)

    def snapshots(self) -> tuple[PluginTelemetry, ...]:
        with self._lock:
            return tuple(self._items[key] for key in sorted(self._items))

    def documents(self) -> list[dict[str, object]]:
        """Return the fixed, public snapshot contract without private detail."""
        return [_telemetry_document(snapshot) for snapshot in self.snapshots()]

    def set_health(self, plugin_id: str, health: PluginTelemetryHealth) -> PluginTelemetry:
        if not isinstance(health, PluginTelemetryHealth):
            raise TypeError("health must be PluginTelemetryHealth")
        return self._update(plugin_id, health=health)

    def set_artifact_readiness(
        self, plugin_id: str, readiness: ArtifactReadiness
    ) -> PluginTelemetry:
        if not isinstance(readiness, ArtifactReadiness):
            raise TypeError("readiness must be ArtifactReadiness")
        return self._update(plugin_id, artifact_readiness=readiness)

    def queue_job(self, plugin_id: str) -> PluginTelemetry:
        with self._lock:
            current = self._required_valid(plugin_id)
            if current.queued_jobs >= MAX_PLUGIN_QUEUE_DEPTH:
                raise ValueError("plugin telemetry queue depth is exhausted")
            return self._store(current, queued_jobs=current.queued_jobs + 1)

    def start_job(self, plugin_id: str) -> PluginTelemetry:
        with self._lock:
            current = self._required_valid(plugin_id)
            if current.queued_jobs <= 0:
                raise ValueError("plugin has no queued job to start")
            if current.active_jobs >= MAX_PLUGIN_ACTIVE_JOBS:
                raise ValueError("plugin active-job telemetry is exhausted")
            return self._store(
                current,
                queued_jobs=current.queued_jobs - 1,
                active_jobs=current.active_jobs + 1,
                current_stage=PipelineStage.COLLECT,
            )

    def advance(self, plugin_id: str, stage: PipelineStage) -> PluginTelemetry:
        if not isinstance(stage, PipelineStage) or stage in {
            PipelineStage.QUEUED,
            PipelineStage.TERMINAL,
        }:
            raise ValueError("active telemetry stage must be runnable")
        with self._lock:
            current = self._required_valid(plugin_id)
            _require_active(current)
            return self._store(current, current_stage=stage)

    def succeed(self, plugin_id: str, *, completed_at_ms: int) -> PluginTelemetry:
        _validate_timestamp(completed_at_ms)
        with self._lock:
            current = self._required_valid(plugin_id)
            _require_active(current)
            return self._store(
                current,
                active_jobs=current.active_jobs - 1,
                current_stage=PipelineStage.TERMINAL,
                health=PluginTelemetryHealth.HEALTHY,
                last_success_at_ms=completed_at_ms,
                successes=_increment(current.successes),
            )

    def fail(
        self,
        plugin_id: str,
        code: str,
        detail: str,
        *,
        completed_at_ms: int,
    ) -> PluginTelemetry:
        _validate_error(code, detail)
        _validate_timestamp(completed_at_ms)
        with self._lock:
            current = self._required_valid(plugin_id)
            _require_active(current)
            return self._store(
                current,
                active_jobs=current.active_jobs - 1,
                current_stage=PipelineStage.TERMINAL,
                health=PluginTelemetryHealth.DEGRADED,
                last_error_code=code,
                last_error_detail=detail[:MAX_TELEMETRY_DETAIL_CHARS],
                last_error_at_ms=completed_at_ms,
                failures=_increment(current.failures),
            )

    def cancel(self, plugin_id: str) -> PluginTelemetry:
        with self._lock:
            current = self._required_valid(plugin_id)
            if current.active_jobs > 0:
                changes = {
                    "active_jobs": current.active_jobs - 1,
                    "current_stage": PipelineStage.TERMINAL,
                }
            elif current.queued_jobs > 0:
                changes = {"queued_jobs": current.queued_jobs - 1}
            else:
                raise ValueError("plugin has no queued or active job to cancel")
            return self._store(
                current,
                cancellations=_increment(current.cancellations),
                **changes,
            )

    def drop(self, plugin_id: str) -> PluginTelemetry:
        with self._lock:
            current = self._required_valid(plugin_id)
            if current.queued_jobs <= 0:
                raise ValueError("plugin has no queued job to drop")
            return self._store(
                current,
                queued_jobs=current.queued_jobs - 1,
                drops=_increment(current.drops),
            )

    def deadline(
        self,
        plugin_id: str,
        detail: str,
        *,
        observed_at_ms: int,
    ) -> PluginTelemetry:
        _validate_error("deadline-exceeded", detail)
        _validate_timestamp(observed_at_ms)
        with self._lock:
            current = self._required_valid(plugin_id)
            _require_active(current)
            return self._store(
                current,
                health=PluginTelemetryHealth.DEGRADED,
                last_error_code="deadline-exceeded",
                last_error_detail=detail[:MAX_TELEMETRY_DETAIL_CHARS],
                last_error_at_ms=observed_at_ms,
                deadline_exceeded=_increment(current.deadline_exceeded),
            )

    def retry(self, plugin_id: str) -> PluginTelemetry:
        with self._lock:
            current = self._required_valid(plugin_id)
            _require_active(current)
            return self._store(current, retries=_increment(current.retries))

    def recover_runtime(self, plugin_id: str) -> PluginTelemetry:
        """Clear stale queue/activity after a process or service restart."""
        return self._update(
            plugin_id,
            queued_jobs=0,
            active_jobs=0,
            current_stage=None,
        )

    def _update(self, plugin_id: str, **changes: object) -> PluginTelemetry:
        with self._lock:
            return self._store(self._required_valid(plugin_id), **changes)

    def _store(self, current: PluginTelemetry, **changes: object) -> PluginTelemetry:
        updated = replace(current, **changes)
        self._items[current.plugin_id] = updated
        return updated

    def _required_valid(self, plugin_id: str) -> PluginTelemetry:
        _validate_plugin_id(plugin_id)
        return self._required(plugin_id)

    def _required(self, plugin_id: str) -> PluginTelemetry:
        try:
            return self._items[plugin_id]
        except KeyError as error:
            raise KeyError(f"plugin telemetry is not registered: {plugin_id}") from error


def _require_active(snapshot: PluginTelemetry) -> None:
    if snapshot.active_jobs <= 0:
        raise ValueError("plugin has no active job")


def _increment(value: int) -> int:
    return min(MAX_TELEMETRY_COUNTER, value + 1)


def _telemetry_document(snapshot: PluginTelemetry) -> dict[str, object]:
    return {
        "id": snapshot.plugin_id,
        "health": snapshot.health.value,
        "stage": snapshot.current_stage.value if snapshot.current_stage is not None else None,
        "artifactReadiness": snapshot.artifact_readiness.value,
        "queuedJobs": snapshot.queued_jobs,
        "activeJobs": snapshot.active_jobs,
        "lastSuccessAt": snapshot.last_success_at_ms,
        "lastErrorCode": snapshot.last_error_code,
        "lastErrorAt": snapshot.last_error_at_ms,
        "deadlineExceeded": snapshot.deadline_exceeded,
        "retries": snapshot.retries,
        "cancellations": snapshot.cancellations,
        "drops": snapshot.drops,
        "successes": snapshot.successes,
        "failures": snapshot.failures,
    }


def _validate_plugin_id(plugin_id: object) -> None:
    if (
        not isinstance(plugin_id, str)
        or not 1 <= len(plugin_id) <= 80
        or _PLUGIN_ID.fullmatch(plugin_id) is None
    ):
        raise ValueError("plugin_id is invalid")


def _validate_error(code: object, detail: object) -> None:
    if not isinstance(code, str) or not 1 <= len(code) <= 80 or _ERROR_CODE.fullmatch(code) is None:
        raise ValueError("telemetry error code is invalid")
    if not isinstance(detail, str):
        raise TypeError("telemetry error detail must be a string")


def _validate_timestamp(value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= MAX_TELEMETRY_TIMESTAMP_MS
    ):
        raise ValueError("telemetry timestamp must be a non-negative JavaScript-safe integer")
