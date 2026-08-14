"""Transport-neutral workload-plugin protocol.

The domain objects deliberately contain only JSON-compatible payloads and
primitive metadata.  They can cross an IPC boundary without importing a
plugin implementation into the OmniTensor service process.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeAlias, runtime_checkable

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = Mapping[str, JsonValue]


class PluginResultStatus(StrEnum):
    """Terminal outcomes returned by a plugin."""

    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"
    FAILED = "failed"


class PluginHealthStatus(StrEnum):
    """Worker readiness independent from accelerator availability."""

    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class PluginContext:
    """Configuration and grants supplied once when a plugin starts."""

    plugin_id: str
    protocol_version: int
    configuration: JsonObject
    permissions: frozenset[str]


@dataclass(frozen=True, slots=True)
class PluginRequest:
    """One bounded unit of work submitted to a plugin."""

    job_id: str
    plugin_id: str
    trigger: str
    payload: JsonObject
    submitted_at_ms: int
    deadline_at_ms: int | None


@dataclass(frozen=True, slots=True)
class PluginProgress:
    """A replaceable progress observation for one job."""

    job_id: str
    stage: str
    fraction: float
    detail: str
    observed_at_ms: int


@dataclass(frozen=True, slots=True)
class PluginResult:
    """Exactly one terminal result for a plugin request."""

    job_id: str
    status: PluginResultStatus
    output: JsonObject
    detail: str
    completed_at_ms: int


@dataclass(frozen=True, slots=True)
class PluginHealth:
    """Current worker health reported to orchestration."""

    status: PluginHealthStatus
    detail: str
    checked_at_ms: int


@runtime_checkable
class CancellationToken(Protocol):
    """Cooperative cancellation observed by plugin code."""

    @property
    def cancelled(self) -> bool: ...

    async def wait(self) -> None: ...

    def raise_if_cancelled(self) -> None: ...


@runtime_checkable
class ProgressReporter(Protocol):
    """Sink for bounded, replaceable progress updates."""

    async def report(self, progress: PluginProgress) -> None: ...


class ScaledProgressReporter:
    """Map provider progress into one bounded workflow stage.

    Provider identity, stage, timestamps, and detail are deliberately not
    forwarded: the workflow owns those public fields and provider detail can
    contain private prompt material.
    """

    def __init__(
        self,
        request: PluginRequest,
        target: ProgressReporter,
        clock_ms: Callable[[], int],
        *,
        stage: str,
        offset: float,
        scale: float,
        error_type: Callable[[str, str], Exception],
    ) -> None:
        self._request = request
        self._target = target
        self._clock_ms = clock_ms
        self._stage = stage
        self._offset = offset
        self._scale = scale
        self._error_type = error_type

    async def report(self, progress: PluginProgress) -> None:
        fraction = progress.fraction
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction)
            or not 0 <= fraction <= 1
        ):
            raise self._error_type("progress-invalid", "provider progress is invalid")
        await self._target.report(
            PluginProgress(
                self._request.job_id,
                self._stage,
                self._offset + (float(fraction) * self._scale),
                "",
                self._clock_ms(),
            )
        )


@runtime_checkable
class PluginLifecycle(Protocol):
    """Lifecycle shared by every workload plugin."""

    async def start(self, context: PluginContext) -> None: ...

    async def health(self) -> PluginHealth: ...

    async def stop(self) -> None: ...


@runtime_checkable
class WorkloadPlugin(PluginLifecycle, Protocol):
    """Executable plugin surface consumed by orchestration."""

    async def execute(
        self,
        request: PluginRequest,
        cancellation: CancellationToken,
        progress: ProgressReporter,
    ) -> PluginResult: ...
