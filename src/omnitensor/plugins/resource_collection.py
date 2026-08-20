"""Bounded cgroup, systemd, and pressure features for scheduling advice.

Pressure stall information is the signal that matters here: utilisation says a
resource is busy, PSI says something is *waiting* for it, and only the second
one distinguishes a machine working efficiently from a machine thrashing.

Unit and slice names are the identities, and they are allowlisted — a scheduler
advisor has no business enumerating every unit a user runs, and the ones it is
asked about are named up front.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum

from omnitensor.telemetry_types import STABLE_ID as _STABLE_ID

from .collection import (
    BoundedCollector,
    CollectionError,
    bounded_number,
)
from .kernel_telemetry import AbsentAggregateSource, KernelAggregateSource

RESOURCE_PLUGIN_ID = "resource-scheduler"
RESOURCE_METADATA_PERMISSION = "read:resource-metadata"
RESOURCE_UNIT_PERMISSION_PREFIX = "read:resource-unit/"
DEFAULT_MAX_UNITS = 64
MAX_PERCENT_MILLI = 100_000
MAX_MEMORY_BYTES = 2**63 - 1
MAX_TASKS = 2**32
MAX_QUEUE_DEPTH = 2**31


class ResourceScope(StrEnum):
    """What the sample describes, which decides how it may be compared."""

    SLICE = "slice"
    SERVICE = "service"
    SCOPE = "scope"
    QUEUE = "queue"


class UnitState(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PressureStall:
    """Share of a window something spent waiting, in milli-percent."""

    some_milli_percent: int
    full_milli_percent: int


@dataclass(frozen=True, slots=True)
class ResourceSample:
    """One unit's bounded consumption and waiting."""

    stable_id: str
    scope: ResourceScope
    state: UnitState
    cpu_usage_milli_percent: int | None
    memory_current_bytes: int | None
    task_count: int | None
    queue_depth: int | None
    cpu_pressure: PressureStall | None
    io_pressure: PressureStall | None
    memory_pressure: PressureStall | None
    observed_at_ms: int


def resource_unit_permission(stable_id: str) -> str:
    if not isinstance(stable_id, str) or not _STABLE_ID.fullmatch(stable_id):
        raise ValueError("resource unit identity is invalid")
    return f"{RESOURCE_UNIT_PERMISSION_PREFIX}{stable_id}"


def resource_sample_error(sample: object) -> str:
    """One stable validation failure for an untrusted source sample."""
    if not isinstance(sample, ResourceSample):
        return "resource sample has an invalid type"
    if not isinstance(sample.scope, ResourceScope):
        return "resource sample scope is invalid"
    if not isinstance(sample.state, UnitState):
        return "resource sample state is invalid"
    reading = _reading_error(sample)
    if reading:
        return reading
    for pressure, name in (
        (sample.cpu_pressure, "cpu"),
        (sample.io_pressure, "io"),
        (sample.memory_pressure, "memory"),
    ):
        error = _pressure_error(pressure, name)
        if error:
            return error
    return ""


class ResourceSchedulerCollector(BoundedCollector[ResourceSample]):
    """Emit only allowlisted, granted units and queues."""

    plugin_id = RESOURCE_PLUGIN_ID
    metadata_permission = RESOURCE_METADATA_PERMISSION
    label = "resource metadata"
    source_name = "cgroup-systemd-pressure"

    def __init__(
        self,
        *args,
        max_items: int = DEFAULT_MAX_UNITS,
        kernel_source: KernelAggregateSource | None = None,
        **changes,
    ) -> None:
        super().__init__(*args, max_items=max_items, **changes)
        # Absent unless a helper is wired in: this collector used to construct
        # the socket adapter itself, so a host with no helper paid a stat and a
        # thread hop per tick to be told, every time, that there is no helper.
        if kernel_source is not None and not isinstance(kernel_source, KernelAggregateSource):
            raise TypeError("kernel_source must implement KernelAggregateSource")
        self._kernel_source: KernelAggregateSource = kernel_source or AbsentAggregateSource()

    async def _output_extensions(self) -> dict[str, object]:
        """Add aggregate run-queue and block-I/O features before validation."""
        if isinstance(self._kernel_source, AbsentAggregateSource):
            # It answers from a constant; a thread hop would cost more than it.
            aggregate = self._kernel_source.read()
        else:
            aggregate = await asyncio.to_thread(self._kernel_source.read)
        return {
            "kernelTelemetry": aggregate.document(),
            "kernelFeatures": aggregate.scheduler_features(),
        }

    def identity_of(self, item: ResourceSample) -> str:
        return item.stable_id

    def item_permission(self, identity: str) -> str:
        return f"{RESOURCE_UNIT_PERMISSION_PREFIX}{identity}"

    def document_of(self, item: ResourceSample) -> dict:
        return {
            "id": item.stable_id,
            "scope": str(item.scope),
            "state": str(item.state),
            "cpuUsageMilliPercent": item.cpu_usage_milli_percent,
            "memoryCurrentBytes": item.memory_current_bytes,
            "taskCount": item.task_count,
            "queueDepth": item.queue_depth,
            "cpuPressure": _pressure_document(item.cpu_pressure),
            "ioPressure": _pressure_document(item.io_pressure),
            "memoryPressure": _pressure_document(item.memory_pressure),
            "observedAtMs": item.observed_at_ms,
        }

    def changed_fields(self, previous: ResourceSample, current: ResourceSample) -> list[str]:
        fields = (
            ("scope", previous.scope, current.scope),
            ("state", previous.state, current.state),
            (
                "cpuUsageMilliPercent",
                previous.cpu_usage_milli_percent,
                current.cpu_usage_milli_percent,
            ),
            ("memoryCurrentBytes", previous.memory_current_bytes, current.memory_current_bytes),
            ("taskCount", previous.task_count, current.task_count),
            ("queueDepth", previous.queue_depth, current.queue_depth),
            ("cpuPressure", previous.cpu_pressure, current.cpu_pressure),
            ("ioPressure", previous.io_pressure, current.io_pressure),
            ("memoryPressure", previous.memory_pressure, current.memory_pressure),
        )
        return [name for name, before, after in fields if before != after]

    sample_error = staticmethod(resource_sample_error)


def _reading_error(sample: ResourceSample) -> str:
    readings = (
        (sample.cpu_usage_milli_percent, "cpu usage", MAX_PERCENT_MILLI),
        (sample.memory_current_bytes, "memory", MAX_MEMORY_BYTES),
        (sample.task_count, "task count", MAX_TASKS),
        (sample.queue_depth, "queue depth", MAX_QUEUE_DEPTH),
    )
    for value, name, maximum in readings:
        if value is None:
            continue
        try:
            bounded_number(value, f"resource {name}", maximum)
        except CollectionError as error:
            return error.detail
    return ""


def _pressure_error(pressure: object, name: str) -> str:
    if pressure is None:
        return ""
    if not isinstance(pressure, PressureStall):
        return f"{name} pressure is invalid"
    for value, label in (
        (pressure.some_milli_percent, "some"),
        (pressure.full_milli_percent, "full"),
    ):
        try:
            bounded_number(value, f"{name} pressure {label}", MAX_PERCENT_MILLI)
        except CollectionError as error:
            return error.detail
    if pressure.full_milli_percent > pressure.some_milli_percent:
        # "full" counts windows where everything stalled, which is a subset of
        # "some"; the reverse would mean the source is not reporting PSI.
        return f"{name} pressure full exceeds some"
    return ""


def _pressure_document(pressure: PressureStall | None) -> dict | None:
    if pressure is None:
        return None
    return {
        "someMilliPercent": pressure.some_milli_percent,
        "fullMilliPercent": pressure.full_milli_percent,
    }
