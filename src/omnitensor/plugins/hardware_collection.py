"""Bounded hardware-health signals: thermal, memory, power, and service state.

Four sources with one thing in common: each is a number the kernel or systemd
already publishes, and none of them is content.  A temperature, a corrected
ECC count, a battery percentage, and whether a unit is running say a great deal
about whether a machine is failing and nothing about what its user is doing.

Everything is read through explicit grants and stable identities, and a sensor
that disappears is reported as a missing source rather than as a zero — a
thermal probe that vanished is not a cool machine, and the difference is the
whole point of collecting this at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .collection import (
    BoundedCollector,
    CollectionError,
    SourceSnapshot,
    bounded_number,
)

HARDWARE_HEALTH_PLUGIN_ID = "hardware-health"
HARDWARE_METADATA_PERMISSION = "read:hardware-health-metadata"
HARDWARE_SENSOR_PERMISSION_PREFIX = "read:hardware-sensor/"
DEFAULT_MAX_SENSORS = 64
MAX_TEMPERATURE_MILLIDEGREES = 200_000
MAX_ERROR_COUNT = 2**53
MAX_PERCENT = 100


class SensorKind(StrEnum):
    """What a reading measures, which decides how it may be interpreted."""

    THERMAL = "thermal"
    FAN = "fan"
    VOLTAGE = "voltage"
    MEMORY_ERRORS = "memory-errors"
    POWER = "power"
    SERVICE = "service"


class SensorHealth(StrEnum):
    """Whether the reading itself can be trusted, separate from its value."""

    OK = "ok"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class HardwareSample:
    """One bounded reading with its unit stated rather than assumed."""

    stable_id: str
    kind: SensorKind
    health: SensorHealth
    value: int | None
    unit: str
    label: str
    observed_at_ms: int


def hardware_sensor_permission(stable_id: str) -> str:
    """The scoped grant for one sensor."""
    from .collection import STABLE_ID  # noqa: PLC0415 - local to keep the module flat

    if not isinstance(stable_id, str) or not STABLE_ID.fullmatch(stable_id):
        raise ValueError("hardware sensor identity is invalid")
    return f"{HARDWARE_SENSOR_PERMISSION_PREFIX}{stable_id}"


def hardware_sample_error(sample: object) -> str:
    """One stable validation failure for an untrusted source sample."""
    shape = _sample_shape_error(sample)
    if shape:
        return shape
    if sample.health is SensorHealth.MISSING:
        # A missing sensor must carry no value at all.  Reporting zero for an
        # absent thermal probe reads as a cool machine, which is the opposite
        # of what happened.
        if sample.value is not None:
            return "a missing hardware sample must not carry a value"
        return ""
    if sample.value is None:
        return "a present hardware sample must carry a value"
    try:
        bounded_number(sample.value, "hardware sample value", _ceiling(sample.kind))
    except CollectionError as error:
        return error.detail
    return ""


def _sample_shape_error(sample: object) -> str:
    if not isinstance(sample, HardwareSample):
        return "hardware sample has an invalid type"
    if not isinstance(sample.kind, SensorKind):
        return "hardware sample kind is invalid"
    if not isinstance(sample.health, SensorHealth):
        return "hardware sample health is invalid"
    if not isinstance(sample.unit, str) or len(sample.unit) > 16:
        return "hardware sample unit is invalid"
    if not isinstance(sample.label, str) or len(sample.label) > 80:
        return "hardware sample label is invalid"
    return ""


class HardwareHealthCollector(BoundedCollector[HardwareSample]):
    """Emit only allowlisted, granted sensors with stable units."""

    metadata_permission = HARDWARE_METADATA_PERMISSION
    label = "hardware health metadata"
    source_name = "hwmon-edac-power-service"

    def __init__(self, *args, max_items: int = DEFAULT_MAX_SENSORS, **changes) -> None:
        super().__init__(*args, max_items=max_items, **changes)

    def identity_of(self, item: HardwareSample) -> str:
        return item.stable_id

    def item_permission(self, identity: str) -> str:
        return f"{HARDWARE_SENSOR_PERMISSION_PREFIX}{identity}"

    def document_of(self, item: HardwareSample) -> dict:
        return {
            "id": item.stable_id,
            "kind": str(item.kind),
            "health": str(item.health),
            "value": item.value,
            "unit": item.unit,
            "label": item.label,
            "observedAtMs": item.observed_at_ms,
        }

    def changed_fields(self, previous: HardwareSample, current: HardwareSample) -> list[str]:
        # observed_at_ms changes on every poll and means nothing on its own, so
        # a churn report driven by it would be noise on every single tick.
        fields = (
            ("kind", previous.kind, current.kind),
            ("health", previous.health, current.health),
            ("value", previous.value, current.value),
            ("unit", previous.unit, current.unit),
            ("label", previous.label, current.label),
        )
        return [name for name, before, after in fields if before != after]

    def _validate_snapshot(self, snapshot: object) -> None:
        # Sample types are checked before the shared checks, which read an
        # identity off each item and would otherwise fault on a malformed one.
        if isinstance(snapshot, SourceSnapshot):
            for sample in snapshot.items:
                error = hardware_sample_error(sample)
                if error:
                    raise CollectionError("source-invalid", error)
        super()._validate_snapshot(snapshot)


def _ceiling(kind: SensorKind) -> int:
    if kind is SensorKind.THERMAL:
        return MAX_TEMPERATURE_MILLIDEGREES
    if kind is SensorKind.MEMORY_ERRORS:
        return MAX_ERROR_COUNT
    if kind in (SensorKind.POWER, SensorKind.SERVICE):
        return MAX_PERCENT
    return MAX_ERROR_COUNT
