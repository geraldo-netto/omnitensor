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

import omnitensor.telemetry_types as _telemetry_types

from .collection import BoundedCollector, CollectionError, SourceSnapshot

HARDWARE_HEALTH_PLUGIN_ID = "hardware-health"
HARDWARE_METADATA_PERMISSION = "read:hardware-health-metadata"
HARDWARE_SENSOR_PERMISSION_PREFIX = "read:hardware-sensor/"
DEFAULT_MAX_SENSORS = 64
MAX_ERROR_COUNT = _telemetry_types.MAX_ERROR_COUNT
MAX_PERCENT = _telemetry_types.MAX_PERCENT
MAX_TEMPERATURE_MILLIDEGREES = _telemetry_types.MAX_TEMPERATURE_MILLIDEGREES
HardwareSample = _telemetry_types.HardwareSample
SensorHealth = _telemetry_types.SensorHealth
SensorKind = _telemetry_types.SensorKind
hardware_sample_error = _telemetry_types.hardware_sample_error


def hardware_sensor_permission(stable_id: str) -> str:
    """The scoped grant for one sensor."""
    if not isinstance(stable_id, str) or not _telemetry_types.STABLE_ID.fullmatch(stable_id):
        raise ValueError("hardware sensor identity is invalid")
    return f"{HARDWARE_SENSOR_PERMISSION_PREFIX}{stable_id}"

class HardwareHealthCollector(BoundedCollector[HardwareSample]):
    """Emit only allowlisted, granted sensors with stable units."""

    plugin_id = HARDWARE_HEALTH_PLUGIN_ID
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
