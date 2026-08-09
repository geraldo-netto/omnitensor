"""Grant-filtered USB and Bluetooth peripheral metadata collection."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from .network_collection import NETWORK_PERIPHERALS_PLUGIN_ID, CollectionPermissionGate
from .pipeline import CollectedOutput
from .triggers import CollectorReadiness, SourceStatus, Trigger

PERIPHERAL_METADATA_PERMISSION = "read:peripheral-metadata"
PERIPHERAL_DEVICE_PERMISSION_PREFIX = "read:peripheral-device/"
DEFAULT_MAX_PERIPHERAL_DEVICES = 32
MAX_PERIPHERAL_DEVICES = 64
MAX_PERIPHERAL_SOURCE_DEVICES = 256
MAX_PERIPHERAL_ERROR_COUNT = 2**31 - 1
_STABLE_ID = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,78}[a-z0-9])?")


class PeripheralBus(StrEnum):
    USB = "usb"
    BLUETOOTH = "bluetooth"


class PeripheralClass(StrEnum):
    INPUT = "input"
    AUDIO = "audio"
    STORAGE = "storage"
    NETWORK = "network"
    SENSOR = "sensor"
    OTHER = "other"


class PeripheralHealth(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class PeripheralCollectionError(ValueError):
    """Stable peripheral collection rejection."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class PeripheralSample:
    """Content-free health observation for one opaque device identity."""

    stable_id: str
    bus: PeripheralBus
    device_class: PeripheralClass
    health: PeripheralHealth
    connected: bool
    authorized: bool
    paired: bool
    trusted: bool
    battery_percent: int | None
    error_count: int
    observed_at_ms: int


@dataclass(frozen=True, slots=True)
class PeripheralSnapshot:
    """Bounded typed source snapshot for a live adapter or replay."""

    status: SourceStatus
    observed_at_ms: int
    devices: tuple[PeripheralSample, ...]


@runtime_checkable
class PeripheralMetadataSource(Protocol):
    """Trusted host adapter for USB and Bluetooth health metadata."""

    async def readiness(self) -> CollectorReadiness: ...

    async def snapshot(self) -> PeripheralSnapshot: ...


class ReplayPeripheralMetadataSource:
    """Advance through typed hotplug snapshots, then repeat the final state."""

    def __init__(self, snapshots: Sequence[PeripheralSnapshot]):
        if isinstance(snapshots, (str, bytes)) or not isinstance(snapshots, Sequence):
            raise TypeError("snapshots must be a sequence")
        self._snapshots = tuple(snapshots)
        self._index = 0

    async def readiness(self) -> CollectorReadiness:
        if not self._snapshots:
            return CollectorReadiness(SourceStatus.UNAVAILABLE, "peripheral replay is empty", 0)
        snapshot = self._snapshots[self._index]
        error = peripheral_snapshot_error(snapshot)
        if error:
            return CollectorReadiness(SourceStatus.UNAVAILABLE, "peripheral replay is invalid", 0)
        return CollectorReadiness(
            snapshot.status,
            _source_health_detail(snapshot.status),
            snapshot.observed_at_ms,
        )

    async def snapshot(self) -> PeripheralSnapshot:
        if not self._snapshots:
            raise PeripheralCollectionError("source-unavailable", "peripheral replay is empty")
        snapshot = self._snapshots[self._index]
        if self._index < len(self._snapshots) - 1:
            self._index += 1
        return snapshot


class PeripheralMetadataCollector:
    """Emit only explicitly allowlisted devices with active per-device grants."""

    def __init__(
        self,
        source: PeripheralMetadataSource,
        permissions: CollectionPermissionGate,
        allowed_device_ids: Sequence[str],
        *,
        max_devices: int = DEFAULT_MAX_PERIPHERAL_DEVICES,
        clock_ms: Callable[[], int] | None = None,
    ):
        if not isinstance(source, PeripheralMetadataSource):
            raise TypeError("source must implement PeripheralMetadataSource")
        if not isinstance(permissions, CollectionPermissionGate):
            raise TypeError("permissions must implement CollectionPermissionGate")
        self._allowed_device_ids = _validated_allowlist(allowed_device_ids)
        if type(max_devices) is not int or not 1 <= max_devices <= MAX_PERIPHERAL_DEVICES:
            raise ValueError(f"max_devices must be an integer from 1 to {MAX_PERIPHERAL_DEVICES}")
        if clock_ms is not None and not callable(clock_ms):
            raise TypeError("clock_ms must be callable")
        self._source = source
        self._permissions = permissions
        self._max_devices = max_devices
        self._clock_ms = clock_ms or _now_ms
        self._previous: dict[str, PeripheralSample] = {}

    async def readiness(self) -> CollectorReadiness:
        if not self._permissions.allows(PERIPHERAL_METADATA_PERMISSION):
            return CollectorReadiness(
                SourceStatus.UNAVAILABLE,
                "peripheral metadata permission is not granted",
                self._clock_ms(),
            )
        try:
            readiness = await self._source.readiness()
        except Exception as error:  # noqa: BLE001 - host adapters can fail arbitrarily
            return CollectorReadiness(
                SourceStatus.UNAVAILABLE,
                f"peripheral source readiness failed: {type(error).__name__}",
                self._clock_ms(),
            )
        if _readiness_error(readiness):
            return CollectorReadiness(
                SourceStatus.UNAVAILABLE,
                "peripheral source readiness is invalid",
                self._clock_ms(),
            )
        assert isinstance(readiness, CollectorReadiness)
        return CollectorReadiness(
            readiness.status,
            _source_health_detail(readiness.status),
            readiness.checked_at_ms,
        )

    async def collect(self, trigger: Trigger) -> CollectedOutput:
        if not isinstance(trigger, Trigger) or trigger.plugin_id != NETWORK_PERIPHERALS_PLUGIN_ID:
            raise PeripheralCollectionError(
                "trigger-invalid", "peripheral collector requires a network-peripherals trigger"
            )
        if not self._permissions.allows(PERIPHERAL_METADATA_PERMISSION):
            raise PeripheralCollectionError(
                "permission-denied", "peripheral metadata permission is not granted"
            )
        snapshot = await self._source.snapshot()
        error = peripheral_snapshot_error(snapshot)
        if error:
            raise PeripheralCollectionError("source-invalid", error)
        eligible = sorted(
            (device for device in snapshot.devices if self._may_emit(device.stable_id)),
            key=lambda device: device.stable_id,
        )
        selected = eligible[: self._max_devices]
        # Churn is computed against every eligible device, not the truncated
        # emission list.  Tracking only what fit reported a device past
        # max_devices as removed while it was still attached, and as added
        # again as soon as it fit — churn the user never experienced.
        current = {device.stable_id: device for device in eligible}
        authorized_previous = {
            stable_id: device
            for stable_id, device in self._previous.items()
            if self._may_emit(stable_id)
        }
        churn = _churn_document(authorized_previous, current)
        self._previous = current
        return CollectedOutput(
            {
                "schemaVersion": 1,
                "source": "usb-bluetooth-health-metadata",
                "sourceHealth": str(snapshot.status),
                "observedAtMs": snapshot.observed_at_ms,
                "devices": [_device_document(device) for device in selected],
                "churn": churn,
                "truncatedDevices": len(eligible) - len(selected),
            }
        )

    def _may_emit(self, stable_id: str) -> bool:
        return stable_id in self._allowed_device_ids and self._permissions.allows(
            peripheral_device_permission(stable_id)
        )


def peripheral_device_permission(stable_id: str) -> str:
    """Return the scoped grant name for one validated stable identity."""
    if not isinstance(stable_id, str) or not _STABLE_ID.fullmatch(stable_id):
        raise ValueError("peripheral stable identity is invalid")
    return f"{PERIPHERAL_DEVICE_PERMISSION_PREFIX}{stable_id}"


def peripheral_snapshot_error(snapshot: object) -> str:
    """Return one stable validation failure for an untrusted source snapshot."""
    error = _snapshot_shape_error(snapshot)
    if error:
        return error
    assert isinstance(snapshot, PeripheralSnapshot)
    identities: set[str] = set()
    for device in snapshot.devices:
        error = _device_error(device, snapshot.observed_at_ms)
        if error:
            return error
        if device.stable_id in identities:
            return f"duplicate peripheral identity: {device.stable_id}"
        identities.add(device.stable_id)
    return ""


def _snapshot_shape_error(snapshot: object) -> str:
    if not isinstance(snapshot, PeripheralSnapshot):
        return "peripheral snapshot is not typed"
    if not isinstance(snapshot.status, SourceStatus):
        return "peripheral source status is invalid"
    if _non_negative_integer_error(snapshot.observed_at_ms):
        return "peripheral snapshot timestamp is invalid"
    if not isinstance(snapshot.devices, tuple):
        return "peripheral devices must be a tuple"
    if len(snapshot.devices) > MAX_PERIPHERAL_SOURCE_DEVICES:
        return f"peripheral snapshot exceeds {MAX_PERIPHERAL_SOURCE_DEVICES} devices"
    return ""


def _device_error(device: object, snapshot_time_ms: int) -> str:
    if not isinstance(device, PeripheralSample):
        return "peripheral sample is not typed"
    if not isinstance(device.stable_id, str) or not _STABLE_ID.fullmatch(device.stable_id):
        return "peripheral stable identity is invalid"
    error = _device_enum_error(device)
    if error:
        return error
    if any(
        type(value) is not bool
        for value in (device.connected, device.authorized, device.paired, device.trusted)
    ):
        return f"peripheral flags are invalid: {device.stable_id}"
    if device.battery_percent is not None and (
        type(device.battery_percent) is not int or not 0 <= device.battery_percent <= 100
    ):
        return f"peripheral battery is invalid: {device.stable_id}"
    if (
        type(device.error_count) is not int
        or not 0 <= device.error_count <= MAX_PERIPHERAL_ERROR_COUNT
    ):
        return f"peripheral error count is invalid: {device.stable_id}"
    if _non_negative_integer_error(device.observed_at_ms) or (
        device.observed_at_ms > snapshot_time_ms
    ):
        return f"peripheral timestamp is invalid: {device.stable_id}"
    return ""


def _device_enum_error(device: PeripheralSample) -> str:
    if not isinstance(device.bus, PeripheralBus):
        return f"peripheral bus is invalid: {device.stable_id}"
    if not isinstance(device.device_class, PeripheralClass):
        return f"peripheral class is invalid: {device.stable_id}"
    if not isinstance(device.health, PeripheralHealth):
        return f"peripheral health is invalid: {device.stable_id}"
    return ""


def _validated_allowlist(identities: object) -> frozenset[str]:
    if isinstance(identities, (str, bytes)) or not isinstance(identities, Sequence):
        raise TypeError("allowed_device_ids must be a sequence")
    if len(identities) > MAX_PERIPHERAL_DEVICES:
        raise ValueError(f"at most {MAX_PERIPHERAL_DEVICES} device identities may be allowed")
    validated = frozenset(identities)
    if len(validated) != len(identities):
        raise ValueError("allowed device identities must be unique")
    for stable_id in validated:
        peripheral_device_permission(stable_id)
    return validated


def _readiness_error(readiness: object) -> bool:
    return (
        not isinstance(readiness, CollectorReadiness)
        or not isinstance(readiness.status, SourceStatus)
        or _non_negative_integer_error(readiness.checked_at_ms)
    )


def _churn_document(
    previous: dict[str, PeripheralSample],
    current: dict[str, PeripheralSample],
) -> dict[str, list[dict[str, object]]]:
    added = [_identity_document(current[key]) for key in sorted(current.keys() - previous.keys())]
    removed = [
        _identity_document(previous[key]) for key in sorted(previous.keys() - current.keys())
    ]
    changed = []
    for key in sorted(previous.keys() & current.keys()):
        fields = _changed_fields(previous[key], current[key])
        if fields:
            changed.append(
                {
                    **_identity_document(current[key]),
                    "fields": fields,
                }
            )
    return {"added": added, "removed": removed, "changed": changed}


def _changed_fields(previous: PeripheralSample, current: PeripheralSample) -> list[str]:
    fields = (
        ("bus", previous.bus, current.bus),
        ("class", previous.device_class, current.device_class),
        ("health", previous.health, current.health),
        ("connected", previous.connected, current.connected),
        ("authorized", previous.authorized, current.authorized),
        ("paired", previous.paired, current.paired),
        ("trusted", previous.trusted, current.trusted),
        ("batteryPercent", previous.battery_percent, current.battery_percent),
        ("errorCount", previous.error_count, current.error_count),
    )
    return [name for name, old, new in fields if old != new]


def _identity_document(device: PeripheralSample) -> dict[str, object]:
    return {"stableId": device.stable_id, "bus": str(device.bus)}


def _device_document(device: PeripheralSample) -> dict[str, object]:
    return {
        **_identity_document(device),
        "class": str(device.device_class),
        "health": str(device.health),
        "connected": device.connected,
        "authorized": device.authorized,
        "paired": device.paired,
        "trusted": device.trusted,
        "batteryPercent": device.battery_percent,
        "errorCount": device.error_count,
        "observedAtMs": device.observed_at_ms,
    }


def _source_health_detail(status: SourceStatus) -> str:
    if status is SourceStatus.READY:
        return "peripheral metadata source is ready"
    if status is SourceStatus.DEGRADED:
        return "peripheral metadata source is degraded"
    return "peripheral metadata source is unavailable"


def _non_negative_integer_error(value: object) -> bool:
    return type(value) is not int or value < 0


def _now_ms() -> int:
    return time.time_ns() // 1_000_000
