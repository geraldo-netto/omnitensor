"""Grant-filtered USB and Bluetooth peripheral metadata collection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from omnitensor.telemetry_types import PERIPHERAL_STABLE_ID as _STABLE_ID

from .collection import (
    BoundedCollector,
    CollectionError,
    CollectionPermissionGate,
    _now_ms,  # noqa: F401 - retained private compatibility alias
)
from .collection import (
    _strict_non_negative_integer_error as _non_negative_integer_error,
)
from .network_collection import NETWORK_PERIPHERALS_PLUGIN_ID
from .triggers import CollectorReadiness, SourceStatus

PERIPHERAL_METADATA_PERMISSION = "read:peripheral-metadata"
PERIPHERAL_DEVICE_PERMISSION_PREFIX = "read:peripheral-device/"
MAX_PERIPHERAL_DEVICES = 64
MAX_PERIPHERAL_SOURCE_DEVICES = 256
MAX_PERIPHERAL_ERROR_COUNT = 2**31 - 1


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


class PeripheralCollectionError(CollectionError):
    """Stable peripheral collection rejection."""


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


class PeripheralMetadataCollector(BoundedCollector[PeripheralSample]):
    """Emit only explicitly allowlisted devices with active per-device grants."""

    plugin_id = NETWORK_PERIPHERALS_PLUGIN_ID
    error_type = PeripheralCollectionError
    metadata_permission = PERIPHERAL_METADATA_PERMISSION
    label = "peripheral metadata"
    source_label = "peripheral metadata source"
    readiness_label = "peripheral source"
    trigger_label = "peripheral collector"
    source_name = "usb-bluetooth-health-metadata"
    items_key = "devices"
    require_allowlist = True
    sort_changed_fields = False

    def __init__(
        self,
        source: PeripheralMetadataSource,
        permissions: CollectionPermissionGate,
        allowed_device_ids: Sequence[str],
        *,
        clock_ms: Callable[[], int] | None = None,
    ):
        if not isinstance(source, PeripheralMetadataSource):
            raise TypeError("source must implement PeripheralMetadataSource")
        if not isinstance(permissions, CollectionPermissionGate):
            raise TypeError("permissions must implement CollectionPermissionGate")
        allowed = _validated_allowlist(allowed_device_ids)
        super().__init__(
            source,
            permissions,
            allowed,
            clock_ms=clock_ms,
        )

    def identity_of(self, item: PeripheralSample) -> str:
        return item.stable_id

    def item_permission(self, identity: str) -> str:
        return peripheral_device_permission(identity)

    def identity_document(self, item: PeripheralSample) -> dict[str, object]:
        return _identity_document(item)

    def document_of(self, item: PeripheralSample) -> dict[str, object]:
        return _device_document(item)

    def changed_fields(self, previous: PeripheralSample, current: PeripheralSample) -> list[str]:
        return _changed_fields(previous, current)

    def _snapshot_items(self, snapshot: object) -> Sequence[PeripheralSample]:
        assert isinstance(snapshot, PeripheralSnapshot)
        return snapshot.devices

    def _validate_snapshot(self, snapshot: object) -> None:
        error = peripheral_snapshot_error(snapshot)
        if error:
            raise PeripheralCollectionError("source-invalid", error)


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
