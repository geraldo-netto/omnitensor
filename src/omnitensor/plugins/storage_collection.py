"""Bounded SMART/NVMe attributes and I/O counters for storage health.

Read-only by construction: SMART has commands that erase a drive and change
its firmware, and none of them are reachable from here — the source protocol
exposes a snapshot and nothing else, so there is no code path that could issue
a write even if a plugin asked for one.

Device identity is the serial-derived stable id rather than the kernel name,
because ``/dev/sda`` is a position and not a device: reboot with a disk removed
and yesterday's ``sda`` is today's different drive.  Trending wear on the wrong
disk is worse than not trending it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from omnitensor.telemetry_types import STABLE_ID as _STABLE_ID

from .collection import (
    BoundedCollector,
    CollectionError,
    SourceSnapshot,
    bounded_number,
)

STORAGE_PLUGIN_ID = "storage-intelligence"
STORAGE_METADATA_PERMISSION = "read:storage-metadata"
STORAGE_DEVICE_PERMISSION_PREFIX = "read:storage-device/"
DEFAULT_MAX_DEVICES = 32
MAX_PERCENT = 100
MAX_TEMPERATURE_CELSIUS = 150
MAX_COUNTER = 2**53
MAX_BYTES = 2**63 - 1


class StorageBus(StrEnum):
    NVME = "nvme"
    SATA = "sata"
    USB = "usb"
    MMC = "mmc"
    OTHER = "other"


class StorageHealth(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    FAILING = "failing"
    UNKNOWN = "unknown"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class StorageCounters:
    """Monotonic I/O totals; never the contents of any read or write."""

    read_bytes: int
    written_bytes: int
    read_errors: int
    write_errors: int


@dataclass(frozen=True, slots=True)
class StorageSample:
    """One device's bounded wear and error state."""

    stable_id: str
    bus: StorageBus
    health: StorageHealth
    present: bool
    temperature_celsius: int | None
    wear_percent: int | None
    power_on_hours: int | None
    reallocated_sectors: int | None
    counters: StorageCounters | None
    observed_at_ms: int


def storage_device_permission(stable_id: str) -> str:
    if not isinstance(stable_id, str) or not _STABLE_ID.fullmatch(stable_id):
        raise ValueError("storage device identity is invalid")
    return f"{STORAGE_DEVICE_PERMISSION_PREFIX}{stable_id}"


def storage_sample_error(sample: object) -> str:
    """One stable validation failure for an untrusted source sample."""
    shape = _shape_error(sample)
    if shape:
        return shape
    if not sample.present:
        # A hot-removed disk reports absence, not zeroed wear: a drive pulled
        # mid-scan is not a drive in perfect health.
        if any(
            value is not None
            for value in (
                sample.temperature_celsius,
                sample.wear_percent,
                sample.power_on_hours,
                sample.reallocated_sectors,
                sample.counters,
            )
        ):
            return "an absent storage device must not carry readings"
        return ""
    return _reading_error(sample)


class StorageIntelligenceCollector(BoundedCollector[StorageSample]):
    """Emit allowlisted, granted devices with stable serial-derived identity."""

    plugin_id = STORAGE_PLUGIN_ID
    metadata_permission = STORAGE_METADATA_PERMISSION
    label = "storage metadata"
    source_name = "smart-nvme-io"

    def __init__(self, *args, max_items: int = DEFAULT_MAX_DEVICES, **changes) -> None:
        super().__init__(*args, max_items=max_items, **changes)

    def identity_of(self, item: StorageSample) -> str:
        return item.stable_id

    def item_permission(self, identity: str) -> str:
        return f"{STORAGE_DEVICE_PERMISSION_PREFIX}{identity}"

    def document_of(self, item: StorageSample) -> dict:
        counters = item.counters
        return {
            "id": item.stable_id,
            "bus": str(item.bus),
            "health": str(item.health),
            "present": item.present,
            "temperatureCelsius": item.temperature_celsius,
            "wearPercent": item.wear_percent,
            "powerOnHours": item.power_on_hours,
            "reallocatedSectors": item.reallocated_sectors,
            "counters": None
            if counters is None
            else {
                "readBytes": counters.read_bytes,
                "writtenBytes": counters.written_bytes,
                "readErrors": counters.read_errors,
                "writeErrors": counters.write_errors,
            },
            "observedAtMs": item.observed_at_ms,
        }

    def changed_fields(self, previous: StorageSample, current: StorageSample) -> list[str]:
        fields = (
            ("bus", previous.bus, current.bus),
            ("health", previous.health, current.health),
            ("present", previous.present, current.present),
            ("temperatureCelsius", previous.temperature_celsius, current.temperature_celsius),
            ("wearPercent", previous.wear_percent, current.wear_percent),
            ("powerOnHours", previous.power_on_hours, current.power_on_hours),
            (
                "reallocatedSectors",
                previous.reallocated_sectors,
                current.reallocated_sectors,
            ),
            ("counters", previous.counters, current.counters),
        )
        return [name for name, before, after in fields if before != after]

    def _validate_snapshot(self, snapshot: object) -> None:
        if isinstance(snapshot, SourceSnapshot):
            for sample in snapshot.items:
                error = storage_sample_error(sample)
                if error:
                    raise CollectionError("source-invalid", error)
        super()._validate_snapshot(snapshot)


def _shape_error(sample: object) -> str:
    if not isinstance(sample, StorageSample):
        return "storage sample has an invalid type"
    if not isinstance(sample.bus, StorageBus):
        return "storage sample bus is invalid"
    if not isinstance(sample.health, StorageHealth):
        return "storage sample health is invalid"
    if not isinstance(sample.present, bool):
        return "storage sample presence is invalid"
    return ""


def _reading_error(sample: StorageSample) -> str:
    checks = (
        (sample.temperature_celsius, "temperature", MAX_TEMPERATURE_CELSIUS),
        (sample.wear_percent, "wear percent", MAX_PERCENT),
        (sample.power_on_hours, "power-on hours", MAX_COUNTER),
        (sample.reallocated_sectors, "reallocated sectors", MAX_COUNTER),
    )
    for value, name, maximum in checks:
        if value is None:
            continue
        try:
            bounded_number(value, f"storage {name}", maximum)
        except CollectionError as error:
            return error.detail
    counters = sample.counters
    if counters is None:
        return ""
    if not isinstance(counters, StorageCounters):
        return "storage counters are invalid"
    totals = (
        (counters.read_bytes, "read bytes", MAX_BYTES),
        (counters.written_bytes, "written bytes", MAX_BYTES),
        (counters.read_errors, "read errors", MAX_COUNTER),
        (counters.write_errors, "write errors", MAX_COUNTER),
    )
    for value, name, maximum in totals:
        try:
            bounded_number(value, f"storage {name}", maximum)
        except CollectionError as error:
            return error.detail
    return ""
