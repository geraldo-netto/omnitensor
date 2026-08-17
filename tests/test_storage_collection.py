from __future__ import annotations

import asyncio
import inspect

import pytest

from omnitensor.plugins.collection import (
    CollectionError,
    MetadataSource,
    ReplaySource,
    SourceSnapshot,
)
from omnitensor.plugins.storage_collection import (
    STORAGE_METADATA_PERMISSION,
    STORAGE_PLUGIN_ID,
    StorageBus,
    StorageCounters,
    StorageHealth,
    StorageIntelligenceCollector,
    StorageSample,
    storage_device_permission,
    storage_sample_error,
)
from omnitensor.plugins.triggers import SourceStatus, Trigger, TriggerKind
from omnitensor.sdk.helpers import PermissionView

NVME = "nvme-sn-abc123"
SATA = "sata-sn-def456"


def permissions(*device_ids, metadata=True):
    scoped = {storage_device_permission(device) for device in device_ids}
    declared = frozenset({STORAGE_METADATA_PERMISSION, *scoped})
    granted = frozenset({*scoped, *({STORAGE_METADATA_PERMISSION} if metadata else set())})
    return PermissionView(declared, granted)


def counters(**changes):
    values = {"read_bytes": 10, "written_bytes": 20, "read_errors": 0, "write_errors": 0}
    values.update(changes)
    return StorageCounters(**values)


def sample(stable_id=NVME, **changes):
    values = {
        "stable_id": stable_id,
        "bus": StorageBus.NVME,
        "health": StorageHealth.OK,
        "present": True,
        "temperature_celsius": 40,
        "wear_percent": 3,
        "power_on_hours": 1200,
        "reallocated_sectors": 0,
        "counters": counters(),
        "observed_at_ms": 9,
    }
    values.update(changes)
    return StorageSample(**values)


def absent(stable_id=NVME):
    return StorageSample(
        stable_id,
        StorageBus.NVME,
        StorageHealth.MISSING,
        False,
        None,
        None,
        None,
        None,
        None,
        9,
    )


def snapshot(*samples, status=SourceStatus.READY, observed_at_ms=10):
    return SourceSnapshot(status, observed_at_ms, tuple(samples) or (sample(),))


def collector(*snapshots, allowed=(NVME,), granted=(NVME,), **changes):
    source = ReplaySource(list(snapshots) or [snapshot()], label="storage")
    return StorageIntelligenceCollector(source, permissions(*granted), allowed, **changes)


def trigger(plugin_id=STORAGE_PLUGIN_ID):
    return Trigger(plugin_id, "storage-1", TriggerKind.MANUAL, {}, 1)


def collect(subject):
    return asyncio.run(subject.collect(trigger())).payload


def test_only_allowlisted_and_granted_devices_are_emitted():
    payload = collect(
        collector(
            snapshot(sample(NVME), sample(SATA, bus=StorageBus.SATA), sample("unlisted-disk")),
            allowed=(NVME, SATA),
            granted=(NVME,),
        )
    )
    assert [item["id"] for item in payload["items"]] == [NVME]


def test_a_hot_removed_disk_reports_absence_rather_than_zeroed_wear():
    """A drive pulled mid-scan is not a drive in perfect health."""
    payload = collect(collector(snapshot(absent())))

    [item] = payload["items"]
    assert item["present"] is False
    assert item["wearPercent"] is None
    assert item["counters"] is None


def test_an_absent_device_carrying_readings_is_refused():
    bad = StorageSample(
        NVME, StorageBus.NVME, StorageHealth.MISSING, False, 40, None, None, None, None, 9
    )
    assert "must not carry readings" in storage_sample_error(bad)


def test_counters_and_wear_are_bounded():
    assert storage_sample_error(sample(wear_percent=101)) != ""
    assert storage_sample_error(sample(temperature_celsius=200)) != ""
    assert storage_sample_error(sample(counters=counters(read_errors=-1))) != ""
    assert storage_sample_error(sample(counters="not-counters")) != ""


def test_identity_is_serial_derived_not_the_kernel_name():
    """/dev/sda is a position; yesterday's sda can be a different drive today."""
    payload = collect(collector(snapshot(sample(NVME))))
    assert payload["items"][0]["id"] == NVME
    assert "sda" not in payload["items"][0]["id"]


def test_the_source_protocol_offers_no_write_path():
    """SMART can erase a drive; nothing reachable from here can issue one."""
    surface = {
        name for name, _value in inspect.getmembers(MetadataSource) if not name.startswith("_")
    }
    assert surface == {"readiness", "snapshot"}


def test_a_disk_that_disappears_between_scans_is_reported_as_removed():
    subject = collector(
        snapshot(sample(NVME), sample(SATA, bus=StorageBus.SATA)),
        snapshot(sample(NVME)),
        allowed=(NVME, SATA),
        granted=(NVME, SATA),
    )
    asyncio.run(subject.collect(trigger()))
    second = asyncio.run(subject.collect(trigger())).payload
    assert [item["id"] for item in second["churn"]["removed"]] == [SATA]


def test_growing_wear_is_reported_as_a_change():
    subject = collector(
        snapshot(sample(NVME, wear_percent=3)),
        snapshot(sample(NVME, wear_percent=4)),
    )
    asyncio.run(subject.collect(trigger()))
    second = asyncio.run(subject.collect(trigger())).payload
    assert second["churn"]["changed"] == [{"id": NVME, "fields": ["wearPercent"]}]


def test_a_new_timestamp_alone_is_not_a_change():
    subject = collector(
        snapshot(sample(NVME)),
        snapshot(sample(NVME, observed_at_ms=99), observed_at_ms=99),
    )
    asyncio.run(subject.collect(trigger()))
    assert asyncio.run(subject.collect(trigger())).payload["churn"]["changed"] == []


def test_collection_without_the_metadata_grant_is_refused():
    subject = StorageIntelligenceCollector(
        ReplaySource([snapshot()], label="storage"),
        permissions(NVME, metadata=False),
        (NVME,),
    )
    with pytest.raises(CollectionError, match="permission-denied"):
        collect(subject)


def test_a_trigger_for_another_plugin_is_refused():
    with pytest.raises(CollectionError) as caught:
        asyncio.run(collector().collect(trigger("other-plugin")))

    assert caught.value.code == "trigger-invalid"
    assert caught.value.detail == ("storage metadata requires a storage-intelligence trigger")


def test_emission_is_bounded_and_truncation_is_reported():
    samples = [sample(f"disk-{index}") for index in range(4)]
    subject = StorageIntelligenceCollector(
        ReplaySource([snapshot(*samples)], label="storage"),
        permissions(*[f"disk-{index}" for index in range(4)]),
        [f"disk-{index}" for index in range(4)],
        max_items=2,
    )
    payload = collect(subject)
    assert len(payload["items"]) == 2
    assert payload["truncatedItems"] == 2


def test_an_invalid_sample_shape_is_refused():
    assert storage_sample_error("not-a-sample") != ""
    assert storage_sample_error(sample(bus="nvme")) != ""
    assert storage_sample_error(sample(health="ok")) != ""
    assert storage_sample_error(sample(present="yes")) != ""


def test_an_invalid_device_identity_has_no_permission():
    with pytest.raises(ValueError, match="storage device identity"):
        storage_device_permission("Bad Disk")


def test_readiness_reports_the_source_health():
    subject = collector(snapshot(status=SourceStatus.DEGRADED))
    assert asyncio.run(subject.readiness()).status is SourceStatus.DEGRADED
