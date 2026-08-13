from __future__ import annotations

import asyncio

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins.collection import (
    MAX_COLLECTED_ITEMS,
    CollectionError,
    ReplaySource,
    SourceSnapshot,
    validated_allowlist,
)
from omnitensor.plugins.hardware_collection import (
    HARDWARE_HEALTH_PLUGIN_ID,
    HARDWARE_METADATA_PERMISSION,
    HardwareHealthCollector,
    HardwareSample,
    SensorHealth,
    SensorKind,
    hardware_sample_error,
    hardware_sensor_permission,
)
from omnitensor.plugins.triggers import SourceStatus, Trigger, TriggerKind
from omnitensor.sdk.helpers import PermissionView

CPU = "cpu-package-0"
DIMM = "dimm-0"


def permissions(*sensor_ids, metadata=True):
    scoped = {hardware_sensor_permission(sensor) for sensor in sensor_ids}
    declared = frozenset({HARDWARE_METADATA_PERMISSION, *scoped})
    granted = frozenset({*scoped, *({HARDWARE_METADATA_PERMISSION} if metadata else set())})
    return PermissionView(declared, granted)


def sample(stable_id=CPU, **changes):
    values = {
        "stable_id": stable_id,
        "kind": SensorKind.THERMAL,
        "health": SensorHealth.OK,
        "value": 45_000,
        "unit": "millidegC",
        "label": "CPU package",
        "observed_at_ms": 9,
    }
    values.update(changes)
    return HardwareSample(**values)


def snapshot(*samples, status=SourceStatus.READY, observed_at_ms=10):
    return SourceSnapshot(status, observed_at_ms, tuple(samples) or (sample(),))


def collector(*samples, allowed=(CPU,), granted=(CPU,), **changes):
    source = ReplaySource([snapshot(*samples)], label="hardware")
    return HardwareHealthCollector(
        source, permissions(*granted), allowed, **changes
    )


def trigger(payload=None, *, plugin_id=HARDWARE_HEALTH_PLUGIN_ID):
    return Trigger(
        plugin_id,
        "hardware-sample-1",
        TriggerKind.MANUAL,
        payload or {},
        1,
    )


def collect(subject):
    return asyncio.run(subject.collect(trigger())).payload


def test_only_allowlisted_and_granted_sensors_are_emitted():
    payload = collect(
        collector(
            sample(CPU),
            sample(DIMM, kind=SensorKind.MEMORY_ERRORS, value=3, unit="count"),
            sample("undeclared-sensor"),
            allowed=(CPU, DIMM),
            granted=(CPU,),
        )
    )

    assert [item["id"] for item in payload["items"]] == [CPU]


def test_a_missing_sensor_is_reported_rather_than_read_as_zero():
    """An absent thermal probe is not a cool machine."""
    payload = collect(
        collector(sample(CPU, health=SensorHealth.MISSING, value=None))
    )

    [item] = payload["items"]
    assert item["health"] == "missing"
    assert item["value"] is None


def test_a_missing_sensor_carrying_a_value_is_refused():
    with pytest.raises(CollectionError, match="must not carry a value"):
        collect(collector(sample(CPU, health=SensorHealth.MISSING, value=0)))


def test_a_present_sensor_without_a_value_is_refused():
    with pytest.raises(CollectionError, match="must carry a value"):
        collect(collector(sample(CPU, value=None)))


def test_units_are_carried_rather_than_assumed():
    payload = collect(collector(sample(CPU, value=45_000, unit="millidegC")))
    assert payload["items"][0]["unit"] == "millidegC"


def test_a_value_beyond_its_ceiling_is_refused():
    with pytest.raises(CollectionError, match="value"):
        collect(collector(sample(CPU, value=500_000)))


def test_a_service_percentage_is_bounded():
    with pytest.raises(CollectionError):
        collect(collector(sample(CPU, kind=SensorKind.SERVICE, value=101, unit="percent")))


def test_collection_without_the_metadata_grant_is_refused():
    source = ReplaySource([snapshot()], label="hardware")
    subject = HardwareHealthCollector(
        source, permissions(CPU, metadata=False), (CPU,)
    )
    with pytest.raises(CollectionError, match="permission-denied"):
        collect(subject)


def test_readiness_reports_a_missing_grant_without_reading_the_source():
    class ExplodingSource(ReplaySource):
        async def readiness(self):
            raise AssertionError("the source must not be read without consent")

    subject = HardwareHealthCollector(
        ExplodingSource([snapshot()], label="hardware"),
        permissions(CPU, metadata=False),
        (CPU,),
    )
    readiness = asyncio.run(subject.readiness())
    assert readiness.status is SourceStatus.UNAVAILABLE
    assert "permission is not granted" in readiness.detail


def test_readiness_contains_a_failing_source():
    class FailingSource(ReplaySource):
        async def readiness(self):
            raise OSError("hwmon is gone")

    subject = HardwareHealthCollector(
        FailingSource([snapshot()], label="hardware"), permissions(CPU), (CPU,)
    )
    readiness = asyncio.run(subject.readiness())
    assert readiness.status is SourceStatus.UNAVAILABLE
    assert "OSError" in readiness.detail


def test_readiness_reports_a_degraded_source():
    source = ReplaySource([snapshot(status=SourceStatus.DEGRADED)], label="hardware")
    subject = HardwareHealthCollector(source, permissions(CPU), (CPU,))
    readiness = asyncio.run(subject.readiness())
    assert readiness.status is SourceStatus.DEGRADED
    assert "degraded" in readiness.detail


def test_emission_is_bounded_and_truncation_is_reported():
    samples = [sample(f"sensor-{index}") for index in range(5)]
    subject = HardwareHealthCollector(
        ReplaySource([snapshot(*samples)], label="hardware"),
        permissions(*[f"sensor-{index}" for index in range(5)]),
        [f"sensor-{index}" for index in range(5)],
        max_items=2,
    )

    payload = collect(subject)

    assert len(payload["items"]) == 2
    assert payload["truncatedItems"] == 3


def test_churn_covers_every_eligible_sensor_not_only_the_emitted_ones():
    """A sensor past the cap is still attached, so it is not a removal."""
    source = ReplaySource(
        [snapshot(sample("sensor-a"), sample("sensor-b"))] * 2, label="hardware"
    )
    subject = HardwareHealthCollector(
        source, permissions("sensor-a", "sensor-b"), ("sensor-a", "sensor-b"), max_items=1
    )

    first = asyncio.run(subject.collect(trigger())).payload
    second = asyncio.run(subject.collect(trigger())).payload

    assert [item["id"] for item in first["churn"]["added"]] == ["sensor-a", "sensor-b"]
    assert second["churn"] == {"added": [], "removed": [], "changed": []}


def test_a_changed_reading_is_reported_but_a_new_timestamp_alone_is_not():
    source = ReplaySource(
        [
            snapshot(sample(CPU, value=45_000)),
            snapshot(sample(CPU, value=45_000, observed_at_ms=99), observed_at_ms=99),
            snapshot(sample(CPU, value=91_000), observed_at_ms=100),
        ],
        label="hardware",
    )
    subject = HardwareHealthCollector(source, permissions(CPU), (CPU,))

    asyncio.run(subject.collect(trigger()))
    unchanged = asyncio.run(subject.collect(trigger())).payload
    changed = asyncio.run(subject.collect(trigger())).payload

    assert unchanged["churn"]["changed"] == []
    assert changed["churn"]["changed"] == [{"id": CPU, "fields": ["value"]}]


def test_a_sensor_that_disappears_is_reported_as_removed():
    source = ReplaySource(
        [snapshot(sample(CPU), sample(DIMM)), snapshot(sample(CPU))], label="hardware"
    )
    subject = HardwareHealthCollector(
        source, permissions(CPU, DIMM), (CPU, DIMM)
    )

    asyncio.run(subject.collect(trigger()))
    second = asyncio.run(subject.collect(trigger())).payload

    assert [item["id"] for item in second["churn"]["removed"]] == [DIMM]


def test_source_health_is_carried_into_the_document():
    payload = collect(
        HardwareHealthCollector(
            ReplaySource([snapshot(status=SourceStatus.DEGRADED)], label="hardware"),
            permissions(CPU),
            (CPU,),
        )
    )
    assert payload["sourceHealth"] == "degraded"


def test_an_invalid_sample_type_is_refused():
    subject = HardwareHealthCollector(
        ReplaySource([SourceSnapshot(SourceStatus.READY, 1, ("not-a-sample",))]),
        permissions(CPU),
        (),
    )
    with pytest.raises(CollectionError, match="source-invalid"):
        collect(subject)


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "thermal"},
        {"health": "ok"},
        {"unit": "x" * 17},
        {"label": "x" * 81},
    ],
)
def test_a_malformed_sample_field_is_refused(changes):
    assert hardware_sample_error(sample(**changes)) != ""


def test_a_non_trigger_is_refused():
    with pytest.raises(CollectionError) as caught:
        asyncio.run(collector().collect("not-a-trigger"))

    assert caught.value.code == "trigger-invalid"
    assert caught.value.detail == "hardware health metadata requires a Trigger"


def test_a_trigger_for_another_plugin_is_refused():
    with pytest.raises(CollectionError) as caught:
        asyncio.run(collector().collect(trigger(plugin_id="other-plugin")))

    assert caught.value.code == "trigger-invalid"
    assert caught.value.detail == (
        "hardware health metadata requires a hardware-health trigger"
    )


@given(plugin_id=st.text().filter(lambda value: value != HARDWARE_HEALTH_PLUGIN_ID))
def test_every_foreign_plugin_identity_is_refused(plugin_id):
    with pytest.raises(CollectionError, match="trigger-invalid"):
        asyncio.run(collector().collect(trigger(plugin_id=plugin_id)))


def test_an_invalid_snapshot_timestamp_is_refused():
    subject = HardwareHealthCollector(
        ReplaySource([SourceSnapshot(SourceStatus.READY, -1, (sample(),))]),
        permissions(CPU),
        (CPU,),
    )
    with pytest.raises(CollectionError, match="timestamp"):
        collect(subject)


def test_an_oversized_snapshot_is_refused():
    items = tuple(sample(f"sensor-{index}") for index in range(MAX_COLLECTED_ITEMS + 1))
    subject = HardwareHealthCollector(
        ReplaySource([SourceSnapshot(SourceStatus.READY, 1, items)]), permissions(CPU), ()
    )
    with pytest.raises(CollectionError, match="more than"):
        collect(subject)


def test_an_empty_replay_is_unavailable():
    subject = HardwareHealthCollector(ReplaySource([]), permissions(CPU), (CPU,))
    assert asyncio.run(subject.readiness()).status is SourceStatus.UNAVAILABLE
    with pytest.raises(CollectionError, match="source-unavailable"):
        collect(subject)


@pytest.mark.parametrize("identity", ["", "Bad Sensor", "x" * 200])
def test_an_invalid_allowlist_identity_is_refused(identity):
    with pytest.raises(ValueError, match="allowlisted identity"):
        validated_allowlist({identity})


def test_an_allowlist_must_be_a_collection():
    with pytest.raises(TypeError, match="collection"):
        validated_allowlist("cpu-package-0")


@pytest.mark.parametrize("bound", [0, MAX_COLLECTED_ITEMS + 1, True, "8"])
def test_the_item_bound_is_validated(bound):
    with pytest.raises(ValueError, match="max_items"):
        HardwareHealthCollector(
            ReplaySource([snapshot()]), permissions(CPU), (CPU,), max_items=bound
        )


def test_the_source_and_gate_types_are_enforced():
    with pytest.raises(TypeError, match="MetadataSource"):
        HardwareHealthCollector("not-a-source", permissions(CPU), (CPU,))
    with pytest.raises(TypeError, match="CollectionPermissionGate"):
        HardwareHealthCollector(ReplaySource([snapshot()]), object(), (CPU,))


def test_the_clock_must_be_callable():
    with pytest.raises(TypeError, match="clock_ms"):
        HardwareHealthCollector(
            ReplaySource([snapshot()]), permissions(CPU), (CPU,), clock_ms="now"
        )


def test_an_invalid_sensor_identity_has_no_permission():
    with pytest.raises(ValueError, match="hardware sensor identity"):
        hardware_sensor_permission("Bad Sensor")
