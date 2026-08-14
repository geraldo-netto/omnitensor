from __future__ import annotations

import asyncio

import pytest

from omnitensor.plugins.collection import CollectionError, ReplaySource, SourceSnapshot
from omnitensor.plugins.kernel_telemetry import parse_aggregate
from omnitensor.plugins.resource_collection import (
    RESOURCE_METADATA_PERMISSION,
    RESOURCE_PLUGIN_ID,
    PressureStall,
    ResourceSample,
    ResourceSchedulerCollector,
    ResourceScope,
    UnitState,
    resource_sample_error,
    resource_unit_permission,
)
from omnitensor.plugins.triggers import SourceStatus, Trigger, TriggerKind
from omnitensor.sdk.helpers import PermissionView

UNIT = "user.slice"
QUEUE = "render-queue"


def permissions(*unit_ids, metadata=True):
    scoped = {resource_unit_permission(unit) for unit in unit_ids}
    declared = frozenset({RESOURCE_METADATA_PERMISSION, *scoped})
    granted = frozenset({*scoped, *({RESOURCE_METADATA_PERMISSION} if metadata else set())})
    return PermissionView(declared, granted)


def sample(stable_id=UNIT, **changes):
    values = {
        "stable_id": stable_id,
        "scope": ResourceScope.SLICE,
        "state": UnitState.ACTIVE,
        "cpu_usage_milli_percent": 12_500,
        "memory_current_bytes": 2**30,
        "task_count": 42,
        "queue_depth": None,
        "cpu_pressure": PressureStall(1_000, 200),
        "io_pressure": None,
        "memory_pressure": None,
        "observed_at_ms": 9,
    }
    values.update(changes)
    return ResourceSample(**values)


def snapshot(*samples, status=SourceStatus.READY):
    return SourceSnapshot(status, 10, tuple(samples) or (sample(),))


def collector(*snapshots, allowed=(UNIT,), granted=(UNIT,), **changes):
    source = ReplaySource(list(snapshots) or [snapshot()], label="resource")
    return ResourceSchedulerCollector(source, permissions(*granted), allowed, **changes)


class KernelSource:
    def read(self):
        return parse_aggregate({
            "version": 1,
            "collectedAtMs": 11,
            "histograms": [
                {"name": "runq_latency_us", "unit": "us", "buckets": [1, 3]},
                {"name": "block_latency_us", "unit": "us", "buckets": [2]},
            ],
            "counters": [],
        })


def trigger(plugin_id=RESOURCE_PLUGIN_ID):
    return Trigger(plugin_id, "resource-1", TriggerKind.PERIODIC, {}, 1)


def collect(subject):
    return asyncio.run(subject.collect(trigger())).payload


def test_only_allowlisted_and_granted_units_are_emitted():
    payload = collect(
        collector(
            snapshot(sample(UNIT), sample(QUEUE, scope=ResourceScope.QUEUE), sample("other.slice")),
            allowed=(UNIT, QUEUE),
            granted=(UNIT,),
        )
    )
    assert [item["id"] for item in payload["items"]] == [UNIT]


def test_pressure_is_carried_because_utilisation_alone_hides_thrashing():
    """PSI distinguishes a busy machine from one waiting on a resource."""
    payload = collect(collector(snapshot(sample(cpu_pressure=PressureStall(9_000, 4_000)))))
    assert payload["items"][0]["cpuPressure"] == {
        "someMilliPercent": 9_000,
        "fullMilliPercent": 4_000,
    }


def test_kernel_histograms_become_bounded_scheduler_features():
    payload = collect(collector(kernel_source=KernelSource()))

    assert payload["kernelTelemetry"]["state"] == "ready"
    assert payload["kernelFeatures"] == {
        "kernelRunQueueSamples": 4.0,
        "kernelRunQueueP50UpperUs": 4.0,
        "kernelRunQueueP95UpperUs": 4.0,
        "kernelBlockIoSamples": 2.0,
        "kernelBlockIoP50UpperUs": 2.0,
        "kernelBlockIoP95UpperUs": 2.0,
    }


def test_invalid_kernel_extension_is_refused_before_churn_commit():
    class MutableKernelSource(KernelSource):
        invalid = False

        def read(self):
            aggregate = super().read()
            if not self.invalid:
                return aggregate

            class InvalidAggregate:
                def document(self):
                    return {**aggregate.document(), "version": 2}

                def scheduler_features(self):
                    return aggregate.scheduler_features()

            return InvalidAggregate()

    kernel = MutableKernelSource()
    changed = snapshot(sample(QUEUE, scope=ResourceScope.QUEUE))
    subject = collector(
        snapshot(sample(UNIT)),
        changed,
        changed,
        allowed=(UNIT, QUEUE),
        granted=(UNIT, QUEUE),
        kernel_source=kernel,
    )
    collect(subject)

    kernel.invalid = True
    with pytest.raises(CollectionError) as caught:
        collect(subject)
    assert caught.value.code == "output-invalid"
    assert caught.value.detail == "kernelTelemetry/version: 1 was expected"

    kernel.invalid = False
    recovered = collect(subject)
    assert recovered["churn"] == {
        "added": [{"id": QUEUE}],
        "removed": [{"id": UNIT}],
        "changed": [],
    }


def test_full_pressure_exceeding_some_is_refused():
    """full counts windows where everything stalled, a subset of some."""
    assert "full exceeds some" in resource_sample_error(
        sample(cpu_pressure=PressureStall(100, 900))
    )


def test_absent_pressure_is_allowed():
    assert resource_sample_error(sample(cpu_pressure=None)) == ""


def test_an_invalid_pressure_shape_is_refused():
    assert resource_sample_error(sample(io_pressure="none")) != ""
    assert resource_sample_error(sample(memory_pressure=PressureStall(-1, 0))) != ""


def test_readings_are_bounded():
    assert resource_sample_error(sample(cpu_usage_milli_percent=100_001)) != ""
    assert resource_sample_error(sample(task_count=-1)) != ""
    assert resource_sample_error(sample(queue_depth=2**31 + 1)) != ""


def test_a_queue_sample_carries_its_depth():
    payload = collect(
        collector(
            snapshot(sample(QUEUE, scope=ResourceScope.QUEUE, queue_depth=7)),
            allowed=(QUEUE,),
            granted=(QUEUE,),
        )
    )
    assert payload["items"][0]["queueDepth"] == 7
    assert payload["items"][0]["scope"] == "queue"


def test_a_failed_unit_is_reported_as_a_change():
    subject = collector(
        snapshot(sample(UNIT, state=UnitState.ACTIVE)),
        snapshot(sample(UNIT, state=UnitState.FAILED)),
    )
    asyncio.run(subject.collect(trigger()))
    second = asyncio.run(subject.collect(trigger())).payload
    assert second["churn"]["changed"] == [{"id": UNIT, "fields": ["state"]}]


def test_collection_without_the_metadata_grant_is_refused():
    subject = ResourceSchedulerCollector(
        ReplaySource([snapshot()], label="resource"),
        permissions(UNIT, metadata=False),
        (UNIT,),
    )
    with pytest.raises(CollectionError, match="permission-denied"):
        collect(subject)


def test_a_trigger_for_another_plugin_is_refused():
    with pytest.raises(CollectionError) as caught:
        asyncio.run(collector().collect(trigger("other-plugin")))

    assert caught.value.code == "trigger-invalid"
    assert caught.value.detail == "resource metadata requires a resource-scheduler trigger"


def test_emission_is_bounded():
    samples = [sample(f"unit-{index}.slice") for index in range(4)]
    subject = ResourceSchedulerCollector(
        ReplaySource([snapshot(*samples)], label="resource"),
        permissions(*[f"unit-{index}.slice" for index in range(4)]),
        [f"unit-{index}.slice" for index in range(4)],
        max_items=2,
    )
    payload = collect(subject)
    assert len(payload["items"]) == 2
    assert payload["truncatedItems"] == 2


def test_an_invalid_sample_shape_is_refused():
    assert resource_sample_error("not-a-sample") != ""
    assert resource_sample_error(sample(scope="slice")) != ""
    assert resource_sample_error(sample(state="active")) != ""


def test_an_invalid_unit_identity_has_no_permission():
    with pytest.raises(ValueError, match="resource unit identity"):
        resource_unit_permission("Bad Unit")


def test_readiness_reports_the_source_health():
    subject = collector(snapshot(status=SourceStatus.DEGRADED))
    assert asyncio.run(subject.readiness()).status is SourceStatus.DEGRADED
