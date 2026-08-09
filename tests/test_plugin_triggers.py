from __future__ import annotations

import asyncio

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins import (
    CollectedOutput,
    CollectionOutcome,
    CollectionStatus,
    Collector,
    CollectorReadiness,
    SourceStatus,
    Trigger,
    TriggerCoordinator,
    TriggerDisposition,
    TriggerKind,
    TriggerValidationError,
)


class FakeCollector:
    def __init__(self, output=None, readiness=None):
        self.output = output or CollectedOutput({"sample": 1})
        self.source_readiness = readiness or CollectorReadiness(SourceStatus.READY, "ready", 1)
        self.triggers = []

    async def readiness(self):
        return self.source_readiness

    async def collect(self, trigger):
        self.triggers.append(trigger)
        return self.output


def coordinator(**overrides):
    instance = TriggerCoordinator(**overrides)
    collector = FakeCollector()
    instance.register_collector("echo-plugin", collector)
    return instance, collector


def test_collector_is_a_structural_async_contract():
    assert isinstance(FakeCollector(), Collector)
    assert not isinstance(object(), Collector)


def test_manual_and_event_triggers_are_queued_in_submission_order():
    instance, collector = coordinator()
    manual = instance.submit_manual("echo-plugin", "same-id", {"manual": True}, created_at_ms=1)
    event = instance.submit_event("echo-plugin", "same-id", {"event": True}, created_at_ms=2)

    assert manual.disposition is TriggerDisposition.ACCEPTED
    assert manual.pending_count == 1
    assert event.disposition is TriggerDisposition.ACCEPTED
    assert event.pending_count == 2

    async def collect():
        first = await instance.collect_next()
        second = await instance.collect_next()
        empty = await instance.collect_next()
        return first, second, empty

    first, second, empty = asyncio.run(collect())
    assert first.status is second.status is CollectionStatus.SUCCEEDED
    assert [trigger.kind for trigger in collector.triggers] == [
        TriggerKind.MANUAL,
        TriggerKind.EVENT,
    ]
    assert empty is None
    assert instance.pending_count == 0


def test_duplicate_pending_trigger_is_coalesced_and_can_be_resubmitted_after_collection():
    instance, collector = coordinator()
    first = instance.submit_event("echo-plugin", "event-1", {"revision": 1}, created_at_ms=1)
    duplicate = instance.submit_event("echo-plugin", "event-1", {"revision": 2}, created_at_ms=2)

    assert first.disposition is TriggerDisposition.ACCEPTED
    assert duplicate.disposition is TriggerDisposition.COALESCED
    assert duplicate.pending_count == 1
    assert duplicate.trigger.payload == {"revision": 2}

    asyncio.run(instance.collect_next())
    again = instance.submit_event("echo-plugin", "event-1", {"revision": 3}, created_at_ms=3)
    assert again.disposition is TriggerDisposition.ACCEPTED
    assert collector.triggers[0].payload == {"revision": 1}


def test_queue_capacity_is_hard_and_does_not_discard_existing_work():
    instance, _collector = coordinator(max_pending=1)
    accepted = instance.submit_manual("echo-plugin", "one", {}, created_at_ms=0)
    rejected = instance.submit_manual("echo-plugin", "two", {}, created_at_ms=0)

    assert accepted.disposition is TriggerDisposition.ACCEPTED
    assert rejected.disposition is TriggerDisposition.QUEUE_FULL
    assert rejected.pending_count == instance.pending_count == 1
    assert rejected.trigger.trigger_id == "two"


def test_periodic_tick_coalesces_missed_intervals_and_orders_plugin_ids():
    instance = TriggerCoordinator()
    collector_a = FakeCollector()
    collector_b = FakeCollector()
    instance.register_collector("plugin-b", collector_b)
    instance.register_collector("plugin-a", collector_a)
    instance.register_periodic(
        "plugin-b",
        interval_ms=10,
        first_due_ms=5,
        payload={"source": "b"},
    )
    instance.register_periodic(
        "plugin-a",
        interval_ms=20,
        first_due_ms=5,
        payload={"source": "a"},
    )

    assert instance.tick(4) == ()
    first = instance.tick(100)
    assert [receipt.trigger.plugin_id for receipt in first] == ["plugin-a", "plugin-b"]
    assert [receipt.trigger.trigger_id for receipt in first] == ["periodic-1", "periodic-1"]
    assert all(receipt.trigger.created_at_ms == 100 for receipt in first)
    assert instance.tick(109) == ()
    second = instance.tick(110)
    assert [receipt.trigger.plugin_id for receipt in second] == ["plugin-b"]
    assert second[0].trigger.trigger_id == "periodic-2"


def test_periodic_registration_is_validated_and_unique():
    instance, _collector = coordinator()
    for invalid_interval in (0, True, 1.5, "1"):
        with pytest.raises(ValueError, match="interval_ms must be a positive integer"):
            instance.register_periodic(
                "echo-plugin",
                interval_ms=invalid_interval,
                first_due_ms=0,
                payload={},
            )
    with pytest.raises(ValueError, match="first_due_ms must be a non-negative integer"):
        instance.register_periodic(
            "echo-plugin",
            interval_ms=1,
            first_due_ms=-1,
            payload={},
        )

    instance.register_periodic(
        "echo-plugin",
        interval_ms=1,
        first_due_ms=0,
        payload={},
    )
    with pytest.raises(ValueError, match="periodic trigger already registered: echo-plugin"):
        instance.register_periodic(
            "echo-plugin",
            interval_ms=1,
            first_due_ms=0,
            payload={},
        )


def test_collection_returns_a_typed_success_with_the_original_trigger():
    instance, collector = coordinator()
    receipt = instance.submit_manual("echo-plugin", "job-1", {"input": 1}, created_at_ms=2)

    outcome = asyncio.run(instance.collect_next())

    assert outcome == CollectionOutcome(
        receipt.trigger,
        CollectionStatus.SUCCEEDED,
        collector.output,
        "",
    )


@pytest.mark.parametrize(
    ("behavior", "status", "detail"),
    [
        ("timeout", CollectionStatus.TIMED_OUT, "collection timed out"),
        ("error", CollectionStatus.FAILED, "collection failed: RuntimeError"),
        ("invalid", CollectionStatus.FAILED, "collector returned an invalid output"),
        ("oversized", CollectionStatus.FAILED, "payload is 17 bytes; limit is 8"),
    ],
)
def test_collection_contains_timeouts_failures_and_invalid_outputs(behavior, status, detail):
    class CollectorWithFailure(FakeCollector):
        async def collect(self, trigger):
            if behavior == "timeout":
                await asyncio.Event().wait()
            if behavior == "error":
                raise RuntimeError("secret detail")
            if behavior == "invalid":
                return {"not": "typed"}
            return CollectedOutput({"value": "12345"})

    instance = TriggerCoordinator(max_trigger_bytes=8, collection_timeout_seconds=0.001)
    instance.register_collector("echo-plugin", CollectorWithFailure())
    receipt = instance.submit_manual("echo-plugin", "job", {}, created_at_ms=0)

    outcome = asyncio.run(instance.collect_next())

    assert outcome.status is status
    assert outcome.trigger == receipt.trigger
    assert outcome.output is None
    assert outcome.detail == detail
    assert "secret" not in outcome.detail


@pytest.mark.parametrize(
    ("behavior", "expected"),
    [
        ("success", CollectorReadiness(SourceStatus.DEGRADED, "sensor missing", 9)),
        ("timeout", CollectorReadiness(SourceStatus.UNAVAILABLE, "readiness timed out", 0)),
        (
            "error",
            CollectorReadiness(SourceStatus.UNAVAILABLE, "readiness failed: RuntimeError", 0),
        ),
        ("invalid", CollectorReadiness(SourceStatus.UNAVAILABLE, "invalid readiness response", 0)),
    ],
)
def test_readiness_is_async_bounded_and_contains_adapter_failures(behavior, expected):
    class ReadinessCollector(FakeCollector):
        async def readiness(self):
            if behavior == "timeout":
                await asyncio.Event().wait()
            if behavior == "error":
                raise RuntimeError("private source")
            if behavior == "invalid":
                return "ready"
            return CollectorReadiness(SourceStatus.DEGRADED, "sensor missing", 9)

    instance = TriggerCoordinator(collection_timeout_seconds=0.001)
    instance.register_collector("echo-plugin", ReadinessCollector())

    assert asyncio.run(instance.readiness("echo-plugin")) == expected


def test_configured_deadline_is_forwarded_to_readiness_and_collection(monkeypatch):
    observed = []

    async def checked_wait_for(awaitable, *, timeout):
        result = await awaitable
        observed.append(timeout)
        return result

    monkeypatch.setattr(asyncio, "wait_for", checked_wait_for)
    instance, _collector = coordinator(collection_timeout_seconds=0.25)
    instance.submit_manual("echo-plugin", "job", {}, created_at_ms=0)

    async def scenario():
        await instance.readiness("echo-plugin")
        await instance.collect_next()

    asyncio.run(scenario())
    assert observed == [0.25, 0.25]


def test_payload_size_uses_exact_compact_utf8_bytes():
    instance, _collector = coordinator(max_trigger_bytes=8)
    assert (
        instance.submit_manual("echo-plugin", "exact", {"x": ""}, created_at_ms=0).disposition
        is TriggerDisposition.ACCEPTED
    )
    with pytest.raises(TriggerValidationError) as too_large:
        instance.submit_manual("echo-plugin", "large", {"x": "a"}, created_at_ms=0)
    assert str(too_large.value) == "payload-too-large: payload is 9 bytes; limit is 8"


def test_payload_size_counts_utf8_bytes_without_ascii_escaping():
    instance, _collector = coordinator(max_trigger_bytes=10)
    receipt = instance.submit_manual("echo-plugin", "unicode", {"x": "é"}, created_at_ms=0)
    assert receipt.disposition is TriggerDisposition.ACCEPTED


@pytest.mark.parametrize("payload", [[], {"value": object()}, {"value": float("nan")}])
def test_payload_must_be_a_finite_json_object(payload):
    instance, _collector = coordinator()
    with pytest.raises(TriggerValidationError) as excinfo:
        instance.submit_manual("echo-plugin", "job", payload, created_at_ms=0)
    assert excinfo.value.code == "invalid-payload"


@pytest.mark.parametrize(
    ("plugin_id", "trigger_id", "timestamp", "code"),
    [
        ("../escape", "job", 0, "invalid-plugin-id"),
        ("echo-plugin", "", 0, "invalid-trigger-id"),
        ("echo-plugin", "x" * 129, 0, "invalid-trigger-id"),
        ("echo-plugin", "job", -1, "timestamp"),
    ],
)
def test_trigger_boundary_metadata_is_validated(plugin_id, trigger_id, timestamp, code):
    instance, _collector = coordinator()
    error_type = ValueError if code == "timestamp" else TriggerValidationError
    with pytest.raises(error_type):
        instance.submit_manual(plugin_id, trigger_id, {}, created_at_ms=timestamp)


def test_generic_submission_rejects_an_unknown_trigger_kind():
    instance, _collector = coordinator()
    with pytest.raises(TriggerValidationError) as excinfo:
        instance.submit(Trigger("echo-plugin", "job", "unknown", {}, 0))
    assert excinfo.value.code == "invalid-trigger-kind"


def test_registration_rejects_missing_contract_duplicates_and_unknown_plugins():
    instance = TriggerCoordinator()
    with pytest.raises(TypeError, match="collector must implement readiness and collect"):
        instance.register_collector("echo-plugin", object())

    instance.register_collector("echo-plugin", FakeCollector())
    with pytest.raises(ValueError, match="collector already registered: echo-plugin"):
        instance.register_collector("echo-plugin", FakeCollector())
    with pytest.raises(KeyError, match="collector not registered: missing-plugin"):
        instance.submit_event("missing-plugin", "event", {}, created_at_ms=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_trigger_bytes": 0},
        {"max_trigger_bytes": True},
        {"max_pending": 0},
        {"max_pending": True},
        {"collection_timeout_seconds": 0},
        {"collection_timeout_seconds": True},
    ],
)
def test_coordinator_limits_are_positive_numbers(kwargs):
    with pytest.raises(ValueError):
        TriggerCoordinator(**kwargs)


@given(trigger_ids=st.lists(st.text(min_size=1, max_size=20), max_size=100))
def test_arbitrary_duplicate_manual_bursts_never_exceed_queue_bound(trigger_ids):
    instance, _collector = coordinator(max_pending=8)
    for timestamp, trigger_id in enumerate(trigger_ids):
        instance.submit_manual("echo-plugin", trigger_id, {}, created_at_ms=timestamp)
    assert instance.pending_count <= 8
