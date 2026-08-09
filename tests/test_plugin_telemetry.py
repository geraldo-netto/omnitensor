from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins import (
    MAX_PLUGIN_ACTIVE_JOBS,
    MAX_PLUGIN_QUEUE_DEPTH,
    MAX_TELEMETRY_COUNTER,
    MAX_TELEMETRY_DETAIL_CHARS,
    ArtifactReadiness,
    PipelineStage,
    PluginTelemetry,
    PluginTelemetryHealth,
    PluginTelemetryRegistry,
)


def test_registration_is_idempotent_sorted_bounded_and_removable():
    registry = PluginTelemetryRegistry(max_plugins=2)

    zeta = registry.register("zeta-plugin")
    alpha = registry.register("alpha-plugin")

    assert registry.register("zeta-plugin") is zeta
    assert registry.snapshots() == (alpha, zeta)
    assert zeta == PluginTelemetry(
        "zeta-plugin",
        PluginTelemetryHealth.INITIALIZING,
        None,
        ArtifactReadiness.UNKNOWN,
        0,
        0,
        None,
        None,
        "",
        None,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    with pytest.raises(ValueError, match="capacity is exhausted"):
        registry.register("third-plugin")
    assert registry.unregister("alpha-plugin")
    assert not registry.unregister("alpha-plugin")
    assert registry.register("third-plugin").plugin_id == "third-plugin"


@pytest.mark.parametrize("plugin_id", ["", "Bad_ID", "a" * 81, None, 1])
def test_plugin_ids_are_strict_at_every_public_lookup(plugin_id):
    registry = PluginTelemetryRegistry()
    for operation in (registry.register, registry.unregister, registry.snapshot):
        with pytest.raises(ValueError, match="plugin_id is invalid"):
            operation(plugin_id)


def test_health_and_artifact_readiness_are_typed_and_replace_snapshots():
    registry = PluginTelemetryRegistry()
    initial = registry.register("health-plugin")

    healthy = registry.set_health("health-plugin", PluginTelemetryHealth.HEALTHY)
    ready = registry.set_artifact_readiness(
        "health-plugin", ArtifactReadiness.READY
    )

    assert initial.health is PluginTelemetryHealth.INITIALIZING
    assert healthy.health is PluginTelemetryHealth.HEALTHY
    assert ready.artifact_readiness is ArtifactReadiness.READY
    assert registry.snapshot("health-plugin") is ready
    with pytest.raises(TypeError, match="health must"):
        registry.set_health("health-plugin", "healthy")
    with pytest.raises(TypeError, match="readiness must"):
        registry.set_artifact_readiness("health-plugin", "ready")


def test_success_lifecycle_tracks_stage_queue_deadline_retry_and_last_success():
    registry = PluginTelemetryRegistry()
    registry.register("pipeline-plugin")

    queued = registry.queue_job("pipeline-plugin")
    started = registry.start_job("pipeline-plugin")
    inferred = registry.advance("pipeline-plugin", PipelineStage.INFER)
    deadline = registry.deadline(
        "pipeline-plugin", "inference exceeded budget", observed_at_ms=10
    )
    retried = registry.retry("pipeline-plugin")
    succeeded = registry.succeed("pipeline-plugin", completed_at_ms=20)

    assert queued.queued_jobs == 1
    assert started.queued_jobs == 0 and started.active_jobs == 1
    assert started.current_stage is PipelineStage.COLLECT
    assert inferred.current_stage is PipelineStage.INFER
    assert deadline.deadline_exceeded == 1
    assert deadline.last_error_code == "deadline-exceeded"
    assert deadline.last_error_detail == "inference exceeded budget"
    assert deadline.last_error_at_ms == 10
    assert retried.retries == 1
    assert succeeded.active_jobs == 0
    assert succeeded.current_stage is PipelineStage.TERMINAL
    assert succeeded.health is PluginTelemetryHealth.HEALTHY
    assert succeeded.last_success_at_ms == 20
    assert succeeded.successes == 1


def test_failure_records_bounded_stable_error_without_erasing_last_success():
    registry = PluginTelemetryRegistry()
    registry.register("failure-plugin")
    registry.queue_job("failure-plugin")
    registry.start_job("failure-plugin")
    registry.succeed("failure-plugin", completed_at_ms=4)
    registry.queue_job("failure-plugin")
    registry.start_job("failure-plugin")

    failed = registry.fail(
        "failure-plugin",
        "executor-failed",
        "x" * (MAX_TELEMETRY_DETAIL_CHARS + 10),
        completed_at_ms=8,
    )

    assert failed.health is PluginTelemetryHealth.DEGRADED
    assert failed.active_jobs == 0
    assert failed.failures == 1
    assert failed.last_success_at_ms == 4
    assert failed.last_error_code == "executor-failed"
    assert failed.last_error_detail == "x" * MAX_TELEMETRY_DETAIL_CHARS
    assert failed.last_error_at_ms == 8


def test_cancel_handles_active_then_queued_and_drop_is_queue_only():
    registry = PluginTelemetryRegistry()
    registry.register("flow-plugin")
    registry.queue_job("flow-plugin")
    registry.queue_job("flow-plugin")
    registry.start_job("flow-plugin")

    active_cancelled = registry.cancel("flow-plugin")
    queued_cancelled = registry.cancel("flow-plugin")
    registry.queue_job("flow-plugin")
    dropped = registry.drop("flow-plugin")

    assert active_cancelled.active_jobs == 0
    assert active_cancelled.queued_jobs == 1
    assert active_cancelled.current_stage is PipelineStage.TERMINAL
    assert queued_cancelled.queued_jobs == 0
    assert queued_cancelled.cancellations == 2
    assert dropped.queued_jobs == 0
    assert dropped.drops == 1
    for operation in (registry.cancel, registry.drop):
        with pytest.raises(ValueError, match="no queued"):
            operation("flow-plugin")


def test_active_operations_reject_impossible_state_and_invalid_inputs():
    registry = PluginTelemetryRegistry()
    registry.register("strict-plugin")

    with pytest.raises(ValueError, match="no queued job to start"):
        registry.start_job("strict-plugin")
    for operation in (
        lambda: registry.advance("strict-plugin", PipelineStage.INFER),
        lambda: registry.succeed("strict-plugin", completed_at_ms=1),
        lambda: registry.retry("strict-plugin"),
        lambda: registry.deadline("strict-plugin", "late", observed_at_ms=1),
    ):
        with pytest.raises(ValueError, match="no active job"):
            operation()
    for stage in (PipelineStage.QUEUED, PipelineStage.TERMINAL, "infer", None):
        with pytest.raises(ValueError, match="stage must be runnable"):
            registry.advance("strict-plugin", stage)


@pytest.mark.parametrize("timestamp", [-1, True, 1.5, "1", None])
def test_telemetry_timestamps_are_non_negative_integers(timestamp):
    registry = PluginTelemetryRegistry()
    registry.register("time-plugin")
    registry.queue_job("time-plugin")
    registry.start_job("time-plugin")
    with pytest.raises(ValueError, match="timestamp"):
        registry.succeed("time-plugin", completed_at_ms=timestamp)
    with pytest.raises(ValueError, match="timestamp"):
        registry.fail("time-plugin", "failed", "detail", completed_at_ms=timestamp)
    with pytest.raises(ValueError, match="timestamp"):
        registry.deadline("time-plugin", "detail", observed_at_ms=timestamp)


@pytest.mark.parametrize("code", ["", "Bad_Code", "a" * 81, None, 1])
def test_error_codes_are_bounded_wire_identifiers(code):
    registry = PluginTelemetryRegistry()
    registry.register("error-plugin")
    registry.queue_job("error-plugin")
    registry.start_job("error-plugin")
    with pytest.raises(ValueError, match="error code"):
        registry.fail("error-plugin", code, "detail", completed_at_ms=1)
    with pytest.raises(TypeError, match="detail must be a string"):
        registry.fail("error-plugin", "valid-code", None, completed_at_ms=1)


def test_queue_active_and_counter_hard_bounds_are_enforced():
    registry = PluginTelemetryRegistry()
    snapshot = registry.register("bounded-plugin")
    registry._items["bounded-plugin"] = replace(
        snapshot,
        queued_jobs=MAX_PLUGIN_QUEUE_DEPTH,
    )
    with pytest.raises(ValueError, match="queue depth is exhausted"):
        registry.queue_job("bounded-plugin")

    registry._items["bounded-plugin"] = replace(
        snapshot,
        queued_jobs=1,
        active_jobs=MAX_PLUGIN_ACTIVE_JOBS,
    )
    with pytest.raises(ValueError, match="active-job telemetry is exhausted"):
        registry.start_job("bounded-plugin")

    registry._items["bounded-plugin"] = replace(
        snapshot,
        active_jobs=1,
        retries=MAX_TELEMETRY_COUNTER,
    )
    assert registry.retry("bounded-plugin").retries == MAX_TELEMETRY_COUNTER


def test_restart_recovery_clears_only_transient_runtime_state():
    registry = PluginTelemetryRegistry()
    registry.register("recovery-plugin")
    registry.queue_job("recovery-plugin")
    registry.start_job("recovery-plugin")
    registry.retry("recovery-plugin")

    recovered = registry.recover_runtime("recovery-plugin")

    assert recovered.queued_jobs == recovered.active_jobs == 0
    assert recovered.current_stage is None
    assert recovered.retries == 1


def test_snapshots_are_frozen_and_unknown_plugins_fail_deterministically():
    registry = PluginTelemetryRegistry()
    snapshot = registry.register("frozen-plugin")
    with pytest.raises(FrozenInstanceError):
        snapshot.health = PluginTelemetryHealth.HEALTHY
    with pytest.raises(KeyError, match="not registered: missing-plugin"):
        registry.snapshot("missing-plugin")


def test_concurrent_queue_updates_do_not_lose_counts():
    registry = PluginTelemetryRegistry()
    registry.register("threaded-plugin")

    with ThreadPoolExecutor(max_workers=8) as pool:
        snapshots = list(
            pool.map(lambda _index: registry.queue_job("threaded-plugin"), range(200))
        )

    assert len(snapshots) == 200
    assert registry.snapshot("threaded-plugin").queued_jobs == 200


@given(events=st.lists(st.sampled_from(["queue", "drop"]), max_size=100))
def test_arbitrary_queue_drop_stream_never_underflows(events):
    registry = PluginTelemetryRegistry()
    registry.register("property-plugin")
    expected = 0
    for event in events:
        if event == "queue":
            registry.queue_job("property-plugin")
            expected += 1
        elif expected:
            registry.drop("property-plugin")
            expected -= 1
        else:
            with pytest.raises(ValueError, match="no queued job"):
                registry.drop("property-plugin")
        assert registry.snapshot("property-plugin").queued_jobs == expected


@pytest.mark.parametrize("max_plugins", [0, -1, True, 129, 1.5, "1"])
def test_registry_capacity_is_strict(max_plugins):
    with pytest.raises(ValueError, match="max_plugins"):
        PluginTelemetryRegistry(max_plugins=max_plugins)
