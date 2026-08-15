"""Worker health and artifact readiness reach the telemetry a client reads.

Both fields existed on the contract and neither was ever populated: nothing
called ``set_health`` or ``set_artifact_readiness``, so a plugin whose worker
never started was indistinguishable from one nobody had asked for anything,
and every artifact reported ``unknown`` forever.
"""

from __future__ import annotations

from dataclasses import dataclass

from omnitensor import telemetry_observation as observation
from omnitensor.inspection import ArtifactReadiness as ArtifactItem
from omnitensor.plugins.supervisor_diagnostics import WorkerState
from omnitensor.plugins.telemetry import (
    ArtifactReadiness,
    PluginTelemetryHealth,
    PluginTelemetryRegistry,
)


@dataclass(frozen=True)
class FakePlugin:
    plugin_id: str
    manifest: dict


@dataclass(frozen=True)
class FakeWorker:
    plugin_id: str
    state: WorkerState
    detail: str = ""


def plugin(identifier: str, artifacts: list[dict] | None = None) -> FakePlugin:
    return FakePlugin(
        identifier,
        {"plugin": {"artifacts": artifacts if artifacts is not None else []}},
    )


@dataclass(frozen=True)
class FakeCatalog:
    """Satisfies PluginCatalogSnapshot structurally; the protocol is checked
    with isinstance, so the attributes are what matter."""

    plugins: tuple


@dataclass(frozen=True)
class FakeSnapshot:
    catalog: FakeCatalog
    workers: tuple


def snapshot(plugins, workers=()) -> FakeSnapshot:
    return FakeSnapshot(FakeCatalog(tuple(plugins)), tuple(workers))


def declared(identifier: str) -> dict:
    return {"id": identifier, "version": "1.0.0", "format": "gguf"}


class TestWorkerHealth:
    def test_a_worker_that_failed_is_unavailable_not_initializing(self):
        registry = PluginTelemetryRegistry()
        registry.register("broken")
        observation.record_worker_health(
            snapshot([plugin("broken")], [FakeWorker("broken", WorkerState.FAILED)]), registry
        )
        assert registry.snapshot("broken").health is PluginTelemetryHealth.UNAVAILABLE

    def test_an_exhausted_worker_is_unavailable_too(self):
        registry = PluginTelemetryRegistry()
        registry.register("spent")
        observation.record_worker_health(
            snapshot([plugin("spent")], [FakeWorker("spent", WorkerState.EXHAUSTED)]), registry
        )
        assert registry.snapshot("spent").health is PluginTelemetryHealth.UNAVAILABLE

    def test_a_stopped_or_exited_worker_reads_stopped(self):
        registry = PluginTelemetryRegistry()
        for identifier, state in (("a", WorkerState.STOPPED), ("b", WorkerState.EXITED)):
            registry.register(identifier)
            observation.record_worker_health(
                snapshot([plugin(identifier)], [FakeWorker(identifier, state)]), registry
            )
            assert registry.snapshot(identifier).health is PluginTelemetryHealth.STOPPED

    def test_transient_states_are_left_to_the_job_outcomes(self):
        # A ready worker has not proved it can serve a job; only a job can.
        registry = PluginTelemetryRegistry()
        registry.register("fine")
        for state in (WorkerState.READY, WorkerState.STARTING, WorkerState.RESTARTING):
            observation.record_worker_health(
                snapshot([plugin("fine")], [FakeWorker("fine", state)]), registry
            )
            assert registry.snapshot("fine").health is PluginTelemetryHealth.INITIALIZING

    def test_a_worker_for_an_unregistered_plugin_is_skipped_quietly(self):
        registry = PluginTelemetryRegistry()
        observation.record_worker_health(
            snapshot([], [FakeWorker("stranger", WorkerState.FAILED)]), registry
        )
        assert registry.snapshots() == ()

    def test_registration_records_health_in_the_same_pass(self):
        registry = PluginTelemetryRegistry()
        observation.register_plugin_telemetry(
            snapshot([plugin("broken")], [FakeWorker("broken", WorkerState.FAILED)]), registry
        )
        assert registry.snapshot("broken").health is PluginTelemetryHealth.UNAVAILABLE


class TestArtifactReadiness:
    def test_a_plugin_with_no_artifacts_has_nothing_missing(self):
        assert observation.artifact_readiness_state(()) is ArtifactReadiness.READY

    def test_every_artifact_ready_reads_ready(self):
        items = (ArtifactItem("a", "1.0.0", "gguf", True, ""),)
        assert observation.artifact_readiness_state(items) is ArtifactReadiness.READY

    def test_an_absent_artifact_reads_missing(self):
        items = (ArtifactItem("a", "1.0.0", "gguf", False, "artifact is not installed"),)
        assert observation.artifact_readiness_state(items) is ArtifactReadiness.MISSING

    def test_a_digest_mismatch_reads_rejected_not_missing(self):
        # The file is there and is the wrong file, which is a different problem.
        items = (ArtifactItem("a", "1.0.0", "gguf", False, "artifact sha256 mismatch"),)
        assert observation.artifact_readiness_state(items) is ArtifactReadiness.REJECTED

    def test_an_incompatible_artifact_says_so(self):
        items = (ArtifactItem("a", "1.0.0", "gguf", False, "unsupported format for this lane"),)
        assert observation.artifact_readiness_state(items) is ArtifactReadiness.INCOMPATIBLE

    def test_readiness_reaches_the_registry(self):
        registry = PluginTelemetryRegistry()
        registry.register("provider")
        resolutions = {"model": type("R", (), {"ready": True, "reason": ""})()}
        observation.record_artifact_readiness(
            snapshot([plugin("provider", [declared("model")])]),
            lambda artifact_id: resolutions[artifact_id],
            registry,
        )
        assert registry.snapshot("provider").artifact_readiness is ArtifactReadiness.READY

    def test_a_store_that_cannot_be_read_is_a_readiness_answer(self):
        registry = PluginTelemetryRegistry()
        registry.register("provider")

        def explode(_artifact_id):
            raise OSError("store is gone")

        observation.record_artifact_readiness(
            snapshot([plugin("provider", [declared("model")])]), explode, registry
        )
        assert registry.snapshot("provider").artifact_readiness is ArtifactReadiness.MISSING
