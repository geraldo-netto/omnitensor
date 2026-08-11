"""Service loop and adapter tests through the injected ports — no D-Bus, no
monkeypatching: fakes are provided via constructor injection."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path

import pytest
from conftest import (
    add_npu,
    add_pcie_tpu,
    sample_manifest,
    sample_plugin_manifest,
    write_workload,
)

from omnitensor.discovery import Device
from omnitensor.plugins.artifacts import ArtifactResolution
from omnitensor.registry import validate_document
from omnitensor.service import (
    FileSnapshotPublisher,
    OmniTensorInterface,
    OmniTensorService,
    SysfsDeviceDiscovery,
    _admit_plugin_job,
    _PluginAwareDispatcher,
)


def tpu_device() -> Device:
    return Device(id="tpu-pcie-0", backend="tpu", name="Coral PCIe Edge TPU", kind="pcie")


def npu_device() -> Device:
    return Device(id="npu-accel0", backend="npu", name="Intel NPU", kind="accel")


def test_plugin_dispatcher_routes_profiles_and_refuses_an_incomplete_plugin_runtime():
    class Inference:
        def __init__(self):
            self.admissions = []
            self.dispatches = []

        def admit(self, workload_id, payload):
            self.admissions.append((workload_id, payload))

        def dispatch(self, job_id, workload_id, payload):
            self.dispatches.append((job_id, workload_id, payload))
            return job_id, workload_id, payload

    class Plugins:
        def __init__(self):
            self.admissions = []
            self.dispatches = []

        def plugin_ids(self):
            return frozenset({"events"})

        def admit(self, workload_id, payload):
            self.admissions.append((workload_id, payload))

        def dispatch(self, job_id, workload_id, payload):
            self.dispatches.append((job_id, workload_id, payload))
            return {"events": []}

    inference = Inference()
    plugins = Plugins()
    dispatcher = _PluginAwareDispatcher(inference, plugins)

    dispatcher.admit("events", {"sources": ["selected"]})
    dispatcher.admit("profile", {"value": 1})
    assert plugins.admissions == [("events", {"sources": ["selected"]})]
    assert inference.admissions == [("profile", {"value": 1})]
    assert dispatcher.dispatch("event-job", "events", {"sources": []}) == {
        "events": []
    }
    assert plugins.dispatches == [("event-job", "events", {"sources": []})]

    assert dispatcher.dispatch("job", "profile", {"value": 1}) == (
        "job",
        "profile",
        {"value": 1},
    )
    assert inference.dispatches == [("job", "profile", {"value": 1})]

    incomplete_plugins = type(
        "Plugins", (), {"plugin_ids": lambda _self: {"events"}}
    )()
    incomplete = _PluginAwareDispatcher(inference, incomplete_plugins)
    with pytest.raises(RuntimeError) as missing_admission:
        incomplete.admit("events", {})
    assert str(missing_admission.value) == "plugin runtime does not implement admission"
    with pytest.raises(RuntimeError) as missing_dispatch:
        incomplete.dispatch("job", "events", {})
    assert str(missing_dispatch.value) == "plugin runtime does not implement dispatch"

    no_inventory = _PluginAwareDispatcher(inference, object())
    assert no_inventory._plugin_ids() == frozenset()


def test_plugin_admission_helper_routes_exact_ports_directly():
    class Inference:
        def __init__(self):
            self.calls = []

        def admit(self, workload_id, payload):
            self.calls.append((workload_id, payload))

        def dispatch(self, *_arguments):
            raise AssertionError("admission must not dispatch")

    class Plugins:
        def __init__(self):
            self.calls = []

        def admit(self, workload_id, payload):
            self.calls.append((workload_id, payload))

    inference = Inference()
    plugins = Plugins()
    provider_calls = []

    def plugin_ids():
        provider_calls.append(True)
        return frozenset({"events"})

    _admit_plugin_job(inference, plugins, plugin_ids, "events", {"source": 1})
    _admit_plugin_job(inference, plugins, plugin_ids, "profile", {"value": 2})

    assert provider_calls == [True, True]
    assert plugins.calls == [("events", {"source": 1})]
    assert inference.calls == [("profile", {"value": 2})]

    with pytest.raises(RuntimeError) as incomplete:
        _admit_plugin_job(inference, object(), plugin_ids, "events", {})
    assert str(incomplete.value) == "plugin runtime does not implement admission"


class FakeDiscovery:
    """DeviceDiscovery fake with a mutable device list."""

    def __init__(self, devices: list[Device], utilization: float | None = None):
        self.devices = devices
        self.utilization_value = utilization
        self.detect_calls = 0

    def detect(self) -> list[Device]:
        self.detect_calls += 1
        return list(self.devices)

    def utilization(self, device: Device) -> float | None:
        return self.utilization_value


class FakePublisher:
    def __init__(self):
        self.published: list[dict] = []
        self.retracted = 0

    def publish(self, snapshot: dict) -> None:
        self.published.append(snapshot)

    def retract(self) -> None:
        self.retracted += 1


class FakeTransport:
    def __init__(self):
        self.handler = None
        self.started = 0
        self.stopped = 0

    async def start(self, handler) -> None:
        self.handler = handler
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1


def build_service(tmp_path, manifests=(), **kwargs):
    workloads_root = tmp_path / "workloads"
    workloads_root.mkdir(exist_ok=True)
    for manifest in manifests:
        write_workload(workloads_root, manifest)
    return OmniTensorService(
        snapshot_path=tmp_path / "state/runtime-snapshot.json",
        policy_path=tmp_path / "state/policy.json",
        workloads_path=workloads_root,
        **kwargs,
    )


def test_run_serves_control_publishes_and_rediscovers(tmp_path):
    discovery = FakeDiscovery([tpu_device()])
    publisher = FakePublisher()
    transport = FakeTransport()

    async def scenario():
        service = build_service(
            tmp_path,
            [sample_manifest()],
            discovery=discovery,
            publisher=publisher,
            transport=transport,
            publish_interval_s=0.01,
            discovery_interval_s=0.01,
        )
        original_tpu = service._executors["tpu"]
        runner = asyncio.get_running_loop().create_task(service.run())
        await asyncio.sleep(0.08)
        discovery.devices.append(npu_device())
        await asyncio.sleep(0.05)
        service._stopping.set()
        await asyncio.wait_for(runner, timeout=2)
        return service, original_tpu

    service, original_tpu = asyncio.run(scenario())
    assert transport.started == 1
    assert transport.stopped == 1
    assert transport.handler is service.runtime_api
    assert len(publisher.published) >= 2
    for snapshot in publisher.published:
        assert validate_document("runtime-snapshot.schema.json", snapshot) == []
    assert discovery.detect_calls >= 2
    backends = {device.backend for device in service._devices}
    assert backends == {"tpu", "npu"}
    assert service._executors["tpu"] is original_tpu
    assert [device["backend"] for device in publisher.published[-1]["devices"]] == ["tpu", "npu"]


def test_service_authorizes_installed_plugins_without_weakening_profile_policy(tmp_path):
    service = build_service(
        tmp_path,
        [sample_manifest()],
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )
    service._plugin_runtime = type(
        "Plugins", (), {"plugin_ids": lambda _self: frozenset({"events"})}
    )()

    assert service._job_authorized("submit", "events") is True
    assert service._job_authorized("cancel", "events") is True
    assert service._job_authorized("submit", "missing") is False
    assert service._job_authorized("cancel", "missing") is False
    assert service._job_authorized("submit", "sample-workload") is True
    service._admits = lambda _workload_id: False
    assert service._job_authorized("submit", "sample-workload") is False
    assert service._job_authorized("cancel", "sample-workload") is True
    assert service._job_authorized("submit", "events") is False
    assert service._job_authorized("cancel", "events") is True

    service._plugin_runtime = object()
    assert service._job_authorized("submit", "missing") is False


def test_run_registers_external_worker_identity_before_serving(tmp_path):
    from types import SimpleNamespace

    plugin = SimpleNamespace(plugin_id="external-worker")
    snapshot = SimpleNamespace(
        catalog=SimpleNamespace(plugins=(plugin,)),
        workers=(),
    )

    class Plugins:
        def __init__(self):
            self.started = 0
            self.stopped = 0

        def plugin_ids(self):
            return frozenset({"external-worker"})

        async def start(self):
            self.started += 1
            return snapshot

        async def stop(self):
            self.stopped += 1
            return ()

    plugins = Plugins()
    transport = FakeTransport()
    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=transport,
        plugin_runtime=plugins,
    )

    service._stopping.set()
    asyncio.run(service.run())

    assert plugins.started == plugins.stopped == 1
    assert transport.started == transport.stopped == 1
    assert "external-worker" in {
        item["id"] for item in service.plugin_telemetry.documents()
    }


def test_rediscovery_hands_the_scheduler_the_current_executors(tmp_path):
    discovery = FakeDiscovery([tpu_device()])

    async def scenario():
        service = build_service(
            tmp_path,
            discovery=discovery,
            publisher=FakePublisher(),
            transport=FakeTransport(),
            publish_interval_s=0.01,
            discovery_interval_s=0.01,
        )
        runner = asyncio.get_running_loop().create_task(service.run())
        await asyncio.sleep(0.05)
        discovery.devices.append(npu_device())
        await asyncio.sleep(0.05)
        service._stopping.set()
        await asyncio.wait_for(runner, timeout=2)
        return service

    service = asyncio.run(scenario())
    # Regression (OMNI-0009): dispatch must use the post-rediscovery executors,
    # not the dict the scheduler was constructed with.
    assert service._scheduler._executors == service._executors
    assert service._executors["npu"]._device_present is True


def test_publisher_loop_retracts_once_when_no_devices_are_present(tmp_path, caplog):
    publisher = FakePublisher()

    async def scenario():
        service = build_service(
            tmp_path,
            discovery=FakeDiscovery([]),
            publisher=publisher,
            transport=FakeTransport(),
            publish_interval_s=0.01,
            discovery_interval_s=60.0,
        )
        runner = asyncio.get_running_loop().create_task(service.run())
        await asyncio.sleep(0.05)
        service._stopping.set()
        await asyncio.wait_for(runner, timeout=2)

    with caplog.at_level("WARNING", logger="omnitensor.service"):
        asyncio.run(scenario())
    assert publisher.published == []
    # Startup with zero devices clears any stale file from a previous run,
    # exactly once, and logs the outage exactly once.
    assert publisher.retracted == 1
    outage_messages = [
        record.getMessage()
        for record in caplog.records
        if "accelerator" in record.getMessage().lower()
    ]
    assert outage_messages == [
        "No accelerator devices present; retracted the runtime snapshot"
        " so readers observe absence instead of stale data",
    ]


def test_publisher_persistence_runs_off_the_event_loop(tmp_path):
    class BlockingPublisher(FakePublisher):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()
            self.publish_thread = None

        def publish(self, snapshot):
            self.publish_thread = threading.get_ident()
            self.started.set()
            assert self.release.wait(timeout=0.5)
            super().publish(snapshot)

    publisher = BlockingPublisher()

    async def scenario():
        service = build_service(
            tmp_path,
            discovery=FakeDiscovery([tpu_device()]),
            publisher=publisher,
            transport=FakeTransport(),
            publish_interval_s=60.0,
            discovery_interval_s=60.0,
        )
        loop_thread = threading.get_ident()
        task = asyncio.create_task(service._publisher())
        assert await asyncio.to_thread(publisher.started.wait, 0.5)
        publisher.release.set()
        service._stopping.set()
        await asyncio.wait_for(task, timeout=1)
        return loop_thread

    loop_thread = asyncio.run(scenario())
    assert publisher.publish_thread != loop_thread


def test_snapshot_retraction_runs_off_the_event_loop(tmp_path):
    class RecordingPublisher(FakePublisher):
        def __init__(self):
            super().__init__()
            self.retract_thread = None

        def retract(self):
            self.retract_thread = threading.get_ident()
            super().retract()

    publisher = RecordingPublisher()

    async def scenario():
        service = build_service(
            tmp_path,
            discovery=FakeDiscovery([]),
            publisher=publisher,
            transport=FakeTransport(),
            publish_interval_s=60.0,
            discovery_interval_s=60.0,
        )
        loop_thread = threading.get_ident()
        task = asyncio.create_task(service._publisher())
        for _ in range(100):
            if publisher.retracted or task.done():
                break
            await asyncio.sleep(0)
        service._stopping.set()
        await asyncio.wait_for(task, timeout=1)
        assert publisher.retracted == 1
        return loop_thread

    loop_thread = asyncio.run(scenario())
    assert publisher.retract_thread != loop_thread


def test_publisher_returns_without_io_when_already_stopping(tmp_path):
    publisher = FakePublisher()

    async def scenario():
        service = build_service(
            tmp_path,
            discovery=FakeDiscovery([tpu_device()]),
            publisher=publisher,
            transport=FakeTransport(),
            publish_interval_s=60.0,
            discovery_interval_s=60.0,
        )
        service._stopping.set()
        await asyncio.wait_for(service._publisher(), timeout=0.1)

    asyncio.run(scenario())
    assert publisher.published == []
    assert publisher.retracted == 0


def test_publisher_retracts_on_device_loss_and_resumes_on_return(tmp_path, caplog):
    publisher = FakePublisher()
    discovery = FakeDiscovery([tpu_device()])

    async def scenario():
        service = build_service(
            tmp_path,
            discovery=discovery,
            publisher=publisher,
            transport=FakeTransport(),
            publish_interval_s=0.01,
            discovery_interval_s=0.01,
        )
        assert service._snapshot_retracted is False
        runner = asyncio.get_running_loop().create_task(service.run())
        await asyncio.sleep(0.05)
        published_before_loss = len(publisher.published)
        assert published_before_loss >= 1
        discovery.devices.clear()
        await asyncio.sleep(0.06)
        assert publisher.retracted == 1
        stale_count = len(publisher.published)
        await asyncio.sleep(0.03)
        # No devices: nothing new is published and the retract is not repeated.
        assert len(publisher.published) == stale_count
        assert publisher.retracted == 1
        discovery.devices.append(tpu_device())
        await asyncio.sleep(0.06)
        assert len(publisher.published) > stale_count
        assert service._snapshot_retracted is False
        service._stopping.set()
        await asyncio.wait_for(runner, timeout=2)

    with caplog.at_level("INFO", logger="omnitensor.service"):
        asyncio.run(scenario())
    warnings = [
        r.getMessage() for r in caplog.records if "No accelerator devices present" in r.message
    ]
    resumes = [
        r.getMessage() for r in caplog.records if "devices returned" in r.message
    ]
    assert warnings == [
        "No accelerator devices present; retracted the runtime snapshot"
        " so readers observe absence instead of stale data",
    ]
    assert resumes == ["Accelerator devices returned; publishing snapshots again"]


def test_publisher_crash_takes_down_rediscover_and_the_whole_run(tmp_path):
    class ExplodingPublisher(FakePublisher):
        def publish(self, snapshot: dict) -> None:
            raise RuntimeError("publisher defect")

    transport = FakeTransport()

    async def scenario():
        service = build_service(
            tmp_path,
            discovery=FakeDiscovery([tpu_device()]),
            publisher=ExplodingPublisher(),
            transport=transport,
            publish_interval_s=0.01,
            discovery_interval_s=0.01,
        )
        with pytest.raises(RuntimeError, match="publisher defect"):
            await asyncio.wait_for(service.run(), timeout=2)
        # Regression (OMNI-0012): the sibling _rediscover task used to be
        # left pending; now nothing survives the crash.
        pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        assert pending == []

    asyncio.run(scenario())
    assert transport.stopped == 1


def test_rediscover_crash_takes_down_the_publisher_too(tmp_path):
    class FailingDiscovery(FakeDiscovery):
        def detect(self):
            devices = super().detect()
            if self.detect_calls > 1:
                raise RuntimeError("sysfs went away")
            return devices

    transport = FakeTransport()

    async def scenario():
        service = build_service(
            tmp_path,
            discovery=FailingDiscovery([tpu_device()]),
            publisher=FakePublisher(),
            transport=transport,
            publish_interval_s=0.01,
            discovery_interval_s=0.01,
        )
        with pytest.raises(RuntimeError, match="sysfs went away"):
            await asyncio.wait_for(service.run(), timeout=2)
        pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        assert pending == []

    asyncio.run(scenario())
    assert transport.stopped == 1


def test_run_stops_cleanly_when_stopped_before_first_interval(tmp_path):
    transport = FakeTransport()

    async def scenario():
        service = build_service(
            tmp_path,
            discovery=FakeDiscovery([tpu_device()]),
            publisher=FakePublisher(),
            transport=transport,
            publish_interval_s=60.0,
            discovery_interval_s=60.0,
        )
        runner = asyncio.get_running_loop().create_task(service.run())
        await asyncio.sleep(0.01)
        service._stopping.set()
        await asyncio.wait_for(runner, timeout=2)

    asyncio.run(scenario())
    assert transport.stopped == 1


def test_kernel_utilization_wins_over_scheduler_load(tmp_path):
    discovery = FakeDiscovery([tpu_device()], utilization=63.5)

    async def scenario():
        service = build_service(
            tmp_path,
            discovery=discovery,
            publisher=FakePublisher(),
            transport=FakeTransport(),
        )
        service._scheduler.start()
        snapshot = service.publish_once()
        await service._scheduler.stop()
        return snapshot

    snapshot = asyncio.run(scenario())
    assert snapshot["devices"][0]["load"] == 63.5


def test_runtime_snapshot_publishes_only_result_summary_documents(tmp_path):
    from omnitensor.plugins import ResultSummaryRegistry, SecretRedactor

    summaries = ResultSummaryRegistry(alert_id_factory=lambda: "alert-runtime")
    summaries.publish(
        plugin_id="hardware-health",
        title="Health warning",
        summary="Bounded evidence",
        timestamp_ms=10,
        confidence=0.8,
        risk_score=0.6,
        result_reference="result-runtime-job",
        redactor=SecretRedactor([]),
    )

    async def scenario():
        service = build_service(
            tmp_path,
            discovery=FakeDiscovery([tpu_device()]),
            publisher=FakePublisher(),
            transport=FakeTransport(),
            result_summaries=summaries,
        )
        service._scheduler.start()
        snapshot = service.publish_once()
        await service._scheduler.stop()
        return snapshot

    snapshot = asyncio.run(scenario())
    assert snapshot["alerts"] == summaries.documents()


def test_runtime_snapshot_registers_and_publishes_installed_plugin_telemetry(tmp_path):
    async def scenario():
        service = build_service(
            tmp_path,
            manifests=[sample_plugin_manifest("observed-plugin")],
            discovery=FakeDiscovery([tpu_device()]),
            publisher=FakePublisher(),
            transport=FakeTransport(),
        )
        service._scheduler.start()
        snapshot = service.publish_once()
        await service._scheduler.stop()
        return service, snapshot

    service, snapshot = asyncio.run(scenario())
    assert snapshot["pluginTelemetry"] == {
        "version": 1,
        "plugins": service.plugin_telemetry.documents(),
    }
    # Every profile, not only version-two manifests. Gating registration on the
    # manifest version left the telemetry list empty on every host that ships
    # only bundled profiles, which is every host.
    published = [item["id"] for item in snapshot["pluginTelemetry"]["plugins"]]
    assert "observed-plugin" in published
    assert set(published) == set(service._workloads)


def test_dbus_interface_is_a_pure_passthrough_shim():
    class FakeControl:
        def __init__(self):
            self.seen: list[str] = []

        async def apply_command_text(self, text: str) -> str:
            self.seen.append(text)
            return '{"echo":true}'

        async def submit_job_text(self, text: str) -> str:
            self.seen.append(f"submit:{text}")
            return '{"submitted":true}'

        async def cancel_job_text(self, text: str) -> str:
            self.seen.append(f"cancel:{text}")
            return '{"cancelled":true}'

    control = FakeControl()
    interface = OmniTensorInterface(control)
    # dbus-fast's @method() wrapper swallows return values on direct calls;
    # the bus dispatches to the wrapped function, so test that path.
    async def jobs():
        reply = await interface.ApplyCommand.__wrapped__(interface, '{"id":"x"}')
        assert reply == '{"echo":true}'
        submitted = await interface.SubmitJob.__wrapped__(interface, '{"id":"y"}')
        cancelled = await interface.CancelJob.__wrapped__(interface, '{"id":"z"}')
        return submitted, cancelled

    assert asyncio.run(jobs()) == ('{"submitted":true}', '{"cancelled":true}')
    assert control.seen == [
        '{"id":"x"}',
        'submit:{"id":"y"}',
        'cancel:{"id":"z"}',
    ]


def test_runtime_job_boundary_authorizes_catalog_and_fails_closed_without_dispatcher(
    tmp_path,
):
    async def scenario():
        service = build_service(tmp_path, [sample_manifest()])
        known = await service.runtime_api.submit_job_text(
            json.dumps(
                {
                    "version": 1,
                    "requestId": "known",
                    "workloadId": "sample-workload",
                    "payload": {},
                }
            )
        )
        unknown = await service.runtime_api.submit_job_text(
            json.dumps(
                {
                    "version": 1,
                    "requestId": "unknown",
                    "workloadId": "missing-workload",
                    "payload": {},
                }
            )
        )
        missing = await service.runtime_api.cancel_job_text(
            json.dumps(
                {
                    "version": 1,
                    "requestId": "cancel",
                    "jobId": "missing-job",
                }
            )
        )
        policy = await service.runtime_api.apply_command_text("not-json")
        return tuple(json.loads(reply) for reply in (known, unknown, missing, policy))

    known, unknown, missing, policy = asyncio.run(scenario())
    assert known["code"] == "dispatch-unavailable"
    assert unknown["code"] == "not-authorized"
    assert missing["code"] == "job-not-found"
    assert policy["status"] == "rejected"


def test_sysfs_discovery_adapter_detects_and_reads_utilization(fake_nodes):
    add_pcie_tpu(fake_nodes)
    add_npu(fake_nodes)
    adapter = SysfsDeviceDiscovery(fake_nodes)
    devices = adapter.detect()
    assert [device.backend for device in devices] == ["tpu", "npu"]
    assert adapter.utilization(devices[0]) is None


def test_default_sysfs_discovery_probes_the_real_root():
    adapter = SysfsDeviceDiscovery()
    assert isinstance(adapter.detect(), list)


def test_file_snapshot_publisher_writes_atomically(tmp_path):
    target = tmp_path / "state/runtime-snapshot.json"
    FileSnapshotPublisher(target).publish({"version": 1})
    assert json.loads(target.read_text()) == {"version": 1}


def test_file_snapshot_publisher_retracts_the_published_file(tmp_path):
    target = tmp_path / "state/runtime-snapshot.json"
    publisher = FileSnapshotPublisher(target)
    publisher.retract()  # no-op when nothing is published
    assert not target.exists()
    publisher.publish({"version": 1})
    publisher.retract()
    assert not target.exists()


def test_service_control_round_trip_through_injected_ports(tmp_path):
    service = build_service(
        tmp_path,
        [sample_manifest()],
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )
    acknowledgement = json.loads(asyncio.run(service.control.apply_command_text(json.dumps({
        "version": 1,
        "id": "cmd-1",
        "issuedAt": 1,
        "expectedRevision": 0,
        "operation": "set-profile-weight",
        "profileId": "sample-workload",
        "value": 5,
    }))))
    assert acknowledgement["status"] == "applied"
    assert service._weight_of("sample-workload") == 5


def test_profile_statuses_with_model_track_running_state(tmp_path):
    from omnitensor.executors.base import Availability, InferenceResult
    from omnitensor.registry import Workload
    from omnitensor.service import profile_statuses
    from omnitensor.state import PolicyState

    class ReadyExecutor:
        backend = "tpu"
        model_formats = frozenset({"tflite-edgetpu"})

        def availability(self):
            return Availability(True)

        def run(self, model_path, inputs):
            return InferenceResult(outputs=[], duration_ms=0.0)

    class StatsOnlyScheduler:
        def __init__(self, running, queued=0):
            self._running = running
            self._queued = queued

        def profile_stats(self):
            if self._running == 0 and self._queued == 0:
                return {}
            return {"sample-workload": {"queued": self._queued, "running": self._running}}

    manifest = sample_manifest(model={
        "id": "sample-model",
        "version": "1.0.0",
        "format": "tflite-edgetpu",
        "fullyQuantized": True,
        "minimumCompilerVersion": "1",
        "minimumRuntimeVersion": "1",
    })
    workloads = {manifest["id"]: Workload(id=manifest["id"], manifest=manifest)}
    executors = {"tpu": ReadyExecutor()}

    policy = PolicyState()
    watching = profile_statuses(workloads, executors, StatsOnlyScheduler(0), policy)
    assert watching["sample-workload"] == {
        "status": "watching",
        "queued": 0,
        "detail": "Serving on tpu",
        "reason": "serving",
    }
    running = profile_statuses(workloads, executors, StatsOnlyScheduler(2, queued=3), policy)
    assert running["sample-workload"] == {
        "status": "running",
        "queued": 3,
        "detail": "Serving on tpu",
        "reason": "serving",
    }
    # Queued-but-not-running is watching, with the real queue depth surfaced.
    backlog = profile_statuses(workloads, executors, StatsOnlyScheduler(0, queued=5), policy)
    assert backlog["sample-workload"] == {
        "status": "watching",
        "queued": 5,
        "detail": "Serving on tpu",
        "reason": "serving",
    }


def policy_command(operation, value, revision, profile_id=None):
    return json.dumps({
        "version": 1,
        "id": f"cmd-{operation}-{revision}",
        "issuedAt": 1,
        "expectedRevision": revision,
        "operation": operation,
        "profileId": profile_id,
        "value": value,
    })


def test_paused_and_disabled_policy_surface_in_the_snapshot(tmp_path):
    service = build_service(
        tmp_path,
        [sample_manifest()],
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )
    disabled = json.loads(asyncio.run(service.control.apply_command_text(
        policy_command("set-profile-enabled", False, 0, profile_id="sample-workload"),
    )))
    assert disabled["status"] == "applied"
    snapshot = service.publish_once()
    assert snapshot["profiles"]["sample-workload"] == {
        "status": "paused",
        "queued": 0,
        "detail": "Profile disabled by policy",
        "reason": "profile-disabled",
    }

    paused = json.loads(asyncio.run(service.control.apply_command_text(
        policy_command("set-paused", True, 1),
    )))
    assert paused["status"] == "applied"
    snapshot = service.publish_once()
    assert snapshot["profiles"]["sample-workload"] == {
        "status": "paused",
        "queued": 0,
        "detail": "Runtime paused by policy",
        "reason": "paused-by-policy",
    }


def test_pause_holds_dispatch_and_resume_command_drains(tmp_path):
    from omnitensor.executors.base import Availability, InferenceResult

    class ReadyExecutor:
        backend = "tpu"
        model_formats = frozenset({"tflite-edgetpu"})

        def __init__(self):
            self.served = []

        def availability(self):
            return Availability(True)

        def run(self, model_path, inputs):
            self.served.append(model_path)
            return InferenceResult(outputs=[inputs], duration_ms=1.0)

    async def scenario():
        service = build_service(
            tmp_path,
            [sample_manifest()],
            discovery=FakeDiscovery([tpu_device()]),
            publisher=FakePublisher(),
            transport=FakeTransport(),
        )
        executor = ReadyExecutor()
        service._executors = {"tpu": executor}
        service._scheduler.update_executors(service._executors)
        service._scheduler.start()

        paused = json.loads(await service.control.apply_command_text(
            policy_command("set-paused", True, 0),
        ))
        assert paused["status"] == "applied"
        assert service._admits("sample-workload") is False
        future = service._scheduler.submit("tpu", "sample-workload", "held-job", [])
        await asyncio.sleep(0.02)
        assert not future.done()
        assert executor.served == []

        # The resume command itself must wake the workers (on_applied -> kick).
        resumed = json.loads(await service.control.apply_command_text(
            policy_command("set-paused", False, 1),
        ))
        assert resumed["status"] == "applied"
        await asyncio.wait_for(future, timeout=2)
        assert executor.served == ["held-job"]
        await service._scheduler.stop()

    asyncio.run(scenario())


def test_admits_reflects_profile_enablement_and_unknown_profiles(tmp_path):
    service = build_service(
        tmp_path,
        [sample_manifest()],
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )
    assert service._admits("unknown-profile") is True
    assert service._admits("sample-workload") is True
    disabled = json.loads(asyncio.run(service.control.apply_command_text(
        policy_command("set-profile-enabled", False, 0, profile_id="sample-workload"),
    )))
    assert disabled["status"] == "applied"
    # A disabled profile is held even though the runtime is not paused.
    assert service._admits("sample-workload") is False
    assert service._admits("unknown-profile") is True


def test_profile_statuses_do_not_leak_running_state_across_profiles():
    from omnitensor.executors.base import Availability, InferenceResult
    from omnitensor.registry import Workload
    from omnitensor.service import profile_statuses
    from omnitensor.state import PolicyState

    class ReadyExecutor:
        backend = "tpu"
        model_formats = frozenset({"tflite-edgetpu"})

        def availability(self):
            return Availability(True)

        def run(self, model_path, inputs):
            return InferenceResult(outputs=[], duration_ms=0.0)

    class OneBusyScheduler:
        def profile_stats(self):
            return {"busy-profile": {"queued": 2, "running": 1}}

    model = {
        "id": "sample-model",
        "version": "1.0.0",
        "format": "tflite-edgetpu",
        "fullyQuantized": True,
        "minimumCompilerVersion": "1",
        "minimumRuntimeVersion": "1",
    }
    workloads = {}
    for workload_id in ("busy-profile", "idle-profile"):
        manifest = sample_manifest(workload_id, model=model)
        workloads[workload_id] = Workload(id=workload_id, manifest=manifest)

    statuses = profile_statuses(
        workloads, {"tpu": ReadyExecutor()}, OneBusyScheduler(), PolicyState(),
    )
    # Regression (OMNI-0019): the global running count used to mark every
    # profile "running"; state is now strictly per profile.
    assert statuses["busy-profile"] == {
        "status": "running",
        "queued": 2,
        "detail": "Serving on tpu",
        "reason": "serving",
    }
    assert statuses["idle-profile"] == {
        "status": "watching",
        "queued": 0,
        "detail": "Serving on tpu",
        "reason": "serving",
    }


def test_env_path_prefers_environment_and_expands_home(monkeypatch):
    from omnitensor.service import _env_path

    monkeypatch.delenv("OMNITENSOR_TEST_PATH", raising=False)
    assert _env_path("OMNITENSOR_TEST_PATH", "~/fallback") == Path.home() / "fallback"
    monkeypatch.setenv("OMNITENSOR_TEST_PATH", "/explicit/path")
    assert _env_path("OMNITENSOR_TEST_PATH", "~/fallback") == Path("/explicit/path")


def test_dbus_transport_stop_without_start_is_a_no_op():
    from omnitensor.service import DbusControlTransport

    asyncio.run(DbusControlTransport().stop())


class NullControl:
    async def apply_command_text(self, text: str) -> str:
        return "{}"


def unique_bus_name() -> str:
    return f"org.cinnamon.OmniTensorTest{os.getpid()}"


def test_dbus_transport_round_trip_on_a_real_session_bus():
    from dbus_fast.aio import MessageBus

    from omnitensor.service import DbusControlTransport

    async def scenario():
        try:
            probe = await MessageBus().connect()
        except Exception as error:  # noqa: BLE001 - environment probe
            pytest.skip(f"no session bus available: {error}")
        probe.disconnect()
        # A unique per-process name: the production BUS_NAME may legitimately
        # be owned by a running omnitensor instance on this machine.
        transport = DbusControlTransport(bus_name=unique_bus_name())
        await transport.start(NullControl())
        await transport.stop()
        await transport.stop()

    asyncio.run(scenario())


def test_dbus_transport_refuses_to_run_as_a_second_instance():
    from dbus_fast.aio import MessageBus

    from omnitensor.service import DbusControlTransport

    async def scenario():
        try:
            probe = await MessageBus().connect()
        except Exception as error:  # noqa: BLE001 - environment probe
            pytest.skip(f"no session bus available: {error}")
        probe.disconnect()
        name = unique_bus_name()
        first = DbusControlTransport(bus_name=name)
        await first.start(NullControl())
        second = DbusControlTransport(bus_name=name)
        with pytest.raises(RuntimeError, match="already owned"):
            await second.start(NullControl())
        # The refused instance never keeps a bus connection around.
        assert second._bus is None
        await second.stop()
        await first.stop()

    asyncio.run(scenario())


class FakeBus:
    def __init__(self, reply):
        self._reply = reply
        self.exported = []
        self.disconnected = 0
        self.requested = []
        self.handlers = []
        self.order = []

    def add_message_handler(self, handler):
        self.handlers.append(handler)
        self.order.append("handler")

    def export(self, path, interface):
        self.exported.append((path, interface))
        self.order.append("export")

    async def request_name(self, name):
        self.requested.append(name)
        return self._reply

    def disconnect(self):
        self.disconnected += 1


@pytest.mark.parametrize("reply_name", ["IN_QUEUE", "EXISTS", "ALREADY_OWNER"])
def test_dbus_transport_fails_loudly_without_primary_ownership(reply_name):
    from dbus_fast import RequestNameReply

    from omnitensor.service import DbusControlTransport

    bus = FakeBus(RequestNameReply[reply_name])

    async def factory():
        return bus

    async def scenario():
        transport = DbusControlTransport(bus_factory=factory)
        with pytest.raises(RuntimeError) as excinfo:
            await transport.start(NullControl())
        assert str(excinfo.value) == (
            "org.cinnamon.OmniTensor1 is already owned"
            f" (request_name reply: {reply_name});"
            " another omnitensor instance is running"
        )
        assert bus.disconnected == 1
        assert transport._bus is None
        await transport.stop()
        assert bus.disconnected == 1

    asyncio.run(scenario())


def test_dbus_transport_keeps_the_bus_as_primary_owner():
    from dbus_fast import RequestNameReply

    from omnitensor.service import DbusControlTransport, OmniTensorInterface

    bus = FakeBus(RequestNameReply.PRIMARY_OWNER)

    async def factory():
        return bus

    async def scenario():
        transport = DbusControlTransport(bus_factory=factory)
        await transport.start(NullControl())
        assert transport._bus is bus
        assert bus.requested == ["org.cinnamon.OmniTensor1"]
        [(path, interface)] = bus.exported
        assert path == "/org/cinnamon/OmniTensor1"
        assert isinstance(interface, OmniTensorInterface)
        assert await interface.ApplyCommand.__wrapped__(interface, "{}") == "{}"
        await transport.stop()
        assert bus.disconnected == 1
        assert transport._bus is None

    asyncio.run(scenario())


@pytest.mark.parametrize("interval_attr", ["_publish_interval_s", "_discovery_interval_s"])
def test_intervals_default_to_module_constants(tmp_path, interval_attr):
    from omnitensor import service as service_module

    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )
    expected = {
        "_publish_interval_s": service_module.PUBLISH_INTERVAL_S,
        "_discovery_interval_s": service_module.DISCOVERY_INTERVAL_S,
    }[interval_attr]
    assert getattr(service, interval_attr) == expected


def test_publisher_survives_an_io_failure_and_retries_next_tick(tmp_path, caplog):
    """A full disk used to tear down job dispatch and D-Bus along with publishing."""

    class IntermittentPublisher(FakePublisher):
        def publish(self, snapshot: dict) -> None:
            if len(self.published) < 2:
                self.published.append(snapshot)
                raise OSError("disk full")
            super().publish(snapshot)

    publisher = IntermittentPublisher()

    async def scenario():
        service = build_service(
            tmp_path,
            discovery=FakeDiscovery([tpu_device()]),
            publisher=publisher,
            publish_interval_s=0.001,
        )
        task = asyncio.create_task(service._publisher())
        for _ in range(500):
            if len(publisher.published) > 2:
                break
            await asyncio.sleep(0.001)
        service._stopping.set()
        await asyncio.wait_for(task, timeout=2)

    with caplog.at_level("ERROR"):
        asyncio.run(scenario())

    assert len(publisher.published) > 2, "publishing never resumed after the failure"
    assert "Could not publish the runtime snapshot" in caplog.text


def test_publisher_survives_an_io_failure_while_retracting(tmp_path, caplog):
    class UnretractablePublisher(FakePublisher):
        def retract(self) -> None:
            self.retracted += 1
            raise OSError("permission denied")

    publisher = UnretractablePublisher()

    async def scenario():
        service = build_service(
            tmp_path,
            discovery=FakeDiscovery([]),
            publisher=publisher,
            publish_interval_s=0.001,
        )
        task = asyncio.create_task(service._publisher())
        for _ in range(500):
            if publisher.retracted > 1:
                break
            await asyncio.sleep(0.001)
        service._stopping.set()
        await asyncio.wait_for(task, timeout=2)

    with caplog.at_level("ERROR"):
        asyncio.run(scenario())

    assert publisher.retracted > 1, "retraction was never retried"
    assert "Could not publish the runtime snapshot" in caplog.text


def test_describe_plugins_answers_a_contract_valid_inventory(tmp_path):
    """The applet needs to say why a profile is idle without running its code."""
    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )

    document = json.loads(service.runtime_api.describe_plugins_text())

    assert validate_document("plugin-inventory.schema.json", document) == []
    assert document["version"] == 1
    assert isinstance(document["plugins"], list)


def test_the_dbus_shim_passes_describe_plugins_through():
    class FakeControl:
        async def apply_command_text(self, text: str) -> str:
            return "{}"

        async def submit_job_text(self, text: str) -> str:
            return "{}"

        async def cancel_job_text(self, text: str) -> str:
            return "{}"

        def describe_plugins_text(self) -> str:
            return '{"inventory":true}'

    interface = OmniTensorInterface(FakeControl())
    assert interface.DescribePlugins.__wrapped__(interface) == '{"inventory":true}'


def test_an_unwired_inspector_reports_an_empty_inventory_rather_than_failing():
    from omnitensor.service import RuntimeAPI

    runtime = RuntimeAPI(control=None, jobs=None)
    document = json.loads(runtime.describe_plugins_text())

    assert document == {"version": 1, "generatedAt": 1, "plugins": []}
    assert validate_document("plugin-inventory.schema.json", document) == []


def test_without_an_artifact_store_inference_stays_fail_closed(tmp_path):
    """Nothing can prove a file is the declared model, so nothing is executed."""
    from omnitensor.jobs import UnavailableJobDispatcher

    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )
    assert isinstance(
        service.jobs._dispatcher.fallback._inference, UnavailableJobDispatcher
    )


def test_an_artifact_store_enables_real_inference_dispatch(tmp_path):
    from omnitensor.dispatch import InferenceJobDispatcher

    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
        artifact_root=tmp_path / "artifacts",
    )
    assert isinstance(service.jobs._dispatcher.fallback._inference, InferenceJobDispatcher)


def test_the_artifact_root_is_configurable_from_the_environment(monkeypatch, tmp_path):
    """Without this the packaged service was permanently fail-closed."""
    from omnitensor.dispatch import InferenceJobDispatcher
    from omnitensor.service import build_service_from_env

    monkeypatch.setenv("OMNITENSOR_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("OMNITENSOR_POLICY_PATH", str(tmp_path / "policy.json"))
    monkeypatch.setenv("OMNITENSOR_WORKLOADS", str(tmp_path / "workloads"))
    monkeypatch.setenv("OMNITENSOR_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    service = build_service_from_env()

    assert isinstance(service.jobs._dispatcher.fallback._inference, InferenceJobDispatcher)
    assert service._artifact_store is not None


def test_the_artifact_root_defaults_to_the_user_share_directory(monkeypatch, tmp_path):
    from omnitensor.service import DEFAULT_ARTIFACT_ROOT, build_service_from_env

    monkeypatch.setenv("OMNITENSOR_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("OMNITENSOR_POLICY_PATH", str(tmp_path / "policy.json"))
    monkeypatch.setenv("OMNITENSOR_WORKLOADS", str(tmp_path / "workloads"))
    monkeypatch.delenv("OMNITENSOR_ARTIFACT_ROOT", raising=False)

    service = build_service_from_env()

    assert str(service._artifact_store._root).endswith("omnitensor/artifacts")
    assert DEFAULT_ARTIFACT_ROOT.endswith("omnitensor/artifacts")


def test_inventory_readiness_uses_the_declared_digest_like_dispatch_does(tmp_path):
    """Resolving by id alone would call an artifact ready that dispatch refuses."""
    manifest = sample_plugin_manifest("digest-workload")
    # The schema binds a tpu accelerator to tflite-edgetpu, so an ncnn model
    # belongs to the GPU lane.
    manifest["requirements"]["accelerator"] = "gpu"
    manifest["requirements"]["acceleratorPreference"] = ["gpu"]
    manifest["requirements"]["model"] = {
        "id": "sample-model",
        "version": "1.2.3",
        "format": "ncnn",
        "fullyQuantized": True,
        "minimumCompilerVersion": "1.0.0",
        "minimumRuntimeVersion": "1.0.0",
    }
    manifest["plugin"]["artifacts"] = [
        {
            "id": "sample-model",
            "version": "1.2.3",
            "format": "ncnn",
            "sha256": "a" * 64,
        }
    ]
    service = build_service(
        tmp_path,
        [manifest],
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
        artifact_root=tmp_path / "artifacts",
    )

    class RecordingStore:
        def __init__(self):
            self.by_reference = []
            self.by_id = []

        def resolve(self, reference):
            self.by_reference.append(reference)
            return ArtifactResolution(False, None, "digest mismatch", 0)

        def resolve_active(self, artifact_id):
            self.by_id.append(artifact_id)
            return ArtifactResolution(True, Path("/models/x"), "", 1)

    store = RecordingStore()
    service._artifact_store = store

    resolution = service._resolve_artifact("sample-model")

    assert [item.sha256 for item in store.by_reference] == ["a" * 64]
    assert store.by_id == []
    assert resolution.ready is False


def test_inventory_readiness_refuses_an_artifact_no_manifest_pins(tmp_path):
    """Readiness answers what dispatch would do, and dispatch refuses this.

    Resolving by id alone would report ready for whatever happens to be
    installed under that name — which is the guarantee the digest exists to
    give, so neither half offers it.
    """
    service = build_service(
        tmp_path,
        [sample_manifest()],
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
        artifact_root=tmp_path / "artifacts",
    )

    class RefusingStore:
        def __init__(self):
            self.asked = []

        def resolve(self, reference):  # pragma: no cover - no digest is declared
            raise AssertionError("no digest is declared for this artifact")

        def resolve_active(self, artifact_id):  # pragma: no cover - never consulted
            raise AssertionError("an unpinned artifact is never resolved by id alone")

    service._artifact_store = RefusingStore()

    resolution = service._resolve_artifact("unknown-model")

    assert resolution.ready is False
    assert "sha256" in resolution.reason


def test_the_sender_is_captured_before_any_method_can_be_dispatched():
    """An exported method reached first would run with an inherited sender."""
    from dbus_fast import RequestNameReply

    from omnitensor.callers import current_sender
    from omnitensor.service import DbusControlTransport

    bus = FakeBus(RequestNameReply.PRIMARY_OWNER)

    async def factory():
        return bus

    async def scenario():
        transport = DbusControlTransport(bus_factory=factory)
        await transport.start(object())
        await transport.stop()

    asyncio.run(scenario())

    assert bus.order == ["handler", "export"]

    class Message:
        sender = ":1.42"

    assert bus.handlers[0](Message()) is None
    assert current_sender() == ":1.42"


def test_the_transport_attaches_the_credential_lookup_to_its_resolver():
    from dbus_fast import RequestNameReply

    from omnitensor.callers import CallerIdentityResolver
    from omnitensor.service import DbusControlTransport

    bus = FakeBus(RequestNameReply.PRIMARY_OWNER)
    resolver = CallerIdentityResolver()

    async def factory():
        return bus

    async def scenario():
        transport = DbusControlTransport(bus_factory=factory, callers=resolver)
        await transport.start(object())
        await transport.stop()
        return await resolver.resolve(":1.7")

    identity = asyncio.run(scenario())

    assert resolver._unix_user is not None
    assert identity.unique_name == ":1.7"


def test_a_submitted_job_is_owned_by_the_caller_that_submitted_it(tmp_path):
    """End to end: the sender the transport bound decides who owns the job."""
    from omnitensor.callers import CallerIdentityResolver, bind_sender
    from omnitensor.service import RuntimeAPI

    class RecordingJobs:
        def __init__(self):
            self.owners = []

        async def submit_job_text(self, text, *, owner):
            self.owners.append(("submit", owner))
            return "{}"

        async def cancel_job_text(self, text, *, owner):
            self.owners.append(("cancel", owner))
            return "{}"

    async def unix_user(name):
        return {":1.7": 1000, ":1.8": 1001}[name]

    resolver = CallerIdentityResolver(unix_user)
    jobs = RecordingJobs()
    api = RuntimeAPI(None, jobs, lambda: "{}", resolver)

    async def scenario():
        bind_sender(":1.7")
        await api.submit_job_text("{}")
        bind_sender(":1.8")
        await api.cancel_job_text("{}")
        bind_sender(None)
        await api.submit_job_text("{}")

    asyncio.run(scenario())

    assert jobs.owners == [
        ("submit", "uid:1000"),
        ("cancel", "uid:1001"),
        ("submit", "anonymous"),
    ]


def test_a_caller_over_quota_gets_a_stable_code_not_an_opaque_failure(tmp_path):
    """The bus reply is the caller's only channel; an exception says nothing."""
    from omnitensor.guard import BusGuard, MethodQuota
    from omnitensor.service import RuntimeAPI

    class Jobs:
        async def submit_job_text(self, text, *, owner):
            return '{"accepted":true}'

        async def cancel_job_text(self, text, *, owner):
            return "{}"

    api = RuntimeAPI(
        None,
        Jobs(),
        lambda: "{}",
        None,
        BusGuard({"SubmitJob": MethodQuota(max_calls=1, max_concurrent=99)}),
    )

    async def scenario():
        return [json.loads(await api.submit_job_text("{}")) for _ in range(2)]

    accepted, refused = asyncio.run(scenario())

    assert accepted == {"accepted": True}
    assert refused["status"] == "rejected"
    assert refused["code"] == "rate-limit-exceeded"
    assert refused["method"] == "SubmitJob"


def test_a_caller_asserted_owner_never_reaches_the_job_service(tmp_path):
    from omnitensor.service import RuntimeAPI

    class Jobs:
        def __init__(self):
            self.calls = []

        async def submit_job_text(self, text, *, owner):
            self.calls.append(owner)
            return "{}"

        async def cancel_job_text(self, text, *, owner):
            return "{}"

    jobs = Jobs()
    api = RuntimeAPI(None, jobs, lambda: "{}")

    reply = json.loads(
        asyncio.run(api.submit_job_text(json.dumps({"version": 1, "owner": "uid:0"})))
    )

    assert reply["code"] == "identity-asserted"
    assert jobs.calls == []


def test_describing_plugins_is_rate_limited_too(tmp_path):
    from omnitensor.guard import BusGuard, MethodQuota
    from omnitensor.service import RuntimeAPI

    api = RuntimeAPI(
        None,
        None,
        lambda: '{"plugins":[]}',
        None,
        BusGuard({"DescribePlugins": MethodQuota(max_calls=1, max_bytes=1)}),
    )

    assert json.loads(api.describe_plugins_text()) == {"plugins": []}
    assert json.loads(api.describe_plugins_text())["code"] == "rate-limit-exceeded"


def test_the_bus_exposes_a_way_to_learn_a_job_outcome(tmp_path):
    """Without this method a caller submits and can never learn what happened."""
    from omnitensor.service import OmniTensorInterface

    exported = [
        name
        for name in dir(OmniTensorInterface)
        if not name.startswith("_") and name[0].isupper()
    ]

    assert "GetJobResult" in exported
    assert {
        "ApplyCommand", "SubmitJob", "CancelJob", "DescribePlugins", "DescribeContract",
    } <= set(exported)


def test_the_service_keeps_a_result_store_so_outcomes_outlive_the_call(tmp_path):
    service = build_service(tmp_path, [sample_manifest()])
    assert service.job_results is not None


def test_a_job_result_request_is_owner_scoped_and_quota_guarded(tmp_path):
    from omnitensor.callers import CallerIdentityResolver, bind_sender
    from omnitensor.guard import BusGuard, MethodQuota
    from omnitensor.service import RuntimeAPI

    class Jobs:
        def __init__(self):
            self.owners = []

        async def job_result_text(self, text, *, owner):
            self.owners.append(owner)
            return '{"state":"running"}'

    async def unix_user(name):
        return {":1.7": 1000}[name]

    jobs = Jobs()
    api = RuntimeAPI(
        None,
        jobs,
        lambda: "{}",
        CallerIdentityResolver(unix_user),
        BusGuard({"GetJobResult": MethodQuota(max_calls=1, max_concurrent=99)}),
    )

    async def scenario():
        bind_sender(":1.7")
        return [json.loads(await api.job_result_text("{}")) for _ in range(2)]

    first, second = asyncio.run(scenario())

    assert first == {"state": "running"}
    assert jobs.owners == ["uid:1000"]
    assert second["code"] == "rate-limit-exceeded"


def test_the_service_routes_submission_through_its_runners(tmp_path):
    """Built runners that nothing routes to leave the pipeline unreachable."""
    from omnitensor.plugins.orchestration import RunnerBackedDispatcher

    service = build_service(tmp_path, [sample_manifest()])

    assert isinstance(service.jobs._dispatcher, RunnerBackedDispatcher)
    assert service.jobs._dispatcher.runners is service.runners


def test_the_inventory_reports_the_grants_a_worker_actually_runs_with(tmp_path):
    """Otherwise every permission reads granted:false while the worker has it."""
    from omnitensor.plugins.loading import InstalledPluginRuntime

    runtime = InstalledPluginRuntime(tmp_path)

    assert runtime.granted_permissions("absent-plugin") == frozenset()
    runtime._granted = {"visual-library": frozenset({"read:visual-library"})}
    assert runtime.granted_permissions("visual-library") == {"read:visual-library"}


def gpu_model_manifest(workload_id="referenced-workload"):
    manifest = sample_manifest(
        workload_id, accelerator="gpu", acceleratorPreference=["gpu"]
    )
    manifest["requirements"]["model"] = {
        "id": "sample-model",
        "version": "1.2.3",
        "format": "ncnn",
        "fullyQuantized": True,
        "minimumCompilerVersion": "1.0.0",
        "minimumRuntimeVersion": "1.0.0",
        # Pinned, and ncnn keeps its weights in the companion: an unpinned
        # profile is refused at admission before its payload is looked at, so
        # a fixture without these would test the wrong refusal.
        "sha256": "c" * 64,
        "companions": {"model.bin": "d" * 64},
    }
    return manifest


def submit_reference(service, document, request_id="ref-1"):
    request = {
        "version": 1,
        "requestId": request_id,
        "workloadId": "referenced-workload",
        "payload": {"inputRefs": [document]},
    }
    reply = asyncio.run(service.runtime_api.submit_job_text(json.dumps(request)))
    parsed = json.loads(reply)
    assert validate_document("runtime-job-acknowledgement.schema.json", parsed) == []
    return parsed


def test_a_reference_outside_the_granted_roots_is_refused_in_the_submission_reply(tmp_path):
    """Verified live against /etc/passwd: SubmitJob answered accepted and the
    refusal only appeared in GetJobResult, having spent a scheduler slot."""
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    service = build_service(
        tmp_path,
        [gpu_model_manifest()],
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
        artifact_root=tmp_path / "artifacts",
        input_roots=[inputs],
    )

    reply = submit_reference(
        service,
        {
            "path": "/etc/passwd",
            "shape": [2],
            "dtype": "float32",
            "sha256": "0" * 64,
        },
    )

    assert reply["status"] == "rejected"
    assert reply["code"] == "input-ref-denied"
    assert reply["jobId"] is None
    assert service.jobs.active_job_ids() == ()


def test_a_digest_that_does_not_match_is_refused_in_the_submission_reply(tmp_path):
    import hashlib
    import struct

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    buffer = inputs / "input.f32"
    buffer.write_bytes(struct.pack("<2f", 1.0, 2.0))
    service = build_service(
        tmp_path,
        [gpu_model_manifest()],
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
        artifact_root=tmp_path / "artifacts",
        input_roots=[inputs],
    )

    wrong = submit_reference(
        service,
        {"path": str(buffer), "shape": [2], "dtype": "float32", "sha256": "0" * 64},
        request_id="ref-wrong",
    )
    right = submit_reference(
        service,
        {
            "path": str(buffer),
            "shape": [2],
            "dtype": "float32",
            "sha256": hashlib.sha256(buffer.read_bytes()).hexdigest(),
        },
        request_id="ref-right",
    )

    assert wrong["status"] == "rejected"
    assert wrong["code"] == "input-ref-mismatch"
    # The matching one is admitted: whether a GPU is present and an artifact is
    # installed is decided at dispatch, and answering it here would refuse a
    # job that would have run.
    assert right["status"] == "accepted"


def test_the_contract_handshake_is_answerable_over_the_bus(tmp_path):
    """A client must be able to ask what this service speaks before it speaks."""
    from omnitensor.service import BUS_METHODS, OmniTensorInterface, RuntimeAPI

    api = RuntimeAPI(None, None, lambda: "{}")
    interface = OmniTensorInterface(api)

    document = json.loads(interface.DescribeContract.__wrapped__(interface))

    assert document["version"] == 1
    assert document["methods"] == sorted(BUS_METHODS)
    assert document["schemas"]["runtime-job-submit"] == 1


def test_the_handshake_is_rate_limited_like_every_other_method(tmp_path):
    """Otherwise asking what the service speaks is a way to keep it busy."""
    from omnitensor.guard import BusGuard, MethodQuota
    from omnitensor.service import RuntimeAPI

    api = RuntimeAPI(
        None,
        None,
        lambda: "{}",
        None,
        BusGuard({"DescribeContract": MethodQuota(max_calls=1, max_bytes=1)}),
    )

    assert json.loads(api.describe_contract_text())["version"] == 1
    assert json.loads(api.describe_contract_text())["code"] == "rate-limit-exceeded"


def test_the_inventory_reports_the_model_a_bundled_profile_declares(tmp_path):
    """The inventory exists to answer "what does this need, and does it have
    it". Every bundled entry answered with silence, including the one profile
    whose model is declared, installed, and served."""
    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )

    # The catalogue is discovered on start, so the inventory before it is
    # empty by design rather than by the defect under test.
    asyncio.run(service._plugin_runtime.start())

    document = json.loads(service.runtime_api.describe_plugins_text())
    declaring = {
        plugin["id"]: plugin["artifacts"]
        for plugin in document["plugins"]
        if plugin["artifacts"]
    }

    assert validate_document("plugin-inventory.schema.json", document) == []
    assert "visual-library" in declaring, "the one profile with a model reports none"
    entry = declaring["visual-library"][0]
    assert entry["format"] == "ncnn"
    # Not installed in this temporary root, and the inventory says so rather
    # than reporting a readiness it never checked.
    assert entry["ready"] is False
    assert entry["reason"] != ""


def test_a_finished_job_moves_the_counters_a_user_can_see(tmp_path):
    """Registering a profile was never the gap on its own.

    No telemetry mutator was called anywhere in the service, so every counter
    would have stayed at zero however many jobs ran — a snapshot that says a
    profile exists and nothing about what it has done.
    """
    from omnitensor.service import TelemetryJobObserver

    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )
    observer = TelemetryJobObserver(service.plugin_telemetry, clock_ms=lambda: 1_700)

    observer.job_started("visual-library")
    started = service.plugin_telemetry.snapshot("visual-library")
    observer.job_finished("visual-library", "succeeded", "")
    finished = service.plugin_telemetry.snapshot("visual-library")

    assert started.active_jobs == 1
    assert (finished.active_jobs, finished.successes) == (0, 1)
    assert finished.last_success_at_ms == 1_700


def test_a_failed_and_a_cancelled_job_are_counted_apart(tmp_path):
    from omnitensor.service import TelemetryJobObserver

    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )
    observer = TelemetryJobObserver(service.plugin_telemetry, clock_ms=lambda: 1_700)

    observer.job_started("visual-library")
    observer.job_finished("visual-library", "failed", "the model refused it")
    failed = service.plugin_telemetry.snapshot("visual-library")
    observer.job_started("visual-library")
    observer.job_finished("visual-library", "cancelled", "")
    cancelled = service.plugin_telemetry.snapshot("visual-library")

    assert (failed.failures, failed.last_error_code) == (1, "job-failed")
    assert "refused" in failed.last_error_detail
    assert (cancelled.cancellations, cancelled.failures) == (1, 1)


def test_bookkeeping_never_fails_the_job_it_is_counting(tmp_path):
    """The registry enforces a state machine and refuses anything out of order.
    Right for the registry, wrong to propagate: a job that failed because its
    counter would not increment is a worse service than a gap in a counter."""
    from omnitensor.service import TelemetryJobObserver

    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )
    observer = TelemetryJobObserver(service.plugin_telemetry)

    # Never registered, and finished without ever having started.
    observer.job_started("no-such-profile")
    observer.job_finished("no-such-profile", "succeeded", "")
    observer.job_finished("visual-library", "succeeded", "")
    observer.job_finished("visual-library", "cancelled", "")

    assert service.plugin_telemetry.snapshot("visual-library").successes == 0


def test_the_job_service_reports_an_outcome_without_a_result_store():
    """A profile's counters must not depend on whether a store is wired."""
    from omnitensor.jobs import JobSubmissionService, NoJobObserver

    seen = []

    class Recording(NoJobObserver):
        def job_finished(self, workload_id, status, detail):
            seen.append((workload_id, status, detail))

    async def scenario():
        service = JobSubmissionService(results=None, observer=Recording())
        finished: asyncio.Future = asyncio.get_running_loop().create_future()
        finished.set_result({"outputs": []})
        service._record_outcome("job-1", "uid:0", "visual-library", finished)

    asyncio.run(scenario())

    assert seen == [("visual-library", "succeeded", "Job completed")]


def test_a_readable_result_becomes_something_the_desktop_can_see(tmp_path):
    """The snapshot is the only surface the applet reads, and nothing ever
    wrote to the summary registry, so every completed job was invisible."""
    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )

    service._deliver_job_output("job-1", {
        "profileId": "visual-library",
        "outputs": [[0.1, 0.9]],
        "reading": {"kind": "classification", "top": [
            {"index": 1, "score": 0.9}, {"index": 0, "score": 0.1},
        ]},
    })

    alerts = service.result_summaries.documents()
    assert len(alerts) == 1
    assert alerts[0]["profileId"] == "visual-library"
    assert alerts[0]["title"] == "class 1"
    assert "best 0.900" in alerts[0]["summary"]
    assert alerts[0]["resultRef"] == "result-job-1"


def test_a_label_is_preferred_over_an_index_when_the_model_supplies_one(tmp_path):
    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )

    service._deliver_job_output("job-2", {
        "profileId": "visual-library",
        "reading": {"kind": "classification", "top": [
            {"index": 7, "score": 0.5, "label": "golden retriever"},
        ]},
    })

    assert service.result_summaries.documents()[0]["title"] == "golden retriever"


def test_a_forecast_becomes_an_advisory_without_invented_semantics(tmp_path, monkeypatch):
    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )
    monkeypatch.setattr("omnitensor.service.time.time", lambda: 1.234)

    service._deliver_job_output(
        "forecast-1",
        {
            "profileId": "resource-scheduler",
            "outputs": [[[0.625]]],
            "reading": {
                "kind": "forecast",
                "targetFeature": "queueDepth",
                "horizon": 3,
                "value": 0.625,
            },
        },
    )

    [alert] = service.result_summaries.documents()
    assert alert["profileId"] == "resource-scheduler"
    assert alert["title"] == "queueDepth forecast"
    assert alert["summary"] == "3 observations ahead: 0.625"
    assert alert["severity"] == "advisory"
    assert alert["timestamp"] == 1234
    assert alert["confidence"] is None
    assert alert["riskScore"] is None
    assert alert["resultRef"] == "result-forecast-1"
    snapshot = service.publish_once()
    assert validate_document("runtime-snapshot.schema.json", snapshot) == []
    assert snapshot["alerts"][0]["title"] == "queueDepth forecast"


def test_a_malformed_forecast_publishes_no_summary(tmp_path):
    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )

    for reading in (
        {"kind": "forecast", "targetFeature": "load", "horizon": 1},
        {
            "kind": "forecast",
            "targetFeature": "load",
            "horizon": 1,
            "value": float("inf"),
        },
    ):
        service._deliver_job_output(
            "forecast-invalid", {"profileId": "resource-scheduler", "reading": reading}
        )

    assert service.result_summaries.documents() == []


def test_a_refused_forecast_summary_is_logged_without_failing_delivery(
    tmp_path, monkeypatch, caplog
):
    from omnitensor.plugins.summaries import SummaryError

    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )

    def refuse(**_fields):
        raise SummaryError("summary-refused", "controlled refusal")

    monkeypatch.setattr(service.result_summaries, "publish", refuse)
    with caplog.at_level("DEBUG", logger="omnitensor.service"):
        service._publish_forecast_summary(
            "forecast-logging",
            "resource-scheduler",
            {
                "kind": "forecast",
                "targetFeature": "load",
                "horizon": 1,
                "value": 0.5,
            },
        )

    assert caplog.messages == [
        "forecast summary not published for forecast-logging: summary-refused: controlled refusal"
    ]


def test_a_result_with_nothing_to_say_publishes_nothing(tmp_path):
    """A thousand raw scores have no sentence in them, and a row per job
    carrying no information is worse than no row."""
    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )

    for output in (
        {"profileId": "visual-library", "outputs": [[0.1, 0.9]]},
        {"profileId": "visual-library", "reading": {"kind": "classification", "top": []}},
        {"reading": {"kind": "classification", "top": [{"index": 1, "score": 0.9}]}},
        {"profileId": "visual-library", "reading": "classification"},
        "not a document",
    ):
        service._deliver_job_output("job-3", output)

    assert service.result_summaries.documents() == []


def test_a_summary_that_cannot_be_published_never_fails_the_job(tmp_path):
    """Delivery has already happened; a bookkeeping refusal here would turn a
    finished job into a failed one."""
    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )
    reading = {"kind": "classification", "top": [{"index": 1, "score": 0.9}]}

    # A profile id the summary registry refuses outright.
    service._deliver_job_output("job-4", {"profileId": "Not An Id", "reading": reading})
    # Two jobs sharing an id: the second reference collides with the first.
    service._deliver_job_output("job-5", {"profileId": "visual-library", "reading": reading})

    assert [item["profileId"] for item in service.result_summaries.documents()] == [
        "visual-library"
    ]


def test_the_published_snapshot_carries_the_summary(tmp_path):
    """End to end through the document the applet actually reads."""
    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )
    service._deliver_job_output("job-6", {
        "profileId": "visual-library",
        "reading": {"kind": "classification", "top": [{"index": 3, "score": 0.42}]},
    })

    snapshot = service.publish_once()

    assert validate_document("runtime-snapshot.schema.json", snapshot) == []
    assert [alert["title"] for alert in snapshot["alerts"]] == ["class 3"]
