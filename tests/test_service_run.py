"""Service loop and adapter tests through the injected ports — no D-Bus, no
monkeypatching: fakes are provided via constructor injection."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from conftest import add_npu, add_pcie_tpu, sample_manifest, write_workload

from omnitensor.discovery import Device
from omnitensor.registry import validate_document
from omnitensor.service import (
    FileSnapshotPublisher,
    OmniTensorInterface,
    OmniTensorService,
    SysfsDeviceDiscovery,
)


def tpu_device() -> Device:
    return Device(id="tpu-pcie-0", backend="tpu", name="Coral PCIe Edge TPU", kind="pcie")


def npu_device() -> Device:
    return Device(id="npu-accel0", backend="npu", name="Intel NPU", kind="accel")


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

    def publish(self, snapshot: dict) -> None:
        self.published.append(snapshot)


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
        runner = asyncio.get_running_loop().create_task(service.run())
        await asyncio.sleep(0.08)
        discovery.devices.append(npu_device())
        await asyncio.sleep(0.05)
        service._stopping.set()
        await asyncio.wait_for(runner, timeout=2)
        return service

    service = asyncio.run(scenario())
    assert transport.started == 1
    assert transport.stopped == 1
    assert transport.handler is service.control
    assert len(publisher.published) >= 2
    for snapshot in publisher.published:
        assert validate_document("runtime-snapshot.schema.json", snapshot) == []
    assert discovery.detect_calls >= 2
    backends = {device.backend for device in service._devices}
    assert backends == {"tpu", "npu"}
    assert [device["backend"] for device in publisher.published[-1]["devices"]] == ["tpu", "npu"]


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


def test_publisher_loop_skips_when_no_devices_are_present(tmp_path):
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

    asyncio.run(scenario())
    assert publisher.published == []


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


def test_dbus_interface_is_a_pure_passthrough_shim():
    class FakeControl:
        def __init__(self):
            self.seen: list[str] = []

        def apply_command_text(self, text: str) -> str:
            self.seen.append(text)
            return '{"echo":true}'

    control = FakeControl()
    interface = OmniTensorInterface(control)
    # dbus-fast's @method() wrapper swallows return values on direct calls;
    # the bus dispatches to the wrapped function, so test that path.
    reply = interface.ApplyCommand.__wrapped__(interface, '{"id":"x"}')
    assert reply == '{"echo":true}'
    assert control.seen == ['{"id":"x"}']


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


def test_service_control_round_trip_through_injected_ports(tmp_path):
    service = build_service(
        tmp_path,
        [sample_manifest()],
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )
    acknowledgement = json.loads(service.control.apply_command_text(json.dumps({
        "version": 1,
        "id": "cmd-1",
        "issuedAt": 1,
        "expectedRevision": 0,
        "operation": "set-profile-weight",
        "profileId": "sample-workload",
        "value": 5,
    })))
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
        def __init__(self, running_profiles):
            self._running_profiles = running_profiles

        def stats(self):
            return {"queueDepth": 0, "runningProfiles": self._running_profiles, "loads": {}}

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
    }
    running = profile_statuses(workloads, executors, StatsOnlyScheduler(2), policy)
    assert running["sample-workload"] == {
        "status": "running",
        "queued": 0,
        "detail": "Serving on tpu",
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
    disabled = json.loads(service.control.apply_command_text(
        policy_command("set-profile-enabled", False, 0, profile_id="sample-workload"),
    ))
    assert disabled["status"] == "applied"
    snapshot = service.publish_once()
    assert snapshot["profiles"]["sample-workload"] == {
        "status": "paused",
        "queued": 0,
        "detail": "Profile disabled by policy",
    }

    paused = json.loads(service.control.apply_command_text(
        policy_command("set-paused", True, 1),
    ))
    assert paused["status"] == "applied"
    snapshot = service.publish_once()
    assert snapshot["profiles"]["sample-workload"] == {
        "status": "paused",
        "queued": 0,
        "detail": "Runtime paused by policy",
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

        paused = json.loads(service.control.apply_command_text(
            policy_command("set-paused", True, 0),
        ))
        assert paused["status"] == "applied"
        assert service._admits("sample-workload") is False
        future = service._scheduler.submit("tpu", "sample-workload", "held-job", [])
        await asyncio.sleep(0.02)
        assert not future.done()
        assert executor.served == []

        # The resume command itself must wake the workers (on_applied -> kick).
        resumed = json.loads(service.control.apply_command_text(
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
    disabled = json.loads(service.control.apply_command_text(
        policy_command("set-profile-enabled", False, 0, profile_id="sample-workload"),
    ))
    assert disabled["status"] == "applied"
    # A disabled profile is held even though the runtime is not paused.
    assert service._admits("sample-workload") is False
    assert service._admits("unknown-profile") is True


def test_env_path_prefers_environment_and_expands_home(monkeypatch):
    from omnitensor.service import _env_path

    monkeypatch.delenv("OMNITENSOR_TEST_PATH", raising=False)
    assert _env_path("OMNITENSOR_TEST_PATH", "~/fallback") == Path.home() / "fallback"
    monkeypatch.setenv("OMNITENSOR_TEST_PATH", "/explicit/path")
    assert _env_path("OMNITENSOR_TEST_PATH", "~/fallback") == Path("/explicit/path")


def test_dbus_transport_stop_without_start_is_a_no_op():
    from omnitensor.service import DbusControlTransport

    asyncio.run(DbusControlTransport().stop())


def test_dbus_transport_round_trip_on_a_real_session_bus():
    from dbus_fast.aio import MessageBus

    from omnitensor.service import DbusControlTransport

    async def scenario():
        try:
            probe = await MessageBus().connect()
        except Exception as error:  # noqa: BLE001 - environment probe
            pytest.skip(f"no session bus available: {error}")
        probe.disconnect()
        transport = DbusControlTransport()

        class NullControl:
            def apply_command_text(self, text: str) -> str:
                return "{}"

        await transport.start(NullControl())
        await transport.stop()
        await transport.stop()

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
