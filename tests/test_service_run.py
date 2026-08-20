"""Service loop and adapter tests through the injected ports — no D-Bus, no
monkeypatching: fakes are provided via constructor injection."""

from __future__ import annotations

import asyncio
import json
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

from omnitensor.control import CONTROL_VERSION
from omnitensor.discovery import Device
from omnitensor.dispatch_routing import PluginAwareDispatcher, admit_plugin_job
from omnitensor.plugins.artifacts import ArtifactResolution
from omnitensor.plugins.loading import InstalledPluginRuntime
from omnitensor.publish_loop import SnapshotPublishLoop
from omnitensor.registry import bundled_workloads_path, validate_document
from omnitensor.service import (
    GRANT_REFRESH_INTERVAL_S,
    FileSnapshotPublisher,
    OmniTensorService,
    SysfsDeviceDiscovery,
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
    dispatcher = PluginAwareDispatcher(inference, plugins)

    dispatcher.admit("events", {"sources": ["selected"]})
    dispatcher.admit("profile", {"value": 1})
    assert plugins.admissions == [("events", {"sources": ["selected"]})]
    assert inference.admissions == [("profile", {"value": 1})]
    assert dispatcher.dispatch("event-job", "events", {"sources": []}) == {"events": []}
    assert plugins.dispatches == [("event-job", "events", {"sources": []})]

    assert dispatcher.dispatch("job", "profile", {"value": 1}) == (
        "job",
        "profile",
        {"value": 1},
    )
    assert inference.dispatches == [("job", "profile", {"value": 1})]
    incomplete_plugins = type("Plugins", (), {"plugin_ids": lambda _self: {"events"}})()
    incomplete = PluginAwareDispatcher(inference, incomplete_plugins)
    with pytest.raises(RuntimeError) as missing_admission:
        incomplete.admit("events", {})
    assert str(missing_admission.value) == "plugin runtime does not implement admission"
    with pytest.raises(RuntimeError) as missing_dispatch:
        incomplete.dispatch("job", "events", {})
    assert str(missing_dispatch.value) == "plugin runtime does not implement dispatch"

    no_inventory = PluginAwareDispatcher(inference, object())
    assert no_inventory._plugin_ids() == frozenset()


def test_plugin_dispatcher_forwards_prepared_inference_lanes_only():
    class Inference:
        def __init__(self):
            self.calls = []

        def prepare_lane(self, workload_id):
            self.calls.append(("prepare", workload_id))
            return "lane", {"id": "model"}

        def dispatch_prepared(self, job_id, workload_id, payload, lane):
            self.calls.append(("dispatch", job_id, workload_id, payload, lane))
            return "prepared-result"

    class Plugins:
        def plugin_ids(self):
            return frozenset({"events"})

    inference = Inference()
    dispatcher = PluginAwareDispatcher(inference, Plugins())

    assert dispatcher.prepare_lane("profile") == ("lane", {"id": "model"})
    assert dispatcher.dispatch_prepared("job", "profile", {"value": 2}, "lane") == (
        "prepared-result"
    )
    assert inference.calls == [
        ("prepare", "profile"),
        ("dispatch", "job", "profile", {"value": 2}, "lane"),
    ]
    with pytest.raises(RuntimeError, match="plugin jobs do not use prepared"):
        dispatcher.prepare_lane("events")
    with pytest.raises(RuntimeError, match="plugin jobs do not use prepared"):
        dispatcher.dispatch_prepared("job", "events", {}, "lane")

    # One named capability: an adapter implementing neither half - or only
    # one of them - is refused by the same check rather than by two probes.
    no_prepared = PluginAwareDispatcher(object(), object())
    with pytest.raises(RuntimeError, match="does not prepare lanes"):
        no_prepared.prepare_lane("profile")
    with pytest.raises(RuntimeError, match="does not prepare lanes"):
        no_prepared.dispatch_prepared("job", "profile", {}, "lane")


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

    # No pool: admission is exactly what it was before plugin jobs queued.
    admit_plugin_job(inference, plugins, plugin_ids, None, "events", {"source": 1})
    admit_plugin_job(inference, plugins, plugin_ids, None, "profile", {"value": 2})

    assert provider_calls == [True, True]
    assert plugins.calls == [("events", {"source": 1})]
    assert inference.calls == [("profile", {"value": 2})]

    with pytest.raises(RuntimeError) as incomplete:
        admit_plugin_job(inference, object(), plugin_ids, None, "events", {})
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
    kwargs.setdefault(
        "plugin_runtime",
        InstalledPluginRuntime(
            workloads_root,
            entry_points_provider=lambda **_kwargs: (),
        ),
    )
    return OmniTensorService(
        snapshot_path=tmp_path / "state/runtime-snapshot.json",
        policy_path=tmp_path / "state/policy.json",
        workloads_path=workloads_root,
        **kwargs,
    )


def test_service_resolves_prepared_devices_by_exact_physical_identity(tmp_path):
    gpu_a = Device("gpu-renderD128", "gpu", "GPU A", "dri")
    gpu_b = Device("gpu-renderD129", "gpu", "GPU B", "dri")
    tpu = tpu_device()
    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([gpu_a, gpu_b, tpu]),
    )
    service.control.state.device_choices["profile"] = gpu_b.id

    assert service._device_identity("profile", "gpu") == gpu_b.id
    assert service._device_identity("profile", "tpu") == tpu.id
    assert (
        service._executor_for_device("gpu", gpu_b.id)
        is (service._executors.device_executors[gpu_b.id])
    )
    assert service._executor_for_device("gpu", "gpu-renderD999") is None
    assert service._executor_for_device("tpu", tpu.id) is service._executors["tpu"]

    # A plain mapping cannot tell two cards apart, so it answers neither
    # question rather than naming a backend as an identity or handing over an
    # executor that may belong to the other card.
    service._executors = {"gpu": object()}
    assert service._device_identity("profile", "gpu") is None
    assert service._executor_for_device("gpu", gpu_b.id) is None


async def wait_until(predicate, *, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0)


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
    backends = {device.backend for device in service._devices.devices}
    assert backends == {"tpu", "npu"}
    assert service._executors["tpu"] is original_tpu
    assert [device["backend"] for device in publisher.published[-1]["devices"]] == ["tpu", "npu"]


def test_slow_plugin_preflight_does_not_block_control_or_snapshot(tmp_path):
    publisher = FakePublisher()
    transport = FakeTransport()

    class SlowPlugins:
        def __init__(self):
            self.started = asyncio.Event()
            self.stopped = 0

        async def start(self):
            self.started.set()
            await asyncio.Event().wait()

        async def stop(self):
            self.stopped += 1

        def plugin_ids(self):
            return frozenset()

    async def scenario():
        plugins = SlowPlugins()
        service = build_service(
            tmp_path,
            [sample_manifest()],
            discovery=FakeDiscovery([tpu_device()]),
            publisher=publisher,
            transport=transport,
            plugin_runtime=plugins,
            publish_interval_s=0.01,
            discovery_interval_s=60,
        )
        runner = asyncio.create_task(service.run())
        await asyncio.wait_for(plugins.started.wait(), timeout=1)
        await asyncio.sleep(0.04)
        assert transport.started == 1
        assert publisher.published
        service._stopping.set()
        await asyncio.wait_for(runner, timeout=2)
        return plugins

    plugins = asyncio.run(scenario())
    assert plugins.stopped == 1


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
    assert "external-worker" in {item["id"] for item in service.plugin_telemetry.documents()}


def test_rediscovery_hands_the_scheduler_the_current_executors(tmp_path):
    discovery = FakeDiscovery([tpu_device()])

    async def scenario():
        service = build_service(
            tmp_path,
            discovery=discovery,
            publisher=FakePublisher(),
            transport=FakeTransport(),
            artifact_root=tmp_path / "artifacts",
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
    assert service._scheduler._executors == service._executors.scheduler_executors()
    assert service._executors["npu"]._device_present is True
    inference = service.jobs._dispatcher.fallback._inference
    assert dict(inference._executors()) == service._executors


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
        await wait_until(publisher.started.is_set, timeout=0.5)
        publisher.release.set()
        service._stopping.set()
        await asyncio.wait_for(task, timeout=1)
        return loop_thread

    loop_thread = asyncio.run(scenario())
    assert publisher.publish_thread != loop_thread


def test_snapshot_construction_runs_off_the_event_loop(tmp_path):
    started = threading.Event()
    release = threading.Event()
    snapshot_thread = None

    async def scenario():
        nonlocal snapshot_thread
        service = build_service(
            tmp_path,
            discovery=FakeDiscovery([tpu_device()]),
            publisher=FakePublisher(),
            transport=FakeTransport(),
            publish_interval_s=60.0,
            discovery_interval_s=60.0,
        )
        build_snapshot = service._build_runtime_snapshot

        def blocking_snapshot(scheduler=None):
            nonlocal snapshot_thread
            snapshot_thread = threading.get_ident()
            started.set()
            assert release.wait(timeout=0.5)
            return build_snapshot(scheduler)

        service._build_runtime_snapshot = blocking_snapshot
        loop_thread = threading.get_ident()
        task = asyncio.create_task(service._publisher())
        await wait_until(started.is_set, timeout=0.5)
        release.set()
        service._stopping.set()
        await asyncio.wait_for(task, timeout=1)
        return loop_thread

    loop_thread = asyncio.run(scenario())
    assert snapshot_thread != loop_thread


def test_publisher_refreshes_grants_once_off_the_event_loop(tmp_path):
    class BlockingGrants:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()
            self.reload_thread = None
            self.reloads = 0

        def reload(self):
            self.reload_thread = threading.get_ident()
            self.reloads += 1
            self.started.set()
            assert self.release.wait(timeout=0.5)

        def changed_on_disk(self):
            return True

        def is_granted(self, _profile_id, _permission, _declared):
            return False

    grants = BlockingGrants()

    async def scenario():
        service = build_service(
            tmp_path,
            discovery=FakeDiscovery([tpu_device()]),
            publisher=FakePublisher(),
            transport=FakeTransport(),
            publish_interval_s=60.0,
            discovery_interval_s=60.0,
        )
        service._grants = grants
        loop_thread = threading.get_ident()
        task = asyncio.create_task(service._publisher())
        await wait_until(grants.started.is_set, timeout=0.5)
        grants.release.set()
        service._stopping.set()
        await asyncio.wait_for(task, timeout=1)
        return loop_thread

    loop_thread = asyncio.run(scenario())
    assert grants.reloads == 1
    assert grants.reload_thread != loop_thread


def test_grant_refresh_interval_includes_the_exact_boundary(tmp_path, monkeypatch):
    class Grants:
        def __init__(self, changed=True):
            self.reloads = 0
            self.changed = changed

        def changed_on_disk(self):
            return self.changed

        def reload(self):
            self.reloads += 1

    class Loop:
        current = 10.0

        def time(self):
            return self.current

    observed = []

    async def run(callback, *args):
        observed.append((callback, args))
        return callback(*args)

    service = build_service(tmp_path, discovery=FakeDiscovery([tpu_device()]))
    grants = Grants()
    service._grants = grants
    service._next_grant_refresh = 10.0
    loop = Loop()
    monkeypatch.setattr("omnitensor.service.asyncio.get_running_loop", lambda: loop)
    monkeypatch.setattr("omnitensor.service.run_off_loop", run)

    asyncio.run(service._refresh_grants())
    assert grants.reloads == 1
    assert service._next_grant_refresh == 10.0 + GRANT_REFRESH_INTERVAL_S
    assert observed == [(grants.reload, ())]

    asyncio.run(service._refresh_grants())
    assert grants.reloads == 1
    loop.current = service._next_grant_refresh
    asyncio.run(service._refresh_grants())
    assert grants.reloads == 2

    # OMNI-0386: an unchanged ledger costs one stat, not a thread hop.
    unchanged = Grants(changed=False)
    service._grants = unchanged
    service._next_grant_refresh = loop.current
    observed.clear()
    asyncio.run(service._refresh_grants())
    assert unchanged.reloads == 0
    assert observed == []
    assert service._next_grant_refresh == 10.0 + 2 * GRANT_REFRESH_INTERVAL_S


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
        assert service._publish.retracted is False
        runner = asyncio.get_running_loop().create_task(service.run())
        await wait_until(lambda: publisher.published)
        published_before_loss = len(publisher.published)
        assert published_before_loss >= 1
        assert all(
            validate_document("runtime-snapshot.schema.json", snapshot) == []
            for snapshot in publisher.published
        )
        discovery.devices.clear()
        await wait_until(lambda: publisher.retracted == 1)
        assert publisher.retracted == 1
        stale_count = len(publisher.published)
        absent_detect_calls = discovery.detect_calls
        await wait_until(lambda: discovery.detect_calls >= absent_detect_calls + 2)
        # No devices: nothing new is published and the retract is not repeated.
        assert len(publisher.published) == stale_count
        assert publisher.retracted == 1
        discovery.devices.append(tpu_device())
        await wait_until(lambda: len(publisher.published) > stale_count)
        assert len(publisher.published) > stale_count
        assert service._publish.retracted is False
        service._stopping.set()
        await asyncio.wait_for(runner, timeout=2)

    with caplog.at_level("INFO", logger="omnitensor.service"):
        asyncio.run(scenario())
    warnings = [
        r.getMessage() for r in caplog.records if "No accelerator devices present" in r.message
    ]
    resumes = [r.getMessage() for r in caplog.records if "devices returned" in r.message]
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
        job_id="result-runtime-job",
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
    add_pcie_tpu(fake_nodes, index=42)
    add_npu(fake_nodes)
    adapter = SysfsDeviceDiscovery(fake_nodes, {"tpu": "tpu-pcie-42"})
    devices = adapter.detect()
    assert [device.backend for device in devices] == ["npu", "tpu"]
    assert devices[1].id == "tpu-pcie-42"
    assert adapter.utilization(devices[1]) is None


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
    acknowledgement = json.loads(
        asyncio.run(
            service.control.apply_command_text(
                json.dumps(
                    {
                        "version": CONTROL_VERSION,
                        "id": "cmd-1",
                        "issuedAt": 1,
                        "expectedRevision": 0,
                        "operation": "set-profile-weight",
                        "profileId": "sample-workload",
                        "value": 5,
                    }
                )
            )
        )
    )
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

        def degraded_backends(self):
            return {}

    manifest = sample_manifest(
        model={
            "id": "sample-model",
            "version": "1.0.0",
            "format": "tflite-edgetpu",
            "fullyQuantized": True,
            "minimumCompilerVersion": "1",
            "minimumRuntimeVersion": "1",
        }
    )
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

    # OMNI-0434: a backend that failed outside job execution used to publish
    # "Serving on tpu" while failing every job handed to it.
    class DegradedScheduler(StatsOnlyScheduler):
        def degraded_backends(self):
            return {"tpu": "ZeroDivisionError"}

    degraded = profile_statuses(workloads, executors, DegradedScheduler(0), policy)
    assert degraded["sample-workload"] == {
        "status": "unavailable",
        "queued": 0,
        "detail": "tpu scheduling failed: ZeroDivisionError",
        "reason": "runtime-unusable",
    }


def policy_command(operation, value, revision, profile_id=None):
    return json.dumps(
        {
            "version": CONTROL_VERSION,
            "id": f"cmd-{operation}-{revision}",
            "issuedAt": 1,
            "expectedRevision": revision,
            "operation": operation,
            "profileId": profile_id,
            "value": value,
        }
    )


def test_paused_and_disabled_policy_surface_in_the_snapshot(tmp_path):
    service = build_service(
        tmp_path,
        [sample_manifest()],
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )
    disabled = json.loads(
        asyncio.run(
            service.control.apply_command_text(
                policy_command("set-profile-enabled", False, 0, profile_id="sample-workload"),
            )
        )
    )
    assert disabled["status"] == "applied"
    snapshot = service.publish_once()
    assert snapshot["profiles"]["sample-workload"] == {
        "status": "paused",
        "queued": 0,
        "detail": "Profile disabled by policy",
        "reason": "profile-disabled",
    }

    paused = json.loads(
        asyncio.run(
            service.control.apply_command_text(
                policy_command("set-paused", True, 1),
            )
        )
    )
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

        paused = json.loads(
            await service.control.apply_command_text(
                policy_command("set-paused", True, 0),
            )
        )
        assert paused["status"] == "applied"
        assert service._admits("sample-workload") is False
        future = service._scheduler.submit("tpu", "sample-workload", "held-job", [])
        await asyncio.sleep(0.02)
        assert not future.done()
        assert executor.served == []

        # The resume command itself must wake the workers (on_applied -> kick).
        resumed = json.loads(
            await service.control.apply_command_text(
                policy_command("set-paused", False, 1),
            )
        )
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
    disabled = json.loads(
        asyncio.run(
            service.control.apply_command_text(
                policy_command("set-profile-enabled", False, 0, profile_id="sample-workload"),
            )
        )
    )
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

        def degraded_backends(self):
            return {}

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
        workloads,
        {"tpu": ReadyExecutor()},
        OneBusyScheduler(),
        PolicyState(),
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
    from omnitensor.composition import _env_path

    monkeypatch.delenv("OMNITENSOR_TEST_PATH", raising=False)
    assert _env_path("OMNITENSOR_TEST_PATH", "~/fallback") == Path.home() / "fallback"
    monkeypatch.setenv("OMNITENSOR_TEST_PATH", "/explicit/path")
    assert _env_path("OMNITENSOR_TEST_PATH", "~/fallback") == Path("/explicit/path")


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
    assert isinstance(service.jobs._dispatcher.fallback._inference, UnavailableJobDispatcher)


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


def test_accelerator_device_ids_are_configurable_from_the_environment(monkeypatch, tmp_path):
    from omnitensor.host import EventDrivenDeviceDiscovery
    from omnitensor.service import SysfsDeviceDiscovery, build_service_from_env

    monkeypatch.setenv("OMNITENSOR_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("OMNITENSOR_POLICY_PATH", str(tmp_path / "policy.json"))
    monkeypatch.setenv("OMNITENSOR_WORKLOADS", str(tmp_path / "workloads"))
    monkeypatch.setenv("OMNITENSOR_GPU_DEVICE", " gpu-renderD902 ")
    monkeypatch.setenv("OMNITENSOR_NPU_DEVICE", "npu-accel23")
    monkeypatch.setenv("OMNITENSOR_TPU_DEVICE", "tpu-pcie-42")

    service = build_service_from_env()

    # Production discovery is the sysfs adapter behind a kernel-event source,
    # so an idle host is woken by udev rather than polled.
    assert isinstance(service._discovery, EventDrivenDeviceDiscovery)
    assert isinstance(service._discovery._discovery, SysfsDeviceDiscovery)
    assert service._discovery._discovery._selected_ids == {
        "gpu": "gpu-renderD902",
        "npu": "npu-accel23",
        "tpu": "tpu-pcie-42",
    }


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
        "sha256": "a" * 64,
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

    jobs = RecordingJobs()
    api = RuntimeAPI(None, jobs, lambda: "{}", CallerIdentityResolver())

    async def scenario():
        bind_sender("peer:1000:1")
        await api.submit_job_text("{}")
        bind_sender("peer:1001:2")
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
    from omnitensor.guard import ControlGuard, GuardRefusedError, MethodQuota
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
        ControlGuard({"submit-job": MethodQuota(max_calls=1, max_concurrent=99)}),
    )

    async def scenario():
        accepted = json.loads(await api.submit_job_text("{}"))
        # The guard runs before the method, so there is no acknowledgement to
        # answer with: the refusal leaves as an exception the transport turns
        # into an envelope error, never as a document in the method's place.
        with pytest.raises(GuardRefusedError) as refused:
            await api.submit_job_text("{}")
        return accepted, refused.value

    accepted, refused = asyncio.run(scenario())

    assert accepted == {"accepted": True}
    assert refused.code == "rate-limit-exceeded"
    assert refused.method == "submit-job"


def test_a_caller_asserted_owner_never_reaches_the_job_service(tmp_path):
    from omnitensor.guard import GuardRefusedError
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

    with pytest.raises(GuardRefusedError) as refused:
        asyncio.run(api.submit_job_text(json.dumps({"version": 1, "owner": "uid:0"})))

    assert refused.value.code == "identity-asserted"
    assert jobs.calls == []


def test_describing_plugins_is_rate_limited_too(tmp_path):
    from omnitensor.guard import ControlGuard, GuardRefusedError, MethodQuota
    from omnitensor.service import RuntimeAPI

    api = RuntimeAPI(
        None,
        None,
        lambda: '{"plugins":[]}',
        None,
        ControlGuard({"describe-plugins": MethodQuota(max_calls=1, max_bytes=1)}),
    )

    assert json.loads(api.describe_plugins_text()) == {"plugins": []}
    with pytest.raises(GuardRefusedError) as refused:
        api.describe_plugins_text()
    assert refused.value.code == "rate-limit-exceeded"


def test_the_control_surface_exposes_a_way_to_learn_a_job_outcome(tmp_path):
    """Without this method a caller submits and can never learn what happened."""
    from omnitensor.service import RUNTIME_METHODS

    assert "get-job-result" in RUNTIME_METHODS
    assert {
        "apply-command",
        "submit-job",
        "cancel-job",
        "describe-plugins",
        "describe-contract",
    } <= set(RUNTIME_METHODS)


def test_the_service_keeps_a_result_store_so_outcomes_outlive_the_call(tmp_path):
    service = build_service(tmp_path, [sample_manifest()])
    assert service.job_results is not None


def test_a_job_result_request_is_owner_scoped_and_quota_guarded(tmp_path):
    from omnitensor.callers import CallerIdentityResolver, bind_sender
    from omnitensor.guard import ControlGuard, GuardRefusedError, MethodQuota
    from omnitensor.service import RuntimeAPI

    class Jobs:
        def __init__(self):
            self.owners = []

        async def job_result_text(self, text, *, owner):
            self.owners.append(owner)
            return '{"state":"running"}'

    jobs = Jobs()
    api = RuntimeAPI(
        None,
        jobs,
        lambda: "{}",
        CallerIdentityResolver(),
        ControlGuard({"get-job-result": MethodQuota(max_calls=1, max_concurrent=99)}),
    )

    async def scenario():
        bind_sender("peer:1000:1")
        first = json.loads(await api.job_result_text("{}"))
        with pytest.raises(GuardRefusedError) as refused:
            await api.job_result_text("{}")
        return first, refused.value

    first, second = asyncio.run(scenario())

    assert first == {"state": "running"}
    assert jobs.owners == ["uid:1000"]
    assert second.code == "rate-limit-exceeded"


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
    manifest = sample_manifest(workload_id, accelerator="gpu", acceleratorPreference=["gpu"])
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


def test_the_contract_handshake_is_answerable_over_the_control_surface(tmp_path):
    """A client must be able to ask what this service speaks before it speaks."""
    from omnitensor.service import RUNTIME_METHODS, RuntimeAPI

    api = RuntimeAPI(None, None, lambda: "{}")

    document = json.loads(api.describe_contract_text())

    assert document["version"] == 1
    assert document["methods"] == sorted(RUNTIME_METHODS)
    assert document["schemas"]["runtime-job-submit"] == 1


def test_the_handshake_is_rate_limited_like_every_other_method(tmp_path):
    """Otherwise asking what the service speaks is a way to keep it busy."""
    from omnitensor.guard import ControlGuard, GuardRefusedError, MethodQuota
    from omnitensor.service import RuntimeAPI

    api = RuntimeAPI(
        None,
        None,
        lambda: "{}",
        None,
        ControlGuard({"describe-contract": MethodQuota(max_calls=1, max_bytes=1)}),
    )

    assert json.loads(api.describe_contract_text())["version"] == 1
    with pytest.raises(GuardRefusedError) as refused:
        api.describe_contract_text()
    assert refused.value.code == "rate-limit-exceeded"


def test_the_inventory_reports_the_model_a_bundled_profile_declares(tmp_path):
    """The inventory exists to answer "what does this need, and does it have
    it". Every bundled entry answered with silence, including the one profile
    whose model is declared, installed, and served."""
    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
        plugin_runtime=InstalledPluginRuntime(
            bundled_workloads_path(),
            entry_points_provider=lambda **_kwargs: (),
        ),
    )

    # The catalogue is discovered on start, so the inventory before it is
    # empty by design rather than by the defect under test.
    asyncio.run(service._plugin_runtime.start())

    document = json.loads(service.runtime_api.describe_plugins_text())
    declaring = {
        plugin["id"]: plugin["artifacts"] for plugin in document["plugins"] if plugin["artifacts"]
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

    service._deliver_job_output(
        "job-1",
        {
            "profileId": "visual-library",
            "outputs": [[0.1, 0.9]],
            "reading": {
                "kind": "classification",
                "top": [
                    {"index": 1, "score": 0.9},
                    {"index": 0, "score": 0.1},
                ],
            },
        },
    )

    alerts = service.result_summaries.documents()
    assert len(alerts) == 1
    assert alerts[0]["profileId"] == "visual-library"
    assert alerts[0]["title"] == "class 1"
    assert "best 0.900" in alerts[0]["summary"]
    assert alerts[0]["jobId"] == "job-1"


def test_a_label_is_preferred_over_an_index_when_the_model_supplies_one(tmp_path):
    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )

    service._deliver_job_output(
        "job-2",
        {
            "profileId": "visual-library",
            "reading": {
                "kind": "classification",
                "top": [
                    {"index": 7, "score": 0.5, "label": "golden retriever"},
                ],
            },
        },
    )

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
    assert alert["jobId"] == "forecast-1"
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
    service._deliver_job_output(
        "job-6",
        {
            "profileId": "visual-library",
            "reading": {"kind": "classification", "top": [{"index": 3, "score": 0.42}]},
        },
    )

    snapshot = service.publish_once()

    assert validate_document("runtime-snapshot.schema.json", snapshot) == []
    assert [alert["title"] for alert in snapshot["alerts"]] == ["class 3"]


def test_idle_publisher_writes_once_and_republishes_after_a_change(tmp_path, monkeypatch):
    """An unchanged runtime must not rewrite its snapshot every tick.

    Republishing an identical document costs a disk write and wakes every
    reader watching the file, for as long as the service runs.
    """
    monkeypatch.setattr("omnitensor.service.SNAPSHOT_HEARTBEAT_INTERVAL_S", 30.0)
    monkeypatch.setattr("omnitensor.service.IDLE_PUBLISH_INTERVAL_S", 0.01)
    discovery = FakeDiscovery([tpu_device()])
    publisher = FakePublisher()

    async def scenario():
        service = build_service(
            tmp_path,
            [sample_manifest()],
            discovery=discovery,
            publisher=publisher,
            transport=FakeTransport(),
            publish_interval_s=0.01,
            discovery_interval_s=0.01,
        )
        runner = asyncio.get_running_loop().create_task(service.run())
        await asyncio.sleep(0.12)
        idle_publishes = len(publisher.published)
        discovery.devices.append(npu_device())
        await asyncio.sleep(0.08)
        service._stopping.set()
        await asyncio.wait_for(runner, timeout=2)
        return idle_publishes

    idle_publishes = asyncio.run(scenario())
    assert idle_publishes == 1
    assert len(publisher.published) == 2
    assert [device["backend"] for device in publisher.published[-1]["devices"]] == ["tpu", "npu"]


def test_idle_publisher_still_heartbeats_so_readers_do_not_call_it_stale(tmp_path, monkeypatch):
    """Readers judge a snapshot stale by age, so proof of life must keep coming."""
    monkeypatch.setattr("omnitensor.service.SNAPSHOT_HEARTBEAT_INTERVAL_S", 0.02)
    monkeypatch.setattr("omnitensor.service.IDLE_PUBLISH_INTERVAL_S", 0.01)
    publisher = FakePublisher()

    async def scenario():
        service = build_service(
            tmp_path,
            [sample_manifest()],
            discovery=FakeDiscovery([tpu_device()]),
            publisher=publisher,
            transport=FakeTransport(),
            publish_interval_s=0.01,
            discovery_interval_s=10.0,
        )
        runner = asyncio.get_running_loop().create_task(service.run())
        await asyncio.sleep(0.15)
        service._stopping.set()
        await asyncio.wait_for(runner, timeout=2)

    asyncio.run(scenario())
    assert len(publisher.published) >= 3
    stamps = [snapshot["generatedAt"] for snapshot in publisher.published]
    assert stamps == sorted(stamps)
    assert [SnapshotPublishLoop.content_of(snapshot) for snapshot in publisher.published[1:]] == [
        SnapshotPublishLoop.content_of(publisher.published[0])
    ] * (len(publisher.published) - 1)


def test_publisher_republishes_after_a_retraction(tmp_path, monkeypatch):
    """A retraction removes the document, so the cached copy cannot suppress it."""
    monkeypatch.setattr("omnitensor.service.SNAPSHOT_HEARTBEAT_INTERVAL_S", 30.0)
    monkeypatch.setattr("omnitensor.service.IDLE_PUBLISH_INTERVAL_S", 0.01)
    discovery = FakeDiscovery([tpu_device()])
    publisher = FakePublisher()

    async def scenario():
        service = build_service(
            tmp_path,
            [sample_manifest()],
            discovery=discovery,
            publisher=publisher,
            transport=FakeTransport(),
            publish_interval_s=0.01,
            discovery_interval_s=0.01,
        )
        runner = asyncio.get_running_loop().create_task(service.run())
        await asyncio.sleep(0.06)
        discovery.devices.clear()
        await asyncio.sleep(0.08)
        discovery.devices.append(tpu_device())
        await asyncio.sleep(0.08)
        service._stopping.set()
        await asyncio.wait_for(runner, timeout=2)

    asyncio.run(scenario())
    assert publisher.retracted == 1
    assert len(publisher.published) == 2


def _count_snapshot_builds(service):
    """Count publisher ticks by the one thing every tick does."""
    builds = []
    original = service._build_runtime_snapshot

    def counted(scheduler=None):
        builds.append(1)
        return original(scheduler)

    service._build_runtime_snapshot = counted
    return builds


def test_idle_runtime_backs_its_tick_off_and_a_change_wakes_it_at_once(tmp_path, monkeypatch):
    """Idle costs a tick every ten seconds; a state change does not wait for it.

    Each tick rereads sysfs, reloads grants, rebuilds the document and
    revalidates it against the schema. Paying that twice a second on a machine
    doing nothing is what an idle desktop feels as power.
    """
    monkeypatch.setattr("omnitensor.service.IDLE_PUBLISH_INTERVAL_S", 30.0)

    async def scenario():
        service = build_service(
            tmp_path,
            [sample_manifest()],
            discovery=FakeDiscovery([tpu_device()]),
            publisher=FakePublisher(),
            transport=FakeTransport(),
            publish_interval_s=0.01,
            discovery_interval_s=30.0,
        )
        builds = _count_snapshot_builds(service)
        runner = asyncio.get_running_loop().create_task(service.run())
        await asyncio.sleep(0.1)
        # Ten fast ticks would have fitted; the loop parked on a 30 s wait
        # after the first found nothing to say.
        parked = len(builds)
        service.request_publish()
        await asyncio.sleep(0.05)
        woken = len(builds)
        service._stopping.set()
        await asyncio.wait_for(runner, timeout=2)
        return parked, woken

    parked, woken = asyncio.run(scenario())
    assert parked <= 2
    assert woken == parked + 1


def test_a_job_starting_wakes_the_publisher_out_of_its_idle_wait(tmp_path, monkeypatch):
    """A reader must see a job start, not wait out the idle interval for it."""
    monkeypatch.setattr("omnitensor.service.IDLE_PUBLISH_INTERVAL_S", 30.0)

    async def scenario():
        service = build_service(
            tmp_path,
            [sample_manifest()],
            discovery=FakeDiscovery([tpu_device()]),
            publisher=FakePublisher(),
            transport=FakeTransport(),
            publish_interval_s=0.01,
            discovery_interval_s=30.0,
        )
        builds = _count_snapshot_builds(service)
        runner = asyncio.get_running_loop().create_task(service.run())
        await asyncio.sleep(0.1)
        parked = len(builds)
        service.jobs._observer.job_started(sample_manifest()["id"])
        await asyncio.sleep(0.05)
        woken = len(builds)
        service._stopping.set()
        await asyncio.wait_for(runner, timeout=2)
        return parked, woken

    parked, woken = asyncio.run(scenario())
    # The wake served a tick, and the change it carried put the loop back on
    # its fast interval, so more may have followed.
    assert woken > parked


def test_stopping_ends_the_publisher_wait_without_burning_the_idle_interval(tmp_path, monkeypatch):
    """Shutdown must not block for the idle interval it happens to be inside."""
    monkeypatch.setattr("omnitensor.service.IDLE_PUBLISH_INTERVAL_S", 30.0)

    async def scenario():
        service = build_service(
            tmp_path,
            [sample_manifest()],
            discovery=FakeDiscovery([tpu_device()]),
            publisher=FakePublisher(),
            transport=FakeTransport(),
            publish_interval_s=0.01,
            discovery_interval_s=30.0,
        )
        runner = asyncio.get_running_loop().create_task(service.run())
        await asyncio.sleep(0.1)
        service._stopping.set()
        await asyncio.wait_for(runner, timeout=1)

    asyncio.run(scenario())


def test_a_failed_startup_still_stops_the_scheduler(tmp_path):
    """`stop()` lived in an inner `finally` the failure never reached.

    Anything raising between `scheduler.start()` and that block left the
    per-backend worker tasks spawned and every queued job's future unresolved.
    """
    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )

    def explode():
        raise RuntimeError("startup exploded")

    service._rediscover = explode

    with pytest.raises(RuntimeError, match="startup exploded"):
        asyncio.run(service.run())

    assert service._scheduler._workers == {}


def test_a_finished_run_leaves_the_service_able_to_run_again(tmp_path):
    """`_stopping` was never cleared and the publish loop never unbound, so a
    second `run()` returned at once and published nothing, silently."""
    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=FakeTransport(),
    )

    service._stopping.set()
    asyncio.run(service.run())

    assert not service._stopping.is_set()
    assert service._publish.loop is None


def test_the_publisher_can_be_woken_while_the_transport_is_still_starting(tmp_path):
    """`_publish_loop` was assigned after `transport.start()`, so every
    `request_publish()` a starting transport made was a silent no-op."""

    class WakingTransport(FakeTransport):
        def __init__(self, service_box):
            super().__init__()
            self._box = service_box
            self.woke = False

        async def start(self, handler):
            await super().start(handler)
            service = self._box[0]
            service.request_publish()
            # `request_publish` hands the wake to the loop, so observe the
            # scheduled callback rather than the event it has not set yet.
            self.woke = any(
                getattr(handle, "_callback", None) == service._publish.wake.set
                for handle in asyncio.get_running_loop()._ready
            )

    box = []
    transport = WakingTransport(box)
    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=transport,
    )
    box.append(service)

    service._stopping.set()
    asyncio.run(service.run())

    assert transport.woke


def test_run_says_it_is_ready_only_once_the_socket_accepts(tmp_path, monkeypatch):
    """XTPU-0191: verification polled blind because nothing announced readiness."""
    from omnitensor import service as service_module

    events: list[str] = []
    transport = FakeTransport()

    original_start = transport.start

    async def watched_start(api):
        events.append("transport-started")
        return await original_start(api)

    transport.start = watched_start
    monkeypatch.setattr(service_module, "notify_ready", lambda: events.append("ready") or True)
    monkeypatch.setattr(
        service_module, "notify_stopping", lambda: events.append("stopping") or True
    )

    service = build_service(
        tmp_path,
        discovery=FakeDiscovery([tpu_device()]),
        publisher=FakePublisher(),
        transport=transport,
    )
    service._stopping.set()
    asyncio.run(service.run())

    assert events == ["transport-started", "ready", "stopping"]
