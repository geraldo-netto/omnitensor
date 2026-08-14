"""Composition and host-port boundaries kept outside the service lifecycle."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from dbus_fast import BusType, RequestNameReply
from hypothesis import given
from hypothesis import strategies as st

from omnitensor import (
    artifact_readiness,
    dispatch_routing,
    profile_selection,
    runtime_api,
    service,
    telemetry_observation,
)
from omnitensor import dbus_transport as dbus_transport_module
from omnitensor import host as host_module
from omnitensor.callers import CallerIdentityResolver
from omnitensor.composition import (
    ServiceEnvironment,
    _env_accelerator_device_ids,
    _env_path,
    _env_paths,
    build_service_from_env,
)
from omnitensor.dbus_transport import DbusControlTransport, OmniTensorInterface
from omnitensor.discovery import Device, DiscoveryPaths
from omnitensor.dispatch import InferenceJobDispatcher
from omnitensor.execution import _build_executor, build_executors
from omnitensor.executors.gpu import CompositeGpuExecutor, GpuExecutor
from omnitensor.executors.npu import NpuExecutor
from omnitensor.executors.tpu import TpuExecutor
from omnitensor.executors.vulkan import VulkanGpuExecutor
from omnitensor.host import (
    FileSnapshotPublisher,
    SysfsDeviceDiscovery,
    build_host_ports,
)
from omnitensor.jobs import UnavailableJobDispatcher
from omnitensor.plugins.artifacts import ArtifactReference, ArtifactResolution
from omnitensor.ports import (
    PluginCatalogSnapshot,
    PluginIdentitySource,
    PluginJobRuntime,
    PluginPermissionSource,
    PluginRuntimeSnapshot,
    PluginSnapshotSource,
    RuntimeHandler,
)


def test_service_public_facade_preserves_extracted_object_identity():
    assert service.RuntimeAPI is runtime_api.RuntimeAPI
    assert service.TelemetryJobObserver is telemetry_observation.TelemetryJobObserver
    assert service.profile_statuses is profile_selection.profile_statuses
    assert service.RuntimeAPI.__module__ == "omnitensor.service"
    assert service.TelemetryJobObserver.__module__ == "omnitensor.service"
    assert service.profile_statuses.__module__ == "omnitensor.service"
    for name in (
        "ARTIFACT_UNAVAILABLE",
        "CONSENT_MISSING",
        "NO_MODEL",
        "PAUSED_BY_POLICY",
        "PROFILE_DISABLED",
        "SERVING",
    ):
        assert getattr(service, name) == getattr(profile_selection, name)


def test_service_drops_only_private_compatibility_owners():
    for name in (
        "_PluginAwareDispatcher",
        "_admit_plugin_job",
        "_build_executor",
        "_env_accelerator_device_ids",
        "_env_path",
        "_env_paths",
        "_profile_status",
    ):
        assert not hasattr(service, name)

    owned = {
        node.name
        for node in ast.parse(Path(service.__file__).read_text(encoding="utf-8")).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert owned == {"OmniTensorService", "main"}


def test_extracted_service_owners_have_no_static_service_backedge():
    owners = (
        artifact_readiness,
        dispatch_routing,
        profile_selection,
        runtime_api,
        telemetry_observation,
    )
    for owner in owners:
        tree = ast.parse(Path(owner.__file__).read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, ast.ImportFrom)
            and node.module in {"service", "omnitensor.service"}
            for node in ast.walk(tree)
        )


def test_artifact_resolution_cache_preserves_the_primary_stamp_contract(tmp_path):
    reference = ArtifactReference("model", "1.0.0", "onnx", "0" * 64)
    resolutions = {}
    stamps = iter(((reference, 7, 11), (reference, 7, 11), (reference, 8, 12)))
    expected = ArtifactResolution(True, tmp_path / "model.onnx", "", 7)

    class Store:
        def __init__(self):
            self.calls = 0

        def resolve(self, actual):
            assert actual is reference
            self.calls += 1
            return expected

    store = Store()
    def call():
        return artifact_readiness.cached_resolution(
            "model", reference, store, resolutions, lambda *_args: next(stamps)
        )

    assert call() is expected
    assert call() is expected
    assert store.calls == 1
    assert call() is expected
    assert store.calls == 2


def test_dispatch_artifact_source_uses_the_stamp_cached_resolver():
    reference = ArtifactReference("model", "1.0.0", "onnx", "0" * 64)
    expected = ArtifactResolution(True, Path("/models/cached.onnx"), "", 7)
    calls = []
    source = dispatch_routing.CachedArtifactSource(
        lambda artifact_id, actual: calls.append((artifact_id, actual)) or expected
    )

    assert source.resolve(reference) is expected
    assert calls == [("model", reference)]


def test_default_dispatcher_wraps_only_a_configured_store_with_the_cache():
    raw_store = object()
    workloads = {}
    scheduler = object()
    executors = {}

    def cached(_artifact_id, _reference):
        return None

    unavailable = dispatch_routing.default_dispatcher({}, None, {}, None, ())
    direct = dispatch_routing.default_dispatcher(
        workloads, scheduler, executors, raw_store, ("/allowed",)
    )
    wrapped = dispatch_routing.default_dispatcher(
        {}, None, {}, raw_store, (), resolve_artifact=cached
    )

    assert isinstance(unavailable, UnavailableJobDispatcher)
    assert isinstance(direct, InferenceJobDispatcher)
    assert direct._workloads is workloads
    assert direct._scheduler is scheduler
    assert direct._executors is executors
    assert direct._artifacts is raw_store
    assert direct._input_roots.roots == (Path("/allowed"),)
    assert isinstance(wrapped, InferenceJobDispatcher)
    assert isinstance(wrapped._artifacts, dispatch_routing.CachedArtifactSource)
    assert wrapped._artifacts._resolve is cached


def test_artifact_owner_builds_plugin_references_in_canonical_companion_order():
    plugin = SimpleNamespace(
        manifest={
            "plugin": {
                "artifacts": [
                    {
                        "id": "plugin-model",
                        "version": "2.0.0",
                        "format": "onnx",
                        "sha256": "a" * 64,
                        "companions": {"z.bin": "c" * 64, "a.bin": "b" * 64},
                    }
                ]
            }
        }
    )

    reference = artifact_readiness.declared_reference("plugin-model", {}, lambda: (plugin,))

    assert reference == ArtifactReference(
        "plugin-model",
        "2.0.0",
        "onnx",
        "a" * 64,
        (("a.bin", "b" * 64), ("z.bin", "c" * 64)),
    )


def test_bundled_artifact_lookup_does_not_require_a_plugin_snapshot(monkeypatch):
    expected = ArtifactReference("bundled-model", "1.0.0", "onnx", "d" * 64)
    model = {"id": "bundled-model"}
    workload = SimpleNamespace(models=(model,))
    monkeypatch.setattr(
        artifact_readiness,
        "declared_artifact_reference",
        lambda actual_workload, actual_model: (
            expected if (actual_workload, actual_model) == (workload, model) else None
        ),
    )

    def unavailable_plugins():
        raise AssertionError("plugin snapshot must stay lazy after a bundled hit")

    assert (
        artifact_readiness.declared_reference(
            "bundled-model", {"profile": workload}, unavailable_plugins
        )
        is expected
    )


def test_public_service_dispatch_seams_remain_live(monkeypatch):
    monkeypatch.setattr(service, "BUS_METHODS", ("CustomMethod",))
    contract_calls = []
    monkeypatch.setattr(
        runtime_api,
        "contract_document_text",
        lambda methods: contract_calls.append(methods) or "contract",
    )
    callers = SimpleNamespace(cached_owner_token=lambda: None)
    api = runtime_api.RuntimeAPI(None, None, callers=callers)
    assert api.describe_contract_text() == "contract"
    assert contract_calls == [("CustomMethod",)]

    monkeypatch.setattr(service, "NO_MODEL", "custom-no-model")
    monkeypatch.setattr(
        profile_selection,
        "select_backend",
        lambda *_args: SimpleNamespace(backend="gpu", reason="", code=""),
    )
    status = profile_selection.profile_status(
        SimpleNamespace(id="profile", models=()),
        {},
        {"queued": 0, "running": 0},
        SimpleNamespace(paused=False, profiles={}),
    )
    assert status["reason"] == "custom-no-model"


def test_telemetry_observers_resolve_live_state_and_preserve_forecast_routing():
    calls = []

    class Registry:
        def __init__(self, label):
            self.label = label

        def publish(self, **document):
            calls.append((self.label, document))

    current = [Registry("first")]
    observer = telemetry_observation.ResultSummaryObserver(
        lambda: current[0],
        clock_ms=lambda: 7,
    )
    observer.summarize(
        "job-1",
        {
            "profileId": "visual-library",
            "reading": {"kind": "classification", "top": [{"index": 2, "score": 0.5}]},
        },
    )
    current[0] = Registry("second")
    observer.summarize(
        "job-2",
        {
            "profileId": "forecast",
            "reading": {
                "kind": "forecast",
                "targetFeature": "load",
                "horizon": 1,
                "value": 3,
            },
        },
        publish_forecast=lambda *values: calls.append(("forecast", values)),
    )
    assert calls[0][0] == "first"
    assert calls[1][0] == "forecast"


def test_invalid_plugin_runtime_returns_empty_inventory_without_reading_clock():
    def forbidden_clock():
        raise AssertionError("fail-closed inventory must not read the clock")

    assert json.loads(
        telemetry_observation.describe_plugins(
            object(),
            lambda _artifact_id: None,
            clock_ms=forbidden_clock,
        )
    ) == {"version": 1, "generatedAt": 1, "plugins": []}


def test_service_lifecycle_wrappers_resolve_owner_functions_at_call_time(monkeypatch):
    workload = object()
    calls = []
    owner = SimpleNamespace(
        _executors={"gpu": object()},
        jobs=SimpleNamespace(note_progress=lambda *values: calls.append(values)),
    )
    monkeypatch.setattr(
        artifact_readiness,
        "selected_backend",
        lambda actual_workload, executors: (
            calls.append((actual_workload, executors)) or "gpu"
        ),
    )

    assert service.OmniTensorService._selected_backend(owner, workload) == "gpu"
    service.OmniTensorService._note_job_progress(owner, "job", "run", 0.5, "half")
    assert calls == [
        (workload, owner._executors),
        ("job", "run", 0.5, "half"),
    ]

    def public_profiles(*_args):
        return {"profile": "patched"}

    monkeypatch.setattr(service, "profile_statuses", public_profiles)
    monkeypatch.setattr(
        telemetry_observation,
        "runtime_snapshot",
        lambda **arguments: arguments["profile_statuses_of"],
    )
    snapshot_owner = SimpleNamespace(
        _devices=(),
        _device_load=lambda *_args: None,
        _workloads={},
        _executors={},
        _scheduler=object(),
        control=SimpleNamespace(state=object()),
        _profile_artifact_ready=lambda _workload: (True, ""),
        _profile_permissions_missing=lambda _workload: (),
        result_summaries=object(),
        plugin_telemetry=object(),
        _input_roots=(),
        _kernel_telemetry_source=object(),
    )
    assert service.OmniTensorService._build_runtime_snapshot(snapshot_owner) is public_profiles


class DiscoveryPort:
    def detect(self):
        return []

    def utilization(self, _device):
        return None


class PublisherPort:
    def publish(self, _snapshot):
        return None

    def retract(self):
        return None


class TransportPort:
    async def start(self, _handler):
        return None

    async def stop(self):
        return None


def test_runtime_and_plugin_ports_match_the_calls_made_through_them():
    assert inspect.iscoroutinefunction(RuntimeHandler.apply_command_text)
    assert inspect.iscoroutinefunction(RuntimeHandler.submit_job_text)
    assert inspect.iscoroutinefunction(RuntimeHandler.cancel_job_text)
    assert inspect.iscoroutinefunction(RuntimeHandler.job_result_text)
    assert callable(RuntimeHandler.describe_plugins_text)
    assert callable(RuntimeHandler.describe_contract_text)

    snapshot = SimpleNamespace(
        catalog=SimpleNamespace(plugins=()),
        workers=(),
    )

    class Plugins:
        def __init__(self, retained_snapshot):
            self.snapshot = retained_snapshot

        def plugin_ids(self):
            return frozenset({"events"})

        def admit(self, _workload_id, _payload):
            return None

        async def dispatch(self, _job_id, _workload_id, _payload):
            return {}

        def granted_permissions(self, _plugin_id):
            return frozenset({"files:read-selected"})

    plugins = Plugins(snapshot)
    assert isinstance(plugins, PluginIdentitySource)
    assert isinstance(plugins, PluginJobRuntime)
    assert isinstance(plugins, PluginSnapshotSource)
    assert isinstance(plugins, PluginPermissionSource)
    assert isinstance(snapshot, PluginRuntimeSnapshot)
    assert isinstance(snapshot.catalog, PluginCatalogSnapshot)


def test_host_port_composition_preserves_every_injected_boundary(tmp_path):
    discovery = DiscoveryPort()
    publisher = PublisherPort()
    transport = TransportPort()

    ports = build_host_ports(
        snapshot_path=tmp_path / "snapshot.json",
        callers=CallerIdentityResolver(),
        discovery=discovery,
        publisher=publisher,
        transport=transport,
    )

    assert ports.discovery is discovery
    assert ports.publisher is publisher
    assert ports.transport is transport


def test_host_port_composition_owns_the_production_adapter_choices(tmp_path):
    callers = CallerIdentityResolver()
    discovery_paths = DiscoveryPaths(tmp_path / "dev", tmp_path / "sys")

    ports = build_host_ports(
        snapshot_path=tmp_path / "snapshot.json",
        callers=callers,
        discovery_paths=discovery_paths,
        accelerator_device_ids={"gpu": "gpu-renderD129"},
    )

    assert isinstance(ports.discovery, SysfsDeviceDiscovery)
    assert ports.discovery._paths is discovery_paths
    assert ports.discovery._selected_ids == {"gpu": "gpu-renderD129"}
    assert isinstance(ports.publisher, FileSnapshotPublisher)
    assert ports.publisher._path == tmp_path / "snapshot.json"
    assert isinstance(ports.transport, DbusControlTransport)
    assert ports.transport._callers is callers


def test_sysfs_adapter_delegates_exact_host_state(monkeypatch, tmp_path):
    paths = DiscoveryPaths(tmp_path / "dev", tmp_path / "sys")
    selected = {"tpu": "tpu-pcie-7"}
    device = Device("tpu-pcie-7", "tpu", "TPU", "pcie")
    calls = []
    monkeypatch.setattr(
        host_module,
        "detect_devices",
        lambda actual_paths, actual_selected: calls.append(
            (actual_paths, actual_selected)
        )
        or [device],
    )
    monkeypatch.setattr(
        host_module,
        "device_utilization",
        lambda actual_paths, actual_device: calls.append(
            (actual_paths, actual_device)
        )
        or 37.5,
    )
    adapter = SysfsDeviceDiscovery(paths, selected)

    assert adapter.detect() == [device]
    assert adapter.utilization(device) == 37.5
    assert calls == [(paths, selected), (paths, device)]


def test_file_snapshot_adapter_publishes_and_retracts_atomically(tmp_path):
    target = tmp_path / "state" / "snapshot.json"
    adapter = FileSnapshotPublisher(target)

    adapter.publish({"version": 1, "label": "שלום"})
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "version": 1,
        "label": "שלום",
    }

    adapter.retract()
    assert not target.exists()


class FakeBus:
    def __init__(self, reply=RequestNameReply.PRIMARY_OWNER):
        self.reply = reply
        self.handlers = []
        self.exports = []
        self.names = []
        self.disconnections = 0

    def add_message_handler(self, handler):
        self.handlers.append(handler)

    def export(self, path, interface):
        self.exports.append((path, interface))

    async def request_name(self, name):
        self.names.append(name)
        return self.reply

    def disconnect(self):
        self.disconnections += 1


def test_dbus_transport_retains_options_and_owns_connection_lifecycle(monkeypatch):
    import asyncio

    bus = FakeBus()
    callers = CallerIdentityResolver()
    lookups = []

    async def lookup(_name):
        return None

    def lookup_factory(actual_bus):
        lookups.append(actual_bus)
        return lookup

    monkeypatch.setattr(dbus_transport_module, "unix_user_lookup", lookup_factory)

    async def factory():
        return bus

    transport = DbusControlTransport(
        BusType.SYSTEM,
        bus_factory=factory,
        bus_name="org.cinnamon.Test",
        callers=callers,
    )
    assert transport._bus_type == BusType.SYSTEM
    assert transport._bus_factory is factory
    assert transport._bus_name == "org.cinnamon.Test"
    assert transport._callers is callers
    assert transport._bus is None

    handler = object()
    asyncio.run(transport.start(handler))
    assert transport._bus is bus
    assert len(bus.handlers) == 1
    assert callable(bus.handlers[0])
    assert bus.names == ["org.cinnamon.Test"]
    assert lookups == [bus]
    assert callers._unix_user is lookup
    [(path, interface)] = bus.exports
    assert path == "/org/cinnamon/OmniTensor1"
    assert isinstance(interface, OmniTensorInterface)
    assert interface._runtime is handler
    asyncio.run(transport.stop())
    assert bus.disconnections == 1
    assert transport._bus is None


def test_dbus_transport_disconnects_a_bus_that_cannot_own_the_name():
    import asyncio

    bus = FakeBus(RequestNameReply.EXISTS)

    async def factory():
        return bus

    transport = DbusControlTransport(bus_factory=factory)
    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(transport.start(object()))
    assert str(excinfo.value) == (
        "org.cinnamon.OmniTensor1 is already owned"
        " (request_name reply: EXISTS);"
        " another omnitensor instance is running"
    )
    assert bus.disconnections == 1
    assert transport._bus is None


def test_dbus_transport_connects_the_requested_bus_type(monkeypatch):
    import asyncio

    connected = object()
    seen = []

    class FakeMessageBus:
        def __init__(self, *, bus_type):
            seen.append(bus_type)

        async def connect(self):
            return connected

    monkeypatch.setattr(dbus_transport_module, "MessageBus", FakeMessageBus)

    transport = DbusControlTransport(BusType.SYSTEM)
    assert asyncio.run(transport._connect()) is connected
    assert seen == [BusType.SYSTEM]


def test_environment_composition_passes_exact_options_to_the_service_factory(tmp_path):
    environment = {
        "OMNITENSOR_STATE_PATH": str(tmp_path / "state.json"),
        "OMNITENSOR_POLICY_PATH": str(tmp_path / "policy.json"),
        "OMNITENSOR_WORKLOADS": str(tmp_path / "workloads"),
        "OMNITENSOR_ARTIFACT_ROOT": str(tmp_path / "artifacts"),
        "OMNITENSOR_MODEL_BINDINGS": str(tmp_path / "bindings"),
        "OMNITENSOR_GRANTS_PATH": str(tmp_path / "grants.json"),
        "OMNITENSOR_INPUT_ROOTS": f"{tmp_path / 'audio'}:{tmp_path / 'תמונה'}",
        "OMNITENSOR_GPU_DEVICE": " gpu-renderD130 ",
    }
    calls = []

    def factory(**options):
        calls.append(options)
        return "service"

    assert build_service_from_env(factory, environment) == "service"
    assert calls == [ServiceEnvironment.read(environment).service_options()]
    assert calls[0]["input_roots"] == (tmp_path / "audio", tmp_path / "תמונה")
    assert calls[0]["accelerator_device_ids"] == {"gpu": "gpu-renderD130"}


def test_environment_composition_uses_the_public_service_by_default(tmp_path):
    environment = {
        "OMNITENSOR_STATE_PATH": str(tmp_path / "state.json"),
        "OMNITENSOR_POLICY_PATH": str(tmp_path / "policy.json"),
        "OMNITENSOR_WORKLOADS": str(tmp_path / "workloads"),
        "OMNITENSOR_ARTIFACT_ROOT": str(tmp_path / "artifacts"),
        "OMNITENSOR_MODEL_BINDINGS": str(tmp_path / "bindings"),
        "OMNITENSOR_GRANTS_PATH": str(tmp_path / "grants.json"),
    }

    built = build_service_from_env(environ=environment)

    assert isinstance(built, service.OmniTensorService)


def test_environment_helpers_honour_defaults_and_all_device_bindings(
    monkeypatch, tmp_path
):
    fallback = str(tmp_path / "fallback")
    explicit = str(tmp_path / "explicit")
    assert _env_path("PATH_SETTING", fallback, {}) == tmp_path / "fallback"
    assert _env_path("PATH_SETTING", fallback, {"PATH_SETTING": explicit}) == (
        tmp_path / "explicit"
    )
    monkeypatch.setenv("PATH_SETTING", explicit)
    assert _env_path("PATH_SETTING", fallback) == tmp_path / "explicit"
    assert _env_paths("ROOTS", {}) == ()

    values = {
        "OMNITENSOR_GPU_DEVICE": " gpu-renderD7 ",
        "OMNITENSOR_NPU_DEVICE": "npu-accel2",
        "OMNITENSOR_TPU_DEVICE": "tpu-pcie-9",
    }
    assert _env_accelerator_device_ids(values) == {
        "gpu": "gpu-renderD7",
        "npu": "npu-accel2",
        "tpu": "tpu-pcie-9",
    }
    assert _env_accelerator_device_ids({"OMNITENSOR_GPU_DEVICE": "   "}) == {}
    monkeypatch.setenv("OMNITENSOR_GPU_DEVICE", "gpu-renderD8")
    monkeypatch.delenv("OMNITENSOR_NPU_DEVICE", raising=False)
    monkeypatch.delenv("OMNITENSOR_TPU_DEVICE", raising=False)
    assert _env_accelerator_device_ids() == {"gpu": "gpu-renderD8"}


def test_executor_factory_binds_each_backend_and_presence_exactly():
    tpu = _build_executor("tpu", True)
    npu = _build_executor("npu", False)
    gpu = _build_executor("gpu", True)

    assert isinstance(tpu, TpuExecutor)
    assert tpu._device_present is True
    assert isinstance(npu, NpuExecutor)
    assert npu._device_present is False
    assert isinstance(gpu, CompositeGpuExecutor)
    assert [type(executor) for executor in gpu._executors] == [
        VulkanGpuExecutor,
        GpuExecutor,
    ]
    assert all(executor._device_present is True for executor in gpu._executors)


def test_executor_composition_reuses_only_complete_unchanged_history():
    old = Device("tpu-old", "tpu", "old", "pcie")
    first = build_executors([old])

    same = build_executors(
        [old], previous_devices=[old], previous_executors=first
    )
    assert all(same[backend] is first[backend] for backend in ("tpu", "npu", "gpu"))

    incomplete_devices = build_executors([old], previous_executors=first)
    incomplete_executors = build_executors([old], previous_devices=[old])
    assert all(
        incomplete_devices[backend] is not first[backend]
        for backend in ("tpu", "npu", "gpu")
    )
    assert all(
        incomplete_executors[backend] is not first[backend]
        for backend in ("tpu", "npu", "gpu")
    )

    replacement = Device("tpu-new", "tpu", "new", "usb")
    changed = build_executors(
        [replacement], previous_devices=[old], previous_executors=same
    )
    assert changed["tpu"] is not same["tpu"]
    assert changed["npu"] is same["npu"]
    assert changed["gpu"] is same["gpu"]
    assert changed["tpu"]._device_present is True


@given(
    roots=st.lists(
        st.from_regex(r"[A-Za-z0-9_-]{1,24}", fullmatch=True),
        min_size=0,
        max_size=12,
    )
)
def test_input_root_environment_round_trips_each_non_empty_segment(roots):
    raw = ":".join(roots)

    assert _env_paths("ROOTS", {"ROOTS": raw}) == tuple(Path(root) for root in roots)


def test_service_module_is_only_the_compatibility_surface_for_host_implementations():
    source = inspect.getsource(service)

    assert "from dbus_fast" not in source
    assert "detect_devices" not in source
    assert "write_snapshot" not in source
    assert "class DbusControlTransport" not in source
    assert "class SysfsDeviceDiscovery" not in source
    assert "def build_service_from_env" not in source
