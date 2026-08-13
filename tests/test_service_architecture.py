"""Composition and host-port boundaries kept outside the service lifecycle."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from dbus_fast import BusType, RequestNameReply
from hypothesis import given
from hypothesis import strategies as st

from omnitensor import host as host_module
from omnitensor import service
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


def test_dbus_transport_retains_options_and_owns_connection_lifecycle():
    import asyncio

    bus = FakeBus()
    callers = CallerIdentityResolver()

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

    asyncio.run(transport.start(object()))
    assert transport._bus is bus
    assert len(bus.handlers) == 1
    assert bus.names == ["org.cinnamon.Test"]
    [(path, interface)] = bus.exports
    assert path == "/org/cinnamon/OmniTensor1"
    assert isinstance(interface, OmniTensorInterface)
    asyncio.run(transport.stop())
    assert bus.disconnections == 1
    assert transport._bus is None


def test_dbus_transport_disconnects_a_bus_that_cannot_own_the_name():
    import asyncio

    bus = FakeBus(RequestNameReply.EXISTS)

    async def factory():
        return bus

    transport = DbusControlTransport(bus_factory=factory)
    with pytest.raises(RuntimeError, match="already owned"):
        asyncio.run(transport.start(object()))
    assert bus.disconnections == 1
    assert transport._bus is None


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
