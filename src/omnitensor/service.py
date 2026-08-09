"""OmniTensor service wiring: discovery, scheduling, publishing, control.

Entry point ``omnitensor`` runs an asyncio loop that
- rediscovers accelerators every ``DISCOVERY_INTERVAL_S``,
- publishes a schema-valid snapshot atomically every ``PUBLISH_INTERVAL_S``,
- owns ``org.cinnamon.OmniTensor1`` on the session bus and answers
  ``ApplyCommand`` through :class:`omnitensor.control.ControlService`.

:class:`OmniTensorService` depends on the ports in :mod:`omnitensor.ports`;
the filesystem, sysfs, and D-Bus adapters defined here are only the default
wiring and can be replaced through constructor injection.

Configuration comes from environment variables:
``OMNITENSOR_STATE_PATH`` (snapshot the applet reads),
``OMNITENSOR_POLICY_PATH`` (persisted policy + revision),
``OMNITENSOR_WORKLOADS`` (manifest directory).
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
from pathlib import Path

from dbus_fast import BusType
from dbus_fast.aio import MessageBus
from dbus_fast.service import ServiceInterface, method

from .control import ControlService
from .discovery import Device, DiscoveryPaths, detect_devices, device_utilization
from .executors.gpu import CompositeGpuExecutor, GpuExecutor
from .executors.npu import NpuExecutor
from .executors.tpu import TpuExecutor
from .executors.vulkan import VulkanGpuExecutor
from .ports import (
    CommandHandler,
    ControlTransport,
    DeviceDiscovery,
    PolicyStorage,
    SnapshotPublisher,
)
from .registry import Workload, load_workloads
from .scheduler import Scheduler, pick_backend
from .snapshot import build_snapshot, write_snapshot
from .state import PolicyStore

BUS_NAME = "org.cinnamon.OmniTensor1"
OBJECT_PATH = "/org/cinnamon/OmniTensor1"
PUBLISH_INTERVAL_S = 2.0
DISCOVERY_INTERVAL_S = 10.0

DEFAULT_STATE_PATH = "~/.local/state/omnitensor/runtime-snapshot.json"
DEFAULT_POLICY_PATH = "~/.local/state/omnitensor/policy.json"
DEFAULT_WORKLOADS_PATH = "~/.local/share/omnitensor/workloads"


class OmniTensorInterface(ServiceInterface):
    """Transport-only D-Bus shim; all logic stays in the injected handler."""

    def __init__(self, control: CommandHandler):
        super().__init__(BUS_NAME)
        self._control = control

    @method()
    def ApplyCommand(self, command: s) -> s:  # noqa: F821, N802 - D-Bus contract names
        return self._control.apply_command_text(command)


class SysfsDeviceDiscovery:
    """:class:`~omnitensor.ports.DeviceDiscovery` over kernel device nodes."""

    def __init__(self, paths: DiscoveryPaths | None = None):
        self._paths = paths or DiscoveryPaths()

    def detect(self) -> list[Device]:
        return detect_devices(self._paths)

    def utilization(self, device: Device) -> float | None:
        return device_utilization(self._paths, device)


class FileSnapshotPublisher:
    """:class:`~omnitensor.ports.SnapshotPublisher` writing atomically to a path."""

    def __init__(self, path: Path):
        self._path = path

    def publish(self, snapshot: dict) -> None:
        write_snapshot(self._path, snapshot)


class DbusControlTransport:
    """:class:`~omnitensor.ports.ControlTransport` over the session bus."""

    def __init__(self, bus_type: BusType = BusType.SESSION):
        self._bus_type = bus_type
        self._bus = None

    async def start(self, handler: CommandHandler) -> None:
        self._bus = await MessageBus(bus_type=self._bus_type).connect()
        self._bus.export(OBJECT_PATH, OmniTensorInterface(handler))
        await self._bus.request_name(BUS_NAME)

    async def stop(self) -> None:
        if self._bus is not None:
            self._bus.disconnect()
            self._bus = None


def build_executors(devices) -> dict:
    present = {device.backend for device in devices}
    gpu_present = "gpu" in present
    return {
        "tpu": TpuExecutor("tpu" in present),
        "npu": NpuExecutor("npu" in present),
        # Vulkan (ncnn) first so any Mesa/RADV/ANV driver serves GPU work;
        # ONNX Runtime with CUDA/ROCm providers is the optional second lane.
        "gpu": CompositeGpuExecutor([
            VulkanGpuExecutor(gpu_present),
            GpuExecutor(gpu_present),
        ]),
    }


def profile_statuses(
    workloads: dict[str, Workload],
    executors: dict,
    scheduler: Scheduler,
) -> dict[str, dict]:
    """Runtime status per profile for the snapshot document."""
    statuses: dict[str, dict] = {}
    stats = scheduler.stats()
    for workload_id, workload in workloads.items():
        backend, reason = pick_backend(workload, executors)
        if backend is None:
            statuses[workload_id] = {
                "status": "unavailable",
                "queued": 0,
                "detail": reason[:240],
            }
        elif workload.model is None:
            statuses[workload_id] = {
                "status": "idle",
                "queued": 0,
                "detail": f"Ready on {backend}; no model bundled",
            }
        else:
            statuses[workload_id] = {
                "status": "watching" if stats["runningProfiles"] == 0 else "running",
                "queued": 0,
                "detail": f"Serving on {backend}",
            }
    return statuses


class OmniTensorService:
    def __init__(
        self,
        snapshot_path: Path,
        policy_path: Path,
        workloads_path: Path,
        discovery_paths: DiscoveryPaths | None = None,
        *,
        discovery: DeviceDiscovery | None = None,
        publisher: SnapshotPublisher | None = None,
        policy_storage: PolicyStorage | None = None,
        transport: ControlTransport | None = None,
        publish_interval_s: float = PUBLISH_INTERVAL_S,
        discovery_interval_s: float = DISCOVERY_INTERVAL_S,
    ):
        self._discovery = discovery or SysfsDeviceDiscovery(discovery_paths)
        self._publisher_port = publisher or FileSnapshotPublisher(snapshot_path)
        self._transport = transport or DbusControlTransport()
        self._publish_interval_s = publish_interval_s
        self._discovery_interval_s = discovery_interval_s
        self._workloads = load_workloads(workloads_path)
        defaults = {
            workload_id: workload.default_policy()
            for workload_id, workload in self._workloads.items()
        }
        storage = policy_storage or PolicyStore(policy_path, defaults)
        self.control = ControlService(storage, defaults)
        self._devices = self._discovery.detect()
        self._executors = build_executors(self._devices)
        self._scheduler = Scheduler(self._executors, self._weight_of)
        self._stopping = asyncio.Event()

    def _device_load(self, device, stats) -> float | None:
        # Prefer the kernel's own utilization counter (covers every consumer
        # of the device); fall back to the scheduler's inference busy EMA.
        utilization = self._discovery.utilization(device)
        return utilization if utilization is not None else stats["loads"].get(device.backend)

    def _weight_of(self, workload_id: str) -> int:
        policy = self.control.state.profiles.get(workload_id)
        return policy.weight if policy is not None else 1

    def publish_once(self) -> dict:
        self._scheduler.tick()
        stats = self._scheduler.stats()
        devices = [
            dataclasses.replace(device, load=self._device_load(device, stats))
            for device in self._devices
        ]
        snapshot = build_snapshot(
            devices=devices,
            metrics=stats,
            profiles=profile_statuses(self._workloads, self._executors, self._scheduler),
        )
        self._publisher_port.publish(snapshot)
        return snapshot

    async def _publisher(self) -> None:
        while not self._stopping.is_set():
            if self._devices:
                self.publish_once()
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self._publish_interval_s)
            except TimeoutError:
                continue

    async def _rediscover(self) -> None:
        while not self._stopping.is_set():
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self._discovery_interval_s)
            except TimeoutError:
                self._devices = self._discovery.detect()
                self._executors = build_executors(self._devices)
                self._scheduler.update_executors(self._executors)

    async def run(self) -> None:
        await self._transport.start(self.control)
        self._scheduler.start()
        try:
            await asyncio.gather(self._publisher(), self._rediscover())
        finally:
            await self._scheduler.stop()
            await self._transport.stop()


def _env_path(name: str, fallback: str) -> Path:
    return Path(os.environ.get(name, fallback)).expanduser()


def build_service_from_env() -> OmniTensorService:
    """The production service instance, configured from the environment."""
    return OmniTensorService(
        snapshot_path=_env_path("OMNITENSOR_STATE_PATH", DEFAULT_STATE_PATH),
        policy_path=_env_path("OMNITENSOR_POLICY_PATH", DEFAULT_POLICY_PATH),
        workloads_path=_env_path("OMNITENSOR_WORKLOADS", DEFAULT_WORKLOADS_PATH),
    )


def main() -> None:  # pragma: no cover - process entry point
    asyncio.run(build_service_from_env().run())
