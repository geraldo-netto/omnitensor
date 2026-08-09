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
import logging
import os
from pathlib import Path

from dbus_fast import BusType, RequestNameReply
from dbus_fast.aio import MessageBus
from dbus_fast.service import ServiceInterface, method

from .control import ControlService
from .discovery import BACKENDS, Device, DiscoveryPaths, detect_devices, device_utilization
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
from .snapshot import build_snapshot, remove_snapshot, write_snapshot
from .state import PolicyState, PolicyStore

LOGGER = logging.getLogger(__name__)

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

    def retract(self) -> None:
        remove_snapshot(self._path)


class DbusControlTransport:
    """:class:`~omnitensor.ports.ControlTransport` over the session bus."""

    def __init__(
        self,
        bus_type: BusType = BusType.SESSION,
        *,
        bus_factory=None,
        bus_name: str = BUS_NAME,
    ):
        self._bus_type = bus_type
        self._bus_factory = bus_factory
        self._bus_name = bus_name
        self._bus = None

    async def start(self, handler: CommandHandler) -> None:
        if self._bus_factory is not None:
            bus = await self._bus_factory()
        else:
            bus = await MessageBus(bus_type=self._bus_type).connect()
        try:
            bus.export(OBJECT_PATH, OmniTensorInterface(handler))
            reply = await bus.request_name(self._bus_name)
            if reply != RequestNameReply.PRIMARY_OWNER:
                # Without primary ownership another instance is serving the
                # bus name and publishing to the same snapshot path; running
                # anyway would double-write. Fail startup loudly instead.
                raise RuntimeError(
                    f"{self._bus_name} is already owned"
                    f" (request_name reply: {reply.name});"
                    " another omnitensor instance is running",
                )
        except BaseException:
            bus.disconnect()
            raise
        self._bus = bus

    async def stop(self) -> None:
        if self._bus is not None:
            self._bus.disconnect()
            self._bus = None


def _build_executor(backend: str, device_present: bool):
    if backend == "tpu":
        return TpuExecutor(device_present)
    if backend == "npu":
        return NpuExecutor(device_present)
    return CompositeGpuExecutor([
        # Vulkan (ncnn) first so any Mesa/RADV/ANV driver serves GPU work;
        # ONNX Runtime with CUDA/ROCm providers is the optional second lane.
        VulkanGpuExecutor(device_present),
        GpuExecutor(device_present),
    ])


def build_executors(
    devices,
    *,
    previous_devices=None,
    previous_executors: dict | None = None,
) -> dict:
    """Build changed backend adapters and preserve unchanged runtime state."""
    current = {device.backend: device for device in devices}
    previous = {device.backend: device for device in previous_devices or []}
    existing = previous_executors or {}
    can_reuse = previous_devices is not None and previous_executors is not None
    return {
        backend: existing[backend]
        if can_reuse and backend in existing and previous.get(backend) == current.get(backend)
        else _build_executor(backend, backend in current)
        for backend in BACKENDS
    }


def profile_statuses(
    workloads: dict[str, Workload],
    executors: dict,
    scheduler: Scheduler,
    policy: PolicyState,
) -> dict[str, dict]:
    """Runtime status per profile for the snapshot document."""
    per_profile = scheduler.profile_stats()
    return {
        workload_id: _profile_status(
            workload,
            executors,
            per_profile.get(workload_id, {"queued": 0, "running": 0}),
            policy,
        )
        for workload_id, workload in workloads.items()
    }


def _profile_status(workload: Workload, executors: dict, counts: dict, policy: PolicyState) -> dict:
    queued = counts["queued"]
    profile_policy = policy.profiles.get(workload.id)
    if policy.paused:
        return {"status": "paused", "queued": queued, "detail": "Runtime paused by policy"}
    if profile_policy is not None and not profile_policy.enabled:
        return {"status": "paused", "queued": queued, "detail": "Profile disabled by policy"}
    backend, reason = pick_backend(workload, executors)
    if backend is None:
        return {"status": "unavailable", "queued": queued, "detail": reason[:240]}
    if workload.model is None:
        detail = f"Ready on {backend}; no model bundled"
        return {"status": "idle", "queued": queued, "detail": detail}
    return {
        "status": "running" if counts["running"] > 0 else "watching",
        "queued": queued,
        "detail": f"Serving on {backend}",
    }


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
        self.control = ControlService(storage, on_applied=self._policy_changed)
        self._devices = self._discovery.detect()
        self._executors = build_executors(self._devices)
        self._scheduler = Scheduler(self._executors, self._weight_of, admits=self._admits)
        self._snapshot_retracted = False
        self._stopping = asyncio.Event()

    def _device_load(self, device, stats) -> float | None:
        # Prefer the kernel's own utilization counter (covers every consumer
        # of the device); fall back to the scheduler's inference busy EMA.
        utilization = self._discovery.utilization(device)
        return utilization if utilization is not None else stats["loads"].get(device.backend)

    def _weight_of(self, workload_id: str) -> int:
        policy = self.control.state.profiles.get(workload_id)
        return policy.weight if policy is not None else 1

    def _admits(self, workload_id: str) -> bool:
        """Dispatch admission: nothing runs while the runtime is paused, and a
        disabled profile's jobs stay queued until it is enabled again."""
        state = self.control.state
        if state.paused:
            return False
        policy = state.profiles.get(workload_id)
        return policy is None or policy.enabled

    def _policy_changed(self) -> None:
        """Applied policy commands wake the workers so held jobs re-evaluate."""
        self._scheduler.kick()

    def _build_runtime_snapshot(self) -> dict:
        self._scheduler.tick()
        stats = self._scheduler.stats()
        devices = [
            dataclasses.replace(device, load=self._device_load(device, stats))
            for device in self._devices
        ]
        return build_snapshot(
            devices=devices,
            metrics=stats,
            profiles=profile_statuses(
                self._workloads, self._executors, self._scheduler, self.control.state,
            ),
        )

    def publish_once(self) -> dict:
        """Synchronously build and publish one snapshot (test/adapter API)."""
        snapshot = self._build_runtime_snapshot()
        self._publisher_port.publish(snapshot)
        return snapshot

    async def _publisher(self) -> None:
        # Honest-absence decision (OMNI-0011): with zero devices no
        # schema-valid snapshot exists (the contract requires >=1 device), so
        # rather than letting the last document go silently stale we retract
        # it — readers observe absence, exactly as when the service is down —
        # and log the outage once.  Publishing resumes when devices return.
        while not self._stopping.is_set():
            if self._devices:
                if self._snapshot_retracted:
                    LOGGER.info("Accelerator devices returned; publishing snapshots again")
                    self._snapshot_retracted = False
                snapshot = self._build_runtime_snapshot()
                await asyncio.to_thread(self._publisher_port.publish, snapshot)
            elif not self._snapshot_retracted:
                await asyncio.to_thread(self._publisher_port.retract)
                self._snapshot_retracted = True
                LOGGER.warning(
                    "No accelerator devices present; retracted the runtime snapshot"
                    " so readers observe absence instead of stale data",
                )
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self._publish_interval_s)
            except TimeoutError:
                continue

    async def _rediscover(self) -> None:
        while not self._stopping.is_set():
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self._discovery_interval_s)
            except TimeoutError:
                devices = self._discovery.detect()
                self._executors = build_executors(
                    devices,
                    previous_devices=self._devices,
                    previous_executors=self._executors,
                )
                self._devices = devices
                self._scheduler.update_executors(self._executors)

    async def run(self) -> None:
        await self._transport.start(self.control)
        self._scheduler.start()
        loop = asyncio.get_running_loop()
        tasks = [
            loop.create_task(self._publisher()),
            loop.create_task(self._rediscover()),
        ]
        try:
            # gather raises on the first loop failure; the finally block then
            # cancels and awaits the sibling, so one crashing loop can never
            # leave the other running as an orphan — the service crashes
            # loudly as a whole.
            await asyncio.gather(*tasks)
        finally:
            self._stopping.set()
            for task in tasks:
                task.cancel()
            await asyncio.wait(tasks)
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
