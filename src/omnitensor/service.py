"""OmniTensor service wiring: discovery, scheduling, publishing, control.

Entry point ``omnitensor`` runs an asyncio loop that
- rediscovers accelerators every ``DISCOVERY_INTERVAL_S``,
- publishes a schema-valid snapshot atomically every ``PUBLISH_INTERVAL_S``,
- owns ``org.cinnamon.OmniTensor1`` on the session bus and answers
  ``ApplyCommand`` through :class:`omnitensor.control.ControlService`.

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

from .control import ControlService, build_control_service
from .discovery import DiscoveryPaths, detect_devices
from .executors.gpu import GpuExecutor
from .executors.npu import NpuExecutor
from .executors.tpu import TpuExecutor
from .registry import Workload, load_workloads
from .scheduler import Scheduler, pick_backend
from .snapshot import build_snapshot, write_snapshot

BUS_NAME = "org.cinnamon.OmniTensor1"
OBJECT_PATH = "/org/cinnamon/OmniTensor1"
PUBLISH_INTERVAL_S = 2.0
DISCOVERY_INTERVAL_S = 10.0

DEFAULT_STATE_PATH = "~/.local/state/omnitensor/runtime-snapshot.json"
DEFAULT_POLICY_PATH = "~/.local/state/omnitensor/policy.json"
DEFAULT_WORKLOADS_PATH = "~/.local/share/omnitensor/workloads"


class OmniTensorInterface(ServiceInterface):
    def __init__(self, control: ControlService):
        super().__init__(BUS_NAME)
        self._control = control

    @method()
    def ApplyCommand(self, command: s) -> s:  # noqa: F821, N802 - D-Bus contract names
        return self._control.apply_command_text(command)


def build_executors(devices) -> dict:
    present = {device.backend for device in devices}
    return {
        "tpu": TpuExecutor("tpu" in present),
        "npu": NpuExecutor("npu" in present),
        "gpu": GpuExecutor("gpu" in present),
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
    ):
        self._snapshot_path = snapshot_path
        self._discovery_paths = discovery_paths or DiscoveryPaths()
        self._workloads = load_workloads(workloads_path)
        defaults = {
            workload_id: workload.default_policy()
            for workload_id, workload in self._workloads.items()
        }
        self.control = build_control_service(policy_path, defaults)
        self._devices = detect_devices(self._discovery_paths)
        self._executors = build_executors(self._devices)
        self._scheduler = Scheduler(self._executors, self._weight_of)
        self._stopping = asyncio.Event()

    def _weight_of(self, workload_id: str) -> int:
        policy = self.control.state.profiles.get(workload_id)
        return policy.weight if policy is not None else 1

    def publish_once(self) -> dict:
        self._scheduler.tick()
        stats = self._scheduler.stats()
        devices = [
            dataclasses.replace(device, load=stats["loads"].get(device.backend))
            for device in self._devices
        ]
        snapshot = build_snapshot(
            devices=devices,
            metrics=stats,
            profiles=profile_statuses(self._workloads, self._executors, self._scheduler),
        )
        write_snapshot(self._snapshot_path, snapshot)
        return snapshot

    async def _publisher(self) -> None:
        while not self._stopping.is_set():
            if self._devices:
                self.publish_once()
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=PUBLISH_INTERVAL_S)
            except TimeoutError:
                continue

    async def _rediscover(self) -> None:
        while not self._stopping.is_set():
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=DISCOVERY_INTERVAL_S)
            except TimeoutError:
                self._devices = detect_devices(self._discovery_paths)
                self._executors = build_executors(self._devices)

    async def run(self) -> None:
        bus = await MessageBus(bus_type=BusType.SESSION).connect()
        bus.export(OBJECT_PATH, OmniTensorInterface(self.control))
        await bus.request_name(BUS_NAME)
        self._scheduler.start()
        try:
            await asyncio.gather(self._publisher(), self._rediscover())
        finally:
            await self._scheduler.stop()
            bus.disconnect()


def _env_path(name: str, fallback: str) -> Path:
    return Path(os.environ.get(name, fallback)).expanduser()


def main() -> None:  # pragma: no cover - process entry point
    service = OmniTensorService(
        snapshot_path=_env_path("OMNITENSOR_STATE_PATH", DEFAULT_STATE_PATH),
        policy_path=_env_path("OMNITENSOR_POLICY_PATH", DEFAULT_POLICY_PATH),
        workloads_path=_env_path("OMNITENSOR_WORKLOADS", DEFAULT_WORKLOADS_PATH),
    )
    asyncio.run(service.run())
