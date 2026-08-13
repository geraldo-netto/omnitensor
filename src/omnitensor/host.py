"""Linux host adapters and their explicit service-port bundle.

The service core consumes only the protocols in :mod:`omnitensor.ports`.
This module owns the concrete sysfs and filesystem choices used by production
wiring, so tests and other hosts can replace them as a group or one at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .callers import CallerIdentityResolver
from .dbus_transport import DbusControlTransport
from .discovery import Device, DiscoveryPaths, detect_devices, device_utilization
from .ports import ControlTransport, DeviceDiscovery, SnapshotPublisher
from .snapshot import remove_snapshot, write_snapshot


class SysfsDeviceDiscovery:
    """:class:`~omnitensor.ports.DeviceDiscovery` over kernel device nodes."""

    def __init__(
        self,
        paths: DiscoveryPaths | None = None,
        selected_ids: dict[str, str] | None = None,
    ) -> None:
        self._paths = paths or DiscoveryPaths()
        self._selected_ids = dict(selected_ids or {})

    def detect(self) -> list[Device]:
        return detect_devices(self._paths, self._selected_ids)

    def utilization(self, device: Device) -> float | None:
        return device_utilization(self._paths, device)


class FileSnapshotPublisher:
    """:class:`~omnitensor.ports.SnapshotPublisher` writing atomically to a path."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def publish(self, snapshot: dict) -> None:
        write_snapshot(self._path, snapshot)

    def retract(self) -> None:
        remove_snapshot(self._path)


@dataclass(frozen=True)
class HostPorts:
    """Concrete host boundary exposed to the service only as protocols."""

    discovery: DeviceDiscovery
    publisher: SnapshotPublisher
    transport: ControlTransport


def build_host_ports(
    *,
    snapshot_path: Path,
    callers: CallerIdentityResolver,
    discovery_paths: DiscoveryPaths | None = None,
    accelerator_device_ids: dict[str, str] | None = None,
    discovery: DeviceDiscovery | None = None,
    publisher: SnapshotPublisher | None = None,
    transport: ControlTransport | None = None,
) -> HostPorts:
    """Compose production adapters while preserving explicitly injected ports."""
    return HostPorts(
        discovery=discovery
        or SysfsDeviceDiscovery(discovery_paths, accelerator_device_ids),
        publisher=publisher or FileSnapshotPublisher(snapshot_path),
        transport=transport or DbusControlTransport(callers=callers),
    )
