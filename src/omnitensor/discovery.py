"""Accelerator discovery over kernel device nodes.

Discovery mirrors the applet's fallback probing: Coral TPU over PCIe
(``/dev/apex_*``) and USB (vendor/product ``18d1:9302`` or DFU
``1a6e:089a``), NPU over the kernel accel subsystem (``/dev/accel/accel*``),
and GPU over DRM render nodes (``/dev/dri/renderD*``).  Vendor
identification is best effort: an unreadable sysfs vendor file never fails
detection of a present node.  Found devices only are reported — absence is
expressed by omission.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

_PCIE_NODE = re.compile(r"^apex_([0-9]{1,6})$")
_ACCEL_NODE = re.compile(r"^accel([0-9]{1,6})$")
_RENDER_NODE = re.compile(r"^renderD([0-9]{1,6})$")

CORAL_USB_IDENTITIES = {
    ("18d1", "9302"): "Coral USB Accelerator",
    ("1a6e", "089a"): "Coral USB Accelerator (DFU)",
}
NPU_VENDOR_NAMES = {
    "0x8086": "Intel NPU",
    "0x1002": "AMD NPU",
    "0x1022": "AMD NPU",
}
GPU_VENDOR_NAMES = {
    "0x10de": "NVIDIA GPU",
    "0x1002": "AMD GPU",
    "0x8086": "Intel GPU",
}

# Routing hierarchy, most preferred first: a profile that states no
# preference of its own is offered these lanes in this order.
BACKENDS = ("gpu", "npu", "tpu")


@dataclass(frozen=True)
class Device:
    id: str
    backend: str
    name: str
    kind: str
    vendor: str = ""
    load: float | None = None
    available: bool = True
    reason: str = ""
    # Runtime-only identity used to bind a DRM render node to the matching
    # Vulkan physical device.  It is deliberately absent from the public
    # snapshot: ``id`` is the stable user-facing selector.
    hardware_id: str = ""
    identity_index: int = 0

    def snapshot_entry(self) -> dict:
        return {
            "id": self.id,
            "backend": self.backend,
            "available": self.available,
            "name": self.name,
            "kind": self.kind,
            "vendor": self.vendor,
            "load": self.load,
            "reason": self.reason,
        }


@dataclass
class DiscoveryPaths:
    """Filesystem roots, injectable for tests."""

    dev: Path = field(default_factory=lambda: Path("/dev"))
    sys: Path = field(default_factory=lambda: Path("/sys"))


def _read_trimmed(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:32].strip().lower()
    except OSError:
        return ""


def _numbered_nodes(root: Path, pattern: re.Pattern[str]) -> tuple[tuple[int, Path], ...]:
    try:
        entries = list(root.iterdir())
    except OSError:
        return ()
    matched = []
    for entry in entries:
        match = pattern.fullmatch(entry.name)
        if match is not None and entry.exists():
            matched.append((int(match.group(1)), entry))
    return tuple(sorted(matched, key=lambda item: (item[0], item[1].name)))


def _selected(candidates: list[Device], selected_id: str | None) -> Device | None:
    if selected_id is None:
        return candidates[0] if candidates else None
    return next((device for device in candidates if device.id == selected_id), None)


def detect_tpu(paths: DiscoveryPaths, selected_id: str | None = None) -> Device | None:
    candidates = []
    for index, _node in _numbered_nodes(paths.dev, _PCIE_NODE):
        suffix = "" if index == 0 else f" {index + 1}"
        candidates.append(
            Device(
                id=f"tpu-pcie-{index}",
                backend="tpu",
                name=f"Coral PCIe Edge TPU{suffix}",
                kind="pcie",
            )
        )
    usb_root = paths.sys / "bus/usb/devices"
    try:
        entries = sorted(usb_root.iterdir())
    except OSError:
        entries = []
    for entry in entries:
        vendor = _read_trimmed(entry / "idVendor")
        product = _read_trimmed(entry / "idProduct")
        name = CORAL_USB_IDENTITIES.get((vendor, product))
        if name is not None:
            candidates.append(
                Device(
                    id="tpu-usb",
                    backend="tpu",
                    name=name,
                    kind="usb",
                    vendor=f"{vendor}:{product}",
                )
            )
            break
    return _selected(candidates, selected_id)


def _detect_node(
    node: Path,
    vendor_file: Path,
    vendor_names: dict[str, str],
    fallback_name: str,
    device_id: str,
    backend: str,
    kind: str,
) -> Device | None:
    if not node.exists():
        return None
    vendor = _read_trimmed(vendor_file)
    return Device(
        id=device_id,
        backend=backend,
        name=vendor_names.get(vendor, fallback_name),
        kind=kind,
        vendor=vendor,
    )


def detect_npu(paths: DiscoveryPaths, selected_id: str | None = None) -> Device | None:
    candidates = []
    for index, node in _numbered_nodes(paths.dev / "accel", _ACCEL_NODE):
        device = _detect_node(
            node,
            paths.sys / f"class/accel/accel{index}/device/vendor",
            NPU_VENDOR_NAMES,
            "NPU accelerator",
            f"npu-accel{index}",
            "npu",
            "accel",
        )
        if device is not None:
            candidates.append(device)
    return _selected(candidates, selected_id)


def detect_gpus(paths: DiscoveryPaths) -> tuple[Device, ...]:
    """Every DRM render node, with a Vulkan-matchable hardware identity."""
    candidates = []
    occurrences: dict[tuple[str, str], int] = {}
    for node_number, node in _numbered_nodes(paths.dev / "dri", _RENDER_NODE):
        vendor_file = paths.sys / f"class/drm/renderD{node_number}/device/vendor"
        hardware_file = paths.sys / f"class/drm/renderD{node_number}/device/device"
        vendor = _read_trimmed(vendor_file)
        hardware_id = _read_trimmed(hardware_file)
        identity = (vendor, hardware_id)
        identity_index = occurrences.get(identity, 0)
        occurrences[identity] = identity_index + 1
        device = _detect_node(
            node,
            vendor_file,
            GPU_VENDOR_NAMES,
            "GPU (render node)",
            f"gpu-renderD{node_number}",
            "gpu",
            "dri",
        )
        if device is not None:
            candidates.append(
                replace(
                    device,
                    hardware_id=hardware_id,
                    identity_index=identity_index,
                )
            )
    return tuple(candidates)


def detect_gpu(paths: DiscoveryPaths, selected_id: str | None = None) -> Device | None:
    return _selected(list(detect_gpus(paths)), selected_id)


def detect_devices(
    paths: DiscoveryPaths | None = None,
    selected_ids: dict[str, str] | None = None,
) -> list[Device]:
    """Detect every selectable GPU and the TPU/NPU defaults, in preference order."""
    resolved = paths or DiscoveryPaths()
    selectors = selected_ids or {}
    gpus = list(detect_gpus(resolved))
    selected_gpu = selectors.get("gpu")
    if selected_gpu is not None:
        gpus.sort(key=lambda device: (device.id != selected_gpu, device.id))
    found = [
        detect_npu(resolved, selectors.get("npu")),
        detect_tpu(resolved, selectors.get("tpu")),
    ]
    return gpus + [device for device in found if device is not None]


def device_utilization(paths: DiscoveryPaths, device: Device) -> float | None:
    """Real device utilization from sysfs, when the kernel exposes it.

    amdgpu (and some other DRM drivers) publish ``gpu_busy_percent`` per
    device; that covers all work on the accelerator, not only OmniTensor's
    own inference, which is the honest number for a load display.  Returns
    ``None`` when the driver does not expose it.
    """
    if device.kind != "dri":
        return None
    node = device.id.removeprefix("gpu-")
    raw = _read_trimmed(paths.sys / f"class/drm/{node}/device/gpu_busy_percent")
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return max(0.0, min(100.0, value))
