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

from dataclasses import dataclass, field
from pathlib import Path

MAX_PCIE_DEVICES = 8
MAX_ACCEL_DEVICES = 8
MAX_RENDER_DEVICES = 8
RENDER_NODE_BASE = 128

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

BACKENDS = ("tpu", "npu", "gpu")


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


def detect_tpu(paths: DiscoveryPaths) -> Device | None:
    for index in range(MAX_PCIE_DEVICES):
        if (paths.dev / f"apex_{index}").exists():
            suffix = "" if index == 0 else f" {index + 1}"
            return Device(
                id=f"tpu-pcie-{index}",
                backend="tpu",
                name=f"Coral PCIe Edge TPU{suffix}",
                kind="pcie",
            )
    usb_root = paths.sys / "bus/usb/devices"
    try:
        entries = sorted(usb_root.iterdir())
    except OSError:
        return None
    for entry in entries:
        vendor = _read_trimmed(entry / "idVendor")
        product = _read_trimmed(entry / "idProduct")
        name = CORAL_USB_IDENTITIES.get((vendor, product))
        if name is not None:
            return Device(
                id="tpu-usb",
                backend="tpu",
                name=name,
                kind="usb",
                vendor=f"{vendor}:{product}",
            )
    return None


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


def detect_npu(paths: DiscoveryPaths) -> Device | None:
    for index in range(MAX_ACCEL_DEVICES):
        device = _detect_node(
            paths.dev / f"accel/accel{index}",
            paths.sys / f"class/accel/accel{index}/device/vendor",
            NPU_VENDOR_NAMES,
            "NPU accelerator",
            f"npu-accel{index}",
            "npu",
            "accel",
        )
        if device is not None:
            return device
    return None


def detect_gpu(paths: DiscoveryPaths) -> Device | None:
    for offset in range(MAX_RENDER_DEVICES):
        node_number = RENDER_NODE_BASE + offset
        device = _detect_node(
            paths.dev / f"dri/renderD{node_number}",
            paths.sys / f"class/drm/renderD{node_number}/device/vendor",
            GPU_VENDOR_NAMES,
            "GPU (render node)",
            f"gpu-renderD{node_number}",
            "gpu",
            "dri",
        )
        if device is not None:
            return device
    return None


def detect_devices(paths: DiscoveryPaths | None = None) -> list[Device]:
    """Detect one device per backend, in tpu > npu > gpu hierarchy order."""
    resolved = paths or DiscoveryPaths()
    found = (detect_tpu(resolved), detect_npu(resolved), detect_gpu(resolved))
    return [device for device in found if device is not None]


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
        return max(0.0, min(100.0, float(raw)))
    except ValueError:
        return None
