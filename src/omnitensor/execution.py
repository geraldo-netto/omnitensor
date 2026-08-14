"""Production accelerator-executor composition."""

from __future__ import annotations

from .discovery import BACKENDS
from .executors.base import DEVICE_ABSENT, Availability
from .executors.gpu import CompositeGpuExecutor, GpuExecutor
from .executors.npu import NpuExecutor
from .executors.tpu import TpuExecutor
from .executors.vulkan import VulkanDeviceRequest, VulkanGpuExecutor


class _UnavailableSelectedGpuExecutor:
    backend = "gpu"
    model_formats = frozenset({"ncnn", "onnx"})

    def __init__(self, device_id: str) -> None:
        self._device_id = device_id

    def availability(self) -> Availability:
        return Availability(
            False,
            f"Selected GPU {self._device_id} is unavailable",
            DEVICE_ABSENT,
        )

    def run(self, _model_path: str, _inputs: list):
        raise RuntimeError(f"Selected GPU {self._device_id} is unavailable")


class ExecutorSet(dict):
    """Legacy backend view plus one executor and queue identity per device."""

    def __init__(self, aliases: dict, devices: tuple, device_executors: dict) -> None:
        super().__init__(aliases)
        self.devices = devices
        self.device_executors = device_executors
        self._gpu_ids = tuple(device.id for device in devices if device.backend == "gpu")

    def for_device(self, gpu_device_id: str | None) -> dict:
        selected = dict(self)
        if gpu_device_id is None:
            return selected
        executor = self.device_executors.get(gpu_device_id)
        selected["gpu"] = (
            executor
            if executor is not None
            else _UnavailableSelectedGpuExecutor(gpu_device_id)
        )
        return selected

    def scheduler_executors(self) -> dict:
        lanes = {backend: self[backend] for backend in ("tpu", "npu")}
        if self._gpu_ids:
            lanes.update(
                (device_id, self.device_executors[device_id])
                for device_id in self._gpu_ids
            )
        else:
            lanes["gpu"] = self["gpu"]
        return lanes

    def lane_key(self, backend: str, gpu_device_id: str | None) -> str:
        if backend != "gpu":
            return backend
        if gpu_device_id is not None:
            return gpu_device_id
        return self._gpu_ids[0] if self._gpu_ids else "gpu"

    def device_id(self, backend: str, gpu_device_id: str | None) -> str | None:
        if backend == "gpu":
            key = self.lane_key(backend, gpu_device_id)
            return None if key == "gpu" else key
        return next(
            (device.id for device in self.devices if device.backend == backend),
            None,
        )


def _vulkan_request(device) -> VulkanDeviceRequest:
    try:
        vendor_id = int(device.vendor, 16)
        hardware_id = int(device.hardware_id, 16)
    except (TypeError, ValueError):
        # A selected render node that cannot be matched to a Vulkan identity
        # must not fall through to ncnn's preferred (possibly other) GPU.
        vendor_id = hardware_id = -1
    return VulkanDeviceRequest(vendor_id, hardware_id, device.identity_index)


def _build_executor(backend: str, device_present: bool, device=None):
    if backend == "tpu":
        return TpuExecutor(device_present)
    if backend == "npu":
        return NpuExecutor(device_present)
    return CompositeGpuExecutor([
        # Vulkan first so any Mesa driver serves GPU work; ONNX Runtime with a
        # CUDA/ROCm provider remains the optional second lane.
        VulkanGpuExecutor(
            device_present,
            requested_device=_vulkan_request(device) if device is not None else None,
        ),
        GpuExecutor(device_present),
    ])


def build_executors(
    devices,
    *,
    previous_devices=None,
    previous_executors: dict | None = None,
) -> ExecutorSet:
    """Build changed backend adapters and preserve unchanged runtime state."""
    current_devices = tuple(devices)
    current = {}
    for device in current_devices:
        current.setdefault(device.backend, device)
    previous = {}
    for device in previous_devices or []:
        previous.setdefault(device.backend, device)
    existing = previous_executors or {}
    can_reuse = previous_devices is not None and previous_executors is not None
    aliases = {
        backend: existing[backend]
        if can_reuse and backend in existing and previous.get(backend) == current.get(backend)
        else _build_executor(
            backend,
            backend in current,
            current.get(backend) if backend == "gpu" else None,
        )
        for backend in BACKENDS
    }
    previous_by_id = (
        previous_executors.device_executors
        if isinstance(previous_executors, ExecutorSet)
        else {}
    )
    previous_devices_by_id = {
        device.id: device for device in previous_devices or []
    }
    device_executors = {}
    for device in current_devices:
        if device.backend != "gpu":
            device_executors[device.id] = aliases[device.backend]
            continue
        if (
            can_reuse
            and previous_devices_by_id.get(device.id) == device
            and device.id in previous_by_id
        ):
            executor = previous_by_id[device.id]
        else:
            executor = _build_executor("gpu", True, device)
        device_executors[device.id] = executor
    gpu_ids = [device.id for device in current_devices if device.backend == "gpu"]
    if gpu_ids:
        aliases["gpu"] = device_executors[gpu_ids[0]]
    return ExecutorSet(aliases, current_devices, device_executors)
