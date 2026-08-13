"""Production accelerator-executor composition."""

from __future__ import annotations

from .discovery import BACKENDS
from .executors.gpu import CompositeGpuExecutor, GpuExecutor
from .executors.npu import NpuExecutor
from .executors.tpu import TpuExecutor
from .executors.vulkan import VulkanGpuExecutor


def _build_executor(backend: str, device_present: bool):
    if backend == "tpu":
        return TpuExecutor(device_present)
    if backend == "npu":
        return NpuExecutor(device_present)
    return CompositeGpuExecutor([
        # Vulkan first so any Mesa driver serves GPU work; ONNX Runtime with a
        # CUDA/ROCm provider remains the optional second lane.
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
