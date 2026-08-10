"""Vulkan GPU executor via ncnn.

Runs ``ncnn`` models (``.param`` + sibling ``.bin``) with Vulkan compute,
so any Mesa/RADV/ANV/NVIDIA Vulkan driver works without CUDA or ROCm.
Device selection honours the no-CPU rule: software Vulkan devices
(llvmpipe, type ``cpu``) are never used — if only a software device is
present the backend reports unavailable instead of silently running on
the host CPU.  Discrete devices are preferred over integrated ones.
"""

from __future__ import annotations

import time
from pathlib import Path

from .base import DEVICE_ABSENT, RUNTIME_MISSING, RUNTIME_UNUSABLE, Availability, InferenceResult

# ncnn VkGpuInfo.type(): 0 discrete, 1 integrated, 2 virtual, 3 cpu (software).
_DEVICE_PREFERENCE = {0: 0, 1: 1, 2: 2}


def _import_ncnn():  # pragma: no cover - trivial import shim
    try:
        import ncnn  # noqa: PLC0415
        return ncnn
    except ImportError:
        return None


class VulkanGpuExecutor:
    backend = "gpu"
    model_formats = frozenset({"ncnn"})

    def __init__(self, device_present: bool, runtime=None):
        self._device_present = device_present
        self._runtime = runtime if runtime is not None else _import_ncnn()

    def _select_device(self) -> tuple[int | None, str]:
        try:
            count = self._runtime.get_gpu_count()
        except Exception as error:  # noqa: BLE001 - loader failures are arbitrary
            return None, f"Vulkan device enumeration failed: {error}"
        candidates: list[tuple[int, int]] = []
        for index in range(count):
            device_type = self._runtime.get_gpu_info(index).type()
            if device_type in _DEVICE_PREFERENCE:
                candidates.append((_DEVICE_PREFERENCE[device_type], index))
        if not candidates:
            if count > 0:
                return None, (
                    "Only a software (CPU) Vulkan device is present; "
                    "CPU execution is not used"
                )
            return None, "No Vulkan device is available"
        return min(candidates)[1], ""

    def availability(self) -> Availability:
        device, reason, code = self._available_device()
        return Availability(device is not None, reason, code)

    def _available_device(self) -> tuple[int | None, str, str]:
        if not self._device_present:
            return None, "No GPU render node detected", DEVICE_ABSENT
        if self._runtime is None:
            return None, "ncnn is not installed", RUNTIME_MISSING
        device, reason = self._select_device()
        return device, reason, "" if device is not None else RUNTIME_UNUSABLE

    def run(self, model_path: str, inputs: list) -> InferenceResult:
        # Resolve availability and device index in one enumeration.  Calling
        # availability() first would enumerate twice and let a hotplug event
        # shift the chosen index before set_vulkan_device().
        device, reason, _code = self._available_device()
        if device is None:
            raise RuntimeError(f"{self.backend} executor unavailable: {reason}")
        param_path = Path(model_path)
        bin_path = param_path.with_suffix(".bin")
        net = self._runtime.Net()
        net.opt.use_vulkan_compute = True
        net.set_vulkan_device(device)
        if net.load_param(str(param_path)) != 0:
            raise RuntimeError(f"Could not load ncnn param: {param_path}")
        if net.load_model(str(bin_path)) != 0:
            raise RuntimeError(f"Could not load ncnn model: {bin_path}")
        extractor = net.create_extractor()
        input_names = net.input_names()
        output_names = net.output_names()
        for name, value in zip(input_names, inputs, strict=True):
            extractor.input(name, self._to_mat(value))
        started = time.monotonic()
        outputs = []
        for name in output_names:
            code, mat = extractor.extract(name)
            if code != 0:
                raise RuntimeError(f"ncnn extraction failed for output {name}")
            outputs.append(self._from_mat(mat))
        duration_ms = (time.monotonic() - started) * 1000
        return InferenceResult(outputs=outputs, duration_ms=duration_ms)

    def _to_mat(self, value):
        if isinstance(value, self._runtime.Mat):
            return value
        import numpy  # noqa: PLC0415 - shipped with the ncnn wheel

        return self._runtime.Mat(numpy.ascontiguousarray(value, dtype=numpy.float32))

    @staticmethod
    def _from_mat(mat):
        return mat.numpy().tolist() if hasattr(mat, "numpy") else list(mat)
