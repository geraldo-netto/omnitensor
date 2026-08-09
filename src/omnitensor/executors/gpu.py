"""GPU executor via ONNX Runtime with CUDA or ROCm execution providers.

The CPU execution provider is deliberately never used: OmniTensor has no
CPU backend by design, so a machine whose onnxruntime build only offers
CPUExecutionProvider reports the GPU backend unavailable instead of
silently loading the host CPU.
"""

from __future__ import annotations

import time

from .base import Availability, InferenceResult, require_available

GPU_PROVIDERS = ("CUDAExecutionProvider", "ROCMExecutionProvider")


def _import_onnxruntime():  # pragma: no cover - trivial import shim
    try:
        import onnxruntime  # noqa: PLC0415
        return onnxruntime
    except ImportError:
        return None


class GpuExecutor:
    backend = "gpu"
    model_formats = frozenset({"onnx"})

    def __init__(self, device_present: bool, runtime=None):
        self._device_present = device_present
        self._runtime = runtime if runtime is not None else _import_onnxruntime()

    def _providers(self) -> list[str]:
        return [
            provider
            for provider in self._runtime.get_available_providers()
            if provider in GPU_PROVIDERS
        ]

    def availability(self) -> Availability:
        if not self._device_present:
            return Availability(False, "No GPU render node detected")
        if self._runtime is None:
            return Availability(False, "onnxruntime is not installed")
        if not self._providers():
            return Availability(
                False,
                "onnxruntime has no CUDA or ROCm execution provider; the CPU provider is not used",
            )
        return Availability(True)

    def run(self, model_path: str, inputs: list) -> InferenceResult:
        require_available(self)
        session = self._runtime.InferenceSession(model_path, providers=self._providers())
        feed = {
            session_input.name: value
            for session_input, value in zip(session.get_inputs(), inputs, strict=True)
        }
        started = time.monotonic()
        outputs = session.run(None, feed)
        duration_ms = (time.monotonic() - started) * 1000
        return InferenceResult(outputs=list(outputs), duration_ms=duration_ms)


class CompositeGpuExecutor:
    """One ``gpu`` backend, several runtimes.

    Tries sub-executors in construction order (Vulkan/ncnn first, ONNX
    Runtime second) so a machine with any Vulkan driver serves GPU work
    without CUDA or ROCm.  ``run`` dispatches on the model file extension:
    ``.param`` is an ncnn model, ``.onnx`` an ONNX model.
    """

    backend = "gpu"

    def __init__(self, executors: list):
        self._executors = [executor for executor in executors if executor is not None]
        self.model_formats = frozenset().union(
            *(executor.model_formats for executor in self._executors),
        )

    def availability(self) -> Availability:
        reasons = []
        for executor in self._executors:
            availability = executor.availability()
            if availability.available:
                return Availability(True)
            reasons.append(availability.reason)
        return Availability(False, "; ".join(reasons) or "No GPU runtime configured")

    def _executor_for(self, model_path: str):
        wanted = "ncnn" if model_path.endswith(".param") else "onnx"
        for executor in self._executors:
            if wanted in executor.model_formats and executor.availability().available:
                return executor
        raise RuntimeError(f"No available GPU runtime for model: {model_path}")

    def run(self, model_path: str, inputs: list) -> InferenceResult:
        return self._executor_for(model_path).run(model_path, inputs)
