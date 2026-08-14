"""GPU executor via ONNX Runtime with CUDA or ROCm execution providers.

The CPU execution provider is deliberately never used: OmniTensor has no
CPU backend by design, so a machine whose onnxruntime build only offers
CPUExecutionProvider reports the GPU backend unavailable instead of
silently loading the host CPU.
"""

from __future__ import annotations

import threading
import time

from .base import (
    DEFAULT_MAX_CACHED_MODELS,
    DEVICE_ABSENT,
    FORMAT_UNSUPPORTED,
    RUNTIME_MISSING,
    RUNTIME_UNUSABLE,
    Availability,
    InferenceResult,
    ModelCache,
    most_actionable,
    require_available,
    supports_model,
)

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

    def __init__(
        self,
        device_present: bool,
        runtime=None,
        *,
        max_cached_models: int = DEFAULT_MAX_CACHED_MODELS,
    ):
        self._device_present = device_present
        self._runtime = runtime if runtime is not None else _import_onnxruntime()
        self._sessions = ModelCache(max_cached_models)
        self._sessions_lock = threading.Lock()

    def _providers(self) -> list[str]:
        return [
            provider
            for provider in self._runtime.get_available_providers()
            if provider in GPU_PROVIDERS
        ]

    def availability(self) -> Availability:
        if not self._device_present:
            return Availability(False, "No GPU render node detected", DEVICE_ABSENT)
        if self._runtime is None:
            return Availability(False, "onnxruntime is not installed", RUNTIME_MISSING)
        if not self._providers():
            return Availability(
                False,
                "onnxruntime has no CUDA or ROCm execution provider; the CPU provider is not used",
                RUNTIME_UNUSABLE,
            )
        return Availability(True)

    def run(self, model_path: str, inputs: list) -> InferenceResult:
        require_available(self)
        session = self._session_for(model_path)
        feed = {
            session_input.name: value
            for session_input, value in zip(session.get_inputs(), inputs, strict=True)
        }
        started = time.monotonic()
        outputs = session.run(None, feed)
        duration_ms = (time.monotonic() - started) * 1000
        return InferenceResult(outputs=list(outputs), duration_ms=duration_ms)

    def _session_for(self, model_path: str):
        with self._sessions_lock:
            return self._sessions.get_or_build(
                model_path,
                lambda: self._runtime.InferenceSession(
                    model_path, providers=self._providers()
                ),
            )


class CompositeGpuExecutor:
    """One ``gpu`` backend, several runtimes.

    Tries sub-executors in construction order (Vulkan/ncnn first, ONNX
    Runtime second) so a machine with any Vulkan driver serves GPU work
    without CUDA or ROCm. Scheduler dispatch uses the declared model format;
    legacy direct ``run`` calls retain their filename-based routing contract.
    """

    backend = "gpu"

    def __init__(self, executors: list):
        self._executors = [executor for executor in executors if executor is not None]
        self.model_formats = frozenset().union(
            *(executor.model_formats for executor in self._executors),
        )

    def availability(self) -> Availability:
        return self._availability_of(self._executors)

    @staticmethod
    def _availability_of(executors: list) -> Availability:
        reasons = []
        codes = []
        for executor in executors:
            availability = executor.availability()
            if availability.available:
                return Availability(True)
            reasons.append(availability.reason)
            codes.append(availability.code)
        return Availability(
            False,
            "; ".join(reasons) or "No GPU runtime configured",
            # One backend, several runtimes: the lane a user can still fix wins
            # over one they cannot, so "install ncnn" is not hidden behind
            # "no ONNX Runtime provider".
            most_actionable(codes),
        )

    def availability_for(self, model: dict | None) -> Availability:
        compatible = [executor for executor in self._executors if supports_model(executor, model)]
        if not compatible:
            return Availability(
                False, "No GPU runtime supports the model format", FORMAT_UNSUPPORTED
            )
        return self._availability_of(compatible)

    def _executor_for(self, model_format: str, model_path: str):
        for executor in self._executors:
            if (
                model_format in executor.model_formats
                and executor.availability().available
            ):
                return executor
        raise RuntimeError(f"No available GPU runtime for model: {model_path}")

    def run_for_format(
        self, model_format: str, model_path: str, inputs: list,
    ) -> InferenceResult:
        """Run the lane selected by the manifest's declared model format."""
        return self._executor_for(model_format, model_path).run(model_path, inputs)

    def run(self, model_path: str, inputs: list) -> InferenceResult:
        model_format = "ncnn" if model_path.endswith(".param") else "onnx"
        return self.run_for_format(model_format, model_path, inputs)
