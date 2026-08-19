"""NPU executor via OpenVINO.

Compiles for OpenVINO's ``NPU`` device only; if the OpenVINO build or the
platform does not expose an NPU plugin the backend reports unavailable
with that reason.  ``openvino`` and ``onnx`` model formats are accepted —
OpenVINO reads both IR and ONNX.
"""

from __future__ import annotations

import threading
import time

from .base import (
    DEFAULT_MAX_CACHED_MODELS,
    RUNTIME_UNUSABLE,
    Availability,
    CachingExecutor,
    InferenceResult,
    require_available,
)

OPENVINO_DEVICE = "NPU"


def _import_openvino():  # pragma: no cover - trivial import shim
    try:
        import openvino  # noqa: PLC0415

        return openvino
    except ImportError:
        return None


class NpuExecutor(CachingExecutor):
    backend = "npu"
    model_formats = frozenset({"openvino", "onnx"})
    device_absent_reason = "No /dev/accel NPU device detected"
    runtime_missing_reason = "openvino is not installed"

    def __init__(
        self,
        device_present: bool,
        runtime=None,
        *,
        max_cached_models: int = DEFAULT_MAX_CACHED_MODELS,
        clock=time.monotonic,
    ):
        super().__init__(
            device_present,
            runtime if runtime is not None else _import_openvino(),
            max_cached_models=max_cached_models,
            clock=clock,
        )
        self._core = None
        self._core_lock = threading.Lock()

    def _ensure_core(self):
        # availability() runs on the event loop thread while run() executes in
        # a dedicated off-loop thread; the lock guarantees exactly one
        # openvino.Core is ever constructed and safely published.
        with self._core_lock:
            if self._core is None:
                self._core = self._runtime.Core()
            return self._core

    def _runtime_health(self) -> Availability:
        try:
            devices = self._ensure_core().available_devices
        except Exception as error:  # noqa: BLE001 - plugin discovery can fail arbitrarily
            return Availability(
                False, f"OpenVINO device discovery failed: {error}", RUNTIME_UNUSABLE
            )
        if OPENVINO_DEVICE not in devices:
            return Availability(
                False, "OpenVINO reports no NPU plugin on this platform", RUNTIME_UNUSABLE
            )
        return Availability(True)

    def run(self, model_path: str, inputs: list) -> InferenceResult:
        require_available(self)
        with self._models.running():
            compiled = self._compiled_model_for(model_path)
            return self._timed(lambda: compiled(inputs).values())

    def _compiled_model_for(self, model_path: str):
        return self._models.get_or_build(
            model_path,
            lambda: self._ensure_core().compile_model(model_path, OPENVINO_DEVICE),
        )
