"""NPU executor via OpenVINO.

Compiles for OpenVINO's ``NPU`` device only; if the OpenVINO build or the
platform does not expose an NPU plugin the backend reports unavailable
with that reason.  ``openvino`` and ``onnx`` model formats are accepted —
OpenVINO reads both IR and ONNX.
"""

from __future__ import annotations

import threading
import time

from .base import Availability, InferenceResult, require_available

OPENVINO_DEVICE = "NPU"


def _import_openvino():  # pragma: no cover - trivial import shim
    try:
        import openvino  # noqa: PLC0415
        return openvino
    except ImportError:
        return None


class NpuExecutor:
    backend = "npu"
    model_formats = frozenset({"openvino", "onnx"})

    def __init__(self, device_present: bool, runtime=None):
        self._device_present = device_present
        self._runtime = runtime if runtime is not None else _import_openvino()
        self._core = None
        self._core_lock = threading.Lock()

    def _ensure_core(self):
        # availability() runs on the event loop thread while run() executes in
        # a worker thread (asyncio.to_thread); the lock guarantees exactly one
        # openvino.Core is ever constructed and safely published.
        with self._core_lock:
            if self._core is None:
                self._core = self._runtime.Core()
            return self._core

    def availability(self) -> Availability:
        if not self._device_present:
            return Availability(False, "No /dev/accel NPU device detected")
        if self._runtime is None:
            return Availability(False, "openvino is not installed")
        try:
            devices = self._ensure_core().available_devices
        except Exception as error:  # noqa: BLE001 - plugin discovery can fail arbitrarily
            return Availability(False, f"OpenVINO device discovery failed: {error}")
        if OPENVINO_DEVICE not in devices:
            return Availability(False, "OpenVINO reports no NPU plugin on this platform")
        return Availability(True)

    def run(self, model_path: str, inputs: list) -> InferenceResult:
        require_available(self)
        core = self._ensure_core()
        compiled = core.compile_model(model_path, OPENVINO_DEVICE)
        started = time.monotonic()
        outputs = compiled(inputs)
        duration_ms = (time.monotonic() - started) * 1000
        return InferenceResult(outputs=list(outputs.values()), duration_ms=duration_ms)
