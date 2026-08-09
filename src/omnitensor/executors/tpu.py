"""Coral Edge TPU executor via tflite-runtime and libedgetpu.

Uses ``tflite_runtime`` with ``load_delegate("libedgetpu.so.1")`` rather
than pycoral: pycoral's packaging is effectively dead and pins old Python,
while the delegate path is the supported minimal surface.  Only fully
quantized, edgetpu-compiled models are compatible.
"""

from __future__ import annotations

import threading
import time

from .base import Availability, InferenceResult, require_available

EDGETPU_DELEGATE = "libedgetpu.so.1"


def _import_tflite():  # pragma: no cover - trivial import shim
    try:
        from tflite_runtime import interpreter as tflite  # noqa: PLC0415
        return tflite
    except ImportError:
        return None


class TpuExecutor:
    backend = "tpu"
    model_formats = frozenset({"tflite-edgetpu"})

    def __init__(self, device_present: bool, runtime=None):
        self._device_present = device_present
        self._runtime = runtime if runtime is not None else _import_tflite()
        # Written by run() in a worker thread (asyncio.to_thread) and read by
        # availability() on the event loop thread; the lock makes the
        # publication of the error text safe across threads.
        self._delegate_error: str | None = None
        self._delegate_error_lock = threading.Lock()
        # Interpreters are expensive model-specific runtime state.  The
        # scheduler serializes TPU work; this lock also preserves that safety
        # when callers use the executor directly from multiple threads.
        self._interpreters: dict[str, object] = {}
        self._interpreter_lock = threading.Lock()

    def availability(self) -> Availability:
        if not self._device_present:
            return Availability(False, "No Coral Edge TPU device detected")
        if self._runtime is None:
            return Availability(False, "tflite-runtime is not installed")
        with self._delegate_error_lock:
            delegate_error = self._delegate_error
        if delegate_error is not None:
            return Availability(False, delegate_error)
        return Availability(True)

    def _interpreter_for(self, model_path: str):
        interpreter = self._interpreters.get(model_path)
        if interpreter is not None:
            return interpreter
        try:
            delegate = self._runtime.load_delegate(EDGETPU_DELEGATE)
        except (ValueError, OSError) as error:
            message = f"Could not load {EDGETPU_DELEGATE}: {error}"
            with self._delegate_error_lock:
                self._delegate_error = message
            raise RuntimeError(message) from error
        interpreter = self._runtime.Interpreter(
            model_path=model_path,
            experimental_delegates=[delegate],
        )
        interpreter.allocate_tensors()
        self._interpreters[model_path] = interpreter
        return interpreter

    def run(self, model_path: str, inputs: list) -> InferenceResult:
        require_available(self)
        with self._interpreter_lock:
            interpreter = self._interpreter_for(model_path)
            input_details = interpreter.get_input_details()
            for detail, value in zip(input_details, inputs, strict=True):
                interpreter.set_tensor(detail["index"], value)
            started = time.monotonic()
            interpreter.invoke()
            duration_ms = (time.monotonic() - started) * 1000
            outputs = [
                interpreter.get_tensor(detail["index"])
                for detail in interpreter.get_output_details()
            ]
        return InferenceResult(outputs=outputs, duration_ms=duration_ms)
