"""Coral Edge TPU executor via tflite-runtime and libedgetpu.

Uses ``tflite_runtime`` with ``load_delegate("libedgetpu.so.1")`` rather
than pycoral: pycoral's packaging is effectively dead and pins old Python,
while the delegate path is the supported minimal surface.  Only fully
quantized, edgetpu-compiled models are compatible.
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

EDGETPU_DELEGATE = "libedgetpu.so.1"
# How long a delegate-load failure is trusted before the next run retries it.
# Latching the failure forever turned one transient error — a device reset, a
# momentarily busy USB accelerator — into a backend that stayed disabled until
# the device set changed.
DELEGATE_RETRY_SECONDS = 30.0


def _import_tflite():  # pragma: no cover - trivial import shim
    try:
        from tflite_runtime import interpreter as tflite  # noqa: PLC0415

        return tflite
    except ImportError:
        return None


class TpuExecutor(CachingExecutor):
    backend = "tpu"
    model_formats = frozenset({"tflite-edgetpu"})
    device_absent_reason = "No Coral Edge TPU device detected"
    runtime_missing_reason = "tflite-runtime is not installed"

    def __init__(
        self,
        device_present: bool,
        runtime=None,
        *,
        delegate_retry_seconds: float = DELEGATE_RETRY_SECONDS,
        clock=time.monotonic,
        max_cached_models: int = DEFAULT_MAX_CACHED_MODELS,
    ):
        super().__init__(
            device_present,
            runtime if runtime is not None else _import_tflite(),
            max_cached_models=max_cached_models,
            clock=clock,
        )
        self._delegate_retry_seconds = delegate_retry_seconds
        # Written by run() in a dedicated off-loop thread and read by
        # availability() on the event loop thread; the lock makes the
        # publication of the error text safe across threads.  The timestamp is
        # what keeps the failure from latching.
        self._delegate_error: tuple[str, float] | None = None
        self._delegate_error_lock = threading.Lock()
        # Interpreters are expensive model-specific runtime state, cached by
        # the base class.  The scheduler serializes TPU work; this lock also
        # preserves that safety when callers use the executor directly from
        # multiple threads.
        self._interpreter_lock = threading.Lock()
        # Held for the executor's life and rebuilt only after a failure: the
        # Coral admits one claim, and this is that claim.
        self._edgetpu_delegate = None

    def _runtime_health(self) -> Availability:
        with self._delegate_error_lock:
            failure = self._delegate_error
        if failure is not None and self._clock() - failure[1] < self._delegate_retry_seconds:
            return Availability(False, failure[0], RUNTIME_UNUSABLE)
        return Availability(True)

    def _interpreter_for(self, model_path: str):
        return self._models.get_or_build(model_path, lambda: self._build_interpreter(model_path))

    def _delegate(self):
        """The one Edge TPU delegate this executor holds.

        One per executor, not one per model: the cache keeps up to four
        interpreters and a delegate each meant four live claims on a device
        that normally admits one, so the second model built could fail for no
        reason the operator could see.
        """
        if self._edgetpu_delegate is None:
            self._edgetpu_delegate = self._runtime.load_delegate(EDGETPU_DELEGATE)
        return self._edgetpu_delegate

    def _build_interpreter(self, model_path: str):
        try:
            interpreter = self._runtime.Interpreter(
                model_path=model_path,
                experimental_delegates=[self._delegate()],
            )
            interpreter.allocate_tensors()
        except (ValueError, OSError, RuntimeError) as error:
            # Latched around the whole sequence, not around load_delegate
            # alone: clearing the error before the interpreter was built left
            # availability() reporting True while every run retried the full
            # delegate load and failed again.
            self._forget_delegate()
            message = f"Could not load {EDGETPU_DELEGATE}: {error}"
            with self._delegate_error_lock:
                self._delegate_error = (message, self._clock())
            raise RuntimeError(message) from error
        with self._delegate_error_lock:
            self._delegate_error = None
        return interpreter

    def _forget_delegate(self) -> None:
        """Drop the delegate so the next attempt claims the device afresh."""
        self._edgetpu_delegate = None

    def close(self) -> None:
        """Release the cached interpreters, then the device claim itself."""
        super().close()
        self._forget_delegate()

    def run(self, model_path: str, inputs: list) -> InferenceResult:
        require_available(self)
        with self._models.running(), self._interpreter_lock:
            interpreter = self._interpreter_for(model_path)
            input_details = interpreter.get_input_details()
            for detail, value in zip(input_details, inputs, strict=True):
                interpreter.set_tensor(detail["index"], value)
            # Reading the output tensors is inside the measured window, the
            # same as every other lane: invoke() alone timed less of the work
            # than the other three backends reported under the same name.
            return self._timed(lambda: self._invoked(interpreter))

    @staticmethod
    def _invoked(interpreter) -> list:
        interpreter.invoke()
        return [
            interpreter.get_tensor(detail["index"]) for detail in interpreter.get_output_details()
        ]
