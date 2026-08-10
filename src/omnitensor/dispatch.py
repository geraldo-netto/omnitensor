"""Turn an admitted job into inference on a real accelerator.

This is the join between four things that each already fail closed on their
own: the workload catalog says which model a profile may run, the artifact
store proves the file on disk is that exact model, the executors say which
backends can run its format and are healthy, and the scheduler serializes work
onto the one physical device.  The dispatcher's whole job is to keep those
guarantees intact across the join rather than re-deciding any of them.

Two rules it never relaxes.  A job runs only the artifact its own manifest
declares — a payload cannot name a model — so a plugin cannot reach another
profile's weights.  And the backend must both support the model's format and
be available for it; there is no CPU fallback when nothing qualifies, only a
stable refusal naming why each candidate was rejected.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from .executors.base import Executor, InferenceResult
from .jobs import JobDispatchError
from .plugins.artifacts import ArtifactReference, ArtifactResolution
from .registry import Workload
from .scheduler import QueueFullError, Scheduler, pick_backend

MAX_INPUT_TENSORS = 64
# One result must stay small enough to cross IPC as a single bounded frame.
MAX_TENSOR_ELEMENTS = 1 << 20


@runtime_checkable
class ArtifactSource(Protocol):
    """The artifact store, narrowed to what dispatch is allowed to ask it."""

    def resolve(self, reference: ArtifactReference) -> ArtifactResolution: ...

    def resolve_active(self, artifact_id: str) -> ArtifactResolution: ...


class InferenceJobDispatcher:
    """:class:`~omnitensor.jobs.JobDispatcher` backed by verified artifacts."""

    def __init__(
        self,
        workloads: Mapping[str, Workload],
        scheduler: Scheduler,
        executors: Mapping[str, Executor],
        artifacts: ArtifactSource,
        *,
        max_input_tensors: int = MAX_INPUT_TENSORS,
        max_input_elements: int = MAX_TENSOR_ELEMENTS,
    ) -> None:
        if max_input_tensors < 1:
            raise ValueError("max_input_tensors must be positive")
        if max_input_elements < 1:
            raise ValueError("max_input_elements must be positive")
        self._workloads = workloads
        self._scheduler = scheduler
        self._executors = executors
        self._artifacts = artifacts
        self._max_input_tensors = max_input_tensors
        self._max_input_elements = max_input_elements

    def dispatch(self, job_id: str, workload_id: str, payload: dict) -> asyncio.Future:
        """Admit, resolve, route, and queue one job; never run it inline."""
        workload = self._workloads.get(workload_id)
        if workload is None:
            raise JobDispatchError(
                "workload-unknown", f"No such workload profile: {workload_id}"
            )
        model = workload.model
        if model is None:
            raise JobDispatchError(
                "workload-has-no-model",
                f"{workload_id} declares no model and cannot run inference",
            )
        inputs = self._inputs(payload)
        backend, reason = pick_backend(workload, dict(self._executors))
        if backend is None:
            # No CPU fallback exists by design, so an unroutable job is refused
            # with the reason each candidate backend was rejected.
            raise JobDispatchError("no-backend-available", reason)
        path = self._model_path(workload, model)
        try:
            return self._scheduler.submit(backend, workload_id, path, inputs)
        except QueueFullError as error:
            raise JobDispatchError("backend-queue-full", str(error)) from error
        except RuntimeError as error:
            raise JobDispatchError("scheduler-unavailable", str(error)) from error

    def _inputs(self, payload: dict) -> list:
        """Validate submitted tensors before anything is queued.

        Unvalidated input reaches the backend as whatever the caller sent, so
        the first thing to reject a malformed tensor is a native library
        running on a shared accelerator.  Validating here keeps that failure a
        stable refusal on the submitting side.
        """
        if not isinstance(payload, dict):
            raise JobDispatchError("payload-invalid", "Job payload must be an object")
        inputs = payload.get("inputs")
        if not isinstance(inputs, list):
            raise JobDispatchError(
                "payload-invalid", "Job payload must contain an inputs array"
            )
        if len(inputs) > self._max_input_tensors:
            raise JobDispatchError(
                "payload-invalid",
                f"At most {self._max_input_tensors} input tensors may be submitted",
            )
        budget = _ElementBudget(
            self._max_input_elements, code="payload-invalid", label="Job payload"
        )
        return [validate_input_tensor(tensor, budget) for tensor in inputs]

    def _model_path(self, workload: Workload, model: dict) -> str:
        """Resolve the model this manifest declares, and only that one."""
        workload_id = workload.id
        reference = declared_artifact_reference(workload, model)
        try:
            if reference is not None:
                resolution = self._artifacts.resolve(reference)
            else:
                # A v1 manifest carries no per-model digest, so the strongest
                # available guarantee is the store's own: the active version was
                # digest-verified when it was installed and is re-verified here.
                resolution = self._artifacts.resolve_active(model["id"])
        except Exception as error:  # noqa: BLE001 - store failures are arbitrary
            raise JobDispatchError(
                "artifact-unavailable",
                f"{workload_id}: artifact store is unreadable: {type(error).__name__}",
            ) from error
        if not resolution.ready or resolution.path is None:
            raise JobDispatchError(
                "artifact-unavailable", f"{workload_id}: {resolution.reason}"
            )
        return str(resolution.path)


def validate_input_tensor(value: object, budget: _ElementBudget) -> object:
    """Accept one rectangular tensor of finite numbers, or refuse it.

    Ragged nesting is rejected because every backend expects a rectangular
    buffer: accepting it here would turn a caller's mistake into an error
    raised deep inside a native library, mid-inference, on a shared device.
    """
    if isinstance(value, bool):
        raise JobDispatchError("payload-invalid", "Input tensors must contain numbers")
    if isinstance(value, (int, float)):
        budget.spend()
        if isinstance(value, float) and not math.isfinite(value):
            raise JobDispatchError(
                "payload-invalid", "Input tensors must contain finite numbers"
            )
        return value
    if not isinstance(value, list):
        raise JobDispatchError(
            "payload-invalid",
            f"Input tensors must be numbers or nested arrays, not {type(value).__name__}",
        )
    encoded = [validate_input_tensor(item, budget) for item in value]
    shapes = {_tensor_shape(item) for item in encoded}
    if len(shapes) > 1:
        raise JobDispatchError("payload-invalid", "Input tensors must be rectangular")
    return encoded


def _tensor_shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    return (len(value),) + (_tensor_shape(value[0]) if value else ())


def declared_artifact_reference(
    workload: Workload, model: dict
) -> ArtifactReference | None:
    """The allowlisted artifact entry matching this workload's model.

    Returns ``None`` when the manifest declares no digest for it, which is the
    v1 shape.  Returning ``None`` rather than inventing a digest keeps the
    weaker guarantee visible at the call site instead of pretending to verify.
    """
    declared = workload.manifest.get("plugin", {}).get("artifacts") or ()
    for entry in declared:
        if (
            entry["id"] == model["id"]
            and entry["version"] == model["version"]
            and entry["format"] == model["format"]
        ):
            return ArtifactReference(
                entry["id"], entry["version"], entry["format"], entry["sha256"]
            )
    return None


def inference_result_payload(
    result: InferenceResult, *, max_elements: int = MAX_TENSOR_ELEMENTS
) -> dict:
    """Map an executor outcome back into a plugin-facing job result.

    Executors return whatever their backend hands them — a numpy array from
    tflite, OpenVINO, and onnxruntime, an ncnn Mat from Vulkan — none of which
    survive JSON encoding.  Converting here rather than inside each executor
    keeps native tensors available to anything that chains stages internally,
    while guaranteeing that what crosses IPC is encodable.
    """
    if not isinstance(result, InferenceResult):
        raise JobDispatchError(
            "executor-result-invalid", "Executor returned a non-result value"
        )
    budget = _ElementBudget(max_elements)
    outputs = [encode_tensor(tensor, budget) for tensor in result.outputs]
    duration = float(result.duration_ms)
    if not math.isfinite(duration):
        raise JobDispatchError(
            "executor-result-invalid", "Executor reported a non-finite duration"
        )
    return {"outputs": outputs, "durationMs": round(duration, 3)}


class _ElementBudget:
    """Bound the total elements one result may carry across all its tensors."""

    def __init__(
        self,
        maximum: int,
        *,
        code: str = "executor-result-invalid",
        label: str = "Inference result",
    ) -> None:
        if maximum < 1:
            raise ValueError("max_elements must be positive")
        self._remaining = maximum
        self._maximum = maximum
        self._code = code
        self._label = label

    def spend(self, count: int = 1) -> None:
        self._remaining -= count
        if self._remaining < 0:
            raise JobDispatchError(
                self._code,
                f"{self._label} exceeds {self._maximum} tensor elements",
            )


def encode_tensor(value: object, budget: _ElementBudget | None = None) -> object:
    """Convert one backend-native tensor into JSON-encodable values.

    Rejects rather than coerces anything unrecognised: silently stringifying a
    tensor would put an unusable value on the wire that only fails much later,
    in the consumer.
    """
    budget = budget or _ElementBudget(MAX_TENSOR_ELEMENTS)
    if isinstance(value, bool):
        budget.spend()
        return value
    if isinstance(value, int):
        budget.spend()
        return value
    if isinstance(value, float):
        budget.spend()
        if not math.isfinite(value):
            raise JobDispatchError(
                "executor-result-invalid",
                "Inference result contains a non-finite value",
            )
        return value
    # numpy, torch, and anything else exposing the array protocol.
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return encode_tensor(value.tolist(), budget)
    # ncnn Mat exposes numpy() rather than tolist().
    if hasattr(value, "numpy"):
        return encode_tensor(value.numpy().tolist(), budget)
    if isinstance(value, (list, tuple)):
        return [encode_tensor(item, budget) for item in value]
    raise JobDispatchError(
        "executor-result-invalid",
        f"Inference result contains an unencodable {type(value).__name__}",
    )
