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
from .tensorcontract import contract_error, declared_inputs, measured_shape
from .tensorref import (
    DenyAllInputRoots,
    InputRootPolicy,
    TensorReference,
    TensorReferenceError,
    load_referenced_tensor,
    parse_references,
    verify_reference,
)

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
        input_roots: InputRootPolicy | None = None,
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
        # Denied by default: referencing a file is a capability, not a default.
        self._input_roots = input_roots or DenyAllInputRoots()

    def admit(self, workload_id: str, payload: dict) -> None:
        """Refuse, in the submitting call, what a running job would refuse later.

        The pipeline reaches this dispatcher from its infer stage, long after
        submission has answered ``accepted``, so every refusal below used to
        arrive as a stored failure a caller had to poll for.  None of these
        decisions needs the accelerator, the queue, or the tensor values, so
        they belong in the caller's own call.

        Deliberately not a resolution or a routing decision: an artifact that
        is momentarily unresolvable or a backend that is momentarily busy is a
        condition of the runtime at dispatch time, and answering it at
        submission would refuse jobs that would have run.
        """
        workload = self._runnable(workload_id)
        specs = declared_inputs(workload.model)
        references = self._references(payload)
        if references is None:
            inputs = self._validated(_inline_inputs(payload, self._max_input_tensors))
            self._agrees(specs, [(measured_shape(tensor), None) for tensor in inputs])
            return
        self._agrees(
            specs,
            [(reference.shape, reference.dtype) for reference in references],
        )
        budget = _ElementBudget(
            self._max_input_elements, code="payload-invalid", label="Job payload"
        )
        for reference in references:
            # From the declared shape, so an oversized tensor is refused before
            # its file is opened rather than after it has been read.
            budget.spend(reference.element_count)
            self._verify(reference)

    def _agrees(self, specs, shapes) -> None:
        """Refuse an input the model cannot accept, in the submitting call.

        The alternative was an opaque native failure from inside the executor,
        after the job had been admitted, queued, and dispatched.
        """
        mismatch = contract_error(specs, shapes)
        if mismatch is not None:
            raise JobDispatchError("input-contract-mismatch", mismatch)

    def dispatch(self, job_id: str, workload_id: str, payload: dict) -> asyncio.Future:
        """Admit, resolve, route, and queue one job; never run it inline."""
        workload = self._runnable(workload_id)
        model = workload.model
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

    def _runnable(self, workload_id: str) -> Workload:
        """The profile this job names, if it exists and can run inference."""
        workload = self._workloads.get(workload_id)
        if workload is None:
            raise JobDispatchError(
                "workload-unknown", f"No such workload profile: {workload_id}"
            )
        if workload.model is None:
            raise JobDispatchError(
                "workload-has-no-model",
                f"{workload_id} declares no model and cannot run inference",
            )
        return workload

    def _references(self, payload: dict) -> tuple[TensorReference, ...] | None:
        """The references a payload declares, or ``None`` for an inline payload."""
        if not isinstance(payload, dict):
            raise JobDispatchError("payload-invalid", "Job payload must be an object")
        try:
            return parse_references(payload, max_tensors=self._max_input_tensors)
        except TensorReferenceError as error:
            # Re-raised as a dispatch failure so the caller sees one error
            # vocabulary regardless of how it supplied the input.
            raise JobDispatchError(error.code, error.detail) from error

    def _verify(self, reference: TensorReference) -> None:
        try:
            verify_reference(reference, self._input_roots)
        except TensorReferenceError as error:
            raise JobDispatchError(error.code, error.detail) from error

    def _inputs(self, payload: dict) -> list:
        """Validate submitted tensors before anything is queued.

        Unvalidated input reaches the backend as whatever the caller sent, so
        the first thing to reject a malformed tensor is a native library
        running on a shared accelerator.  Validating here keeps that failure a
        stable refusal on the submitting side.

        Referenced inputs are read and re-digested here even when admission
        already verified them, because the file may have changed since; what
        runs is the tensor these bytes contain.
        """
        references = self._references(payload)
        if references is None:
            return self._validated(_inline_inputs(payload, self._max_input_tensors))
        try:
            loaded = [
                load_referenced_tensor(reference, self._input_roots)
                for reference in references
            ]
        except TensorReferenceError as error:
            raise JobDispatchError(error.code, error.detail) from error
        # Already shaped, digest-checked, and bounded on the way in, so it
        # rejoins the inline path here and is validated identically.
        return self._validated(loaded)

    def _validated(self, inputs: list) -> list:
        """The one place tensor shape and element budget are enforced.

        Inline and referenced inputs both arrive here, so a tensor cannot be
        admitted by one route under rules the other would refuse.
        """
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


def _inline_inputs(payload: dict, max_input_tensors: int) -> list:
    """The tensors a payload carries inline, bounded by count."""
    inputs = payload.get("inputs")
    if not isinstance(inputs, list):
        raise JobDispatchError(
            "payload-invalid", "Job payload must contain an inputs array"
        )
    if len(inputs) > max_input_tensors:
        raise JobDispatchError(
            "payload-invalid",
            f"At most {max_input_tensors} input tensors may be submitted",
        )
    return inputs


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
    digest = model.get("sha256")
    if digest:
        return ArtifactReference(
            model["id"], model["version"], model["format"], digest
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
