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
from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable

from .dispatch_lane import PreparedDispatchLane
from .executors.base import (
    Executor,
    InferenceResult,
    availability_for_model,
    supports_model,
)
from .job_ports import JobDispatchError
from .plugins.artifacts import ArtifactReference, ArtifactResolution
from .registry import Workload
from .scheduler import QueueFullError, Scheduler, runnable_model, select_backend
from .tensorcontract import contract_error, declared_inputs, measured_shape
from .tensorref import (
    DenyAllInputRoots,
    InputRootPolicy,
    TensorReference,
    TensorReferenceError,
    parse_references,
    referenced_inputs,
    verify_reference,
)

MAX_INPUT_TENSORS = 64
# The default bound on what a *caller* may select as input. A result carries no
# element ceiling: the frame that must stay crossable is bounded independently
# by socket_transport.MAX_FRAME_BYTES, and destroying a completed correct
# inference — any 1024x1024x3 image is 3.1M elements — is not a protection.
MAX_TENSOR_ELEMENTS = 1 << 20


def _executor_snapshot(
    source: Mapping[str, Executor] | Callable[[], Mapping[str, Executor]],
) -> dict[str, Executor]:
    """Resolve one routing snapshot from a static or rediscovery-backed source."""
    return dict(source() if callable(source) else source)


@runtime_checkable
class ArtifactSource(Protocol):
    """The artifact store, narrowed to what dispatch is allowed to ask it.

    ``resolve`` is the whole contract. It once also declared ``resolve_active``,
    which dispatch never calls and the adapter the service always wires
    (``CachedArtifactSource``) never implemented, so the protocol described
    something no participant honoured.
    """

    def resolve(self, reference: ArtifactReference) -> ArtifactResolution: ...


class InferenceJobDispatcher:
    """:class:`~omnitensor.jobs.JobDispatcher` backed by verified artifacts."""

    def __init__(
        self,
        workloads: Mapping[str, Workload],
        scheduler: Scheduler,
        executors: Mapping[str, Executor] | Callable[[], Mapping[str, Executor]],
        artifacts: ArtifactSource,
        *,
        executor_view: Callable[[str], Mapping[str, Executor]] | None = None,
        scheduler_lane: Callable[[str, str], str] | None = None,
        device_identity: Callable[[str, str], str | None] | None = None,
        executor_for_device: Callable[[str, str], Executor | None] | None = None,
        max_input_tensors: int = MAX_INPUT_TENSORS,
        max_input_elements: int = MAX_TENSOR_ELEMENTS,
        input_roots: InputRootPolicy | None = None,
    ) -> None:
        if not isinstance(artifacts, ArtifactSource):
            # Checked at wiring time rather than at the first dispatch, where
            # a missing method surfaced as an opaque `internal-error`.
            raise TypeError("artifacts must implement the ArtifactSource protocol")
        if max_input_tensors < 1:
            raise ValueError("max_input_tensors must be positive")
        if max_input_elements < 1:
            raise ValueError("max_input_elements must be positive")
        self._workloads = workloads
        self._scheduler = scheduler
        self._executors = executors
        self._artifacts = artifacts
        self._executor_view = executor_view
        self._scheduler_lane = scheduler_lane
        self._device_identity = device_identity
        self._executor_for_device = executor_for_device
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
        self._pinned(workload)
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

    def _pinned(self, workload: Workload) -> None:
        """Refuse an unpinned manifest in the submitting call.

        Which files a manifest vouches for is a fact of the manifest, stable
        from load to shutdown and knowable without touching the artifact store
        — so it belongs here rather than arriving as a stored failure the
        caller has to poll for, which is the shape of refusal this method
        exists to remove.
        """
        for model in workload.models:
            _refuse_unpinned(workload.id, declared_artifact_reference(workload, model))

    def _agrees(self, specs, shapes) -> None:
        """Refuse an input the model cannot accept, in the submitting call.

        The alternative was an opaque native failure from inside the executor,
        after the job had been admitted, queued, and dispatched.
        """
        mismatch = contract_error(specs, shapes)
        if mismatch is not None:
            raise JobDispatchError("input-contract-mismatch", mismatch)

    def dispatch(self, job_id: str, workload_id: str, payload: dict) -> asyncio.Future:
        """Prepare and queue one job in a single non-interleaved call."""
        workload = self._runnable(workload_id)
        inputs = self._inputs(payload)
        prepared, model = self._prepare_lane(workload_id, workload)
        return self._queue_prepared(workload_id, workload, inputs, prepared, model)

    def prepare_lane(self, workload_id: str) -> tuple[PreparedDispatchLane, dict]:
        """Freeze the exact available backend, physical device, and model."""
        workload = self._runnable(workload_id)
        return self._prepare_lane(workload_id, workload)

    def _prepare_lane(
        self, workload_id: str, workload: Workload
    ) -> tuple[PreparedDispatchLane, dict]:
        executors = (
            dict(self._executor_view(workload_id))
            if self._executor_view is not None
            else _executor_snapshot(self._executors)
        )
        choice = select_backend(workload, executors)
        if choice.backend is None:
            # No CPU fallback exists by design, so an unroutable job is refused
            # with the reason each candidate backend was rejected.
            raise JobDispatchError("no-backend-available", choice.reason)
        # The lane is chosen first and the artifact follows it: a profile may
        # declare one model per format, and which one runs is decided by which
        # accelerator won, never the other way round.
        model = runnable_model(workload, choice.backend, executors)
        if model is None:
            raise JobDispatchError(
                "no-backend-available",
                f"{workload_id}: {choice.backend} declares no model this profile can run",
            )
        backend = choice.backend
        scheduler_lane = (
            self._scheduler_lane(workload_id, backend)
            if self._scheduler_lane is not None
            else backend
        )
        device_id = (
            self._device_identity(workload_id, backend)
            if self._device_identity is not None
            else scheduler_lane
        )
        if not isinstance(device_id, str) or not device_id:
            raise JobDispatchError(
                "prepared-lane-unavailable",
                f"{workload_id}: {backend} has no stable device identity",
            )
        reference = declared_artifact_reference(workload, model)
        _refuse_unpinned(workload_id, reference)
        assert reference is not None
        prepared = PreparedDispatchLane(backend, device_id, reference)
        if scheduler_lane != prepared.scheduler_lane:
            raise JobDispatchError(
                "prepared-lane-unavailable",
                f"{workload_id}: {backend} queue does not match its stable device identity",
            )
        return prepared, model

    def dispatch_prepared(
        self,
        job_id: str,
        workload_id: str,
        payload: dict,
        prepared: PreparedDispatchLane,
    ) -> asyncio.Future:
        """Validate and enqueue only the lane frozen during preparation."""
        workload = self._runnable(workload_id)
        inputs = self._inputs(payload)
        model = self._prepared_model(workload, prepared)
        return self._queue_prepared(workload_id, workload, inputs, prepared, model)

    def _queue_prepared(
        self,
        workload_id: str,
        workload: Workload,
        inputs: list,
        prepared: PreparedDispatchLane,
        model: dict,
    ) -> asyncio.Future:
        executor = self._prepared_executor(workload_id, prepared)
        availability = availability_for_model(executor, model)
        if not availability.available:
            raise JobDispatchError(
                "prepared-lane-unavailable",
                f"{workload_id}: {prepared.device_id} is unavailable: {availability.reason}",
            )
        path = self._model_path(workload, model, prepared.model_reference)
        try:
            return self._scheduler.submit(
                prepared.scheduler_lane,
                workload_id,
                path,
                inputs,
                model_format=model["format"],
            )
        except QueueFullError as error:
            raise JobDispatchError("backend-queue-full", str(error)) from error
        except RuntimeError as error:
            raise JobDispatchError("scheduler-unavailable", str(error)) from error
        except KeyError as error:
            raise JobDispatchError(
                "prepared-lane-unavailable",
                f"{workload_id}: prepared scheduler lane {prepared.scheduler_lane} disappeared",
            ) from error

    def _prepared_model(self, workload: Workload, prepared: PreparedDispatchLane) -> dict:
        if not isinstance(prepared, PreparedDispatchLane):
            raise JobDispatchError("prepared-lane-invalid", "Prepared dispatch lane is invalid")
        for model in workload.models:
            if declared_artifact_reference(workload, model) == prepared.model_reference:
                return model
        raise JobDispatchError(
            "prepared-lane-invalid",
            f"{workload.id}: prepared model is not declared by this workload",
        )

    def _prepared_executor(self, workload_id: str, prepared: PreparedDispatchLane) -> Executor:
        if self._executor_for_device is not None:
            executor = self._executor_for_device(prepared.backend, prepared.device_id)
        else:
            executors = (
                dict(self._executor_view(workload_id))
                if self._executor_view is not None
                else _executor_snapshot(self._executors)
            )
            executor = executors.get(prepared.backend)
        declared_format = {"format": prepared.model_reference.format}
        if executor is None or not supports_model(executor, declared_format):
            raise JobDispatchError(
                "prepared-lane-unavailable",
                f"{workload_id}: prepared device {prepared.device_id} disappeared",
            )
        return executor

    def _runnable(self, workload_id: str) -> Workload:
        """The profile this job names, if it exists and can run inference."""
        workload = self._workloads.get(workload_id)
        if workload is None:
            raise JobDispatchError("workload-unknown", f"No such workload profile: {workload_id}")
        if not workload.models:
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
        if not isinstance(payload, dict):
            raise JobDispatchError("payload-invalid", "Job payload must be an object")
        try:
            loaded = referenced_inputs(
                payload,
                self._input_roots,
                max_tensors=self._max_input_tensors,
            )
        except TensorReferenceError as error:
            raise JobDispatchError(error.code, error.detail) from error
        if loaded is None:
            return self._validated(_inline_inputs(payload, self._max_input_tensors))
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

    def _model_path(
        self,
        workload: Workload,
        model: dict,
        expected_reference: ArtifactReference,
    ) -> str:
        """Resolve the model this manifest declares, and only that one."""
        workload_id = workload.id
        reference = declared_artifact_reference(workload, model)
        _refuse_unpinned(workload_id, reference)
        if reference != expected_reference:
            raise JobDispatchError(
                "prepared-lane-invalid",
                f"{workload_id}: prepared model reference changed before dispatch",
            )
        assert reference is not None
        try:
            resolution = self._artifacts.resolve(reference)
        except Exception as error:  # noqa: BLE001 - store failures are arbitrary
            raise JobDispatchError(
                "artifact-unavailable",
                f"{workload_id}: artifact store is unreadable: {type(error).__name__}",
            ) from error
        if not resolution.ready or resolution.path is None:
            raise JobDispatchError("artifact-unavailable", f"{workload_id}: {resolution.reason}")
        return str(resolution.path)


def _refuse_unpinned(workload_id: str, reference: ArtifactReference | None) -> None:
    """Refuse a model whose manifest does not say which files it means."""
    if reference is None:
        # The store's own digest proves the file has not changed since it was
        # installed; it cannot prove the publisher meant *this* file. Anything
        # installed under the same id and version would satisfy an unpinned
        # manifest, which is the one guarantee the digest exists to give.
        raise JobDispatchError(
            "model-unpinned",
            f"{workload_id}: the manifest declares a model without a sha256, so the "
            "runtime cannot tell which file the publisher meant",
        )
    unpinned = reference.unpinned_companions()
    if unpinned:
        # For ncnn and OpenVINO IR the weights live in the companion, so a
        # manifest that names only the primary file has vouched for the graph
        # and not for the numbers it runs.
        raise JobDispatchError(
            "companion-unpinned",
            f"{workload_id}: the manifest declares no digest for "
            f"{', '.join(unpinned)}, which is where this format keeps its weights",
        )


def _inline_inputs(payload: dict, max_input_tensors: int) -> list:
    """The tensors a payload carries inline, bounded by count."""
    inputs = payload.get("inputs")
    if not isinstance(inputs, list):
        raise JobDispatchError("payload-invalid", "Job payload must contain an inputs array")
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

    Nesting is walked recursively, so a deeply nested payload exhausts the
    interpreter stack.  That is still a malformed payload and is answered as
    one: letting the RecursionError escape reaches the blanket handler in
    ``job_lifecycle`` and tells the caller ``internal-error``, which it cannot
    tell apart from a broken daemon.
    """
    try:
        return _validate_input_tensor(value, budget)
    except RecursionError as error:
        raise JobDispatchError("payload-invalid", "Input tensors are nested too deeply") from error


def _validate_input_tensor(value: object, budget: _ElementBudget) -> object:
    if isinstance(value, bool):
        raise JobDispatchError("payload-invalid", "Input tensors must contain numbers")
    if isinstance(value, (int, float)):
        budget.spend()
        if isinstance(value, float) and not math.isfinite(value):
            raise JobDispatchError("payload-invalid", "Input tensors must contain finite numbers")
        return value
    if not isinstance(value, list):
        raise JobDispatchError(
            "payload-invalid",
            f"Input tensors must be numbers or nested arrays, not {type(value).__name__}",
        )
    encoded = [_validate_input_tensor(item, budget) for item in value]
    shapes = {_tensor_shape(item) for item in encoded}
    if len(shapes) > 1:
        raise JobDispatchError("payload-invalid", "Input tensors must be rectangular")
    return encoded


def _tensor_shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    return (len(value),) + (_tensor_shape(value[0]) if value else ())


def declared_artifact_reference(workload: Workload, model: dict) -> ArtifactReference | None:
    """The allowlisted artifact entry matching this workload's model.

    Returns ``None`` when the manifest declares no digest for it, which is the
    v1 shape.  Returning ``None`` rather than inventing a digest keeps the
    weaker guarantee visible at the call site instead of pretending to verify.
    """
    companions = _declared_companions(model)
    declared = workload.manifest.get("plugin", {}).get("artifacts") or ()
    for entry in declared:
        if (
            entry["id"] == model["id"]
            and entry["version"] == model["version"]
            and entry["format"] == model["format"]
        ):
            return ArtifactReference(
                entry["id"],
                entry["version"],
                entry["format"],
                entry["sha256"],
                companions,
                entry.get("sourceUri", ""),
                entry.get("licenseSpdx", ""),
            )
    digest = model.get("sha256")
    if digest:
        return ArtifactReference(model["id"], model["version"], model["format"], digest, companions)
    return None


def _declared_companions(model: dict) -> tuple[tuple[str, str], ...]:
    """The companion digests a manifest publishes, in a stable order."""
    companions = model.get("companions")
    if not isinstance(companions, dict):
        return ()
    return tuple(sorted((str(name), str(digest)) for name, digest in companions.items()))


def inference_result_payload(result: InferenceResult, *, max_elements: int | None = None) -> dict:
    """Map an executor outcome back into a plugin-facing job result.

    Executors return whatever their backend hands them — a numpy array from
    tflite, OpenVINO, and onnxruntime, an ncnn Mat from Vulkan — none of which
    survive JSON encoding.  Converting here rather than inside each executor
    keeps native tensors available to anything that chains stages internally,
    while guaranteeing that what crosses IPC is encodable.
    """
    if not isinstance(result, InferenceResult):
        raise JobDispatchError("executor-result-invalid", "Executor returned a non-result value")
    budget = _ElementBudget(max_elements)
    outputs = [encode_tensor(tensor, budget) for tensor in result.outputs]
    duration = float(result.duration_ms)
    if not math.isfinite(duration):
        raise JobDispatchError("executor-result-invalid", "Executor reported a non-finite duration")
    return {"outputs": outputs, "durationMs": round(duration, 3)}


class _ElementBudget:
    """Bound the total elements one payload may carry across all its tensors.

    Used for what a caller selected as input, where the bound is real, and
    optionally for a result, where ``None`` means unbounded.
    """

    def __init__(
        self,
        maximum: int | None,
        *,
        code: str = "executor-result-invalid",
        label: str = "Inference result",
    ) -> None:
        if maximum is not None and maximum < 1:
            raise ValueError("max_elements must be positive")
        self._remaining = maximum
        self._maximum = maximum
        self._code = code
        self._label = label

    def spend(self, count: int = 1) -> None:
        if self._remaining is None:
            return
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
    in the consumer.  A result nested deeper than the interpreter stack allows
    is refused the same way, rather than escaping as a RecursionError the
    caller reads as a broken daemon.
    """
    try:
        return _encode_tensor(value, budget or _ElementBudget(None))
    except RecursionError as error:
        raise JobDispatchError(
            "executor-result-invalid", "Inference result is nested too deeply"
        ) from error


def _encode_tensor(value: object, budget: _ElementBudget) -> object:
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
        return _encode_tensor(value.tolist(), budget)
    # ncnn Mat exposes numpy() rather than tolist().
    if hasattr(value, "numpy"):
        return _encode_tensor(value.numpy().tolist(), budget)
    if isinstance(value, (list, tuple)):
        return [_encode_tensor(item, budget) for item in value]
    raise JobDispatchError(
        "executor-result-invalid",
        f"Inference result contains an unencodable {type(value).__name__}",
    )
