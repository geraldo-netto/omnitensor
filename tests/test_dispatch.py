from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from conftest import sample_manifest

from omnitensor.dispatch import (
    InferenceJobDispatcher,
    declared_artifact_reference,
    inference_result_payload,
)
from omnitensor.executors.base import Availability, InferenceResult
from omnitensor.jobs import JobDispatchError
from omnitensor.plugins.artifacts import ArtifactReference, ArtifactResolution
from omnitensor.registry import Workload
from omnitensor.scheduler import Scheduler
from omnitensor.tensorref import OptedInInputRoots

MODEL = {
    "id": "sample-model",
    "version": "1.2.3",
    "format": "ncnn",
    "fullyQuantized": True,
    "minimumCompilerVersion": "1.0.0",
    "minimumRuntimeVersion": "1.0.0",
    "sha256": "f" * 64,
}
UNPINNED = {name: value for name, value in MODEL.items() if name != "sha256"}


def workload(
    workload_id="sample-workload",
    *,
    model=None,
    accelerator="gpu",
    preference=("gpu",),
    artifacts=None,
):
    manifest = sample_manifest(
        workload_id, accelerator=accelerator, acceleratorPreference=list(preference)
    )
    manifest["requirements"]["model"] = dict(model) if model else None
    if artifacts is not None:
        manifest["plugin"] = {"artifacts": list(artifacts)}
    return Workload(id=workload_id, manifest=manifest)


class FakeExecutor:
    backend = "gpu"
    model_formats = frozenset({"ncnn"})

    def __init__(self, available=True, reason="", outputs=None):
        self._availability = Availability(available, reason)
        self._outputs = outputs if outputs is not None else [[1]]
        self.ran = []

    def availability(self):
        return self._availability

    def run(self, model_path, inputs):
        self.ran.append((model_path, inputs))
        return InferenceResult(outputs=self._outputs, duration_ms=1.5)


class FakeArtifacts:
    def __init__(self, resolution=None, error=None):
        self._resolution = resolution or ArtifactResolution(
            True, Path("/models/sample.ncnn.param"), "", 64
        )
        self._error = error
        self.resolved = []
        self.resolved_active = []

    def resolve(self, reference):
        self.resolved.append(reference)
        if self._error is not None:
            raise self._error
        return self._resolution

    def resolve_active(self, artifact_id):
        self.resolved_active.append(artifact_id)
        if self._error is not None:
            raise self._error
        return self._resolution


def dispatcher(workloads, executors=None, artifacts=None, scheduler=None, input_roots=None):
    executor = executors if executors is not None else {"gpu": FakeExecutor()}
    return InferenceJobDispatcher(
        {item.id: item for item in workloads},
        scheduler or Scheduler(executor, lambda _p: 1),
        executor,
        artifacts or FakeArtifacts(),
        input_roots=input_roots,
    )


def test_an_unknown_workload_is_refused():
    with pytest.raises(JobDispatchError) as excinfo:
        dispatcher([]).dispatch("job-1", "missing", {"inputs": []})
    assert excinfo.value.code == "workload-unknown"


def test_a_workload_without_a_model_cannot_run_inference():
    """The bundled v1 profiles declare model: null and must not silently route."""
    with pytest.raises(JobDispatchError) as excinfo:
        dispatcher([workload()]).dispatch("job-1", "sample-workload", {"inputs": []})
    assert excinfo.value.code == "workload-has-no-model"


@pytest.mark.parametrize(
    "payload", [None, [], {"tensor": 1}, {"inputs": "not-a-list"}, "text"]
)
def test_a_payload_without_an_inputs_array_is_refused(payload):
    with pytest.raises(JobDispatchError) as excinfo:
        dispatcher([workload(model=MODEL)]).dispatch("job-1", "sample-workload", payload)
    assert excinfo.value.code == "payload-invalid"


def test_too_many_input_tensors_are_refused():
    with pytest.raises(JobDispatchError) as excinfo:
        InferenceJobDispatcher(
            {"sample-workload": workload(model=MODEL)},
            Scheduler({"gpu": FakeExecutor()}, lambda _p: 1),
            {"gpu": FakeExecutor()},
            FakeArtifacts(),
            max_input_tensors=2,
        ).dispatch("job-1", "sample-workload", {"inputs": [1, 2, 3]})
    assert excinfo.value.code == "payload-invalid"


def test_the_input_tensor_bound_must_be_positive():
    with pytest.raises(ValueError, match="max_input_tensors"):
        InferenceJobDispatcher({}, None, {}, FakeArtifacts(), max_input_tensors=0)


def test_an_incompatible_format_leaves_no_backend_and_never_falls_back_to_cpu():
    """No CPU lane exists, so an unroutable job is refused with the reasons."""
    incompatible = {"gpu": FakeExecutor()}
    incompatible["gpu"].model_formats = frozenset({"onnx"})

    with pytest.raises(JobDispatchError) as excinfo:
        dispatcher(
            [workload(model=MODEL)], executors=incompatible
        ).dispatch("job-1", "sample-workload", {"inputs": []})

    assert excinfo.value.code == "no-backend-available"
    assert "model format not supported" in excinfo.value.message


def test_an_unavailable_backend_is_refused_with_its_reason():
    unavailable = {"gpu": FakeExecutor(available=False, reason="No Vulkan device")}
    with pytest.raises(JobDispatchError) as excinfo:
        dispatcher([workload(model=MODEL)], executors=unavailable).dispatch(
            "job-1", "sample-workload", {"inputs": []}
        )
    assert excinfo.value.code == "no-backend-available"
    assert "No Vulkan device" in excinfo.value.message


def test_an_unresolvable_artifact_is_refused_before_queueing():
    artifacts = FakeArtifacts(
        ArtifactResolution(False, None, "artifact has no active version", 0)
    )
    with pytest.raises(JobDispatchError) as excinfo:
        dispatcher([workload(model=MODEL)], artifacts=artifacts).dispatch(
            "job-1", "sample-workload", {"inputs": []}
        )
    assert excinfo.value.code == "artifact-unavailable"
    assert "no active version" in excinfo.value.message


def test_an_unreadable_artifact_store_is_a_stable_refusal():
    artifacts = FakeArtifacts(error=OSError("store is gone"))
    with pytest.raises(JobDispatchError) as excinfo:
        dispatcher([workload(model=MODEL)], artifacts=artifacts).dispatch(
            "job-1", "sample-workload", {"inputs": []}
        )
    assert excinfo.value.code == "artifact-unavailable"
    assert "OSError" in excinfo.value.message


def test_a_declared_digest_is_used_to_resolve_the_artifact():
    """A payload can never name a model; only the manifest's allowlist can."""
    declared = [{**{k: MODEL[k] for k in ("id", "version", "format")}, "sha256": "a" * 64}]
    artifacts = FakeArtifacts()

    async def scenario():
        target = workload(model=MODEL, artifacts=declared)
        dispatch = dispatcher([target], artifacts=artifacts)
        dispatch._scheduler.start()
        future = dispatch.dispatch("job-1", "sample-workload", {"inputs": [[1]]})
        await asyncio.wait_for(future, timeout=5)
        await dispatch._scheduler.stop()

    asyncio.run(scenario())

    assert artifacts.resolved == [
        ArtifactReference("sample-model", "1.2.3", "ncnn", "a" * 64)
    ]
    assert artifacts.resolved_active == []


def test_a_manifest_without_a_digest_does_not_run_at_all():
    """The store's digest proves the file has not changed since installation.

    It cannot prove the publisher meant this file: anything installed under the
    same id and version satisfies an unpinned manifest, which is the one thing
    the digest exists to establish.  So an unpinned model is refused rather than
    run on the weaker guarantee, and nothing is asked of the store.
    """
    artifacts = FakeArtifacts()

    with pytest.raises(JobDispatchError) as excinfo:
        dispatcher([workload(model=UNPINNED)], artifacts=artifacts).dispatch(
            "job-1", "sample-workload", {"inputs": [[1]]}
        )

    assert excinfo.value.code == "model-unpinned"
    assert "which file the publisher meant" in str(excinfo.value)
    assert artifacts.resolved == []
    assert artifacts.resolved_active == []


def test_the_declared_reference_matches_only_an_exact_entry():
    declared = [
        {"id": "sample-model", "version": "9.9.9", "format": "ncnn", "sha256": "b" * 64},
        {"id": "other-model", "version": "1.2.3", "format": "ncnn", "sha256": "c" * 64},
    ]
    # No allowlist entry matches, so the model's own digest is what remains.
    target = workload(model=MODEL, artifacts=declared)
    assert declared_artifact_reference(target, MODEL) == ArtifactReference(
        "sample-model", "1.2.3", "ncnn", "f" * 64
    )
    unpinned = workload(model=UNPINNED, artifacts=declared)
    assert declared_artifact_reference(unpinned, UNPINNED) is None

    exact = [{"id": "sample-model", "version": "1.2.3", "format": "ncnn", "sha256": "d" * 64}]
    assert declared_artifact_reference(
        workload(model=MODEL, artifacts=exact), MODEL
    ) == ArtifactReference("sample-model", "1.2.3", "ncnn", "d" * 64)


def test_a_dispatched_job_reaches_the_executor_and_returns_its_outcome():
    executor = FakeExecutor(outputs=[[7]])

    async def scenario():
        dispatch = dispatcher([workload(model=MODEL)], executors={"gpu": executor})
        dispatch._scheduler.start()
        result = await asyncio.wait_for(
            dispatch.dispatch("job-1", "sample-workload", {"inputs": [[3]]}), timeout=5
        )
        await dispatch._scheduler.stop()
        return result

    result = asyncio.run(scenario())

    assert executor.ran == [("/models/sample.ncnn.param", [[3]])]
    assert inference_result_payload(result) == {"outputs": [[7]], "durationMs": 1.5}


def test_a_full_backend_queue_is_a_stable_refusal():
    async def scenario():
        scheduler = Scheduler(
            {"gpu": FakeExecutor()}, lambda _p: 1, max_backend_queue_depth=1
        )
        dispatch = dispatcher([workload(model=MODEL)], scheduler=scheduler)
        dispatch.dispatch("job-1", "sample-workload", {"inputs": []})
        with pytest.raises(JobDispatchError) as excinfo:
            dispatch.dispatch("job-2", "sample-workload", {"inputs": []})
        return excinfo.value

    error = asyncio.run(scenario())
    assert error.code == "backend-queue-full"


def test_a_stopped_scheduler_is_a_stable_refusal():
    async def scenario():
        scheduler = Scheduler({"gpu": FakeExecutor()}, lambda _p: 1)
        scheduler.start()
        await scheduler.stop()
        dispatch = dispatcher([workload(model=MODEL)], scheduler=scheduler)
        with pytest.raises(JobDispatchError) as excinfo:
            dispatch.dispatch("job-1", "sample-workload", {"inputs": []})
        return excinfo.value

    error = asyncio.run(scenario())
    assert error.code == "scheduler-unavailable"


def test_a_non_result_from_an_executor_is_refused():
    with pytest.raises(JobDispatchError) as excinfo:
        inference_result_payload({"outputs": []})
    assert excinfo.value.code == "executor-result-invalid"


def result(outputs, duration_ms=1.0):
    return InferenceResult(outputs=outputs, duration_ms=duration_ms)


def test_a_numpy_output_is_encoded_as_json_values():
    """tflite, OpenVINO, and onnxruntime all hand back numpy arrays."""
    numpy = pytest.importorskip("numpy")
    payload = inference_result_payload(result([numpy.array([[1.5, 2.0], [3.0, 4.0]])]))
    assert payload["outputs"] == [[[1.5, 2.0], [3.0, 4.0]]]
    assert json.dumps(payload)


def test_an_ncnn_mat_is_encoded_through_its_numpy_view():
    class FakeMat:
        def numpy(self):
            import numpy

            return numpy.array([1.0, 2.0])

    pytest.importorskip("numpy")
    assert inference_result_payload(result([FakeMat()]))["outputs"] == [[1.0, 2.0]]


def test_nested_sequences_and_scalars_pass_through():
    payload = inference_result_payload(result([[1, 2], (3.5, True), 7]))
    assert payload["outputs"] == [[1, 2], [3.5, True], 7]


def test_a_non_finite_output_is_refused():
    with pytest.raises(JobDispatchError) as excinfo:
        inference_result_payload(result([[float("nan")]]))
    assert excinfo.value.code == "executor-result-invalid"


def test_a_non_finite_duration_is_refused():
    with pytest.raises(JobDispatchError) as excinfo:
        inference_result_payload(result([[1]], duration_ms=float("inf")))
    assert excinfo.value.code == "executor-result-invalid"


def test_an_unencodable_output_is_refused_rather_than_stringified():
    with pytest.raises(JobDispatchError) as excinfo:
        inference_result_payload(result([{"tensor": 1}]))
    assert excinfo.value.code == "executor-result-invalid"
    assert "unencodable dict" in excinfo.value.message


def test_the_total_element_count_is_bounded_across_all_tensors():
    with pytest.raises(JobDispatchError) as excinfo:
        inference_result_payload(result([[1, 2], [3, 4]]), max_elements=3)
    assert "exceeds 3 tensor elements" in excinfo.value.message


def test_a_result_at_the_element_bound_is_accepted():
    assert inference_result_payload(result([[1, 2]]), max_elements=2)["outputs"] == [[1, 2]]


def test_bytes_are_not_treated_as_a_tensor():
    with pytest.raises(JobDispatchError):
        inference_result_payload(result([b"raw"]))


def submit(payload):
    return dispatcher([workload(model=MODEL)]).dispatch("job-1", "sample-workload", payload)


def test_a_ragged_input_tensor_is_refused_before_it_reaches_a_backend():
    """Every backend expects a rectangular buffer; ragged input fails natively."""
    with pytest.raises(JobDispatchError) as excinfo:
        submit({"inputs": [[[1, 2], [3]]]})
    assert excinfo.value.code == "payload-invalid"
    assert "rectangular" in excinfo.value.message


@pytest.mark.parametrize(
    "tensor",
    [
        [float("nan")],
        [float("inf")],
        ["text"],
        [None],
        [{"value": 1}],
        [True],
    ],
)
def test_a_tensor_that_is_not_finite_numbers_is_refused(tensor):
    with pytest.raises(JobDispatchError) as excinfo:
        submit({"inputs": [tensor]})
    assert excinfo.value.code == "payload-invalid"


def test_the_total_input_element_count_is_bounded():
    dispatch = InferenceJobDispatcher(
        {"sample-workload": workload(model=MODEL)},
        Scheduler({"gpu": FakeExecutor()}, lambda _p: 1),
        {"gpu": FakeExecutor()},
        FakeArtifacts(),
        max_input_elements=3,
    )
    with pytest.raises(JobDispatchError) as excinfo:
        dispatch.dispatch("job-1", "sample-workload", {"inputs": [[1, 2], [3, 4]]})
    assert excinfo.value.code == "payload-invalid"
    assert "exceeds 3 tensor elements" in excinfo.value.message


def test_the_input_element_bound_must_be_positive():
    with pytest.raises(ValueError, match="max_input_elements"):
        InferenceJobDispatcher({}, None, {}, FakeArtifacts(), max_input_elements=0)


def test_a_well_formed_nested_tensor_is_accepted():
    executor = FakeExecutor()

    async def scenario():
        dispatch = dispatcher([workload(model=MODEL)], executors={"gpu": executor})
        dispatch._scheduler.start()
        await asyncio.wait_for(
            dispatch.dispatch(
                "job-1", "sample-workload", {"inputs": [[[1.0, 2.0], [3.0, 4.0]]]}
            ),
            timeout=5,
        )
        await dispatch._scheduler.stop()

    asyncio.run(scenario())
    assert executor.ran == [("/models/sample.ncnn.param", [[[1.0, 2.0], [3.0, 4.0]]])]


def test_an_empty_tensor_list_is_accepted():
    async def scenario():
        return submit({"inputs": []})

    assert asyncio.run(scenario()) is not None


def test_a_v1_manifest_may_declare_its_model_digest():
    """Without this a v1 profile could only trust the install-time digest."""
    digested = dict(MODEL, sha256="e" * 64)
    target = workload(model=digested)

    assert declared_artifact_reference(target, digested) == ArtifactReference(
        "sample-model", "1.2.3", "ncnn", "e" * 64
    )


def test_a_declared_artifact_entry_still_wins_over_the_model_digest():
    """The plugin allowlist is the stronger statement, so it is preferred."""
    digested = dict(MODEL, sha256="e" * 64)
    declared = [
        {"id": "sample-model", "version": "1.2.3", "format": "ncnn", "sha256": "f" * 64}
    ]
    target = workload(model=digested, artifacts=declared)

    assert declared_artifact_reference(target, digested).sha256 == "f" * 64


def test_a_digested_v1_model_is_resolved_by_reference_not_by_id():
    artifacts = FakeArtifacts()

    async def scenario():
        dispatch = dispatcher(
            [workload(model=dict(MODEL, sha256="e" * 64))], artifacts=artifacts
        )
        dispatch._scheduler.start()
        await asyncio.wait_for(
            dispatch.dispatch("job-1", "sample-workload", {"inputs": [[1]]}), timeout=5
        )
        await dispatch._scheduler.stop()

    asyncio.run(scenario())

    assert [item.sha256 for item in artifacts.resolved] == ["e" * 64]
    assert artifacts.resolved_active == []


def buffer_reference(tmp_path, values, shape):
    import hashlib
    import struct

    payload = struct.pack(f"<{len(values)}f", *values)
    path = tmp_path / "input.f32"
    path.write_bytes(payload)
    return {
        "path": str(path),
        "shape": list(shape),
        "dtype": "float32",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_a_referenced_input_is_dispatched_like_an_inline_one(tmp_path):
    """An input too large to inline reaches the executor by the same route."""
    from omnitensor.tensorref import OptedInInputRoots

    reference = buffer_reference(tmp_path, [1.0, 2.0, 3.0, 4.0], (2, 2))

    async def scenario():
        dispatch = dispatcher(
            [workload(model=MODEL)], input_roots=OptedInInputRoots([tmp_path])
        )
        dispatch._scheduler.start()
        future = dispatch.dispatch("job-1", "sample-workload", {"inputRefs": [reference]})
        await asyncio.wait_for(future, timeout=5)
        await dispatch._scheduler.stop()
        return dispatch

    dispatch = asyncio.run(scenario())
    [(_path, inputs)] = dispatch._executors["gpu"].ran
    # The executor received the reshaped buffer, not a path or a reference.
    assert inputs == [[[1.0, 2.0], [3.0, 4.0]]]


def test_referencing_a_file_is_denied_until_roots_are_configured(tmp_path):
    """The capability is off by default, so a default install cannot be used
    to read a caller-named file."""
    reference = buffer_reference(tmp_path, [1.0, 2.0], (2,))
    subject = dispatcher([workload(model=MODEL)])

    with pytest.raises(JobDispatchError) as failure:
        subject.dispatch("job-1", "sample-workload", {"inputRefs": [reference]})

    assert failure.value.code == "input-ref-denied"


def test_a_malformed_reference_is_one_error_vocabulary(tmp_path):
    """However the input arrived, the caller sees a dispatch failure."""
    from omnitensor.tensorref import OptedInInputRoots

    subject = dispatcher([workload(model=MODEL)], input_roots=OptedInInputRoots([tmp_path]))

    with pytest.raises(JobDispatchError) as failure:
        subject.dispatch("job-1", "sample-workload", {"inputRefs": [{"path": "/x"}]})

    assert failure.value.code == "input-ref-invalid"


def test_admission_refuses_a_reference_outside_the_granted_roots(tmp_path):
    """The refusal belongs in the submission reply, not in a stored result."""
    reference = buffer_reference(tmp_path, [1.0, 2.0], (2,))
    subject = dispatcher([workload(model=MODEL)])

    with pytest.raises(JobDispatchError) as failure:
        subject.admit("sample-workload", {"inputRefs": [reference]})

    assert failure.value.code == "input-ref-denied"


def test_admission_refuses_a_digest_that_does_not_match(tmp_path):
    from omnitensor.tensorref import OptedInInputRoots

    reference = buffer_reference(tmp_path, [1.0, 2.0], (2,))
    reference["sha256"] = "0" * 64
    subject = dispatcher([workload(model=MODEL)], input_roots=OptedInInputRoots([tmp_path]))

    with pytest.raises(JobDispatchError) as failure:
        subject.admit("sample-workload", {"inputRefs": [reference]})

    assert failure.value.code == "input-ref-mismatch"


def test_admission_accepts_a_reference_it_does_not_read(tmp_path):
    """Admission proves the file; the buffer is still read at dispatch."""
    from omnitensor.tensorref import OptedInInputRoots

    reference = buffer_reference(tmp_path, [1.0, 2.0, 3.0, 4.0], (2, 2))
    subject = dispatcher([workload(model=MODEL)], input_roots=OptedInInputRoots([tmp_path]))

    assert subject.admit("sample-workload", {"inputRefs": [reference]}) is None


def test_admission_refuses_a_referenced_tensor_over_the_element_budget(tmp_path):
    """Bounded from the declared shape, before the file is opened."""
    from omnitensor.tensorref import OptedInInputRoots

    reference = buffer_reference(tmp_path, [1.0, 2.0, 3.0, 4.0], (2, 2))
    subject = InferenceJobDispatcher(
        {"sample-workload": workload(model=MODEL)},
        Scheduler({"gpu": FakeExecutor()}, lambda _p: 1),
        {"gpu": FakeExecutor()},
        FakeArtifacts(),
        max_input_elements=3,
        input_roots=OptedInInputRoots([tmp_path]),
    )

    with pytest.raises(JobDispatchError) as failure:
        subject.admit("sample-workload", {"inputRefs": [reference]})

    assert failure.value.code == "payload-invalid"


@pytest.mark.parametrize(
    ("workload_id", "payload", "expected"),
    [
        ("missing", {"inputs": []}, "workload-unknown"),
        ("no-model", {"inputs": []}, "workload-has-no-model"),
        ("sample-workload", "text", "payload-invalid"),
        ("sample-workload", {"tensor": 1}, "payload-invalid"),
        ("sample-workload", {"inputs": [[1.0, float("inf")]]}, "payload-invalid"),
        ("sample-workload", {"inputRefs": [{"path": "/x"}]}, "input-ref-invalid"),
    ],
)
def test_admission_answers_with_the_same_codes_dispatch_would(workload_id, payload, expected):
    subject = dispatcher([workload(model=MODEL), workload("no-model")])

    with pytest.raises(JobDispatchError) as admission:
        subject.admit(workload_id, payload)
    with pytest.raises(JobDispatchError) as dispatched:
        subject.dispatch("job-1", workload_id, payload)

    assert admission.value.code == expected
    assert dispatched.value.code == expected




def test_admission_leaves_routing_and_resolution_to_dispatch():
    """A backend that is busy or an artifact that is briefly unresolvable is a
    condition of the runtime, not a reason to refuse the caller's submission."""
    unavailable = {"gpu": FakeExecutor(available=False, reason="device is warming up")}
    subject = dispatcher(
        [workload(model=MODEL)],
        executors=unavailable,
        artifacts=FakeArtifacts(ArtifactResolution(False, None, "not installed", 0)),
    )

    assert subject.admit("sample-workload", {"inputs": [[1.0]]}) is None
    with pytest.raises(JobDispatchError) as failure:
        subject.dispatch("job-1", "sample-workload", {"inputs": [[1.0]]})
    assert failure.value.code == "no-backend-available"


CONTRACT = {
    **MODEL,
    "tensorContract": {
        "inputs": [{"shape": [1, 2, 2], "dtype": "float32", "layout": "NCHW"}],
    },
}


def test_a_shape_the_model_cannot_accept_is_refused_in_the_submitting_call():
    """It used to surface as `ncnn extraction failed for output prob`, raised
    inside the executor after the job was admitted, queued, and dispatched."""
    dispatch = dispatcher([workload(model=CONTRACT)])

    with pytest.raises(JobDispatchError) as excinfo:
        dispatch.admit("sample-workload", {"inputs": [[[1.0, 2.0, 3.0]]]})

    assert excinfo.value.code == "input-contract-mismatch"
    assert "[1, 2, 2]" in str(excinfo.value)


def test_an_input_the_model_declares_is_admitted():
    dispatch = dispatcher([workload(model=CONTRACT)])

    # One tensor of shape (1, 2, 2): the batch dimension is part of the
    # declared contract, and the reference path that works today carries it.
    admitted = dispatch.admit("sample-workload", {"inputs": [[[[1.0, 2.0], [3.0, 4.0]]]]})

    assert admitted is None


def test_a_referenced_input_is_checked_against_the_contract_before_it_is_read(tmp_path):
    """The reference declares its own shape and dtype, so the disagreement is
    found without opening the file at all."""
    dispatch = dispatcher([workload(model=CONTRACT)], input_roots=OptedInInputRoots([tmp_path]))
    absent = tmp_path / "never-opened.f32"

    with pytest.raises(JobDispatchError) as excinfo:
        dispatch.admit("sample-workload", {"inputRefs": [{
            "path": str(absent), "shape": [1, 3, 8, 8],
            "dtype": "float32", "sha256": "0" * 64,
        }]})

    assert excinfo.value.code == "input-contract-mismatch"
    assert absent.exists() is False


def test_a_referenced_dtype_the_model_does_not_take_is_refused(tmp_path):
    dispatch = dispatcher([workload(model=CONTRACT)], input_roots=OptedInInputRoots([tmp_path]))

    with pytest.raises(JobDispatchError) as excinfo:
        dispatch.admit("sample-workload", {"inputRefs": [{
            "path": str(tmp_path / "x.bin"), "shape": [1, 2, 2],
            "dtype": "int32", "sha256": "0" * 64,
        }]})

    assert excinfo.value.code == "input-contract-mismatch"
    assert "int32" in str(excinfo.value)


def test_a_model_declaring_no_contract_admits_what_it_always_did():
    """Absent means no check: every profile written before the field behaves
    exactly as it did."""
    dispatch = dispatcher([workload(model=MODEL)])

    assert dispatch.admit("sample-workload", {"inputs": [[[1.0, 2.0, 3.0]]]}) is None


def test_an_unpinned_model_is_refused_before_the_store_is_touched():
    """The refusal is a routing decision, not a resolution failure.

    Reaching the store first would report "artifact-unavailable" for a manifest
    that is simply missing a field, sending somebody to look for a file that is
    installed and fine.
    """
    class ExplodingArtifacts:
        def resolve(self, reference):  # pragma: no cover - never reached
            raise AssertionError("the store is not consulted for an unpinned model")

        def resolve_active(self, artifact_id):  # pragma: no cover - never reached
            raise AssertionError("the store is not consulted for an unpinned model")

    with pytest.raises(JobDispatchError) as excinfo:
        dispatcher([workload(model=UNPINNED)], artifacts=ExplodingArtifacts()).dispatch(
            "job-1", "sample-workload", {"inputs": [[1]]}
        )

    assert excinfo.value.code == "model-unpinned"
