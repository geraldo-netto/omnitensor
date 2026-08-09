from __future__ import annotations

import asyncio
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

MODEL = {
    "id": "sample-model",
    "version": "1.2.3",
    "format": "ncnn",
    "fullyQuantized": True,
    "minimumCompilerVersion": "1.0.0",
    "minimumRuntimeVersion": "1.0.0",
}


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


def dispatcher(workloads, executors=None, artifacts=None, scheduler=None):
    executor = executors if executors is not None else {"gpu": FakeExecutor()}
    return InferenceJobDispatcher(
        {item.id: item for item in workloads},
        scheduler or Scheduler(executor, lambda _p: 1),
        executor,
        artifacts or FakeArtifacts(),
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


def test_a_manifest_without_a_digest_falls_back_to_the_active_version():
    artifacts = FakeArtifacts()

    async def scenario():
        dispatch = dispatcher([workload(model=MODEL)], artifacts=artifacts)
        dispatch._scheduler.start()
        future = dispatch.dispatch("job-1", "sample-workload", {"inputs": [[1]]})
        await asyncio.wait_for(future, timeout=5)
        await dispatch._scheduler.stop()

    asyncio.run(scenario())

    assert artifacts.resolved == []
    assert artifacts.resolved_active == ["sample-model"]


def test_the_declared_reference_matches_only_an_exact_entry():
    declared = [
        {"id": "sample-model", "version": "9.9.9", "format": "ncnn", "sha256": "b" * 64},
        {"id": "other-model", "version": "1.2.3", "format": "ncnn", "sha256": "c" * 64},
    ]
    target = workload(model=MODEL, artifacts=declared)
    assert declared_artifact_reference(target, MODEL) is None

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
