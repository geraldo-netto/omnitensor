from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins.cancellation import (
    Cancellation,
    CancellationReason,
    JobCancellationRegistry,
)
from omnitensor.plugins.orchestration import (
    JobSubmission,
    OrchestrationError,
    build_plugin_runners,
    inference_stages,
    recover_interrupted_jobs,
    with_recovery,
)
from omnitensor.plugins.pipeline import PipelineStage, PreprocessedOutput
from omnitensor.plugins.protocol import PluginResultStatus
from omnitensor.plugins.runner import STAGE_ORDER
from omnitensor.registry import Workload


def workload(profile_id="visual-library", *, model=True, permissions=("read:visual-library",)):
    """A manifest in the shape the registry actually loads."""
    return Workload(
        profile_id,
        {
            "id": profile_id,
            "manifestVersion": 2,
            "requirements": {
                "runtimeApi": 1,
                "accelerator": "gpu",
                "acceleratorPreference": ["tpu", "npu", "gpu"],
                "minimumDevices": 1,
                "model": (
                    {
                        "id": f"{profile_id}-model",
                        "version": "1.0.0",
                        "format": "ncnn",
                    }
                    if model
                    else None
                ),
            },
            "defaults": {"enabled": True, "weight": 2},
            "plugin": {"permissions": list(permissions)},
        },
    )


class Resolution:
    def __init__(self, ready=True, path="/var/lib/omnitensor/model.ncnn", detail=""):
        self.ready = ready
        self.path = path
        self.detail = detail
        self.reference = None


class Dispatcher:
    def __init__(self, result=None, error=None):
        self.calls = []
        self._result = result if result is not None else {"outputs": [[1.0]], "durationMs": 2.0}
        self._error = error

    def dispatch(self, job_id, workload_id, payload):
        self.calls.append((job_id, workload_id, payload))
        future = asyncio.get_running_loop().create_future()
        if self._error is not None:
            future.set_exception(self._error)
        else:
            future.set_result(self._result)
        return future


def runners(tmp_path, workloads=None, *, dispatcher=None, resolve=None, **changes):
    registry = JobCancellationRegistry(tmp_path / "cancellations.json")
    options = {
        "dispatcher": dispatcher or Dispatcher(),
        "resolve_artifact": resolve or (lambda _artifact_id: Resolution()),
        "encode_result": lambda value: dict(value),
        "cancellations": registry,
        "is_paused": lambda: False,
        "is_enabled": lambda _profile: True,
    }
    options.update(changes)
    catalog = workloads if workloads is not None else {"visual-library": workload()}
    return build_plugin_runners(catalog, **options), registry


def run(built, job_id="job-1", payload=None, profile_id="visual-library"):
    return asyncio.run(
        built.submit(profile_id, job_id, payload if payload is not None else {"inputs": [[1, 2]]})
    )


def test_a_runner_is_built_for_each_executable_profile(tmp_path):
    catalog = {"visual-library": workload("visual-library"), "hardware": workload("hardware")}
    built, _registry = runners(tmp_path, catalog)

    assert sorted(built.runners) == ["hardware", "visual-library"]
    assert built.skipped == {}


def test_a_job_runs_every_stage_and_delivers(tmp_path):
    dispatcher = Dispatcher()
    built, _registry = runners(tmp_path, dispatcher=dispatcher)

    result = run(built)

    assert result.status is PluginResultStatus.SUCCEEDED
    assert dispatcher.calls == [("job-1", "visual-library", {"inputs": [[1, 2]]})]


def test_a_delivered_result_reaches_the_delivery_sink(tmp_path):
    delivered = []
    built, _registry = runners(tmp_path, deliver=lambda job_id, output: delivered.append(job_id))

    run(built)

    assert delivered == ["job-1"]


def test_a_profile_without_a_model_is_recorded_rather_than_given_a_runner(tmp_path):
    """A stage failure is not something a user can act on; this reason is."""
    catalog = {"desktop-context": workload("desktop-context", model=False)}
    built, _registry = runners(tmp_path, catalog)

    assert built.runners == {}
    assert "declares no model" in built.skipped["desktop-context"]
    assert built.get("desktop-context") is None


def test_stages_can_be_composed_directly_for_a_model_profile():
    stages = inference_stages(
        workload(),
        dispatcher=Dispatcher(),
        resolve_artifact=lambda _artifact_id: Resolution(),
        encode_result=dict,
    )
    assert tuple(stages) == STAGE_ORDER


def test_composing_stages_for_a_profile_without_a_model_is_refused():
    with pytest.raises(OrchestrationError, match="profile-has-no-model"):
        inference_stages(
            workload(model=False),
            dispatcher=Dispatcher(),
            resolve_artifact=lambda _artifact_id: Resolution(),
            encode_result=dict,
        )


def test_an_unready_artifact_stops_the_job_before_dispatch(tmp_path):
    dispatcher = Dispatcher()
    built, _registry = runners(
        tmp_path,
        dispatcher=dispatcher,
        resolve=lambda _artifact_id: Resolution(False, detail="digest mismatch"),
    )

    result = run(built)

    assert result.status is PluginResultStatus.FAILED
    assert "digest mismatch" in result.detail
    assert dispatcher.calls == []


def test_a_paused_runtime_stops_the_job_as_a_refusal_not_a_fault(tmp_path):
    built, _registry = runners(tmp_path, is_paused=lambda: True)

    result = run(built)

    assert result.status is PluginResultStatus.CANCELLED


def test_a_disabled_profile_stops_the_job(tmp_path):
    built, _registry = runners(tmp_path, is_enabled=lambda _profile: False)
    assert run(built).status is PluginResultStatus.CANCELLED


def test_a_missing_permission_stops_the_job(tmp_path):
    built, _registry = runners(tmp_path, allows_permission=lambda _profile, _permission: False)
    assert run(built).status is PluginResultStatus.CANCELLED


def test_a_dispatch_failure_ends_one_job_not_the_runner(tmp_path):
    built, _registry = runners(tmp_path, dispatcher=Dispatcher(error=RuntimeError("queue full")))

    failed = run(built, "job-1")
    assert failed.status is PluginResultStatus.FAILED

    healthy, _registry = runners(tmp_path)
    assert run(healthy, "job-2").status is PluginResultStatus.SUCCEEDED


def test_a_request_that_is_not_a_submission_is_refused(tmp_path):
    built, _registry = runners(tmp_path)
    runner = built.get("visual-library")

    result = asyncio.run(runner.run("job-1", {"inputs": []}))

    assert result.status is PluginResultStatus.FAILED
    assert "JobSubmission" in result.detail


def test_a_payload_that_is_not_an_object_is_refused(tmp_path):
    built, _registry = runners(tmp_path)
    result = asyncio.run(
        built.get("visual-library").run("job-1", JobSubmission("job-1", ["inputs"]))
    )
    assert result.status is PluginResultStatus.FAILED


def test_concurrent_jobs_of_one_profile_do_not_share_a_job_id(tmp_path):
    """A cell shared by every job would attribute one job's inputs to another."""
    dispatcher = Dispatcher()
    built, _registry = runners(tmp_path, dispatcher=dispatcher)

    async def both():
        return await asyncio.gather(
            built.submit("visual-library", "job-a", {"inputs": [[1]]}),
            built.submit("visual-library", "job-b", {"inputs": [[2]]}),
        )

    asyncio.run(both())

    assert sorted(dispatcher.calls) == [
        ("job-a", "visual-library", {"inputs": [[1]]}),
        ("job-b", "visual-library", {"inputs": [[2]]}),
    ]


def test_each_profile_gets_its_own_flow_controller(tmp_path):
    """One plugin's backlog must not consume every other plugin's admission."""
    catalog = {"a": workload("a"), "b": workload("b")}
    built, _registry = runners(tmp_path, catalog)

    first = built.get("a")
    second = built.get("b")

    assert first is not second
    assert first._flow is not second._flow


def test_a_job_can_be_cancelled_through_its_runner(tmp_path):
    built, registry = runners(tmp_path)
    runner = built.get("visual-library")

    async def scenario():
        task = asyncio.create_task(
            runner.run("job-1", JobSubmission("job-1", {"inputs": [[1]]}))
        )
        await asyncio.sleep(0)
        cancelled = runner.cancel("job-1", "user asked")
        return cancelled, await task

    cancelled, result = asyncio.run(scenario())

    assert cancelled is True
    assert result.status is PluginResultStatus.CANCELLED
    assert registry.recover() == ()


def test_interrupted_jobs_are_reconciled_at_startup(tmp_path):
    """Whatever owned their stage is gone, so they are not resumable."""
    journal = tmp_path / "cancellations.json"
    journal.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": {"job-1": "visual-library"},
            }
        )
    )
    registry = JobCancellationRegistry(journal)

    interrupted = recover_interrupted_jobs(registry)

    assert [item.job_id for item in interrupted] == ["job-1"]
    assert interrupted[0].reason is CancellationReason.INTERRUPTED


def test_reconciliation_runs_once_so_a_crash_does_not_haunt_every_start(tmp_path):
    journal = tmp_path / "cancellations.json"
    journal.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": {"job-1": "visual-library"},
            }
        )
    )
    registry = JobCancellationRegistry(journal)

    assert len(recover_interrupted_jobs(registry)) == 1
    assert recover_interrupted_jobs(registry) == ()


def test_a_clean_start_reconciles_nothing(tmp_path):
    built, registry = runners(tmp_path)
    assert with_recovery(built, registry).interrupted == ()


def test_recovery_is_attached_to_the_runner_set(tmp_path):
    journal = tmp_path / "cancellations.json"
    journal.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": {"job-9": "visual-library"},
            }
        )
    )
    registry = JobCancellationRegistry(journal)
    built = build_plugin_runners(
        {"visual-library": workload()},
        dispatcher=Dispatcher(),
        resolve_artifact=lambda _artifact_id: Resolution(),
        encode_result=dict,
        cancellations=registry,
        is_paused=lambda: False,
        is_enabled=lambda _profile: True,
    )

    recovered = with_recovery(built, registry)
    document = recovered.document()

    assert document["runners"] == ["visual-library"]
    assert document["interrupted"][0]["jobId"] == "job-9"
    assert document["skipped"] == {}


@pytest.mark.parametrize("registry", [object(), None])
def test_the_cancellation_registry_is_required(tmp_path, registry):
    with pytest.raises(OrchestrationError) as build_error:
        build_plugin_runners(
            {"visual-library": workload()},
            dispatcher=Dispatcher(),
            resolve_artifact=lambda _artifact_id: Resolution(),
            encode_result=dict,
            cancellations=registry,
            is_paused=lambda: False,
            is_enabled=lambda _profile: True,
        )
    with pytest.raises(OrchestrationError) as recovery_error:
        recover_interrupted_jobs(registry)
    for error in (build_error.value, recovery_error.value):
        assert (error.code, error.detail, str(error)) == (
            "cancellations-invalid",
            "cancellations must be a JobCancellationRegistry",
            "cancellations-invalid: cancellations must be a JobCancellationRegistry",
        )


def test_recovery_accepts_a_structural_cancellation_journal():
    interrupted = (
        Cancellation(
            "job-structural",
            CancellationReason.INTERRUPTED,
            "the previous service stopped",
        ),
    )

    class Journal:
        def recover(self):
            return interrupted

    assert recover_interrupted_jobs(Journal()) is interrupted


def test_the_number_of_wired_profiles_is_bounded(tmp_path):
    catalog = {f"profile-{index}": workload(f"profile-{index}") for index in range(300)}
    with pytest.raises(OrchestrationError, match="too-many-profiles"):
        runners(tmp_path, catalog)


def test_a_profile_declaring_no_permissions_still_runs(tmp_path):
    bare = workload("bare", permissions=())
    built, _registry = runners(tmp_path, {"bare": bare})
    assert run(built, profile_id="bare").status is PluginResultStatus.SUCCEEDED


def test_permissions_are_read_from_a_flat_manifest_too(tmp_path):
    flat = Workload(
        "flat",
        {
            "id": "flat",
            "manifestVersion": 2,
            "requirements": {"accelerator": "gpu", "model": {"id": "m", "version": "1.0.0"}},
            "defaults": {"enabled": True, "weight": 1},
            "permissions": ["read:flat"],
        },
    )
    built, _registry = runners(
        tmp_path, {"flat": flat}, allows_permission=lambda _profile, _p: False
    )
    assert run(built, profile_id="flat").status is PluginResultStatus.CANCELLED


def test_the_resolved_path_is_the_one_the_store_reported(tmp_path):
    seen = {}

    def resolve(artifact_id):
        seen["artifactId"] = artifact_id
        return Resolution(path="/var/lib/omnitensor/artifacts/model.ncnn")

    built, _registry = runners(tmp_path, resolve=resolve)
    run(built)

    assert seen["artifactId"] == "visual-library-model"


@given(st.integers(min_value=0, max_value=1))
def test_resolve_stage_carries_the_selected_model_lane(index):
    models = (
        {"id": "gpu-model", "version": "1.0.0", "format": "ncnn"},
        {"id": "npu-model", "version": "1.0.0", "format": "openvino"},
    )
    selected = models[index]
    subject = workload()
    requirements = subject.manifest["requirements"]
    requirements.pop("model")
    requirements["models"] = list(models)
    stages = inference_stages(
        subject,
        dispatcher=Dispatcher(),
        resolve_artifact=lambda artifact_id: Resolution(path=f"/models/{artifact_id}"),
        encode_result=dict,
        select_model=lambda _workload: selected,
    )

    resolved = asyncio.run(
        stages[PipelineStage.RESOLVE](PreprocessedOutput({}, {}))
    )

    assert resolved.artifact.id == selected["id"]
    assert resolved.path == Path(f"/models/{selected['id']}")
    assert resolved.model == selected


def test_preferred_lane_owns_resolution_and_labels_end_to_end(tmp_path):
    gpu_model = {
        "id": "gpu-model",
        "version": "1.0.0",
        "format": "ncnn",
        "outputContract": {
            "kind": "classification",
            "topK": 1,
            "labels": "labels.txt",
        },
    }
    npu_model = {
        **gpu_model,
        "id": "npu-model",
        "format": "openvino",
    }
    subject = workload()
    requirements = subject.manifest["requirements"]
    requirements.pop("model")
    requirements["models"] = [gpu_model, npu_model]
    npu_root = tmp_path / "npu"
    npu_root.mkdir()
    (npu_root / "labels.txt").write_text("npu-low\nnpu-high\n", encoding="utf-8")
    resolved = []

    def resolve(artifact_id):
        resolved.append(artifact_id)
        if artifact_id != "npu-model":
            raise AssertionError("the unselected GPU artifact must not be resolved")
        return Resolution(path=npu_root / "model.xml")

    built, _registry = runners(
        tmp_path,
        {subject.id: subject},
        dispatcher=Dispatcher({"outputs": [[0.1, 0.9]]}),
        resolve=resolve,
        select_model=lambda _workload: npu_model,
    )

    result = run(built)

    assert result.output["reading"]["top"] == [
        {"index": 1, "score": 0.9, "label": "npu-high"}
    ]
    assert resolved == ["npu-model", "npu-model"]


def test_a_stage_reached_without_a_bound_job_is_refused():
    stages = inference_stages(
        workload(),
        dispatcher=Dispatcher(),
        resolve_artifact=lambda _artifact_id: Resolution(),
        encode_result=dict,
    )

    async def skip_preprocess():
        from omnitensor.plugins.pipeline import ResolvedOutput

        return await stages[PipelineStage.INFER](
            ResolvedOutput(None, Path("/var/lib/omnitensor/model.ncnn"))
        )

    with pytest.raises(OrchestrationError, match="job-context-missing"):
        asyncio.run(skip_preprocess())


def test_submitting_to_a_profile_with_no_runner_reports_why(tmp_path):
    catalog = {"desktop-context": workload("desktop-context", model=False)}
    built, _registry = runners(tmp_path, catalog)

    with pytest.raises(OrchestrationError, match="profile-not-runnable"):
        asyncio.run(built.submit("desktop-context", "job-1", {"inputs": []}))
    with pytest.raises(OrchestrationError, match="no runner is wired"):
        asyncio.run(built.submit("absent", "job-1", {"inputs": []}))


def test_submitting_a_payload_that_is_not_an_object_is_refused(tmp_path):
    built, _registry = runners(tmp_path)
    with pytest.raises(OrchestrationError, match="request-invalid"):
        asyncio.run(built.submit("visual-library", "job-1", ["inputs"]))


def test_cancelling_through_the_runner_set_reaches_the_job(tmp_path):
    built, _registry = runners(tmp_path)

    async def scenario():
        task = asyncio.create_task(built.submit("visual-library", "job-1", {"inputs": [[1]]}))
        await asyncio.sleep(0)
        return built.cancel("visual-library", "job-1", "user asked"), await task

    cancelled, result = asyncio.run(scenario())

    assert cancelled is True
    assert result.status is PluginResultStatus.CANCELLED


def test_cancelling_an_unwired_profile_is_false(tmp_path):
    built, _registry = runners(tmp_path)
    assert built.cancel("absent", "job-1") is False


def test_a_resolution_that_names_its_reference_keeps_it(tmp_path):
    from omnitensor.plugins.artifacts import ArtifactReference

    declared = ArtifactReference("visual-library-model", "1.0.0", "ncnn", "a" * 64)

    def resolve(_artifact_id):
        resolution = Resolution()
        resolution.reference = declared
        return resolution

    built, _registry = runners(tmp_path, resolve=resolve)
    assert run(built).status is PluginResultStatus.SUCCEEDED


def test_a_manifest_the_registry_would_never_produce_is_skipped(tmp_path):
    built, _registry = runners(tmp_path, {"odd": Workload("odd", ["not", "a", "mapping"])})
    assert built.runners == {}
    assert "no readable requirements" in built.skipped["odd"]


def test_a_profile_declaring_permissions_nowhere_still_runs(tmp_path):
    """Every bundled manifest is this shape, so it is the common path."""
    quiet = Workload(
        "quiet",
        {
            "id": "quiet",
            "manifestVersion": 2,
            "requirements": {"accelerator": "gpu", "model": {"id": "m", "version": "1.0.0"}},
            "defaults": {"enabled": True, "weight": 1},
        },
    )
    built, _registry = runners(
        tmp_path, {"quiet": quiet}, allows_permission=lambda _profile, _p: False
    )
    assert run(built, profile_id="quiet").status is PluginResultStatus.SUCCEEDED


def test_a_profile_with_a_runner_is_dispatched_through_its_pipeline(tmp_path):
    """Runners nobody routes to are why the pipeline layer stayed unreachable."""
    from omnitensor.plugins.orchestration import RunnerBackedDispatcher

    inner = Dispatcher()
    built, _registry = runners(tmp_path, dispatcher=inner)

    class Fallback:
        def __init__(self):
            self.calls = []

        def dispatch(self, job_id, workload_id, payload):
            self.calls.append(job_id)
            future = asyncio.get_running_loop().create_future()
            future.set_result({})
            return future

    fallback = Fallback()
    routed = RunnerBackedDispatcher(built, fallback)

    output = asyncio.run(routed.dispatch("job-1", "visual-library", {"inputs": [[1]]}))

    assert fallback.calls == []
    assert inner.calls == [("job-1", "visual-library", {"inputs": [[1]]})]
    assert output["profileId"] == "visual-library"


def test_a_profile_without_a_runner_falls_back_to_direct_dispatch(tmp_path):
    from omnitensor.plugins.orchestration import RunnerBackedDispatcher

    catalog = {"desktop-context": workload("desktop-context", model=False)}
    built, _registry = runners(tmp_path, catalog)

    class Fallback:
        def __init__(self):
            self.calls = []

        def dispatch(self, job_id, workload_id, payload):
            self.calls.append(workload_id)
            return "dispatched-directly"

    fallback = Fallback()
    routed = RunnerBackedDispatcher(built, fallback)

    assert routed.dispatch("job-1", "desktop-context", {}) == "dispatched-directly"
    assert fallback.calls == ["desktop-context"]


def test_a_pipeline_that_does_not_succeed_raises_rather_than_claiming_success(tmp_path):
    from omnitensor.plugins.orchestration import PipelineFailedError, RunnerBackedDispatcher

    built, _registry = runners(tmp_path, is_paused=lambda: True)
    routed = RunnerBackedDispatcher(built, Dispatcher())

    with pytest.raises(PipelineFailedError) as failure:
        asyncio.run(routed.dispatch("job-1", "visual-library", {"inputs": [[1]]}))

    assert failure.value.status == "cancelled"


def test_the_routed_runner_set_must_be_a_runner_set():
    from omnitensor.plugins.orchestration import RunnerBackedDispatcher

    with pytest.raises(OrchestrationError, match="runners-invalid"):
        RunnerBackedDispatcher(object(), Dispatcher())


def test_the_router_reports_what_it_wraps(tmp_path):
    from omnitensor.plugins.orchestration import RunnerBackedDispatcher

    built, _registry = runners(tmp_path)
    fallback = Dispatcher()
    routed = RunnerBackedDispatcher(built, fallback)

    assert routed.fallback is fallback
    assert routed.runners is built


def test_each_stage_announces_itself_as_it_starts(tmp_path):
    """An update that lands only on completion says nothing about a stuck stage."""
    reported = []
    def note(job_id, stage, fraction, detail):
        reported.append((stage, fraction))

    built, _registry = runners(tmp_path, progress=note)

    run(built)

    assert [stage for stage, _fraction in reported] == [
        "collect",
        "preprocess",
        "resolve",
        "infer",
        "postprocess",
        "deliver",
    ]
    assert [fraction for _stage, fraction in reported] == [0.0] + [
        index / 6 for index in range(1, 6)
    ]


def test_a_job_that_stops_early_reports_only_the_stages_it_reached(tmp_path):
    reported = []
    built, _registry = runners(
        tmp_path,
        resolve=lambda _artifact_id: Resolution(False, detail="not ready"),
        progress=lambda job_id, stage, fraction, detail: reported.append(stage),
    )

    run(built)

    assert reported == ["collect", "preprocess", "resolve"]


def test_progress_reporting_is_optional(tmp_path):
    built, _registry = runners(tmp_path)
    assert run(built).status is PluginResultStatus.SUCCEEDED


def test_a_referenced_input_survives_the_pipeline(tmp_path):
    """Narrowing the payload to its inputs dropped inputRefs entirely, so a
    referenced input reached the executor as nothing at all."""
    dispatcher = Dispatcher()
    built, _registry = runners(tmp_path, dispatcher=dispatcher)

    reference = {"path": "/x/frame.f32", "shape": [1], "dtype": "float32", "sha256": "a" * 64}
    asyncio.run(built.submit("visual-library", "job-1", {"inputRefs": [reference]}))

    [(_job, _profile, payload)] = dispatcher.calls
    assert payload["inputRefs"] == [reference]


def test_an_inline_input_still_reaches_the_dispatcher_unchanged(tmp_path):
    dispatcher = Dispatcher()
    built, _registry = runners(tmp_path, dispatcher=dispatcher)

    asyncio.run(built.submit("visual-library", "job-1", {"inputs": [[1, 2]]}))

    [(_job, _profile, payload)] = dispatcher.calls
    assert payload["inputs"] == [[1, 2]]
