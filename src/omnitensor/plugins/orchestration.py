"""Build one runner per active profile, and reconcile what a crash left behind.

:class:`~omnitensor.plugins.runner.PipelineRunner` holds policy, flow, and
cancellation around a job, but a runner nobody constructs runs nothing.  This
is the wiring: for each profile that can actually execute, it composes the six
stages out of collaborators the service already owns, and it gives every
profile its own flow controller so one plugin's backlog cannot occupy the
admission budget of another.

Startup reconciliation lives here too.  A journal entry that survived a restart
describes a job whose executor is gone, so it is not resumable — reporting it
as interrupted and clearing the journal is the only honest outcome, and doing
it once at startup keeps one crash from haunting every later start.

A profile with no model gets no runner and a recorded reason rather than a
runner that fails at its first stage: "this profile does not run inference" is
something a user can act on, and a stage failure is not.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..outputcontract import declared_output, parse_labels, reduce_output
from ..registry import Workload
from .artifacts import ArtifactReference
from .cancellation import Cancellation, JobCancellationRegistry
from .flow import PluginFlowController
from .pipeline import (
    CollectedOutput,
    DeliveredOutput,
    InferenceOutput,
    PipelineStage,
    PostprocessedOutput,
    PreprocessedOutput,
    ResolvedOutput,
)
from .policy import PipelinePolicyGate
from .runner import STAGE_ORDER, PipelineRunner

MAX_RUNNERS = 256

# A stage sees only the previous stage's output, and those outputs are a closed
# set the state machine validates — so the job's identity travels beside the
# chain rather than through it.  It is set once at submission, in the caller's
# own context, because each stage runs in a task of its own and a value set
# inside one of those tasks would not be visible to the next.  Per-task
# contexts are also what keeps two concurrent jobs of one profile apart.
_RUNNING_JOB: ContextVar[tuple[str, object]] = ContextVar("omnitensor_running_job")


def _current_job() -> tuple[str, object]:
    try:
        return _RUNNING_JOB.get()
    except LookupError as error:
        raise OrchestrationError(
            "job-context-missing", "submit through the runner set so the job identity is bound"
        ) from error


class OrchestrationError(ValueError):
    """Stable wiring failure, raised before any job is accepted."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class JobSubmission:
    """What a runner is handed: the job's identity and its payload.

    The stages are built once per profile and reused for every job, so the job
    id has to travel with the request rather than being captured in a closure.
    """

    job_id: str
    payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class RunnerSet:
    """The runners that were built, why the rest were not, and what a crash left."""

    runners: Mapping[str, PipelineRunner]
    skipped: Mapping[str, str]
    interrupted: tuple[Cancellation, ...] = ()

    def get(self, profile_id: str) -> PipelineRunner | None:
        return self.runners.get(profile_id)

    async def submit(
        self, profile_id: str, job_id: str, payload: Mapping[str, object]
    ) -> object:
        """Run one job on its profile's runner, binding the job's identity.

        This is the entry point rather than the runner itself, because binding
        has to happen in the caller's context: every stage runs in a task of
        its own, so a value set inside one stage is invisible to the next.
        """
        runner = self.runners.get(profile_id)
        if runner is None:
            raise OrchestrationError(
                "profile-not-runnable",
                self.skipped.get(profile_id, f"no runner is wired for {profile_id}"),
            )
        if not isinstance(payload, Mapping):
            raise OrchestrationError("request-invalid", "job payload must be an object")
        # The whole payload, not just its inputs: how a caller supplies a
        # tensor is the dispatcher's business, and narrowing it here silently
        # dropped inputRefs so a referenced input reached the executor as
        # nothing at all.
        _RUNNING_JOB.set((job_id, dict(payload)))
        return await runner.run(job_id, JobSubmission(job_id, dict(payload)))

    def cancel(self, profile_id: str, job_id: str, detail: str = "") -> bool:
        runner = self.runners.get(profile_id)
        return False if runner is None else runner.cancel(job_id, detail)

    def document(self) -> dict:
        return {
            "runners": sorted(self.runners),
            "skipped": dict(sorted(self.skipped.items())),
            "interrupted": [
                {"jobId": item.job_id, "reason": str(item.reason), "detail": item.detail}
                for item in self.interrupted
            ],
        }


class ArtifactResolver(Protocol):
    """Resolve a profile's declared model to a verified path."""

    def __call__(self, artifact_id: str): ...


class ProgressSink(Protocol):
    """Record where a running job has got to, against its owner."""

    def __call__(self, job_id: str, stage: str, fraction: float, detail: str) -> None: ...


class Dispatcher(Protocol):
    """Queue one job onto an accelerator and return the pending future."""

    def dispatch(self, job_id: str, workload_id: str, payload: dict): ...


def inference_stages(
    workload: Workload,
    *,
    dispatcher: Dispatcher,
    resolve_artifact: ArtifactResolver,
    encode_result: Callable[[object], dict],
    deliver: Callable[[str, dict], None] | None = None,
    progress: ProgressSink | None = None,
) -> dict[PipelineStage, Callable]:
    """Compose the six stages of an inference profile from existing parts."""
    try:
        model = workload.model
    except (TypeError, KeyError, AttributeError) as error:
        # The registry validates manifests, so this is a profile assembled by
        # something else; skipping it beats a runner that cannot describe why.
        raise OrchestrationError(
            "profile-manifest-invalid", f"{workload.id} has no readable requirements: {error}"
        ) from error
    if model is None:
        raise OrchestrationError(
            "profile-has-no-model", f"{workload.id} declares no model and cannot run inference"
        )
    stages = {
        PipelineStage.COLLECT: _collect_stage(),
        PipelineStage.PREPROCESS: _preprocess_stage(workload),
        PipelineStage.RESOLVE: _resolve_stage(model, resolve_artifact),
        PipelineStage.INFER: _infer_stage(workload, dispatcher, encode_result),
        PipelineStage.POSTPROCESS: _postprocess_stage(
            workload, _label_reader(model, resolve_artifact)
        ),
        PipelineStage.DELIVER: _deliver_stage(deliver),
    }
    if progress is None:
        return stages
    return {
        stage: _reporting(stage, index, call, progress)
        for index, (stage, call) in enumerate(stages.items())
    }


def _reporting(stage: PipelineStage, index: int, call: Callable, progress: ProgressSink):
    """Announce a stage as it starts, so a slow stage is not silence.

    Reported before the stage runs rather than after: the interesting case is
    the stage that has not finished, and an update that only lands on
    completion says nothing about the one a job is stuck in.
    """
    fraction = index / len(STAGE_ORDER)

    async def reported(carried: object):
        with contextlib.suppress(OrchestrationError):
            progress(_current_job()[0], str(stage), fraction, f"Running {stage}")
        return await call(carried)

    return reported


def _collect_stage() -> Callable:
    async def collect(request: object) -> CollectedOutput:
        if not isinstance(request, JobSubmission):
            raise OrchestrationError("request-invalid", "request must be a JobSubmission")
        if not isinstance(request.payload, Mapping):
            raise OrchestrationError("request-invalid", "job payload must be an object")
        return CollectedOutput({"jobId": request.job_id, "payload": dict(request.payload)})

    return collect


def _preprocess_stage(workload: Workload) -> Callable:
    async def preprocess(collected: CollectedOutput) -> PreprocessedOutput:
        payload = collected.payload["payload"]
        return PreprocessedOutput(
            # Carried whole for the same reason: preprocess must not decide
            # which ways of supplying an input are legitimate.
            dict(payload),
            {"jobId": collected.payload["jobId"], "profileId": workload.id},
        )

    return preprocess


def _resolve_stage(model: Mapping[str, object], resolve_artifact: ArtifactResolver) -> Callable:
    async def resolve(preprocessed: PreprocessedOutput) -> ResolvedOutput:
        resolution = resolve_artifact(model["id"])
        if not getattr(resolution, "ready", False):
            raise OrchestrationError(
                "artifact-unavailable", getattr(resolution, "detail", "artifact is not ready")
            )
        return ResolvedOutput(_reference_of(resolution, model), Path(resolution.path))

    return resolve


def _infer_stage(
    workload: Workload, dispatcher: Dispatcher, encode_result: Callable[[object], dict]
) -> Callable:
    async def infer(resolved: ResolvedOutput) -> InferenceOutput:
        job_id, job_payload = _current_job()
        # The dispatcher owns admission, routing, and queueing; re-deciding any
        # of that here would let two code paths disagree about the same job.
        future = dispatcher.dispatch(job_id, workload.id, job_payload)
        return InferenceOutput(encode_result(await future))

    return infer


def _postprocess_stage(workload: Workload, read_labels: Callable[[], tuple]) -> Callable:
    spec = declared_output(workload.model)

    async def postprocess(inferred: InferenceOutput) -> PostprocessedOutput:
        output = {"profileId": workload.id, **dict(inferred.tensors)}
        reading = reduce_output(spec, output.get("outputs") or (), read_labels())
        if reading is not None:
            # Added beside the tensors, never in place of them: a consumer that
            # wants the raw scores must not lose them to a reduction, and a
            # reduction that replaced them would be unrecoverable.
            output["reading"] = reading
        return PostprocessedOutput(output)

    return postprocess


def _label_reader(model: Mapping[str, object], resolve_artifact: ArtifactResolver) -> Callable:
    """Read the labels companion once, from the directory the resolver verified.

    Read lazily rather than when the runner is built: an artifact installed
    after startup would otherwise never be seen, and reading it at build time
    would make a profile with no artifact fail to get a runner at all.
    """
    spec = declared_output(model)
    cached: list[tuple] = []

    def read() -> tuple:
        if cached:
            return cached[0]
        if spec is None or not spec.labels:
            cached.append(())
            return cached[0]
        resolution = resolve_artifact(model["id"])
        path = getattr(resolution, "path", None)
        if not getattr(resolution, "ready", False) or path is None:
            # Not cached: the artifact may be installed a moment from now, and
            # remembering an empty list would report indices forever.
            return ()
        try:
            text = (Path(path).parent / spec.labels).read_text(encoding="utf-8")
        except (OSError, ValueError):
            # A labels file that cannot be read is a result reported by index,
            # which is exactly what a profile declaring none reports.
            return ()
        cached.append(parse_labels(text))
        return cached[0]

    return read


def _deliver_stage(deliver: Callable[[str, dict], None] | None) -> Callable:
    async def deliver_stage(postprocessed: PostprocessedOutput) -> DeliveredOutput:
        if deliver is not None:
            deliver(_current_job()[0], dict(postprocessed.output))
        return DeliveredOutput(dict(postprocessed.output))

    return deliver_stage


def _reference_of(resolution: object, model: Mapping[str, object]):
    reference = getattr(resolution, "reference", None)
    if reference is not None:
        return reference
    return ArtifactReference(
        model["id"], model.get("version", "0.0.0"), model.get("format", "unknown"), ""
    )


def build_plugin_runners(
    workloads: Mapping[str, Workload],
    *,
    dispatcher: Dispatcher,
    resolve_artifact: ArtifactResolver,
    encode_result: Callable[[object], dict],
    cancellations: JobCancellationRegistry,
    is_paused: Callable[[], bool],
    is_enabled: Callable[[str], bool],
    allows_permission: Callable[[str, str], bool] = lambda _profile, _permission: True,
    deliver: Callable[[str, dict], None] | None = None,
    progress: ProgressSink | None = None,
    flow_options: Mapping[str, object] | None = None,
    clock_ms: Callable[[], int] = lambda: int(time.time() * 1000),
) -> RunnerSet:
    """One runner per profile that can execute; a reason for each that cannot."""
    if not isinstance(cancellations, JobCancellationRegistry):
        raise OrchestrationError(
            "cancellations-invalid", "cancellations must be a JobCancellationRegistry"
        )
    if len(workloads) > MAX_RUNNERS:
        raise OrchestrationError("too-many-profiles", f"at most {MAX_RUNNERS} profiles are wired")
    runners: dict[str, PipelineRunner] = {}
    skipped: dict[str, str] = {}
    for profile_id, workload in sorted(workloads.items()):
        try:
            stages = inference_stages(
                workload,
                dispatcher=dispatcher,
                resolve_artifact=resolve_artifact,
                encode_result=encode_result,
                deliver=deliver,
                progress=progress,
            )
        except OrchestrationError as error:
            # A profile that cannot run is recorded, not given a runner that
            # would fail at its first stage with a less useful message.
            skipped[profile_id] = error.detail
            continue
        runners[profile_id] = PipelineRunner(
            profile_id,
            stages,
            policy=PipelinePolicyGate(
                profile_id,
                is_paused=is_paused,
                is_enabled=is_enabled,
                required_permissions=required_permissions(workload),
                allows_permission=allows_permission,
                artifact_ready=lambda _profile: (True, ""),
            ),
            # Each profile gets its own controller so one plugin's backlog
            # cannot consume the admission budget of every other plugin.
            flow=PluginFlowController(profile_id, **dict(flow_options or {})),
            cancellations=cancellations,
            clock_ms=clock_ms,
        )
    return RunnerSet(runners, skipped)


def required_permissions(workload: Workload) -> tuple[str, ...]:
    """Permissions the profile declares, from either manifest shape.

    A workload manifest carries none; a plugin manifest nests them under
    ``plugin``.  Reading both means a profile gains its gate the moment it
    declares one, without a second wiring change.
    """
    # Reached only for a manifest whose requirements already parsed, so it
    # supports lookup; the unreadable ones were skipped before this.
    manifest = workload.manifest
    nested = manifest.get("plugin")
    declared = manifest.get("permissions")
    if isinstance(nested, Mapping) and isinstance(nested.get("permissions"), list):
        declared = nested["permissions"]
    if not isinstance(declared, list):
        return ()
    return tuple(sorted(item for item in declared if isinstance(item, str)))


def recover_interrupted_jobs(cancellations: JobCancellationRegistry) -> tuple[Cancellation, ...]:
    """Reconcile jobs the previous process left claiming to be in flight."""
    if not isinstance(cancellations, JobCancellationRegistry):
        raise OrchestrationError(
            "cancellations-invalid", "cancellations must be a JobCancellationRegistry"
        )
    return cancellations.recover()


def with_recovery(runner_set: RunnerSet, cancellations: JobCancellationRegistry) -> RunnerSet:
    """Attach the startup reconciliation to a freshly built runner set."""
    return RunnerSet(
        runner_set.runners, runner_set.skipped, recover_interrupted_jobs(cancellations)
    )


__all__ = [
    "JobSubmission",
    "PipelineFailedError",
    "RunnerBackedDispatcher",
    "OrchestrationError",
    "RunnerSet",
    "STAGE_ORDER",
    "build_plugin_runners",
    "inference_stages",
    "recover_interrupted_jobs",
    "with_recovery",
]


class RunnerBackedDispatcher:
    """Send a job through its profile's pipeline, or straight to the accelerator.

    Building runners nobody routes to is how the pipeline layer stayed
    unreachable from the bus: submission dispatched directly and the policy,
    flow, and cancellation controls the runner holds never applied to a real
    job.  This sits in front of the raw dispatcher and prefers the runner
    whenever one exists for the profile.

    The runner's own infer stage calls the *raw* dispatcher, not this one, so
    routing here cannot loop back into itself.
    """

    def __init__(self, runners: RunnerSet, fallback: Dispatcher) -> None:
        if not isinstance(runners, RunnerSet):
            raise OrchestrationError("runners-invalid", "runners must be a RunnerSet")
        self._runners = runners
        self._fallback = fallback

    @property
    def fallback(self) -> Dispatcher:
        """Where a profile without a runner is dispatched."""
        return self._fallback

    @property
    def runners(self) -> RunnerSet:
        return self._runners

    def dispatch(self, job_id: str, workload_id: str, payload: dict):
        if self._runners.get(workload_id) is None:
            # A profile with no runner is not an error: it has no model and is
            # dispatched exactly as it was before runners existed.
            return self._fallback.dispatch(job_id, workload_id, payload)
        return self._run(job_id, workload_id, payload)

    async def _run(self, job_id: str, workload_id: str, payload: dict) -> dict:
        result = await self._runners.submit(workload_id, job_id, payload)
        status = str(getattr(result, "status", ""))
        if status != "succeeded":
            # Surfaced as a failure so the job service records why, rather than
            # storing a terminal result that claims success with no output.
            raise PipelineFailedError(status or "unknown", getattr(result, "detail", ""))
        output = getattr(result, "output", {})
        return dict(output) if isinstance(output, Mapping) else {"value": output}


class PipelineFailedError(RuntimeError):
    """A job that ran through a pipeline and did not succeed."""

    def __init__(self, status: str, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(f"{status}: {detail}" if detail else status)
