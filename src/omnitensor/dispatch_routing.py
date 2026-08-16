"""Dispatch routing and policy/permission gates for the service."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path

from .dispatch import InferenceJobDispatcher, inference_result_payload
from .job_ports import JobAdmission, JobDispatcher, JobDispatchError
from .jobs import UnavailableJobDispatcher
from .plugin_admission import PluginAdmissionQueue
from .plugins.orchestration import RunnerSet, build_plugin_runners, required_permissions
from .ports import PluginIdentitySource, PluginRuntime
from .registry import Workload
from .scheduler import runnable_model, select_backend
from .tensorref import OptedInInputRoots


class CachedArtifactSource:
    """Adapt the dispatch artifact port to the service's stamp cache."""

    def __init__(self, resolve: Callable) -> None:
        self._resolve = resolve

    def resolve(self, reference):
        return self._resolve(reference.id, reference)


def admit_plugin_job(
    inference: JobDispatcher,
    plugins: PluginRuntime,
    plugin_ids_provider: Callable[[], frozenset[str]],
    queue: PluginAdmissionQueue | None,
    workload_id: str,
    payload: dict,
) -> None:
    if workload_id in plugin_ids_provider():
        if not isinstance(plugins, JobAdmission):
            raise RuntimeError("plugin runtime does not implement admission")
        if queue is not None and not queue.has_room():
            # Refused here rather than queued: admission runs before an id is
            # minted, so a caller learns now instead of polling a job that will
            # never be served.
            raise JobDispatchError(
                "plugin-queue-full", "Too many plugin jobs are waiting for a worker slot"
            )
        plugins.admit(workload_id, payload)
        return
    if isinstance(inference, JobAdmission):
        inference.admit(workload_id, payload)


class PluginAwareDispatcher:
    """Route installed plugin IDs to workers and model profiles to inference."""

    def __init__(
        self,
        inference: JobDispatcher,
        plugins: PluginRuntime,
        queue: PluginAdmissionQueue | None = None,
    ) -> None:
        self._inference = inference
        self._plugins = plugins
        self._queue = queue
        self.admit = partial(
            admit_plugin_job,
            self._inference,
            self._plugins,
            self._plugin_ids,
            self._queue,
        )

    def _plugin_ids(self) -> frozenset[str]:
        if not isinstance(self._plugins, PluginIdentitySource):
            return frozenset()
        return frozenset(self._plugins.plugin_ids())

    def dispatch(self, job_id: str, workload_id: str, payload: dict):
        if workload_id in self._plugin_ids():
            if not isinstance(self._plugins, JobDispatcher):
                raise RuntimeError("plugin runtime does not implement dispatch")
            if self._queue is None:
                return self._plugins.dispatch(job_id, workload_id, payload)
            # The worker call is started by the queue, once this profile's
            # weight has earned it a slot — never here, or the pool would bound
            # nothing and weight would order nothing.
            return self._queue.run(
                workload_id,
                lambda: self._plugins.dispatch(job_id, workload_id, payload),
            )
        return self._inference.dispatch(job_id, workload_id, payload)

    def prepare_lane(self, workload_id: str):
        if workload_id in self._plugin_ids():
            raise RuntimeError("plugin jobs do not use prepared inference lanes")
        prepare = getattr(self._inference, "prepare_lane", None)
        if not callable(prepare):
            raise RuntimeError("inference dispatcher does not prepare lanes")
        return prepare(workload_id)

    def dispatch_prepared(self, job_id: str, workload_id: str, payload: dict, lane):
        if workload_id in self._plugin_ids():
            raise RuntimeError("plugin jobs do not use prepared inference lanes")
        dispatch = getattr(self._inference, "dispatch_prepared", None)
        if not callable(dispatch):
            raise RuntimeError("inference dispatcher does not consume prepared lanes")
        return dispatch(job_id, workload_id, payload, lane)


def policy_weight(state, workload_id: str) -> int:
    policy = state.profiles.get(workload_id)
    return policy.weight if policy is not None else 1


def policy_admits(state, workload_id: str) -> bool:
    if state.paused:
        return False
    policy = state.profiles.get(workload_id)
    return policy is None or policy.enabled


def job_authorized(
    action: str,
    workload_id: str,
    workloads: Mapping[str, Workload],
    plugins: PluginRuntime,
    admits: Callable[[str], bool],
) -> bool:
    plugin_ids = plugins.plugin_ids() if isinstance(plugins, PluginIdentitySource) else ()
    if workload_id not in workloads and workload_id not in plugin_ids:
        return False
    return action == "cancel" or admits(workload_id)


def default_dispatcher(
    workloads: dict[str, Workload],
    scheduler,
    executors: Callable[[], dict],
    artifact_store,
    input_roots: Sequence[Path | str],
    *,
    resolve_artifact=None,
    executor_view=None,
    scheduler_lane=None,
    device_identity=None,
    executor_for_device=None,
) -> JobDispatcher:
    if artifact_store is None:
        return UnavailableJobDispatcher()
    artifacts = (
        CachedArtifactSource(resolve_artifact) if resolve_artifact is not None else artifact_store
    )
    return InferenceJobDispatcher(
        workloads,
        scheduler,
        executors,
        artifacts,
        executor_view=executor_view,
        scheduler_lane=scheduler_lane,
        device_identity=device_identity,
        executor_for_device=executor_for_device,
        input_roots=OptedInInputRoots(input_roots),
    )


def build_runners(
    workloads: dict[str, Workload],
    *,
    dispatcher: JobDispatcher,
    resolve_artifact,
    cancellations,
    is_paused,
    is_enabled,
    allows_permission,
    deliver,
    progress,
) -> RunnerSet:
    return build_plugin_runners(
        workloads,
        dispatcher=dispatcher,
        resolve_artifact=resolve_artifact,
        encode_result=inference_result_payload,
        cancellations=cancellations,
        is_paused=is_paused,
        is_enabled=is_enabled,
        allows_permission=allows_permission,
        deliver=deliver,
        progress=progress,
    )


def select_runtime_model(workload: Workload, executors: dict):
    """Model for the first available preferred lane, matching dispatch."""
    choice = select_backend(workload, executors)
    if choice.backend is None:
        return None
    return runnable_model(workload, choice.backend, executors)


def profile_permissions_missing(workload: Workload, grants) -> tuple[str, ...]:
    declared = required_permissions(workload)
    if not declared:
        return ()
    return tuple(
        permission
        for permission in declared
        if not grants.is_granted(workload.id, permission, set(declared))
    )


def profile_permitted(
    profile_id: str,
    permission: str,
    workloads: Mapping[str, Workload],
    grants,
) -> bool:
    workload = workloads.get(profile_id)
    if workload is None:
        return False
    declared = set(required_permissions(workload))
    if permission not in declared:
        return False
    return grants.is_granted(profile_id, permission, declared)


__all__ = [
    "CachedArtifactSource",
    "PluginAwareDispatcher",
    "admit_plugin_job",
    "build_runners",
    "default_dispatcher",
    "job_authorized",
    "policy_admits",
    "policy_weight",
    "profile_permissions_missing",
    "profile_permitted",
]
