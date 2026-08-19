"""OmniTensor service wiring: discovery, scheduling, publishing, control.

Entry point ``omnitensor`` runs an asyncio loop that
- rediscovers accelerators when the kernel reports a device change, falling
  back to polling every ``DISCOVERY_INTERVAL_S`` where no event source exists,
- publishes a schema-valid snapshot atomically every ``PUBLISH_INTERVAL_S``,
- serves the control socket (framed msgpack over a unix-domain socket,
  default ``$XDG_RUNTIME_DIR/omnitensor/control.sock``) and answers
  ``apply-command``, ``submit-job``, and ``cancel-job`` through a versioned
  facade.

:class:`OmniTensorService` depends on the ports in :mod:`omnitensor.ports`;
the filesystem, sysfs, and socket adapters live in dedicated host modules and
are supplied through an explicit port bundle or constructor injection.

Configuration comes from environment variables:
``OMNITENSOR_CONTROL_SOCKET`` (control socket path override),
``OMNITENSOR_STATE_PATH`` (snapshot the applet reads),
``OMNITENSOR_POLICY_PATH`` (persisted policy + revision),
``OMNITENSOR_WORKLOADS`` (manifest directory),
``OMNITENSOR_MODEL_BINDINGS`` (restricted local model overrides),
``OMNITENSOR_ARTIFACT_ROOT`` (verified model artifact store).
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import artifact_readiness as artifacts
from . import dispatch_routing as routing
from . import telemetry_observation as observation
from .artifact_readiness import ArtifactResolver
from .callers import CallerIdentityResolver
from .composition import (
    DEFAULT_ARTIFACT_ROOT as DEFAULT_ARTIFACT_ROOT,
)
from .composition import (
    DEFAULT_GRANTS_PATH,
)
from .composition import (
    DEFAULT_MODEL_BINDINGS_PATH as DEFAULT_MODEL_BINDINGS_PATH,
)
from .composition import (
    DEFAULT_POLICY_PATH as DEFAULT_POLICY_PATH,
)
from .composition import (
    DEFAULT_STATE_PATH as DEFAULT_STATE_PATH,
)
from .composition import (
    DEFAULT_WORKLOADS_PATH as DEFAULT_WORKLOADS_PATH,
)
from .composition import (
    build_service_from_env as build_service_from_env,
)
from .contract import (
    RUNTIME_METHODS as RUNTIME_METHODS,
)
from .control import ControlService
from .device_lanes import DeviceLanes
from .discovery import DiscoveryPaths
from .execution import build_executors as build_executors
from .execution import close_executors
from .host import FileSnapshotPublisher as FileSnapshotPublisher
from .host import SysfsDeviceDiscovery as SysfsDeviceDiscovery
from .host import build_host_ports
from .host_pressure import read_pressure
from .job_ports import JobAdmission, JobDispatcher
from .jobs import (
    JobSubmissionService,
    PredicateJobAuthorizer,
)
from .plugin_admission import DEFAULT_MAX_CONCURRENT, PluginAdmissionQueue
from .plugins.artifact_installation import ArtifactInstaller
from .plugins.artifacts import ArtifactReference, ArtifactResolution
from .plugins.cancellation import JobCancellationRegistry
from .plugins.grants import GrantLedger
from .plugins.job_results import JobResultStore
from .plugins.kernel_telemetry import UnixSocketAggregateSource
from .plugins.loading import InstalledPluginRuntime
from .plugins.offloop import run_off_loop
from .plugins.orchestration import (
    RunnerBackedDispatcher,
    RunnerSet,
    with_recovery,
)
from .plugins.summaries import ResultSummaryRegistry
from .plugins.telemetry import PluginTelemetryRegistry
from .ports import (
    AcceleratorReloadableRuntime,
    ControlTransport,
    DeviceChangeSource,
    DeviceDiscovery,
    PluginIdentitySource,
    PluginRuntime,
    PolicyStorage,
    SnapshotPublisher,
)
from .profile_selection import (
    ARTIFACT_UNAVAILABLE as ARTIFACT_UNAVAILABLE,
)
from .profile_selection import (
    CONSENT_MISSING as CONSENT_MISSING,
)
from .profile_selection import (
    MAX_PUBLISHED_PROFILES as MAX_PUBLISHED_PROFILES,
)
from .profile_selection import (
    NO_MODEL as NO_MODEL,
)
from .profile_selection import (
    PAUSED_BY_POLICY as PAUSED_BY_POLICY,
)
from .profile_selection import (
    PROFILE_DISABLED as PROFILE_DISABLED,
)
from .profile_selection import (
    SERVING as SERVING,
)
from .profile_selection import (
    ReasonCodes as ReasonCodes,
)
from .profile_selection import (
    plugin_profile_statuses as plugin_profile_statuses,
)
from .profile_selection import (
    profile_statuses as profile_statuses,
)
from .registry import Workload, bundled_workloads_path, load_workload_catalog
from .runtime_api import RuntimeAPI as RuntimeAPI
from .scheduler import Scheduler
from .state import PolicyStore
from .telemetry_observation import TelemetryJobObserver as TelemetryJobObserver

LOGGER = logging.getLogger(__name__)

PUBLISH_INTERVAL_S = 2.0
# Only the fallback: a host whose discovery port can be woken by the kernel
# never reaches this timer, and one that cannot says so in the log with the
# wakeups it is spending.
DISCOVERY_INTERVAL_S = 10.0
# A runtime with nothing to report has nothing to gain from a fast tick: each
# one rereads sysfs, reloads grants, rebuilds the document and revalidates it
# against the schema — about 6 ms of CPU to conclude that nothing happened.
# State changes wake the loop through ``request_publish``, so backing an idle
# machine off this far costs no responsiveness.
IDLE_PUBLISH_INTERVAL_S = 10.0
# Readers treat a snapshot older than 15 s as stale, so an idle runtime still
# owes them proof of life — but only that, not a fresh copy of an unchanged
# document twice a second. Publishing on change, plus this heartbeat, keeps the
# staleness verdict correct while an idle machine stops writing to disk.
SNAPSHOT_HEARTBEAT_INTERVAL_S = 10.0
GRANT_REFRESH_INTERVAL_S = 0.25
# Artifact readiness re-reads and re-digests every declared artifact, which is
# 0.25 ms for a small model and 25 ms at the 64 MiB ceiling — affordable
# occasionally, not twice a second beside the snapshot. Thirty seconds is short
# enough that installing an artifact shows up while someone is still looking
# for it.
PLUGIN_READINESS_INTERVAL_S = 30.0


@dataclass(frozen=True, slots=True)
class _SchedulerObservation:
    """A scheduler's published figures, frozen on the event loop.

    Snapshot building runs on a worker thread (`run_off_loop`), and reading the
    live scheduler from there walked `profiles` and `_running` while the loop
    deleted drained profiles and finished jobs, raising `RuntimeError` past a
    handler that catches only `OSError` and `ValueError` — one unlucky
    interleaving tore the daemon down mid-job. `tick()` also folds busy time
    into the load EMA and resets it, which is a mutation a function named
    "build a snapshot" should not perform, and which raced the running job
    still adding to it. Both now happen on the loop; the thread only reads.
    """

    _stats: dict
    _profile_stats: dict

    @classmethod
    def of(cls, scheduler) -> _SchedulerObservation:
        scheduler.tick()
        return cls(scheduler.stats(), scheduler.profile_stats())

    def tick(self) -> None:
        """Already ticked on the event loop; building a snapshot only reads."""

    def stats(self) -> dict:
        return self._stats

    def profile_stats(self) -> dict:
        return self._profile_stats


class OmniTensorService:
    def __init__(
        self,
        snapshot_path: Path,
        policy_path: Path,
        workloads_path: Path,
        discovery_paths: DiscoveryPaths | None = None,
        *,
        discovery: DeviceDiscovery | None = None,
        publisher: SnapshotPublisher | None = None,
        policy_storage: PolicyStorage | None = None,
        plugin_runtime: PluginRuntime | None = None,
        job_dispatcher: JobDispatcher | None = None,
        artifact_root: Path | None = None,
        model_bindings_path: Path | None = None,
        grants_path: Path | str = DEFAULT_GRANTS_PATH,
        grants: GrantLedger | None = None,
        result_summaries: ResultSummaryRegistry | None = None,
        job_results: JobResultStore | None = None,
        plugin_telemetry: PluginTelemetryRegistry | None = None,
        kernel_telemetry_source=None,
        transport: ControlTransport | None = None,
        cancellation_journal_path: Path | None = None,
        input_roots: Sequence[Path | str] = (),
        publish_interval_s: float = PUBLISH_INTERVAL_S,
        discovery_interval_s: float = DISCOVERY_INTERVAL_S,
        accelerator_device_ids: dict[str, str] | None = None,
        plugin_slots: int = DEFAULT_MAX_CONCURRENT,
    ):
        self._callers = CallerIdentityResolver()
        host = build_host_ports(
            snapshot_path=snapshot_path,
            discovery_paths=discovery_paths,
            accelerator_device_ids=accelerator_device_ids,
            discovery=discovery,
            publisher=publisher,
            transport=transport,
        )
        self._discovery = host.discovery
        self._publisher_port = host.publisher
        self._transport = host.transport
        self._input_roots = tuple(input_roots)
        self._kernel_telemetry_source = kernel_telemetry_source or UnixSocketAggregateSource()
        # Consent has a home now.  The ledger was written, tested, and never
        # constructed, so `active_permissions` came from a deny-all stub: every
        # declared permission read as ungranted and nothing could change that.
        self._grants = grants if grants is not None else GrantLedger(grants_path)
        self._artifacts = ArtifactResolver(
            artifact_root,
            ArtifactInstaller(artifact_root) if artifact_root else None,
            workloads_of=lambda: self._workloads,
            plugins_of=lambda: self._plugin_runtime.snapshot.catalog.plugins,
        )
        self._plugin_runtime = plugin_runtime or InstalledPluginRuntime(
            bundled_workloads_path(),
            grant_source=self._grants,
            selected_files_root=snapshot_path.parent / "plugin-inputs",
            worker_state_root=snapshot_path.parent / "plugin-state",
            resolve_artifact=self._resolve_plugin_artifact,
            accelerator_devices=self._plugin_accelerator_devices,
            profile_accelerator_devices=self._plugin_accelerator_devices,
            profile_model_choice=self._plugin_model_choice,
            progress_sink=lambda progress: self._note_job_progress(
                progress.job_id,
                progress.stage,
                progress.fraction,
                progress.detail,
            ),
        )
        self._publish_interval_s = publish_interval_s
        self._discovery_interval_s = discovery_interval_s
        self._workloads = load_workload_catalog(
            workloads_path,
            model_bindings_root=model_bindings_path,
        )
        defaults = {
            workload_id: workload.default_policy()
            for workload_id, workload in self._workloads.items()
        }
        storage = policy_storage or PolicyStore(policy_path, defaults)
        self.control = ControlService(
            storage,
            on_applied=self._policy_changed,
            profile_exists=self._profile_exists,
            gpu_device_ids=self._gpu_device_ids,
            profile_models=self._profile_models,
        )
        self._devices = self._discovery.detect()
        self._executors = build_executors(self._devices)
        self._lanes = DeviceLanes(lambda: self._executors, self._gpu_device_choice)
        self._scheduler = Scheduler(
            self._executors.scheduler_executors(),
            self._weight_of,
            admits=self._admits,
        )
        self._last_device_choices = dict(self.control.state.device_choices)
        self._last_model_choices = dict(self.control.state.model_choices)
        self._pending_device_profiles: set[str] = set()
        self._plugin_reload_requested = False
        self._plugin_reload_task: asyncio.Task | None = None
        self._plugin_lifecycle_lock = asyncio.Lock()
        # Plugin jobs run in worker processes rather than scheduler lanes, so
        # this pool is where a plugin profile's weight decides anything at all.
        # Its width answers to the host: the per-worker ceilings that used to
        # guess what the machine could hold are gone, and what replaced them is
        # one measurement of how full the machine actually is.
        self._plugin_queue = PluginAdmissionQueue(
            weight_of=self._weight_of,
            admits=self._admits,
            max_concurrent=plugin_slots,
            pressure=read_pressure,
        )
        self._job_dispatcher = routing.PluginAwareDispatcher(
            job_dispatcher or self._default_dispatcher(),
            self._plugin_runtime,
            self._plugin_queue,
        )
        # Constructed here because submission answers with an acceptance: if no
        # store outlives the call, the outcome has nowhere to be kept and a
        # caller can never learn what happened to the job it submitted.
        self.job_results = job_results or JobResultStore()
        self._cancellations = JobCancellationRegistry(
            cancellation_journal_path or (snapshot_path.parent / "cancellations.json")
        )
        self.runners = self._build_runners()
        self.plugin_telemetry = plugin_telemetry or PluginTelemetryRegistry()
        # Every profile, not only version-two manifests.  Gating on the
        # manifest version meant no bundled profile was ever registered, so the
        # snapshot carried an empty telemetry list on every host and a user
        # could not tell a profile that had run a thousand jobs from one that
        # had never run at all.
        for workload in self._workloads.values():
            self.plugin_telemetry.register(workload.id)
        self.jobs = JobSubmissionService(
            # Routed through the runners, so a profile that has a pipeline runs
            # its pipeline; without this the runners were built and never used.
            RunnerBackedDispatcher(self.runners, self._job_dispatcher),
            PredicateJobAuthorizer(self._job_authorized),
            results=self.job_results,
            # The raw dispatcher, not the runner-backed one: routing through a
            # pipeline is what defers every refusal past the submission reply,
            # and this is the same object that would have raised them.
            admission=(
                self._job_dispatcher if isinstance(self._job_dispatcher, JobAdmission) else None
            ),
            observer=observation.PublishingJobObserver(
                TelemetryJobObserver(self.plugin_telemetry), self.request_publish
            ),
        )
        self.runtime_api = RuntimeAPI(
            self.control, self.jobs, self._describe_plugins, self._callers
        )
        self.result_summaries = result_summaries or ResultSummaryRegistry(
            alert_id_factory=lambda: f"alert-{secrets.token_hex(16)}"
        )
        self._summary_observer = observation.ResultSummaryObserver(
            lambda: self.result_summaries,
            clock_ms=lambda: max(1, int(time.time() * 1000)),
            logger=LOGGER,
        )
        self._snapshot_retracted = False
        self._last_published_snapshot: dict | None = None
        self._next_snapshot_heartbeat = 0.0
        self._idle = False
        self._publish_wake = asyncio.Event()
        self._publish_loop: asyncio.AbstractEventLoop | None = None
        self._next_grant_refresh = 0.0
        self._next_readiness_refresh = 0.0
        self._stopping = asyncio.Event()

    def _device_load(self, device, stats) -> float | None:
        return observation.device_load(self._discovery, device, stats)

    def _weight_of(self, workload_id: str) -> int:
        return routing.policy_weight(self.control.state, workload_id)

    def _admits(self, workload_id: str) -> bool:
        return workload_id not in self._pending_device_profiles and routing.policy_admits(
            self.control.state, workload_id
        )

    def _job_authorized(self, action: str, workload_id: str) -> bool:
        return routing.job_authorized(
            action,
            workload_id,
            self._workloads,
            self._plugin_runtime,
            self._admits,
        )

    def _policy_changed(self) -> None:
        self.request_publish()
        self._scheduler.kick()
        # Same reason the scheduler is kicked: a plugin job held because its
        # profile was disabled must be re-evaluated now, not when whatever is
        # running happens to finish.
        self._plugin_queue.kick()
        changed_profiles = self._changed_choices() | self._changed_models()
        plugin_profiles = changed_profiles & self._plugin_ids()
        if plugin_profiles:
            self._schedule_plugin_reload(plugin_profiles)

    def _changed_choices(self) -> set[str]:
        choices = dict(self.control.state.device_choices)
        if choices == self._last_device_choices:
            return set()
        changed = {
            profile_id
            for profile_id in set(choices) | set(self._last_device_choices)
            if choices.get(profile_id) != self._last_device_choices.get(profile_id)
        }
        self._last_device_choices = choices
        return changed

    def _changed_models(self) -> set[str]:
        """Which workloads were told to run a different model.

        A worker holds its model for as long as it lives — the weights are
        mapped at load — so changing the choice means replacing the worker, in
        exactly the way changing its card does. Without this a person would
        choose a model, see the command accepted, and keep getting answers from
        the old one until something else happened to restart the worker.
        """
        chosen = dict(self.control.state.model_choices)
        if chosen == self._last_model_choices:
            return set()
        changed = {
            profile_id
            for profile_id in set(chosen) | set(self._last_model_choices)
            if chosen.get(profile_id) != self._last_model_choices.get(profile_id)
        }
        self._last_model_choices = chosen
        return changed

    def _plugin_ids(self) -> frozenset[str]:
        return (
            frozenset(self._plugin_runtime.plugin_ids())
            if isinstance(self._plugin_runtime, PluginIdentitySource)
            else frozenset()
        )

    def _schedule_plugin_reload(self, profile_ids: set[str] | frozenset[str] | None = None) -> None:
        if not isinstance(self._plugin_runtime, AcceleratorReloadableRuntime):
            return
        affected = self._plugin_ids() if profile_ids is None else frozenset(profile_ids)
        if not affected:
            return
        # A worker's device mount is immutable.  Once policy points elsewhere,
        # new jobs must wait until the replacement worker owns that exact lease;
        # a failed replacement remains unavailable rather than using the old GPU.
        self._pending_device_profiles.update(affected)
        self._plugin_reload_requested = True
        if self._plugin_reload_task is None or self._plugin_reload_task.done():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            self._plugin_reload_task = loop.create_task(
                self._reload_plugin_accelerators(),
                name="omnitensor-plugin-accelerator-reload",
            )

    def _profile_exists(self, profile_id: str) -> bool:
        return profile_id in self._workloads or profile_id in self._plugin_ids()

    def _plugin_model_choice(self, profile_id: str) -> str | None:
        """The model this workload was told to run, or nothing.

        Asked of the policy store each time a worker is built rather than
        cached, for the same reason the device choice is: a worker started
        after a change must get the change.
        """
        return self.control.state.model_choices.get(profile_id)

    def _profile_models(self, profile_id: str) -> tuple[str, ...]:
        """The artifacts this workload's manifest pins, by id.

        The bound on what a person may choose. A model outside it has no
        verified digest for this workload, so choosing it could only ever end
        at a worker that refuses to start — better refused here, where the
        refusal has somewhere to be read.
        """
        declared = getattr(self._plugin_runtime, "declared_artifacts", None)
        return tuple(declared(profile_id)) if callable(declared) else ()

    def _gpu_device_ids(self) -> tuple[str, ...]:
        return tuple(device.id for device in self._devices if device.backend == "gpu")

    def _gpu_device_choice(self, profile_id: str) -> str | None:
        return self.control.state.device_choices.get(profile_id)

    def _executor_view(self, profile_id: str) -> dict:
        return self._lanes.executors_for(profile_id)

    def _scheduler_lane(self, profile_id: str, backend: str) -> str:
        return self._lanes.lane(profile_id, backend)

    def _device_identity(self, profile_id: str, backend: str) -> str | None:
        return self._lanes.identity(profile_id, backend)

    def _executor_for_device(self, backend: str, device_id: str):
        return self._lanes.executor_for(backend, device_id)

    def _default_dispatcher(self) -> JobDispatcher:
        return routing.default_dispatcher(
            self._workloads,
            self._scheduler,
            lambda: self._executors,
            self._artifact_store,
            self._input_roots,
            resolve_artifact=self._cached_resolution,
            executor_view=self._executor_view,
            scheduler_lane=self._scheduler_lane,
            device_identity=self._device_identity,
            executor_for_device=self._executor_for_device,
        )

    def _build_runners(self) -> RunnerSet:
        return routing.build_runners(
            self._workloads,
            dispatcher=self._job_dispatcher,
            resolve_artifact=self._resolve_artifact,
            cancellations=self._cancellations,
            is_paused=lambda: self.control.state.paused,
            is_enabled=self._admits,
            allows_permission=self._permitted,
            deliver=self._deliver_job_output,
            progress=self._note_job_progress,
        )

    def _profile_permissions_missing(self, workload: Workload) -> tuple[str, ...]:
        return routing.profile_permissions_missing(workload, self._grants)

    def _permitted(self, profile_id: str, permission: str) -> bool:
        return routing.profile_permitted(
            profile_id,
            permission,
            self._workloads,
            self._grants,
        )

    def _note_job_progress(self, job_id: str, stage: str, fraction: float, detail: str) -> None:
        # Late-bound: runners are built before the job service they report to.
        self.jobs.note_progress(job_id, stage, fraction, detail)

    def _deliver_job_output(self, job_id: str, output: dict) -> None:
        self.jobs.note_progress(job_id, "deliver", 1.0, "Result delivered")
        self._summarize(job_id, output)

    def _summarize(self, job_id: str, output: dict) -> None:
        self._summary_observer.summarize(
            job_id,
            output,
            publish_forecast=self._publish_forecast_summary,
        )

    def _publish_forecast_summary(
        self,
        job_id: str,
        profile_id: str,
        reading: dict,
    ) -> None:
        self._summary_observer.publish_forecast(job_id, profile_id, reading)

    def reconcile_interrupted_jobs(self) -> RunnerSet:
        """Report the jobs a previous process left in flight, once, at startup.

        Whatever owned their stage is gone, so they are not resumable; saying
        so and clearing the journal is the only honest outcome.
        """
        self.runners = with_recovery(self.runners, self._cancellations)
        for cancellation in self.runners.interrupted:
            LOGGER.warning(
                "job %s was interrupted by a previous shutdown: %s",
                cancellation.job_id,
                cancellation.detail,
            )
        for profile_id, reason in sorted(self.runners.skipped.items()):
            LOGGER.info("no pipeline runner for %s: %s", profile_id, reason)
        return self.runners

    def _describe_plugins(self) -> str:
        return observation.describe_plugins(
            self._plugin_runtime,
            self._resolve_artifact,
            clock_ms=lambda: int(time.time() * 1000),
        )

    def _workload_executors(self, workload: Workload) -> dict:
        """The executors this workload may use, which is its own lane's view.

        Both callers used to open with the same six lines guarding
        `getattr(self, "_executor_view", None)` and `callable(...)` — a method
        this class defines, so the guard could not fail and the fallback to
        every executor could not run. What the duplication hid is that the two
        methods differ only in what they do with the answer.
        """
        return self._lanes.executors_for(workload.id)

    def _profile_artifact_ready(self, workload: Workload) -> tuple[bool, str]:
        return artifacts.profile_artifact_ready(
            workload,
            self._workload_executors(workload),
            self._resolve_artifact,
            self._selected_backend,
        )

    def _selected_backend(self, workload: Workload) -> str:
        return artifacts.selected_backend(workload, self._workload_executors(workload))

    @property
    def _artifact_store(self):
        return self._artifacts.store

    @_artifact_store.setter
    def _artifact_store(self, store) -> None:
        self._artifacts.store = store

    def _resolve_artifact(self, artifact_id: str) -> ArtifactResolution:
        return self._artifacts.resolve(artifact_id)

    def _resolve_plugin_artifact(
        self,
        reference: ArtifactReference,
    ) -> ArtifactResolution:
        return self._artifacts.resolve_plugin(reference)

    def _plugin_accelerator_devices(self, profile_id: str | None = None) -> dict[str, Path]:
        return artifacts.plugin_accelerator_devices(
            self._devices,
            self._gpu_device_choice(profile_id) if profile_id is not None else None,
        )

    def _cached_resolution(
        self,
        artifact_id: str,
        reference: ArtifactReference,
    ) -> ArtifactResolution:
        return self._artifacts.cached(artifact_id, reference)

    def _build_runtime_snapshot(self, scheduler=None) -> dict:
        return observation.runtime_snapshot(
            devices=self._devices,
            device_load_of=self._device_load,
            workloads=self._workloads,
            executors=self._executors,
            scheduler=self._scheduler if scheduler is None else scheduler,
            policy=self.control.state,
            artifact_ready=self._profile_artifact_ready,
            permissions_missing=self._profile_permissions_missing,
            result_summaries=self.result_summaries,
            plugin_telemetry=self.plugin_telemetry,
            input_roots=self._input_roots,
            kernel_telemetry_source=self._kernel_telemetry_source,
            profile_statuses_of=self._profile_statuses,
        )

    def _profile_statuses(
        self,
        workloads,
        executors,
        scheduler,
        policy,
        artifact_ready,
        permissions_missing,
    ) -> dict[str, dict]:
        """Catalog profiles as before, plus the installed plugins.

        Both are governed by the same policy, so both belong in the published
        profile set: a reader that finds only catalog profiles there cannot
        tell a plugin whose policy it may set from one the runtime has never
        heard of, and drew its controls dead for both.
        """
        reasons = ReasonCodes(
            paused_by_policy=PAUSED_BY_POLICY,
            profile_disabled=PROFILE_DISABLED,
            no_model=NO_MODEL,
            artifact_unavailable=ARTIFACT_UNAVAILABLE,
            serving=SERVING,
            consent_missing=CONSENT_MISSING,
        )
        statuses = profile_statuses(
            workloads,
            executors,
            scheduler,
            policy,
            artifact_ready,
            permissions_missing,
            reasons,
        )
        statuses.update(
            plugin_profile_statuses(
                self._plugin_ids(),
                self._plugin_queue.profile_stats(),
                policy,
                reasons=reasons,
            )
        )
        return statuses

    def publish_once(self) -> dict:
        """Synchronously build and publish one snapshot (test/adapter API)."""
        snapshot = self._build_runtime_snapshot()
        self._publisher_port.publish(snapshot)
        self._record_published(snapshot)
        return snapshot

    @staticmethod
    def _content_of(snapshot: dict) -> dict:
        """The snapshot's content, apart from when it was built."""
        return {key: value for key, value in snapshot.items() if key != "generatedAt"}

    def _snapshot_changed(self, snapshot: dict) -> bool:
        """Whether this document says anything the last published one did not.

        ``generatedAt`` moves on every build, so it is compared out: it is the
        proof of life the heartbeat carries, never a change in its own right.
        """
        if self._last_published_snapshot is None:
            return True
        return self._content_of(snapshot) != self._content_of(self._last_published_snapshot)

    def _heartbeat_due(self) -> bool:
        return self._monotonic() >= self._next_snapshot_heartbeat

    def request_publish(self) -> None:
        """Ask the publisher for a tick now rather than at its next deadline.

        Backing an idle runtime off to a slow tick would otherwise delay every
        state change by that interval. The service knows when it has changed
        something, so it says so instead of being polled for it.
        """
        loop = self._publish_loop
        if loop is None or self._publish_wake.is_set():
            return
        try:
            # Runners deliver progress from their own threads, and
            # ``Event.set`` is not safe to call from one.
            loop.call_soon_threadsafe(self._publish_wake.set)
        except RuntimeError:
            # The loop is closed; there is no publisher left to wake.
            return

    async def _await_next_publish(self) -> None:
        """Sleep until the next deadline, a state change, or shutdown.

        An idle runtime has nothing to say, so it waits far longer between
        ticks than a busy one — rebuilding, revalidating, and rewriting an
        unchanged document costs real power on a machine doing nothing.
        """
        interval = IDLE_PUBLISH_INTERVAL_S if self._idle else self._publish_interval_s
        waits = {
            asyncio.ensure_future(self._stopping.wait()),
            asyncio.ensure_future(self._publish_wake.wait()),
        }
        try:
            await asyncio.wait(waits, timeout=interval, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for wait in waits:
                wait.cancel()
            await asyncio.gather(*waits, return_exceptions=True)
        self._publish_wake.clear()

    def _record_published(self, snapshot: dict) -> None:
        self._last_published_snapshot = snapshot
        self._next_snapshot_heartbeat = self._monotonic() + SNAPSHOT_HEARTBEAT_INTERVAL_S

    @staticmethod
    def _monotonic() -> float:
        try:
            return asyncio.get_running_loop().time()
        except RuntimeError:
            # ``publish_once`` is a synchronous test/adapter entry point that
            # can run with no loop; the heartbeat still needs a clock.
            return time.monotonic()

    async def _publisher(self) -> None:
        # Honest-absence decision (OMNI-0011): with zero devices no
        # schema-valid snapshot exists (the contract requires >=1 device), so
        # rather than letting the last document go silently stale we retract
        # it — readers observe absence, exactly as when the service is down —
        # and log the outage once.  Publishing resumes when devices return.
        while not self._stopping.is_set():
            # A publish failure is an outage of one output, not of the service.
            # Letting an OSError escape here propagates through the gather that
            # supervises the loops and takes down job dispatch and D-Bus with
            # it, so a full disk would stop far more than snapshot publishing.
            try:
                if self._devices:
                    await self._refresh_grants()
                    if self._snapshot_retracted:
                        LOGGER.info("Accelerator devices returned; publishing snapshots again")
                        self._snapshot_retracted = False
                    await self._refresh_plugin_readiness()
                    # Ticked and read on the event loop, then handed to the
                    # worker thread as an immutable value. Reading the live
                    # scheduler off-loop walked mappings the loop was deleting
                    # from, and `tick()` reset busy time a running job was
                    # still adding to.
                    scheduler_view = _SchedulerObservation.of(self._scheduler)
                    snapshot = await run_off_loop(self._build_runtime_snapshot, scheduler_view)
                    # An idle runtime rebuilds the same document every tick.
                    # Rewriting it anyway costs a disk write and wakes every
                    # reader watching the file, forever, to say nothing changed.
                    changed = self._snapshot_changed(snapshot)
                    if changed or self._heartbeat_due():
                        await run_off_loop(self._publisher_port.publish, snapshot)
                        self._record_published(snapshot)
                    self._idle = not changed
                elif not self._snapshot_retracted:
                    await run_off_loop(self._publisher_port.retract)
                    self._snapshot_retracted = True
                    self._last_published_snapshot = None
                    self._next_snapshot_heartbeat = 0.0
                    LOGGER.warning(
                        "No accelerator devices present; retracted the runtime snapshot"
                        " so readers observe absence instead of stale data",
                    )
            except (OSError, ValueError):
                LOGGER.exception("Could not publish the runtime snapshot; retrying next tick")
                self._idle = False
            await self._await_next_publish()

    async def _refresh_grants(self) -> None:
        now = asyncio.get_running_loop().time()
        if now < self._next_grant_refresh:
            return
        await run_off_loop(self._grants.reload)
        self._next_grant_refresh = now + GRANT_REFRESH_INTERVAL_S

    async def _rediscover(self) -> None:
        """Rediscover when the kernel says so, and only poll when it cannot.

        Hardware that is not changing has nothing to rediscover, so with an
        event source this loop is asleep on a socket: an idle machine spends
        no wakeups, no sysfs reads, and no CPU on it at all.  Polling remains
        the fallback, and it is announced with what it costs rather than
        quietly resumed.
        """
        changes = self._discovery if isinstance(self._discovery, DeviceChangeSource) else None
        polling = changes is None
        if polling:
            self._warn_discovery_polling("this host exposes no device-change events")
        while not self._stopping.is_set():
            if polling:
                if await self._sleep_until_stopping(self._discovery_interval_s):
                    return
            else:
                if await self._await_device_change(changes):
                    if self._stopping.is_set():
                        return
                else:
                    if self._stopping.is_set():
                        return
                    polling = True
                    self._warn_discovery_polling("the kernel event source closed")
                    continue
            self._apply_discovery()

    def _warn_discovery_polling(self, reason: str) -> None:
        interval = self._discovery_interval_s
        LOGGER.warning(
            "Device discovery is polling sysfs every %.1f s because %s."
            " That is about %d wakeups a day on a machine whose hardware never"
            " changes; event-driven discovery costs none of them.",
            interval,
            reason,
            int(86400 // interval) if interval > 0 else 0,
        )

    async def _await_device_change(self, changes: DeviceChangeSource) -> bool:
        """Wait for a device event or shutdown; ``False`` means fall back."""
        change = asyncio.ensure_future(changes.wait_for_change())
        stopping = asyncio.ensure_future(self._stopping.wait())
        try:
            done, _pending = await asyncio.wait(
                (change, stopping), return_when=asyncio.FIRST_COMPLETED
            )
            if change not in done:
                return True
            try:
                return bool(change.result())
            except Exception:  # noqa: BLE001 - event adapters are host code
                LOGGER.exception("Kernel device-event source failed")
                return False
        finally:
            for task in (change, stopping):
                task.cancel()
            await asyncio.gather(change, stopping, return_exceptions=True)

    async def _sleep_until_stopping(self, timeout: float) -> bool:
        """Sleep ``timeout`` seconds; ``True`` when shutdown arrived first."""
        try:
            await asyncio.wait_for(self._stopping.wait(), timeout=timeout)
        except TimeoutError:
            return False
        return True

    def _apply_discovery(self) -> None:
        devices = self._discovery.detect()
        devices_changed = devices != self._devices
        self._executors = build_executors(
            devices,
            previous_devices=self._devices,
            previous_executors=self._executors,
        )
        self._devices = devices
        self._scheduler.update_executors(self._executors.scheduler_executors())
        if devices_changed:
            self.request_publish()
            self._schedule_plugin_reload()

    async def _close_discovery(self) -> None:
        """Release the discovery event source, when the adapter holds one."""
        if not isinstance(self._discovery, DeviceChangeSource):
            return
        try:
            await self._discovery.aclose()
        except Exception:  # noqa: BLE001 - host adapters are external
            LOGGER.exception("Could not close the device-event source")

    async def run(self) -> None:
        try:
            await self._transport.start(self.runtime_api)
            self.reconcile_interrupted_jobs()
            self._scheduler.start()
            loop = asyncio.get_running_loop()
            self._publish_loop = loop
            loop_tasks = [
                loop.create_task(self._publisher()),
                loop.create_task(self._rediscover()),
            ]
            loops = asyncio.gather(*loop_tasks)
            plugin_start = loop.create_task(self._start_plugin_runtime())
            try:
                done, _pending = await asyncio.wait(
                    (loops, plugin_start), return_when=asyncio.FIRST_COMPLETED
                )
                if plugin_start in done:
                    await plugin_start
                    await loops
                else:
                    await loops
                    if plugin_start.done():
                        await plugin_start
            finally:
                self._stopping.set()
                plugin_start.cancel()
                loops.cancel()
                for task in loop_tasks:
                    task.cancel()
                await asyncio.gather(plugin_start, loops, *loop_tasks, return_exceptions=True)
                await self._scheduler.stop()
        finally:
            await self.jobs.stop()
            reload_task = self._plugin_reload_task
            if reload_task is not None:
                reload_task.cancel()
                await asyncio.gather(reload_task, return_exceptions=True)
            await self._stop_plugin_runtime()
            await self._transport.stop()
            await self._close_discovery()
            # Last: the scheduler has stopped, so no job still holds a
            # runtime, and the accelerator memory is handed back in a chosen
            # order rather than whenever the collector notices.
            close_executors(self._executors)

    async def _start_plugin_runtime(self) -> None:
        async with self._plugin_lifecycle_lock:
            plugin_snapshot = await self._plugin_runtime.start()
        observation.register_plugin_telemetry(
            plugin_snapshot,
            self.plugin_telemetry,
        )
        await self._adopt_plugin_profiles()
        await self._record_plugin_readiness()

    async def _adopt_plugin_profiles(self) -> None:
        """Give every discovered plugin a policy, once it is known to exist.

        Policy is seeded from the workload catalog at construction, and plugins
        are discovered after that: until they are adopted, every command naming
        one is refused as an unknown profile and the published profile set
        omits them, so a surface can only draw controls that cannot work.
        """
        plugin_ids = self._plugin_ids()
        if not plugin_ids:
            return
        adopted = await self.control.adopt_profiles(plugin_ids)
        if adopted:
            LOGGER.info("Adopted policy for %d plugin profile(s)", len(adopted))

    async def _record_plugin_readiness(self) -> None:
        """Record artifact readiness and worker health for every plugin.

        Resolving an artifact re-reads and re-digests the file, so this is not
        something the publish loop can do twice a second; it runs when the
        plugin runtime changes and then on its own slow interval, which is what
        `_refresh_plugin_readiness` is for.
        """
        snapshot = getattr(self._plugin_runtime, "snapshot", None)
        if snapshot is None:
            return
        await run_off_loop(
            observation.record_artifact_readiness,
            snapshot,
            self._resolve_plugin_artifact_id,
            self.plugin_telemetry,
        )
        observation.record_worker_health(snapshot, self.plugin_telemetry)
        self._next_readiness_refresh = time.monotonic() + PLUGIN_READINESS_INTERVAL_S

    async def _refresh_plugin_readiness(self) -> None:
        """Re-record readiness once the interval has passed.

        Worker health is refreshed every time, because reading a state a
        supervisor already holds costs nothing; artifacts are the expensive
        half and wait for the interval.
        """
        snapshot = getattr(self._plugin_runtime, "snapshot", None)
        if snapshot is None:
            return
        observation.record_worker_health(snapshot, self.plugin_telemetry)
        if time.monotonic() < self._next_readiness_refresh:
            return
        await self._record_plugin_readiness()

    def _resolve_plugin_artifact_id(self, artifact_id: str):
        """Resolve a plugin's declared artifact by id, as the inventory does."""
        return self._resolve_artifact(artifact_id)

    async def _stop_plugin_runtime(self) -> None:
        async with self._plugin_lifecycle_lock:
            await self._plugin_runtime.stop()

    async def _reload_plugin_accelerators(self) -> None:
        while self._plugin_reload_requested:
            self._plugin_reload_requested = False
            try:
                async with self._plugin_lifecycle_lock:
                    plugin_snapshot = await self._plugin_runtime.reload_accelerator_devices()
            except Exception:  # noqa: BLE001 - plugin adapters are external
                LOGGER.exception("Could not reload plugin accelerator leases")
                return
            observation.register_plugin_telemetry(
                plugin_snapshot,
                self.plugin_telemetry,
            )
            await self._record_plugin_readiness()
        self._pending_device_profiles.clear()


def main() -> None:  # pragma: no cover - process entry point
    asyncio.run(build_service_from_env().run())
