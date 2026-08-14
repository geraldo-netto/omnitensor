"""OmniTensor service wiring: discovery, scheduling, publishing, control.

Entry point ``omnitensor`` runs an asyncio loop that
- rediscovers accelerators every ``DISCOVERY_INTERVAL_S``,
- publishes a schema-valid snapshot atomically every ``PUBLISH_INTERVAL_S``,
- owns ``org.cinnamon.OmniTensor1`` on the session bus and answers
  ``ApplyCommand``, ``SubmitJob``, and ``CancelJob`` through a versioned facade.

:class:`OmniTensorService` depends on the ports in :mod:`omnitensor.ports`;
the filesystem, sysfs, and D-Bus adapters live in dedicated host modules and
are supplied through an explicit port bundle or constructor injection.

Configuration comes from environment variables:
``OMNITENSOR_STATE_PATH`` (snapshot the applet reads),
``OMNITENSOR_POLICY_PATH`` (persisted policy + revision),
``OMNITENSOR_WORKLOADS`` (manifest directory),
``OMNITENSOR_MODEL_BINDINGS`` (restricted local model overrides),
``OMNITENSOR_ARTIFACT_ROOT`` (verified model artifact store).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import secrets
import time
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

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
    _env_accelerator_device_ids as _env_accelerator_device_ids,
)
from .composition import (
    _env_path as _env_path,
)
from .composition import (
    _env_paths as _env_paths,
)
from .composition import (
    build_service_from_env as build_service_from_env,
)
from .contract import contract_document_text
from .control import ControlService
from .dbus_transport import (
    BUS_METHODS as BUS_METHODS,
)
from .dbus_transport import (
    BUS_NAME as BUS_NAME,
)
from .dbus_transport import (
    OBJECT_PATH as OBJECT_PATH,
)
from .dbus_transport import (
    DbusControlTransport as DbusControlTransport,
)
from .dbus_transport import (
    OmniTensorInterface as OmniTensorInterface,
)
from .discovery import DiscoveryPaths
from .dispatch import (
    InferenceJobDispatcher,
    declared_artifact_reference,
    inference_result_payload,
)
from .execution import _build_executor as _build_executor
from .execution import build_executors
from .forecastresult import parse_forecast_reading
from .guard import BusGuard, GuardRefusedError, guarded
from .host import FileSnapshotPublisher as FileSnapshotPublisher
from .host import SysfsDeviceDiscovery as SysfsDeviceDiscovery
from .host import build_host_ports
from .inspection import PLUGIN_INVENTORY_VERSION, build_plugin_inventory
from .job_ports import JobAdmission, JobDispatcher
from .jobs import (
    JobSubmissionService,
    PredicateJobAuthorizer,
    UnavailableJobDispatcher,
)
from .plugins.artifact_installation import ArtifactInstaller
from .plugins.artifacts import ArtifactReference, ArtifactResolution, artifact_filename
from .plugins.cancellation import JobCancellationRegistry
from .plugins.grants import GrantLedger
from .plugins.job_results import JobResultStore
from .plugins.kernel_telemetry import UnixSocketAggregateSource
from .plugins.loading import InstalledPluginRuntime
from .plugins.orchestration import (
    RunnerBackedDispatcher,
    RunnerSet,
    build_plugin_runners,
    required_permissions,
    with_recovery,
)
from .plugins.secrets import SecretRedactor
from .plugins.summaries import ResultSummaryRegistry, SummaryError
from .plugins.telemetry import PluginTelemetryRegistry
from .ports import (
    ControlTransport,
    DeviceDiscovery,
    PluginCatalogSnapshot,
    PluginIdentitySource,
    PluginPermissionSource,
    PluginRuntime,
    PluginRuntimeSnapshot,
    PluginSnapshotSource,
    PolicyStorage,
    SnapshotPublisher,
)
from .registry import Workload, bundled_workloads_path, load_workload_catalog
from .scheduler import Scheduler, runnable_model, select_backend
from .snapshot import build_snapshot, input_roots_document
from .state import PolicyState, PolicyStore
from .tensorref import OptedInInputRoots

LOGGER = logging.getLogger(__name__)

PUBLISH_INTERVAL_S = 2.0
DISCOVERY_INTERVAL_S = 10.0


def _no_inventory() -> str:
    """Fail closed: an unwired inspector reports an empty inventory, not an error."""
    return json.dumps(
        {"version": PLUGIN_INVENTORY_VERSION, "generatedAt": 1, "plugins": []},
        separators=(",", ":"),
    )


def _admit_plugin_job(
    inference: JobDispatcher,
    plugins: PluginRuntime,
    plugin_ids_provider: Callable[[], frozenset[str]],
    workload_id: str,
    payload: dict,
) -> None:
    plugin_ids = plugin_ids_provider()
    installed_plugin = workload_id in plugin_ids
    if installed_plugin:
        if not isinstance(plugins, JobAdmission):
            raise RuntimeError("plugin runtime does not implement admission")
        plugins.admit(workload_id, payload)
        return
    if isinstance(inference, JobAdmission):
        inference.admit(workload_id, payload)


class _PluginAwareDispatcher:
    """Route installed plugin IDs to workers and model profiles to inference."""

    def __init__(self, inference: JobDispatcher, plugins: PluginRuntime) -> None:
        self._inference = inference
        self._plugins = plugins
        self.admit = partial(
            _admit_plugin_job,
            self._inference,
            self._plugins,
            self._plugin_ids,
        )

    def _plugin_ids(self) -> frozenset[str]:
        if not isinstance(self._plugins, PluginIdentitySource):
            return frozenset()
        return frozenset(self._plugins.plugin_ids())

    def dispatch(self, job_id: str, workload_id: str, payload: dict):
        if workload_id in self._plugin_ids():
            if not isinstance(self._plugins, JobDispatcher):
                raise RuntimeError("plugin runtime does not implement dispatch")
            return self._plugins.dispatch(job_id, workload_id, payload)
        return self._inference.dispatch(job_id, workload_id, payload)


class RuntimeAPI:
    """Transport-neutral facade retaining policy control while adding jobs."""

    def __init__(
        self,
        control: ControlService,
        jobs: JobSubmissionService,
        inspector: Callable[[], str] | None = None,
        callers: CallerIdentityResolver | None = None,
        guard: BusGuard | None = None,
    ) -> None:
        self._control = control
        self._jobs = jobs
        self._inspector = inspector or _no_inventory
        # Resolving here rather than in the store keeps every transport-shaped
        # concern on the transport side of the facade.
        self._callers = callers or CallerIdentityResolver()
        self._guard = guard or BusGuard()

    async def apply_command_text(self, text: str) -> str:
        return await self._guarded("ApplyCommand", text, self._control.apply_command_text)

    async def submit_job_text(self, text: str) -> str:
        owner = await self._callers.owner_token()
        return await self._guarded(
            "SubmitJob",
            text,
            lambda request: self._jobs.submit_job_text(request, owner=owner),
            owner=owner,
        )

    async def cancel_job_text(self, text: str) -> str:
        owner = await self._callers.owner_token()
        return await self._guarded(
            "CancelJob",
            text,
            lambda request: self._jobs.cancel_job_text(request, owner=owner),
            owner=owner,
        )

    async def job_result_text(self, text: str) -> str:
        owner = await self._callers.owner_token()
        return await self._guarded(
            "GetJobResult",
            text,
            lambda request: self._jobs.job_result_text(request, owner=owner),
            owner=owner,
        )

    def describe_plugins_text(self) -> str:
        owner = self._callers.cached_owner_token()
        try:
            with guarded(self._guard, "DescribePlugins", owner):
                return self._inspector()
        except GuardRefusedError as refusal:
            return refusal.text()

    def describe_contract_text(self) -> str:
        """Answer what this service speaks, before it is asked to speak it.

        Guarded like every other method: a handshake a caller can issue
        without limit is a way to keep the service busy answering handshakes.
        """
        owner = self._callers.cached_owner_token()
        try:
            with guarded(self._guard, "DescribeContract", owner):
                return contract_document_text(BUS_METHODS)
        except GuardRefusedError as refusal:
            return refusal.text()

    async def _guarded(self, method: str, text: str, call, *, owner: str | None = None) -> str:
        """Admit one call, run it, and release the slot however it ends.

        A refusal is returned as text rather than raised: the bus reply is the
        only channel the caller has, and an exception here would surface as an
        opaque internal error instead of a code they can branch on.
        """
        resolved = owner if owner is not None else await self._callers.owner_token()
        try:
            with guarded(self._guard, method, resolved, text):
                return await call(text)
        except GuardRefusedError as refusal:
            return refusal.text()


class TelemetryJobObserver:
    """Adapter from the job lifecycle onto per-profile telemetry counters.

    The registry enforces a state machine — a job is queued, then started,
    then ends — and refuses anything out of order.  That is right for the
    registry and wrong to propagate: bookkeeping must never fail the job it is
    counting, so a refusal is logged and dropped.  A gap in a counter is a
    worse report; a job that failed because its counter would not increment is
    a worse service.
    """

    def __init__(self, telemetry: PluginTelemetryRegistry, clock_ms=None) -> None:
        self._telemetry = telemetry
        self._clock_ms = clock_ms or (lambda: max(1, int(time.time() * 1000)))

    def job_started(self, workload_id: str) -> None:
        # Queued and started together: the scheduler owns the queue, and the
        # moment between the two is not observable from here.
        self._guarded(workload_id, self._telemetry.queue_job)
        self._guarded(workload_id, self._telemetry.start_job)

    def job_finished(self, workload_id: str, status: str, detail: str) -> None:
        if status == "succeeded":
            self._guarded(
                workload_id,
                lambda plugin_id: self._telemetry.succeed(
                    plugin_id, completed_at_ms=self._clock_ms()
                ),
            )
        elif status == "cancelled":
            self._guarded(workload_id, self._telemetry.cancel)
        else:
            self._guarded(
                workload_id,
                lambda plugin_id: self._telemetry.fail(
                    plugin_id,
                    "job-failed",
                    detail or "the job failed",
                    completed_at_ms=self._clock_ms(),
                ),
            )

    def _guarded(self, workload_id: str, call) -> None:
        try:
            call(workload_id)
        except (KeyError, ValueError) as error:
            LOGGER.debug("plugin telemetry not updated for %s: %s", workload_id, error)


def profile_statuses(
    workloads: dict[str, Workload],
    executors: dict,
    scheduler: Scheduler,
    policy: PolicyState,
    artifact_ready: Callable[[Workload], tuple[bool, str]] | None = None,
    permissions_missing: Callable[[Workload], tuple[str, ...]] | None = None,
) -> dict[str, dict]:
    """Runtime status per profile for the snapshot document."""
    per_profile = scheduler.profile_stats()
    return {
        workload_id: _profile_status(
            workload,
            executors,
            per_profile.get(workload_id, {"queued": 0, "running": 0}),
            policy,
            artifact_ready,
            permissions_missing,
        )
        for workload_id, workload in workloads.items()
    }


# Why a profile is in the state it is, as a code rather than a sentence.  The
# applet chooses which remedy to offer from this; it used to choose by matching
# substrings of ``detail``, so rewording a message here silently cost a user
# their remedy.  Every branch below sets one, and the snapshot schema closes the
# enum, so adding a state without a code fails a gate rather than a popup.
PAUSED_BY_POLICY = "paused-by-policy"
PROFILE_DISABLED = "profile-disabled"
NO_MODEL = "no-model"
ARTIFACT_UNAVAILABLE = "artifact-unavailable"
SERVING = "serving"
CONSENT_MISSING = "consent-missing"


def _profile_status(
    workload: Workload,
    executors: dict,
    counts: dict,
    policy: PolicyState,
    artifact_ready: Callable[[Workload], tuple[bool, str]] | None = None,
    permissions_missing: Callable[[Workload], tuple[str, ...]] | None = None,
) -> dict:
    queued = counts["queued"]
    profile_policy = policy.profiles.get(workload.id)
    if policy.paused:
        return {
            "status": "paused",
            "queued": queued,
            "detail": "Runtime paused by policy",
            "reason": PAUSED_BY_POLICY,
        }
    if profile_policy is not None and not profile_policy.enabled:
        return {
            "status": "paused",
            "queued": queued,
            "detail": "Profile disabled by policy",
            "reason": PROFILE_DISABLED,
        }
    choice = select_backend(workload, executors)
    if choice.backend is None:
        return {
            "status": "unavailable",
            "queued": queued,
            "detail": choice.reason[:240],
            "reason": choice.code,
        }
    if not workload.models:
        return {
            "status": "idle",
            "queued": queued,
            "detail": f"Ready on {choice.backend}; no model bundled",
            "reason": NO_MODEL,
        }
    ungranted = permissions_missing(workload) if permissions_missing is not None else ()
    if ungranted:
        # The gate refuses this profile's jobs, so saying "serving" would
        # promise something the first job would be refused for — and the
        # remedy is a command, which is worth naming where a user reads it.
        return {
            "status": "unavailable",
            "queued": queued,
            "detail": f"needs consent for {', '.join(ungranted)}"[:240],
            "reason": CONSENT_MISSING,
        }
    if artifact_ready is not None:
        ready, reason = artifact_ready(workload)
        if not ready:
            # Declaring a model is not the same as having one.  Reporting
            # "serving" here told a user a profile was working on every host
            # where the manifest shipped but the artifact had not been
            # installed — which is every fresh checkout.
            return {
                "status": "unavailable",
                "queued": queued,
                "detail": f"{choice.backend}: {reason}"[:240],
                "reason": ARTIFACT_UNAVAILABLE,
            }
    return {
        "status": "running" if counts["running"] > 0 else "watching",
        "queued": queued,
        "detail": f"Serving on {choice.backend}",
        "reason": SERVING,
    }


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
    ):
        self._callers = CallerIdentityResolver()
        host = build_host_ports(
            snapshot_path=snapshot_path,
            callers=self._callers,
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
        self._artifact_root_path = artifact_root
        self._resolutions: dict[str, tuple] = {}
        self._artifact_store = ArtifactInstaller(artifact_root) if artifact_root else None
        self._plugin_runtime = plugin_runtime or InstalledPluginRuntime(
            bundled_workloads_path(),
            grant_source=self._grants,
            selected_files_root=snapshot_path.parent / "plugin-inputs",
            worker_state_root=snapshot_path.parent / "plugin-state",
            resolve_artifact=self._resolve_plugin_artifact,
            accelerator_devices=self._plugin_accelerator_devices,
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
        self.control = ControlService(storage, on_applied=self._policy_changed)
        self._devices = self._discovery.detect()
        self._executors = build_executors(self._devices)
        self._scheduler = Scheduler(self._executors, self._weight_of, admits=self._admits)
        self._job_dispatcher = _PluginAwareDispatcher(
            job_dispatcher or self._default_dispatcher(), self._plugin_runtime
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
                self._job_dispatcher
                if isinstance(self._job_dispatcher, JobAdmission)
                else None
            ),
            observer=TelemetryJobObserver(self.plugin_telemetry),
        )
        self.runtime_api = RuntimeAPI(
            self.control, self.jobs, self._describe_plugins, self._callers
        )
        self.result_summaries = result_summaries or ResultSummaryRegistry(
            alert_id_factory=lambda: f"alert-{secrets.token_hex(16)}"
        )
        self._snapshot_retracted = False
        self._stopping = asyncio.Event()

    def _device_load(self, device, stats) -> float | None:
        # Prefer the kernel's own utilization counter (covers every consumer
        # of the device); fall back to the scheduler's inference busy EMA.
        utilization = self._discovery.utilization(device)
        return utilization if utilization is not None else stats["loads"].get(device.backend)

    def _weight_of(self, workload_id: str) -> int:
        policy = self.control.state.profiles.get(workload_id)
        return policy.weight if policy is not None else 1

    def _admits(self, workload_id: str) -> bool:
        """Dispatch admission: nothing runs while the runtime is paused, and a
        disabled profile's jobs stay queued until it is enabled again."""
        state = self.control.state
        if state.paused:
            return False
        policy = state.profiles.get(workload_id)
        return policy is None or policy.enabled

    def _job_authorized(self, action: str, workload_id: str) -> bool:
        plugin_ids = (
            self._plugin_runtime.plugin_ids()
            if isinstance(self._plugin_runtime, PluginIdentitySource)
            else ()
        )
        known_workload = workload_id in self._workloads or workload_id in plugin_ids
        if not known_workload:
            return False
        return action == "cancel" or self._admits(workload_id)

    def _policy_changed(self) -> None:
        """Applied policy commands wake the workers so held jobs re-evaluate."""
        self._scheduler.kick()

    def _default_dispatcher(self) -> JobDispatcher:
        """Route inference only when an artifact store is configured.

        Without one there is nothing that can prove a model file is the model
        the manifest declares, so the fail-closed dispatcher stays in place
        rather than executing an unverified file.
        """
        if self._artifact_store is None:
            return UnavailableJobDispatcher()
        return InferenceJobDispatcher(
            self._workloads,
            self._scheduler,
            lambda: self._executors,
            self._artifact_store,
            # Referencing a file is a capability the operator grants, so the
            # roots come from configuration and default to none.
            input_roots=OptedInInputRoots(self._input_roots),
        )

    def _build_runners(self) -> RunnerSet:
        """One pipeline runner per profile that can actually execute.

        Built here rather than lazily so a profile that cannot run says so at
        startup, in one place, instead of at the first job a user submits.
        """
        return build_plugin_runners(
            self._workloads,
            dispatcher=self._job_dispatcher,
            resolve_artifact=self._resolve_artifact,
            encode_result=inference_result_payload,
            cancellations=self._cancellations,
            is_paused=lambda: self.control.state.paused,
            is_enabled=self._admits,
            allows_permission=self._permitted,
            # Without a delivery sink the final stage is a no-op, so a job that
            # reached DELIVER left nothing a caller could collect.
            deliver=self._deliver_job_output,
            progress=self._note_job_progress,
        )

    def _profile_permissions_missing(self, workload: Workload) -> tuple[str, ...]:
        """Permissions this profile declares and has not been granted.

        Reported in the snapshot rather than only refused at job time: a
        profile whose jobs will all be refused should say so before somebody
        submits one, and the remedy — a command — is worth naming where it is
        read.
        """
        declared = required_permissions(workload)
        if not declared:
            return ()
        self._grants.reload()
        return tuple(
            permission
            for permission in declared
            if not self._grants.is_granted(workload.id, permission, set(declared))
        )

    def _permitted(self, profile_id: str, permission: str) -> bool:
        """Whether *this* profile holds an active grant for this permission.

        Asked of the profile rather than of the catalogue.  Answering "does any
        installed plugin hold a grant for this permission name" made consent
        transitive: a grant a user issued to one plugin satisfied every other
        plugin's gate for the same string, which is the opposite of what
        granting one thing means.

        Re-read first: the ledger in memory is a snapshot taken at
        construction, so a grant revoked by the CLI while a job sits in a queue
        would otherwise stay in force until the service restarted — which is
        the opposite of what revoking means.
        """
        workload = self._workloads.get(profile_id)
        if workload is None:
            return False
        declared = set(required_permissions(workload))
        if permission not in declared:
            # Never declared, so never grantable: the manifest bounds what
            # consent can be given for.
            return False
        self._grants.reload()
        return self._grants.is_granted(profile_id, permission, declared)

    def _note_job_progress(self, job_id: str, stage: str, fraction: float, detail: str) -> None:
        # Late-bound: runners are built before the job service they report to.
        self.jobs.note_progress(job_id, stage, fraction, detail)

    def _deliver_job_output(self, job_id: str, output: dict) -> None:
        self.jobs.note_progress(job_id, "deliver", 1.0, "Result delivered")
        self._summarize(job_id, output)

    def _summarize(self, job_id: str, output: dict) -> None:
        """Publish what a finished job found, so a desktop can see it happened.

        The snapshot is the only surface the applet reads, and nothing ever
        wrote to this registry, so every completed job was invisible there.
        Only a reading is summarised: a thousand raw scores have no sentence in
        them, and publishing "a job finished" for each of those would fill the
        surface with rows carrying no information.
        """
        reading = output.get("reading") if isinstance(output, dict) else None
        profile_id = output.get("profileId") if isinstance(output, dict) else None
        if not isinstance(reading, dict) or not isinstance(profile_id, str):
            return
        forecast = parse_forecast_reading(reading)
        if forecast is not None:
            self._publish_forecast_summary(job_id, profile_id, forecast)
            return
        top = reading.get("top") or []
        if not top:
            return
        best = top[0]
        named = best.get("label") or f"class {best.get('index')}"
        score = best.get("score")
        try:
            self.result_summaries.publish(
                plugin_id=profile_id,
                title=str(named)[:120],
                summary=f"{len(top)} candidates, best {score:.3f}"
                if isinstance(score, (int, float))
                else f"{len(top)} candidates",
                timestamp_ms=max(1, int(time.time() * 1000)),
                # Not a risk and not a probability the runtime can vouch for: a
                # softmax score is the model's confidence in its own ranking,
                # which is a different claim from the one these fields make.
                confidence=None,
                risk_score=None,
                result_reference=f"result-{job_id}",
                redactor=SecretRedactor(()),
            )
        except (SummaryError, TypeError, ValueError) as error:
            LOGGER.debug("result summary not published for %s: %s", job_id, error)

    def _publish_forecast_summary(self, job_id: str, profile_id: str, reading: dict) -> None:
        """Publish an advisory reading without inventing units or certainty."""
        try:
            self.result_summaries.publish(
                plugin_id=profile_id,
                title=f"{reading['targetFeature']} forecast",
                summary=f"{reading['horizon']} observations ahead: {reading['value']:g}",
                timestamp_ms=max(1, int(time.time() * 1000)),
                confidence=None,
                risk_score=None,
                result_reference=f"result-{job_id}",
                redactor=SecretRedactor(()),
            )
        except (SummaryError, TypeError, ValueError) as error:
            LOGGER.debug("forecast summary not published for %s: %s", job_id, error)

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
        """Answer DescribePlugins from installed metadata only.

        Building this from the discovery snapshot means it never imports a
        plugin module: asking what a plugin requires must not run its code.
        """
        if not isinstance(self._plugin_runtime, PluginSnapshotSource):
            return _no_inventory()
        snapshot = self._plugin_runtime.snapshot
        if not isinstance(snapshot, PluginRuntimeSnapshot) or not isinstance(
            snapshot.catalog, PluginCatalogSnapshot
        ):
            return _no_inventory()
        states = {status.plugin_id: str(status.state) for status in snapshot.workers}
        granted_permissions = (
            self._plugin_runtime.granted_permissions
            if isinstance(self._plugin_runtime, PluginPermissionSource)
            else lambda _plugin_id: ()
        )
        document = build_plugin_inventory(
            snapshot.catalog.plugins,
            resolve_artifact=self._resolve_artifact,
            # Without this every permission reports granted:false, so a user
            # reading the inventory sees a plugin as unauthorised while its
            # worker is running with the grant.
            granted_permissions=granted_permissions,
            worker_states=states.get,
            generated_at_ms=int(time.time() * 1000),
        )
        return json.dumps(document, separators=(",", ":"))

    def _profile_artifact_ready(self, workload: Workload) -> tuple[bool, str]:
        """Whether the model this profile would actually run is installed.

        Asked of the lane dispatch would choose, not of the first entry the
        manifest happens to list.  A profile declaring one model per
        accelerator has as many answers as it has entries, and reading only
        the first reported unavailable for a profile the runtime would serve —
        or serving for one whose chosen lane refuses — which is the
        snapshot-versus-dispatch disagreement this resolver exists to prevent.
        """
        if not workload.models:
            return True, ""
        model = runnable_model(workload, self._selected_backend(workload), self._executors)
        if model is None:
            # No lane can run any declared model; `select_backend` already
            # reports that as the profile's status, so readiness stays quiet
            # rather than adding a second, vaguer reason for the same fact.
            return True, ""
        if not model.get("sha256"):
            # Said here as well as at dispatch, so a profile that can never run
            # says why in the snapshot instead of looking installable and
            # refusing the first job somebody submits.
            return False, (
                "the manifest declares this model without a sha256, so the runtime "
                "cannot tell which file it means"
            )
        resolution = self._resolve_artifact(model["id"])
        if getattr(resolution, "ready", False):
            return True, ""
        return False, getattr(resolution, "reason", "") or "model artifact is not installed"

    def _selected_backend(self, workload: Workload) -> str:
        """The lane dispatch would take for this profile, or its first choice.

        Falling back to the preference head keeps readiness answerable on a
        host where nothing is available: the profile is unavailable for that
        reason anyway, and asking about a lane is better than asking about
        none.
        """
        choice = select_backend(workload, self._executors)
        return choice.backend or (workload.preference[0] if workload.preference else "")

    def _resolve_artifact(self, artifact_id: str) -> ArtifactResolution:
        """Report readiness exactly as dispatch would decide it.

        Resolving by id alone would call an artifact ready that dispatch then
        refuses for a digest mismatch, so the inventory would tell a user the
        opposite of what the runtime does.  When a manifest declares a digest,
        that is what is checked.
        """
        if self._artifact_store is None:
            return ArtifactResolution(
                False, None, "no artifact store is configured for this service", 0
            )
        reference = self._declared_reference(artifact_id)
        if reference is None:
            return ArtifactResolution(
                False, None, "no manifest declares a sha256 for this artifact", 0
            )
        return self._cached_resolution(artifact_id, reference)

    def _resolve_plugin_artifact(self, reference: ArtifactReference) -> ArtifactResolution:
        if self._artifact_store is None:
            return ArtifactResolution(
                False, None, "no artifact store is configured for this service", 0
            )
        return self._artifact_store.resolve(reference)

    def _plugin_accelerator_devices(self) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for device in self._devices:
            if device.backend == "gpu" and device.kind == "dri":
                paths.setdefault("gpu", Path("/dev/dri") / device.id.removeprefix("gpu-"))
            elif device.backend == "npu" and device.kind == "accel":
                paths.setdefault("npu", Path("/dev/accel") / device.id.removeprefix("npu-"))
        return paths

    def _cached_resolution(self, artifact_id: str, reference) -> ArtifactResolution:
        """Resolve, but not by re-reading every model file twice a second.

        Resolution re-digests the artifact and its companions, and the
        publisher asks for every declared profile on every tick — so a
        catalogue of large models is hashed end to end at the publish
        interval, for an answer that changes only when a file does.

        Keyed on what a change would move: the reference itself plus the
        primary file's size and mtime.  A file replaced in place is a
        different mtime, and one replaced with identical bytes resolves the
        same either way, so the cache cannot report ready for something the
        store would now refuse.
        """
        stamp = self._artifact_stamp(artifact_id, reference)
        cached = self._resolutions.get(artifact_id)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        resolution = self._artifact_store.resolve(reference)
        self._resolutions[artifact_id] = (stamp, resolution)
        return resolution

    def _artifact_stamp(self, artifact_id: str, reference) -> tuple:
        """What must be unchanged for a previous resolution to still hold."""
        try:
            status = (
                Path(self._artifact_root_path)
                / artifact_id
                / reference.version
                / artifact_filename(reference.format)
            ).stat()
        except (OSError, ValueError):
            # Unreadable, absent, or an unsupported format: let the store say
            # so itself rather than caching a guess.
            return None
        return (reference, status.st_size, status.st_mtime_ns)

    def _declared_reference(self, artifact_id: str):
        """The digest a bundled or installed manifest declares for this artifact."""
        for workload in self._workloads.values():
            for model in workload.models:
                if model["id"] != artifact_id:
                    continue
                reference = declared_artifact_reference(workload, model)
                if reference is not None:
                    return reference
        for plugin in self._plugin_runtime.snapshot.catalog.plugins:
            for entry in plugin.manifest["plugin"]["artifacts"]:
                if entry["id"] == artifact_id:
                    return ArtifactReference(
                        entry["id"],
                        entry["version"],
                        entry["format"],
                        entry["sha256"],
                        tuple(sorted((entry.get("companions") or {}).items())),
                    )
        return None

    def _build_runtime_snapshot(self) -> dict:
        self._scheduler.tick()
        stats = self._scheduler.stats()
        devices = [
            dataclasses.replace(device, load=self._device_load(device, stats))
            for device in self._devices
        ]
        return build_snapshot(
            devices=devices,
            metrics=stats,
            profiles=profile_statuses(
                self._workloads,
                self._executors,
                self._scheduler,
                self.control.state,
                self._profile_artifact_ready,
                self._profile_permissions_missing,
            ),
            alerts=self.result_summaries.documents(),
            plugin_telemetry=self.plugin_telemetry.documents(),
            # Published every tick rather than once at startup: the roots are
            # a capability, and a consumer that cached them would keep offering
            # a directory this service has stopped reading.
            inputs=input_roots_document(self._input_roots),
            kernel_telemetry=self._kernel_telemetry_source.read().document(),
        )

    def publish_once(self) -> dict:
        """Synchronously build and publish one snapshot (test/adapter API)."""
        snapshot = self._build_runtime_snapshot()
        self._publisher_port.publish(snapshot)
        return snapshot

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
                    if self._snapshot_retracted:
                        LOGGER.info("Accelerator devices returned; publishing snapshots again")
                        self._snapshot_retracted = False
                    snapshot = self._build_runtime_snapshot()
                    await asyncio.to_thread(self._publisher_port.publish, snapshot)
                elif not self._snapshot_retracted:
                    await asyncio.to_thread(self._publisher_port.retract)
                    self._snapshot_retracted = True
                    LOGGER.warning(
                        "No accelerator devices present; retracted the runtime snapshot"
                        " so readers observe absence instead of stale data",
                    )
            except OSError:
                LOGGER.exception("Could not publish the runtime snapshot; retrying next tick")
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self._publish_interval_s)
            except TimeoutError:
                continue

    async def _rediscover(self) -> None:
        while not self._stopping.is_set():
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self._discovery_interval_s)
            except TimeoutError:
                devices = self._discovery.detect()
                self._executors = build_executors(
                    devices,
                    previous_devices=self._devices,
                    previous_executors=self._executors,
                )
                self._devices = devices
                self._scheduler.update_executors(self._executors)

    async def run(self) -> None:
        try:
            await self._transport.start(self.runtime_api)
            self.reconcile_interrupted_jobs()
            self._scheduler.start()
            loop = asyncio.get_running_loop()
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
            await self._plugin_runtime.stop()
            await self._transport.stop()

    async def _start_plugin_runtime(self) -> None:
        plugin_snapshot = await self._plugin_runtime.start()
        if not isinstance(plugin_snapshot, PluginRuntimeSnapshot) or not isinstance(
            plugin_snapshot.catalog, PluginCatalogSnapshot
        ):
            return
        for plugin in plugin_snapshot.catalog.plugins:
            self.plugin_telemetry.register(plugin.plugin_id)


def main() -> None:  # pragma: no cover - process entry point
    asyncio.run(build_service_from_env().run())
