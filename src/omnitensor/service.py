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
import logging
import secrets
import time
from collections.abc import Sequence
from pathlib import Path

from . import artifact_readiness as artifacts
from . import dispatch_routing as routing
from . import telemetry_observation as observation
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
from .execution import build_executors as build_executors
from .host import FileSnapshotPublisher as FileSnapshotPublisher
from .host import SysfsDeviceDiscovery as SysfsDeviceDiscovery
from .host import build_host_ports
from .job_ports import JobAdmission, JobDispatcher
from .jobs import (
    JobSubmissionService,
    PredicateJobAuthorizer,
)
from .plugins.artifact_installation import ArtifactInstaller
from .plugins.artifacts import ArtifactReference, ArtifactResolution
from .plugins.cancellation import JobCancellationRegistry
from .plugins.grants import GrantLedger
from .plugins.job_results import JobResultStore
from .plugins.kernel_telemetry import UnixSocketAggregateSource
from .plugins.loading import InstalledPluginRuntime
from .plugins.orchestration import (
    RunnerBackedDispatcher,
    RunnerSet,
    with_recovery,
)
from .plugins.summaries import ResultSummaryRegistry
from .plugins.telemetry import PluginTelemetryRegistry
from .ports import (
    ControlTransport,
    DeviceDiscovery,
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
    profile_statuses as profile_statuses,
)
from .registry import Workload, bundled_workloads_path, load_workload_catalog
from .runtime_api import RuntimeAPI as RuntimeAPI
from .scheduler import Scheduler
from .state import PolicyStore
from .telemetry_observation import TelemetryJobObserver as TelemetryJobObserver

LOGGER = logging.getLogger(__name__)

PUBLISH_INTERVAL_S = 2.0
DISCOVERY_INTERVAL_S = 10.0
GRANT_REFRESH_INTERVAL_S = 0.25


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
        self._job_dispatcher = routing.PluginAwareDispatcher(
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
                self._job_dispatcher if isinstance(self._job_dispatcher, JobAdmission) else None
            ),
            observer=TelemetryJobObserver(self.plugin_telemetry),
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
        self._next_grant_refresh = 0.0
        self._stopping = asyncio.Event()

    def _device_load(self, device, stats) -> float | None:
        return observation.device_load(self._discovery, device, stats)

    def _weight_of(self, workload_id: str) -> int:
        return routing.policy_weight(self.control.state, workload_id)

    def _admits(self, workload_id: str) -> bool:
        return routing.policy_admits(self.control.state, workload_id)

    def _job_authorized(self, action: str, workload_id: str) -> bool:
        return routing.job_authorized(
            action,
            workload_id,
            self._workloads,
            self._plugin_runtime,
            self._admits,
        )

    def _policy_changed(self) -> None:
        self._scheduler.kick()

    def _default_dispatcher(self) -> JobDispatcher:
        return routing.default_dispatcher(
            self._workloads,
            self._scheduler,
            lambda: self._executors,
            self._artifact_store,
            self._input_roots,
            resolve_artifact=self._cached_resolution,
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

    def _profile_artifact_ready(self, workload: Workload) -> tuple[bool, str]:
        return artifacts.profile_artifact_ready(
            workload,
            self._executors,
            self._resolve_artifact,
            self._selected_backend,
        )

    def _selected_backend(self, workload: Workload) -> str:
        return artifacts.selected_backend(workload, self._executors)

    def _resolve_artifact(self, artifact_id: str) -> ArtifactResolution:
        return artifacts.resolve_artifact(
            artifact_id,
            self._artifact_store,
            self._declared_reference,
            self._cached_resolution,
        )

    def _resolve_plugin_artifact(
        self,
        reference: ArtifactReference,
    ) -> ArtifactResolution:
        return artifacts.resolve_plugin_artifact(reference, self._artifact_store)

    def _plugin_accelerator_devices(self) -> dict[str, Path]:
        return artifacts.plugin_accelerator_devices(self._devices)

    def _cached_resolution(
        self,
        artifact_id: str,
        reference: ArtifactReference,
    ) -> ArtifactResolution:
        return artifacts.cached_resolution(
            artifact_id,
            reference,
            self._artifact_store,
            self._resolutions,
            self._artifact_stamp,
        )

    def _artifact_stamp(
        self,
        artifact_id: str,
        reference: ArtifactReference,
    ) -> tuple | None:
        return artifacts.artifact_stamp(
            self._artifact_root_path,
            artifact_id,
            reference,
        )

    def _declared_reference(self, artifact_id: str) -> ArtifactReference | None:
        return artifacts.declared_reference(
            artifact_id,
            self._workloads,
            lambda: self._plugin_runtime.snapshot.catalog.plugins,
        )

    def _build_runtime_snapshot(self) -> dict:
        return observation.runtime_snapshot(
            devices=self._devices,
            device_load_of=self._device_load,
            workloads=self._workloads,
            executors=self._executors,
            scheduler=self._scheduler,
            policy=self.control.state,
            artifact_ready=self._profile_artifact_ready,
            permissions_missing=self._profile_permissions_missing,
            result_summaries=self.result_summaries,
            plugin_telemetry=self.plugin_telemetry,
            input_roots=self._input_roots,
            kernel_telemetry_source=self._kernel_telemetry_source,
            profile_statuses_of=profile_statuses,
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
                    await self._refresh_grants()
                    if self._snapshot_retracted:
                        LOGGER.info("Accelerator devices returned; publishing snapshots again")
                        self._snapshot_retracted = False
                    snapshot = await asyncio.to_thread(self._build_runtime_snapshot)
                    await asyncio.to_thread(self._publisher_port.publish, snapshot)
                elif not self._snapshot_retracted:
                    await asyncio.to_thread(self._publisher_port.retract)
                    self._snapshot_retracted = True
                    LOGGER.warning(
                        "No accelerator devices present; retracted the runtime snapshot"
                        " so readers observe absence instead of stale data",
                    )
            except (OSError, ValueError):
                LOGGER.exception("Could not publish the runtime snapshot; retrying next tick")
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self._publish_interval_s)
            except TimeoutError:
                continue

    async def _refresh_grants(self) -> None:
        now = asyncio.get_running_loop().time()
        if now < self._next_grant_refresh:
            return
        await asyncio.to_thread(self._grants.reload)
        self._next_grant_refresh = now + GRANT_REFRESH_INTERVAL_S

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
        observation.register_plugin_telemetry(
            plugin_snapshot,
            self.plugin_telemetry,
        )


def main() -> None:  # pragma: no cover - process entry point
    asyncio.run(build_service_from_env().run())
