"""OmniTensor service wiring: discovery, scheduling, publishing, control.

Entry point ``omnitensor`` runs an asyncio loop that
- rediscovers accelerators every ``DISCOVERY_INTERVAL_S``,
- publishes a schema-valid snapshot atomically every ``PUBLISH_INTERVAL_S``,
- owns ``org.cinnamon.OmniTensor1`` on the session bus and answers
  ``ApplyCommand``, ``SubmitJob``, and ``CancelJob`` through a versioned facade.

:class:`OmniTensorService` depends on the ports in :mod:`omnitensor.ports`;
the filesystem, sysfs, and D-Bus adapters defined here are only the default
wiring and can be replaced through constructor injection.

Configuration comes from environment variables:
``OMNITENSOR_STATE_PATH`` (snapshot the applet reads),
``OMNITENSOR_POLICY_PATH`` (persisted policy + revision),
``OMNITENSOR_WORKLOADS`` (manifest directory),
``OMNITENSOR_ARTIFACT_ROOT`` (verified model artifact store).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import secrets
import time
from collections.abc import Callable
from pathlib import Path

from dbus_fast import BusType, RequestNameReply
from dbus_fast.aio import MessageBus
from dbus_fast.service import ServiceInterface, method

from .callers import CallerIdentityResolver, caller_capture_handler, unix_user_lookup
from .control import ControlService
from .discovery import BACKENDS, Device, DiscoveryPaths, detect_devices, device_utilization
from .dispatch import (
    InferenceJobDispatcher,
    declared_artifact_reference,
    inference_result_payload,
)
from .executors.gpu import CompositeGpuExecutor, GpuExecutor
from .executors.npu import NpuExecutor
from .executors.tpu import TpuExecutor
from .executors.vulkan import VulkanGpuExecutor
from .guard import BusGuard, GuardRefusedError, guarded
from .inspection import PLUGIN_INVENTORY_VERSION, build_plugin_inventory
from .jobs import (
    JobDispatcher,
    JobSubmissionService,
    PredicateJobAuthorizer,
    UnavailableJobDispatcher,
)
from .plugins.artifact_installation import ArtifactInstaller
from .plugins.artifacts import ArtifactReference, ArtifactResolution
from .plugins.cancellation import JobCancellationRegistry
from .plugins.loading import InstalledPluginRuntime
from .plugins.orchestration import (
    RunnerBackedDispatcher,
    RunnerSet,
    build_plugin_runners,
    with_recovery,
)
from .plugins.results import JobResultStore
from .plugins.summaries import ResultSummaryRegistry
from .plugins.telemetry import PluginTelemetryRegistry
from .ports import (
    ControlTransport,
    DeviceDiscovery,
    PluginRuntime,
    PolicyStorage,
    RuntimeHandler,
    SnapshotPublisher,
)
from .registry import Workload, bundled_workloads_path, load_workload_catalog
from .scheduler import Scheduler, pick_backend
from .snapshot import build_snapshot, remove_snapshot, write_snapshot
from .state import PolicyState, PolicyStore

LOGGER = logging.getLogger(__name__)

BUS_NAME = "org.cinnamon.OmniTensor1"
OBJECT_PATH = "/org/cinnamon/OmniTensor1"
PUBLISH_INTERVAL_S = 2.0
DISCOVERY_INTERVAL_S = 10.0

DEFAULT_STATE_PATH = "~/.local/state/tpu-workload-manager/state.json"
DEFAULT_POLICY_PATH = "~/.local/state/omnitensor/policy.json"
DEFAULT_WORKLOADS_PATH = "~/.local/share/omnitensor/workloads"
DEFAULT_ARTIFACT_ROOT = "~/.local/share/omnitensor/artifacts"


def _no_inventory() -> str:
    """Fail closed: an unwired inspector reports an empty inventory, not an error."""
    return json.dumps(
        {"version": PLUGIN_INVENTORY_VERSION, "generatedAt": 1, "plugins": []},
        separators=(",", ":"),
    )


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


class OmniTensorInterface(ServiceInterface):
    """Transport-only D-Bus shim; all logic stays in the injected handler."""

    def __init__(self, runtime: RuntimeAPI):
        super().__init__(BUS_NAME)
        self._runtime = runtime

    @method()
    async def ApplyCommand(self, command: s) -> s:  # noqa: F821, N802 - D-Bus contract names
        return await self._runtime.apply_command_text(command)

    @method()
    async def SubmitJob(self, request: s) -> s:  # noqa: F821, N802 - D-Bus contract names
        return await self._runtime.submit_job_text(request)

    @method()
    async def CancelJob(self, request: s) -> s:  # noqa: F821, N802 - D-Bus contract names
        return await self._runtime.cancel_job_text(request)

    @method()
    async def GetJobResult(self, request: s) -> s:  # noqa: F821, N802 - D-Bus contract names
        return await self._runtime.job_result_text(request)

    @method()
    def DescribePlugins(self) -> s:  # noqa: F821, N802 - D-Bus contract names
        return self._runtime.describe_plugins_text()


class SysfsDeviceDiscovery:
    """:class:`~omnitensor.ports.DeviceDiscovery` over kernel device nodes."""

    def __init__(self, paths: DiscoveryPaths | None = None):
        self._paths = paths or DiscoveryPaths()

    def detect(self) -> list[Device]:
        return detect_devices(self._paths)

    def utilization(self, device: Device) -> float | None:
        return device_utilization(self._paths, device)


class FileSnapshotPublisher:
    """:class:`~omnitensor.ports.SnapshotPublisher` writing atomically to a path."""

    def __init__(self, path: Path):
        self._path = path

    def publish(self, snapshot: dict) -> None:
        write_snapshot(self._path, snapshot)

    def retract(self) -> None:
        remove_snapshot(self._path)


class DbusControlTransport:
    """:class:`~omnitensor.ports.ControlTransport` over the session bus."""

    def __init__(
        self,
        bus_type: BusType = BusType.SESSION,
        *,
        bus_factory=None,
        bus_name: str = BUS_NAME,
        callers: CallerIdentityResolver | None = None,
    ):
        self._bus_type = bus_type
        self._bus_factory = bus_factory
        self._bus_name = bus_name
        self._callers = callers
        self._bus = None

    async def start(self, handler: RuntimeHandler) -> None:
        if self._bus_factory is not None:
            bus = await self._bus_factory()
        else:
            bus = await MessageBus(bus_type=self._bus_type).connect()
        try:
            # Installed before the interface is exported so no message can be
            # dispatched with an unbound — that is, inherited — sender.
            bus.add_message_handler(caller_capture_handler())
            if self._callers is not None:
                self._callers.attach(unix_user_lookup(bus))
            bus.export(OBJECT_PATH, OmniTensorInterface(handler))
            reply = await bus.request_name(self._bus_name)
            if reply != RequestNameReply.PRIMARY_OWNER:
                # Without primary ownership another instance is serving the
                # bus name and publishing to the same snapshot path; running
                # anyway would double-write. Fail startup loudly instead.
                raise RuntimeError(
                    f"{self._bus_name} is already owned"
                    f" (request_name reply: {reply.name});"
                    " another omnitensor instance is running",
                )
        except BaseException:
            bus.disconnect()
            raise
        self._bus = bus

    async def stop(self) -> None:
        if self._bus is not None:
            self._bus.disconnect()
            self._bus = None


def _build_executor(backend: str, device_present: bool):
    if backend == "tpu":
        return TpuExecutor(device_present)
    if backend == "npu":
        return NpuExecutor(device_present)
    return CompositeGpuExecutor([
        # Vulkan (ncnn) first so any Mesa/RADV/ANV driver serves GPU work;
        # ONNX Runtime with CUDA/ROCm providers is the optional second lane.
        VulkanGpuExecutor(device_present),
        GpuExecutor(device_present),
    ])


def build_executors(
    devices,
    *,
    previous_devices=None,
    previous_executors: dict | None = None,
) -> dict:
    """Build changed backend adapters and preserve unchanged runtime state."""
    current = {device.backend: device for device in devices}
    previous = {device.backend: device for device in previous_devices or []}
    existing = previous_executors or {}
    can_reuse = previous_devices is not None and previous_executors is not None
    return {
        backend: existing[backend]
        if can_reuse and backend in existing and previous.get(backend) == current.get(backend)
        else _build_executor(backend, backend in current)
        for backend in BACKENDS
    }


def profile_statuses(
    workloads: dict[str, Workload],
    executors: dict,
    scheduler: Scheduler,
    policy: PolicyState,
) -> dict[str, dict]:
    """Runtime status per profile for the snapshot document."""
    per_profile = scheduler.profile_stats()
    return {
        workload_id: _profile_status(
            workload,
            executors,
            per_profile.get(workload_id, {"queued": 0, "running": 0}),
            policy,
        )
        for workload_id, workload in workloads.items()
    }


def _profile_status(workload: Workload, executors: dict, counts: dict, policy: PolicyState) -> dict:
    queued = counts["queued"]
    profile_policy = policy.profiles.get(workload.id)
    if policy.paused:
        return {"status": "paused", "queued": queued, "detail": "Runtime paused by policy"}
    if profile_policy is not None and not profile_policy.enabled:
        return {"status": "paused", "queued": queued, "detail": "Profile disabled by policy"}
    backend, reason = pick_backend(workload, executors)
    if backend is None:
        return {"status": "unavailable", "queued": queued, "detail": reason[:240]}
    if workload.model is None:
        detail = f"Ready on {backend}; no model bundled"
        return {"status": "idle", "queued": queued, "detail": detail}
    return {
        "status": "running" if counts["running"] > 0 else "watching",
        "queued": queued,
        "detail": f"Serving on {backend}",
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
        result_summaries: ResultSummaryRegistry | None = None,
        job_results: JobResultStore | None = None,
        plugin_telemetry: PluginTelemetryRegistry | None = None,
        transport: ControlTransport | None = None,
        cancellation_journal_path: Path | None = None,
        publish_interval_s: float = PUBLISH_INTERVAL_S,
        discovery_interval_s: float = DISCOVERY_INTERVAL_S,
    ):
        self._discovery = discovery or SysfsDeviceDiscovery(discovery_paths)
        self._publisher_port = publisher or FileSnapshotPublisher(snapshot_path)
        self._callers = CallerIdentityResolver()
        self._transport = transport or DbusControlTransport(callers=self._callers)
        self._plugin_runtime = plugin_runtime or InstalledPluginRuntime(
            bundled_workloads_path()
        )
        self._publish_interval_s = publish_interval_s
        self._discovery_interval_s = discovery_interval_s
        self._workloads = load_workload_catalog(workloads_path)
        defaults = {
            workload_id: workload.default_policy()
            for workload_id, workload in self._workloads.items()
        }
        storage = policy_storage or PolicyStore(policy_path, defaults)
        self.control = ControlService(storage, on_applied=self._policy_changed)
        self._devices = self._discovery.detect()
        self._executors = build_executors(self._devices)
        self._scheduler = Scheduler(self._executors, self._weight_of, admits=self._admits)
        self._artifact_store = ArtifactInstaller(artifact_root) if artifact_root else None
        self._job_dispatcher = job_dispatcher or self._default_dispatcher()
        # Constructed here because submission answers with an acceptance: if no
        # store outlives the call, the outcome has nowhere to be kept and a
        # caller can never learn what happened to the job it submitted.
        self.job_results = job_results or JobResultStore()
        self._cancellations = JobCancellationRegistry(
            cancellation_journal_path or (snapshot_path.parent / "cancellations.json")
        )
        self.runners = self._build_runners()
        self.jobs = JobSubmissionService(
            # Routed through the runners, so a profile that has a pipeline runs
            # its pipeline; without this the runners were built and never used.
            RunnerBackedDispatcher(self.runners, self._job_dispatcher),
            PredicateJobAuthorizer(self._job_authorized),
            results=self.job_results,
        )
        self.runtime_api = RuntimeAPI(
            self.control, self.jobs, self._describe_plugins, self._callers
        )
        self.result_summaries = result_summaries or ResultSummaryRegistry(
            alert_id_factory=lambda: f"alert-{secrets.token_hex(16)}"
        )
        self.plugin_telemetry = plugin_telemetry or PluginTelemetryRegistry()
        for workload in self._workloads.values():
            if workload.manifest["manifestVersion"] >= 2:
                self.plugin_telemetry.register(workload.id)
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
        if workload_id not in self._workloads:
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
            self._workloads, self._scheduler, self._executors, self._artifact_store
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
            # Without a delivery sink the final stage is a no-op, so a job that
            # reached DELIVER left nothing a caller could collect.
            deliver=self._deliver_job_output,
            progress=self._note_job_progress,
        )

    def _note_job_progress(self, job_id: str, stage: str, fraction: float, detail: str) -> None:
        # Late-bound: runners are built before the job service they report to.
        self.jobs.note_progress(job_id, stage, fraction, detail)

    def _deliver_job_output(self, job_id: str, output: dict) -> None:
        self.jobs.note_progress(job_id, "deliver", 1.0, "Result delivered")

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
        snapshot = self._plugin_runtime.snapshot
        states = {status.plugin_id: str(status.state) for status in snapshot.workers}
        document = build_plugin_inventory(
            snapshot.catalog.plugins,
            resolve_artifact=self._resolve_artifact,
            # Without this every permission reports granted:false, so a user
            # reading the inventory sees a plugin as unauthorised while its
            # worker is running with the grant.
            granted_permissions=getattr(
                self._plugin_runtime, "granted_permissions", lambda _plugin_id: ()
            ),
            worker_states=states.get,
            generated_at_ms=int(time.time() * 1000),
        )
        return json.dumps(document, separators=(",", ":"))

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
        if reference is not None:
            return self._artifact_store.resolve(reference)
        return self._artifact_store.resolve_active(artifact_id)

    def _declared_reference(self, artifact_id: str):
        """The digest a bundled or installed manifest declares for this artifact."""
        for workload in self._workloads.values():
            model = workload.model
            if model is not None and model["id"] == artifact_id:
                reference = declared_artifact_reference(workload, model)
                if reference is not None:
                    return reference
        for plugin in self._plugin_runtime.snapshot.catalog.plugins:
            for entry in plugin.manifest["plugin"]["artifacts"]:
                if entry["id"] == artifact_id:
                    return ArtifactReference(
                        entry["id"], entry["version"], entry["format"], entry["sha256"]
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
                self._workloads, self._executors, self._scheduler, self.control.state,
            ),
            alerts=self.result_summaries.documents(),
            plugin_telemetry=self.plugin_telemetry.documents(),
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
        await self._transport.start(self.runtime_api)
        try:
            self.reconcile_interrupted_jobs()
            await self._plugin_runtime.start()
            self._scheduler.start()
            loop = asyncio.get_running_loop()
            tasks = [
                loop.create_task(self._publisher()),
                loop.create_task(self._rediscover()),
            ]
            try:
                # gather raises on the first loop failure; the finally block then
                # cancels and awaits the sibling, so one crashing loop can never
                # leave the other running as an orphan — the service crashes
                # loudly as a whole.
                await asyncio.gather(*tasks)
            finally:
                self._stopping.set()
                for task in tasks:
                    task.cancel()
                await asyncio.wait(tasks)
                await self._scheduler.stop()
        finally:
            await self.jobs.stop()
            await self._plugin_runtime.stop()
            await self._transport.stop()


def _env_path(name: str, fallback: str) -> Path:
    return Path(os.environ.get(name, fallback)).expanduser()


def build_service_from_env() -> OmniTensorService:
    """The production service instance, configured from the environment."""
    return OmniTensorService(
        snapshot_path=_env_path("OMNITENSOR_STATE_PATH", DEFAULT_STATE_PATH),
        policy_path=_env_path("OMNITENSOR_POLICY_PATH", DEFAULT_POLICY_PATH),
        workloads_path=_env_path("OMNITENSOR_WORKLOADS", DEFAULT_WORKLOADS_PATH),
        artifact_root=_env_path("OMNITENSOR_ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT),
    )


def main() -> None:  # pragma: no cover - process entry point
    asyncio.run(build_service_from_env().run())
