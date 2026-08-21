"""Turn identity-validated installed plugins into supervised workers."""

from __future__ import annotations

import asyncio
import copy
import functools
import logging
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Protocol, runtime_checkable

import jsonschema

from ..job_ports import JobDispatchError
from . import loading_accelerator as _accelerator
from . import loading_staging as _staging
from . import worker_specs as _specs
from .artifacts import ArtifactReference as _ArtifactReference
from .artifacts import ArtifactResolution as _ArtifactResolution
from .configuration_spec import manifest_configuration_spec
from .discovery import PluginSource, discover_plugin_metadata
from .identity import PluginCatalog, ResolvedPlugin, resolve_plugin_identities
from .manifest_compatibility import resolve_plugin_compatibility
from .offloop import run_off_loop as _run_off_loop
from .protocol import JsonObject, PluginProgress, PluginRequest, PluginResultStatus
from .sandbox import SELECTED_FILES_PERMISSION
from .settings import (
    PluginConfigurationSpec,
    PluginSettings,
    PluginSettingsError,
    PluginSettingsStore,
)
from .supervisor import PluginWorkerSupervisor
from .supervisor_diagnostics import WorkerState, WorkerStatus
from .supervisor_process import AsyncioSubprocessLauncher
from .supervisor_process import WorkerSpec as _WorkerSpec
from .supervisor_session import PluginWorkerError

ArtifactReference = _ArtifactReference
ArtifactResolution = _ArtifactResolution
WorkerSpec = _WorkerSpec

MAX_WORKER_IMPORT_PATHS = _specs.MAX_WORKER_IMPORT_PATHS
WORKER_CAPABILITIES = _specs.WORKER_CAPABILITIES
MAX_SELECTED_SOURCES = _staging.MAX_SELECTED_SOURCES
MAX_SELECTED_SOURCE_BYTES = _staging.MAX_SELECTED_SOURCE_BYTES
_STAGING_COMPONENT = _staging._STAGING_COMPONENT
_ACCELERATOR_PERMISSIONS = _accelerator.ACCELERATOR_PERMISSIONS
_VULKAN_METADATA_ROOTS = _accelerator.VULKAN_METADATA_ROOTS
_TIMEZONE_METADATA_ROOT = _specs.TIMEZONE_METADATA_ROOT
_SYS_CHAR_ROOT = _accelerator.SYS_CHAR_ROOT
_SYS_DEVICES_ROOT = _accelerator.SYS_DEVICES_ROOT

ArtifactProvider = _specs.ArtifactProvider
external_worker_specs = _specs.external_worker_specs
_external_worker_spec = _specs.external_worker_spec
_worker_argv = _specs.worker_argv
_artifact_bootstrap = _specs.artifact_bootstrap
_artifact_paths = _specs.artifact_paths
_prepare_selected_files_root = _staging.prepare_selected_files_root
_prepare_worker_state_root = _specs.prepare_worker_state_root
_plugin_state_path = _specs.plugin_state_path
_accelerator_lease_path = _accelerator.accelerator_lease_path
_resolved_artifacts = _specs.resolved_artifacts
_accelerator_paths = _accelerator.accelerator_paths
_vulkan_sysfs_resources = _accelerator.vulkan_sysfs_resources
_staged_source_name = _staging.staged_source_name
_canonical_selected_source = _staging.canonical_selected_source
_open_selected_source = _staging.open_selected_source
_selected_source_changed = _staging.selected_source_changed
entry_points_from_distributions = _specs.entry_points_from_distributions
_executable = _specs.executable_path
_import_paths = _specs.worker_import_paths_tuple
_trusted_runtime_paths = _specs.trusted_runtime_paths
LOGGER = logging.getLogger(__name__)
GRANT_REFRESH_SECONDS = 0.25
# How long a worker with nothing to do keeps its process, and its model, alive.
# Long enough that a person working through a document does not pay the model
# load again between questions; short enough that a machine left alone gets its
# memory and its accelerator lease back.  It bounds nothing a caller is told:
# the next request starts the worker again and is answered in full.
WORKER_IDLE_TIMEOUT_S = 300.0
IDLE_EXIT_DETAIL = "worker stopped after being idle; starts on demand"


async def _cleanup_abandoned_staging(staging: asyncio.Task) -> None:
    try:
        _worker_payload, abandoned = await asyncio.shield(staging)
    except BaseException:  # noqa: BLE001 - containment: this must not escape into the caller
        return
    await _run_off_loop(shutil.rmtree, abandoned, True)


class PermissionGrantSource(Protocol):
    def active_permissions(
        self,
        plugin_id: str,
        declared_permissions: set[str],
    ) -> frozenset[str]: ...


@runtime_checkable
class ReloadablePermissionGrantSource(Protocol):
    """Optional grant source capability for refreshing persisted consent."""

    def reload(self) -> object: ...


# How long a job waits for a worker that is starting or restarting before it
# is told the worker is not there. Long enough for the largest installed model
# to load onto the GPU, which is what the wait is actually for; a worker that
# has failed is refused at once rather than at the end of this.
WORKER_READY_WAIT_SECONDS = 120.0
# States — and the absence of a state — that a moment's patience resolves.
# Anything else is a verdict, and waiting on one only delays the news.
_COMING_UP = frozenset({WorkerState.STARTING, WorkerState.RESTARTING})
_READY_POLL_SECONDS = 0.05


@runtime_checkable
class OnDemandWorkerSupervisor(Protocol):
    """Optional supervisor capability: launch and stop one worker at a time.

    A supervisor without it is started eagerly, exactly as before; one with it
    lets an idle machine hold no plugin processes at all.
    """

    async def register(self, specs) -> object: ...

    async def ensure_ready(self, plugin_id: str) -> object: ...

    async def stop_worker(self, plugin_id: str, detail: str) -> object: ...


@runtime_checkable
class WorkerRevoker(Protocol):
    """Optional supervisor capability for stopping one compromised worker."""

    async def revoke(self, plugin_id: str, detail: str) -> None: ...


class _DenyAllGrants:
    def active_permissions(
        self,
        plugin_id: str,
        declared_permissions: set[str],
    ) -> frozenset[str]:
        return frozenset()


@dataclass(frozen=True, slots=True)
class InstalledPluginSnapshot:
    """One bounded discovery/start result retained for inspection."""

    catalog: PluginCatalog
    workers: tuple[WorkerStatus, ...]


class InstalledPluginRuntime:
    """Discover external distributions and supervise their isolated workers."""

    def __init__(
        self,
        bundled_root: Path,
        supervisor: PluginWorkerSupervisor | None = None,
        *,
        entry_points_provider: Callable[..., Iterable] = metadata.entry_points,
        python_executable: str | Path = sys.executable,
        worker_import_paths: Sequence[Path] = (),
        grant_source: PermissionGrantSource | None = None,
        progress_sink: Callable[[PluginProgress], None] | None = None,
        clock_ms: Callable[[], int] | None = None,
        selected_files_root: Path | None = None,
        worker_state_root: Path | None = None,
        resolve_artifact: ArtifactProvider | None = None,
        accelerator_devices: Callable[[], Mapping[str, Path]] | None = None,
        profile_accelerator_devices: Callable[[str], Mapping[str, Path]] | None = None,
        profile_model_choice: Callable[[str], str | None] | None = None,
        settings_store: PluginSettingsStore | None = None,
        worker_idle_timeout_s: float | None = WORKER_IDLE_TIMEOUT_S,
        sleep: Callable[[float], object] | None = None,
    ) -> None:
        self._bundled_root = Path(bundled_root)
        # Workers run unbounded: no delegated cgroup, no CPU or memory ceiling.
        self._supervisor = supervisor or PluginWorkerSupervisor(AsyncioSubprocessLauncher())
        self._entry_points_provider = entry_points_provider
        self._python_executable = _executable(python_executable)
        self._worker_import_paths = _import_paths(worker_import_paths)
        self._grant_source = grant_source or _DenyAllGrants()
        self._progress_sink = progress_sink
        self._clock_ms = clock_ms or (lambda: max(1, int(time.time() * 1000)))
        self._selected_files_root = _prepare_selected_files_root(selected_files_root)
        self._worker_state_root = _prepare_worker_state_root(worker_state_root)
        self._resolve_artifact = resolve_artifact
        self._accelerator_devices = accelerator_devices or (lambda: {})
        self._profile_accelerator_devices = profile_accelerator_devices
        # Which model each workload was told to run, asked of the host rather
        # than stored, so a worker started after a change gets the change.
        self._profile_model_choice = profile_model_choice
        # Where a workload's tuning is held. Read here rather than handed in
        # by the host: the spec it is read against comes from this runtime's
        # own manifests, so a host callback for the values could only ask this
        # object for the contract and then answer it back.
        self._settings_store = settings_store
        self._granted: dict[str, frozenset[str]] = {}
        self._revoked_workers: set[str] = set()
        self._snapshot = InstalledPluginSnapshot(PluginCatalog((), ()), ())
        self._grant_monitor: asyncio.Task | None = None
        self._grant_changes: dict[str, set[asyncio.Event]] = {}
        # On-demand worker lifecycle: how much work each plugin is currently
        # doing, and the task that will stop its worker once that reaches zero.
        self._worker_idle_timeout_s = worker_idle_timeout_s
        self._sleep = sleep or asyncio.sleep
        self._inflight: dict[str, int] = {}
        self._idle_exits: dict[str, asyncio.Task] = {}
        self._idle_stops: dict[str, asyncio.Task] = {}

    @property
    def snapshot(self) -> InstalledPluginSnapshot:
        return InstalledPluginSnapshot(self._snapshot.catalog, self._supervisor.statuses())

    def plugin_ids(self) -> frozenset[str]:
        """Accepted executable identities, without importing their code."""
        return frozenset(
            plugin.plugin_id
            for plugin in self._snapshot.catalog.plugins
            if plugin.source is PluginSource.EXTERNAL
        )

    def declared_artifacts(self, plugin_id: str) -> tuple[str, ...]:
        """The artifact ids a person may choose for this plugin, in declared order.

        Every id here has a pinned digest this service verifies, and anything
        else has none — but a manifest pins every artifact the workload
        *uses*, and most workloads use one nobody picks: the embedder behind a
        document question, the named model behind one target language, the
        transcriber behind a recording. Those declare `selectable: false`, and
        leaving them in meant a model dropdown offering an embedder and a
        `set-profile-model` that accepted it, ending at a worker that cannot
        do the work.
        """
        plugin = self._plugin(plugin_id)
        if plugin is None:
            return ()
        artifacts = plugin.manifest.get("plugin", {}).get("artifacts", ())
        return tuple(
            str(artifact["id"])
            for artifact in artifacts
            if isinstance(artifact, dict)
            and isinstance(artifact.get("id"), str)
            and artifact.get("selectable", True) is not False
        )

    def configuration_spec(self, plugin_id: str) -> PluginConfigurationSpec | None:
        """The configuration contract this plugin's manifest declares.

        `None` when the plugin is unknown or declares no contract at all. A
        manifest that declares one this runtime cannot hold raises instead:
        that is a defect in the distribution, and refusing by name is the only
        way anybody learns which manifest is wrong.
        """
        plugin = self._plugin(plugin_id)
        if plugin is None:
            return None
        try:
            return manifest_configuration_spec(plugin.plugin_id, plugin.manifest)
        except PluginSettingsError as error:
            if error.code == "configuration-undeclared":
                return None
            raise

    def stored_settings(self, plugin_id: str) -> PluginSettings | None:
        """The stored settings this workload runs on: revision and values.

        ``None`` when the manifest declares no configuration contract, or when
        the store refuses — a plugin upgraded across a contract with no
        migration, an unreadable document — because a revision this runtime
        cannot read is one it must not invent. A declared contract with no
        store, or no stored file, answers revision 0 on the manifest defaults,
        which is exactly what its worker starts on.
        """
        spec = self.configuration_spec(plugin_id)
        if spec is None:
            return None
        if self._settings_store is None:
            return PluginSettings(
                spec.plugin_id,
                spec.plugin_version,
                0,
                copy.deepcopy(dict(spec.defaults)),
            )
        try:
            return self._settings_store.load(spec)
        except (PluginSettingsError, OSError):
            LOGGER.warning(
                "Could not read the stored configuration for %s; its worker starts on the"
                " defaults its manifest declares",
                plugin_id,
                exc_info=True,
            )
            return None

    def stored_configuration(self, plugin_id: str) -> JsonObject | None:
        """What this workload was tuned to, or nothing.

        A store that refuses starts the worker on the defaults its manifest
        declares rather than not at all: a workload lost to a tunable it does
        not need would be the worse answer, and the log names it.
        """
        if self._settings_store is None:
            return None
        settings = self.stored_settings(plugin_id)
        return None if settings is None else settings.configuration

    def granted_permissions(self, plugin_id: str) -> frozenset[str]:
        """What this plugin was actually granted when its worker started."""
        return self._granted.get(plugin_id, frozenset())

    def admit(self, plugin_id: str, payload: dict) -> None:
        """Refuse jobs synchronously unless the declared worker is ready."""
        plugin = self._plugin(plugin_id)
        if plugin is None:
            raise JobDispatchError("workload-unknown", f"No installed plugin: {plugin_id}")
        status = next(
            (item for item in self._supervisor.statuses() if item.plugin_id == plugin_id), None
        )
        # A status the supervisor has not written yet is a moment rather than
        # a fault — seconds after a restart the catalog already lists a plugin
        # whose worker has not been registered — so it is admitted here,
        # waited for in `_acquire_worker`, and failed there if it never
        # arrives.
        if status is not None and not self._servable(status):
            raise JobDispatchError("worker-unavailable", "Plugin worker is not ready")
        capabilities = set(plugin.manifest["plugin"]["protocol"].get("capabilities", ()))
        if "execute" not in capabilities:
            raise JobDispatchError("worker-incompatible", "Plugin does not declare execution")
        self._admit_permissions(plugin)
        if not isinstance(payload, dict):
            raise JobDispatchError("payload-invalid", "Plugin payload must be an object")
        try:
            schema = plugin.manifest["plugin"]["schemas"]["input"]
            jsonschema.Draft202012Validator.check_schema(schema)
            violations = tuple(jsonschema.Draft202012Validator(schema).iter_errors(payload))
        except jsonschema.SchemaError as error:
            raise JobDispatchError(
                "plugin-contract-invalid", "Plugin input schema is invalid"
            ) from error
        if violations:
            raise JobDispatchError("payload-invalid", "Payload violates the plugin input contract")

    def _servable(self, status: WorkerStatus) -> bool:
        """Whether a job may be accepted for the worker in this state.

        A worker that is deliberately not running is not an unavailable one.
        Refusing here would turn the first request after an idle exit into a
        failure, when what it should do is pay the start-up cost.

        The same is true of one that is on its way up. This gate is
        synchronous — it answers the submitting call — so it cannot wait, but
        it can accept: `_acquire_worker` does the waiting a moment later, and
        a worker that never arrives fails the job there with what the
        supervisor says about it. Refusing here instead meant that for the
        first half-minute after a restart every submission came back
        `worker-unavailable`, which reads like a fault and is a queue.
        """
        if status.state is WorkerState.READY:
            return True
        if status.state in {WorkerState.STARTING, WorkerState.RESTARTING}:
            return True
        return status.state is WorkerState.IDLE and self._on_demand

    @property
    def _on_demand(self) -> bool:
        return isinstance(self._supervisor, OnDemandWorkerSupervisor)

    def _admit_permissions(self, plugin: ResolvedPlugin) -> None:
        plugin_id = plugin.plugin_id
        declared = set(plugin.manifest["plugin"]["permissions"])
        current = self._current_permissions(plugin_id, declared)
        if plugin_id in self._revoked_workers:
            raise JobDispatchError(
                "consent-revoked", "Plugin worker was stopped after a grant change"
            )
        if current != self.granted_permissions(plugin_id) or declared - current:
            raise JobDispatchError("consent-missing", "Plugin permissions are not granted")

    def dispatch(self, job_id: str, plugin_id: str, payload: dict):
        self.admit(plugin_id, payload)
        return self._dispatch(job_id, plugin_id, payload)

    async def _dispatch(self, job_id: str, plugin_id: str, payload: dict) -> dict:
        await self._acquire_worker(plugin_id)
        try:
            result = await self._dispatch_to_worker(job_id, plugin_id, payload)
        finally:
            self._release_worker(plugin_id)
        if result.status is PluginResultStatus.SUCCEEDED:
            return dict(result.output)
        if result.status is PluginResultStatus.CANCELLED:
            raise asyncio.CancelledError(result.detail)
        raise PluginWorkerError("plugin-failed", result.detail or "plugin request failed")

    async def _dispatch_to_worker(self, job_id: str, plugin_id: str, payload: dict):
        worker_payload, staged = await self._stage_payload(job_id, plugin_id, payload)
        try:
            return await self._execute_with_live_grants(
                PluginRequest(job_id, plugin_id, "manual", worker_payload, self._clock_ms(), None)
            )
        finally:
            if staged is not None:
                await _run_off_loop(shutil.rmtree, staged, True)

    async def _acquire_worker(self, plugin_id: str) -> None:
        """Make sure a worker is serving this plugin, starting one if needed."""
        if not self._on_demand:
            # Eagerly started workers still take time to come up — a
            # multi-gigabyte model has to reach the GPU first — and for that
            # window every job was refused `worker-unavailable`, which reads
            # like a fault and is a queue. Waited for here rather than in the
            # supervisor: how long a caller will wait for a worker is this
            # layer's decision, and the supervisor's job is to say what the
            # worker is doing.
            await self._await_worker(plugin_id)
            return
        self._cancel_idle_exit(plugin_id)
        self._inflight[plugin_id] = self._inflight.get(plugin_id, 0) + 1
        try:
            await self._await_idle_stop(plugin_id)
            status = await self._ready_or_waited(plugin_id)
        except BaseException:
            self._release_worker(plugin_id)
            raise
        if getattr(status, "state", None) is not WorkerState.READY:
            detail = getattr(status, "detail", "") or "plugin worker could not be started"
            self._release_worker(plugin_id)
            raise PluginWorkerError("worker-unavailable", detail)

    async def _ready_or_waited(self, plugin_id: str):
        """Ask for the worker, and keep asking while the answer is "not yet".

        `ensure_ready` answers with what the supervisor knows *now*, and right
        after a restart that is `None` — the supervisor has not adopted the
        specs yet, so there is nothing to be ready. That is a moment rather
        than a refusal, and it used to end the job with "plugin worker could
        not be started" three seconds after a restart.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + WORKER_READY_WAIT_SECONDS
        while True:
            status = await self._supervisor.ensure_ready(plugin_id)
            state = getattr(status, "state", None)
            if state is WorkerState.READY:
                return status
            if state is not None and state not in _COMING_UP:
                return status
            if loop.time() >= deadline:
                return status
            await asyncio.sleep(_READY_POLL_SECONDS)

    def _status_of(self, plugin_id: str):
        """What the supervisor says about this worker now, if it will say."""
        statuses = getattr(self._supervisor, "statuses", None)
        if statuses is None:
            return None
        return next((status for status in statuses() if status.plugin_id == plugin_id), None)

    async def _await_worker(self, plugin_id: str) -> None:
        """Give a worker that is on its way up the time to arrive.

        Waited for here rather than in the supervisor because this is the
        layer that knows the plugin exists: seconds after a restart the
        catalog already lists it and the supervisor has no status for it at
        all, which is the same "not yet" as `starting` and was reported as
        the same fault. A worker that reaches a state waiting cannot mend —
        failed, exhausted, exited — ends the wait at once.
        """
        if getattr(self._supervisor, "statuses", None) is None:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + WORKER_READY_WAIT_SECONDS
        while True:
            status = self._status_of(plugin_id)
            if status is not None and status.state not in _COMING_UP:
                return
            if loop.time() >= deadline:
                return
            await asyncio.sleep(_READY_POLL_SECONDS)

    def _release_worker(self, plugin_id: str) -> None:
        """Give the worker back, and arm its idle exit once nothing is left."""
        remaining = self._inflight.get(plugin_id, 0) - 1
        if remaining > 0:
            self._inflight[plugin_id] = remaining
            return
        self._inflight.pop(plugin_id, None)
        if not self._on_demand or self._worker_idle_timeout_s is None:
            return
        self._cancel_idle_exit(plugin_id)
        self._idle_exits[plugin_id] = asyncio.ensure_future(self._exit_when_idle(plugin_id))

    def _cancel_idle_exit(self, plugin_id: str) -> None:
        pending = self._idle_exits.pop(plugin_id, None)
        if pending is not None:
            pending.cancel()

    async def _await_idle_stop(self, plugin_id: str) -> None:
        """Wait out a stop already tearing this worker down.

        A request that arrives after `_exit_when_idle` has re-checked
        `_inflight` cancels a task the shield no longer lets die, and used to
        race it to the supervisor: when `ensure_ready` won, the job was
        answered READY off the still-present slot and the stop then killed
        the worker under it. The stop cannot be called off any more, so the
        request waits for it to finish and starts a fresh worker afterwards.
        A stop that failed is the stop's problem, logged where it ran; this
        job's outcome is decided by `ensure_ready`, not by the corpse.
        """
        stopping = self._idle_stops.get(plugin_id)
        if stopping is not None:
            await asyncio.wait((stopping,))

    async def _exit_when_idle(self, plugin_id: str) -> None:
        """Stop a worker that has had nothing to do for the idle timeout.

        A job that arrives once the stop is underway waits it out in
        `_await_idle_stop` and starts the worker back up afterwards, so it
        is answered in full either way.
        """
        try:
            try:
                await self._sleep(self._worker_idle_timeout_s)
            except asyncio.CancelledError:
                return
            if self._inflight.get(plugin_id):
                return
            # Registered in the same step as the inflight check — no await
            # between them — so every later request sees either no stop at
            # all or the one it must wait out. Shielded because a request
            # arriving mid-stop cancels this task, and a half-stopped worker
            # would leak its process and its lease; the done callback both
            # clears the registry and retrieves the outcome, so a failure is
            # logged even when the cancellation lands on the shield first.
            stopping = asyncio.ensure_future(
                self._supervisor.stop_worker(plugin_id, IDLE_EXIT_DETAIL)
            )
            self._idle_stops[plugin_id] = stopping
            stopping.add_done_callback(functools.partial(self._idle_stop_finished, plugin_id))
            try:
                await asyncio.shield(stopping)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - already logged by the callback
                pass
        finally:
            if self._idle_exits.get(plugin_id) is asyncio.current_task():
                self._idle_exits.pop(plugin_id, None)

    def _idle_stop_finished(self, plugin_id: str, stopping: asyncio.Task) -> None:
        if self._idle_stops.get(plugin_id) is stopping:
            self._idle_stops.pop(plugin_id, None)
        if stopping.cancelled():
            return
        if stopping.exception() is not None:
            LOGGER.error(
                "Could not stop the idle worker for %s",
                plugin_id,
                exc_info=stopping.exception(),
            )

    async def _stage_payload(
        self, job_id: str, plugin_id: str, payload: dict
    ) -> tuple[dict, Path | None]:
        if SELECTED_FILES_PERMISSION not in self.granted_permissions(plugin_id):
            return dict(payload), None
        if self._selected_files_root is None:
            raise PluginWorkerError(
                "selected-files-unavailable", "selected-file broker is not configured"
            )
        staging = asyncio.create_task(
            _run_off_loop(
                _stage_selected_sources,
                self._selected_files_root,
                plugin_id,
                job_id,
                payload,
            )
        )
        try:
            return await asyncio.shield(staging)
        except BaseException as staging_error:
            cleanup = asyncio.create_task(_cleanup_abandoned_staging(staging))
            # Staging must finish being torn down before this returns, so a
            # cancellation delivered while waiting is remembered rather than
            # honoured immediately -- and then re-raised, because a job whose
            # caller withdrew it must not continue as a plain failure.
            cancelled: asyncio.CancelledError | None = None
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError as cancel:
                    cancelled = cancel
            await cleanup
            if cancelled is not None:
                raise cancelled from staging_error
            raise

    async def _execute_with_live_grants(self, request: PluginRequest):
        execution = asyncio.create_task(
            self._supervisor.execute(request, _ProgressSink(self._progress_sink))
        )
        plugin = self._plugin(request.plugin_id)
        assert plugin is not None
        declared = set(plugin.manifest["plugin"]["permissions"])
        change = asyncio.Event()
        changes = self._grant_changes.setdefault(request.plugin_id, set())
        changes.add(change)
        local_monitor = None
        await asyncio.sleep(0)
        if (
            not execution.done()
            and self._grant_monitor is None
            and isinstance(self._grant_source, ReloadablePermissionGrantSource)
        ):
            local_monitor = asyncio.ensure_future(self._monitor_permission_grants())
        changed = asyncio.ensure_future(change.wait())
        try:
            done, _pending = await asyncio.wait(
                (execution, changed), return_when=asyncio.FIRST_COMPLETED
            )
            if changed in done:
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
                raise PluginWorkerError(
                    "consent-revoked", "Plugin permission changed while work was active"
                )
            await self._refresh_permission_grants()
            if self._current_permissions(request.plugin_id, declared) != self.granted_permissions(
                request.plugin_id
            ):
                await self._revoke_worker(request.plugin_id)
                raise PluginWorkerError(
                    "consent-revoked", "Plugin permission changed while work was active"
                )
            return await execution
        except BaseException:
            if not execution.done():
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
            raise
        finally:
            changed.cancel()
            await asyncio.gather(changed, return_exceptions=True)
            if local_monitor is not None:
                local_monitor.cancel()
                await asyncio.gather(local_monitor, return_exceptions=True)
            changes.discard(change)
            if not changes:
                self._grant_changes.pop(request.plugin_id, None)

    async def _revoke_worker(self, plugin_id: str) -> None:
        for change in tuple(self._grant_changes.get(plugin_id, ())):
            change.set()
        if plugin_id in self._revoked_workers:
            return
        self._revoked_workers.add(plugin_id)
        if isinstance(self._supervisor, WorkerRevoker):
            await self._supervisor.revoke(
                plugin_id, "worker stopped because its permission grant changed"
            )

    async def _monitor_permission_grants(self) -> None:
        try:
            while True:
                await self._reconcile_permission_grants()
                await asyncio.sleep(GRANT_REFRESH_SECONDS)
        except asyncio.CancelledError:
            return

    async def _reconcile_permission_grants(self) -> None:
        external = tuple(
            plugin
            for plugin in self._snapshot.catalog.plugins
            if plugin.source is PluginSource.EXTERNAL
        )
        try:
            await self._refresh_permission_grants()
        except Exception:  # noqa: BLE001 - persisted grant adapters vary
            LOGGER.exception("Could not refresh plugin grants; revoking workers")
            for plugin in external:
                await self._revoke_worker(plugin.plugin_id)
            return
        for plugin in external:
            declared = set(plugin.manifest["plugin"]["permissions"])
            current = self._current_permissions(plugin.plugin_id, declared)
            if current != self.granted_permissions(plugin.plugin_id):
                await self._revoke_worker(plugin.plugin_id)

    async def _refresh_permission_grants(self) -> None:
        if isinstance(self._grant_source, ReloadablePermissionGrantSource):
            await _run_off_loop(self._grant_source.reload)

    def _current_permissions(self, plugin_id: str, declared: set[str]) -> frozenset[str]:
        if isinstance(self._grant_source, _DenyAllGrants):
            return self.granted_permissions(plugin_id)
        return self._grant_source.active_permissions(plugin_id, declared)

    def _plugin(self, plugin_id: str) -> ResolvedPlugin | None:
        return next(
            (
                plugin
                for plugin in self._snapshot.catalog.plugins
                if plugin.source is PluginSource.EXTERNAL and plugin.plugin_id == plugin_id
            ),
            None,
        )

    async def start(self) -> InstalledPluginSnapshot:
        await self._refresh_permission_grants()
        candidates = discover_plugin_metadata(
            bundled_root=self._bundled_root,
            entry_points_provider=self._entry_points_provider,
        )
        catalog = resolve_plugin_compatibility(resolve_plugin_identities(candidates))
        granted_permissions = {
            plugin.plugin_id: self._grant_source.active_permissions(
                plugin.plugin_id,
                set(plugin.manifest["plugin"]["permissions"]),
            )
            for plugin in catalog.plugins
            if plugin.source is PluginSource.EXTERNAL
        }
        self._granted = dict(granted_permissions)
        self._snapshot = InstalledPluginSnapshot(catalog, ())
        self._revoked_workers.clear()
        self._grant_changes.clear()
        spec_options = {
            "python_executable": self._python_executable,
            "worker_import_paths": self._worker_import_paths,
            "granted_permissions": granted_permissions,
            "selected_files_root": self._selected_files_root,
        }
        if self._worker_state_root is not None:
            spec_options["worker_state_root"] = self._worker_state_root
        if self._resolve_artifact is not None:
            spec_options["resolve_artifact"] = self._resolve_artifact
        devices = self._accelerator_devices()
        if devices:
            spec_options["accelerator_devices"] = devices
        if self._profile_accelerator_devices is not None:
            spec_options["accelerator_devices_by_plugin"] = {
                plugin.plugin_id: self._profile_accelerator_devices(plugin.plugin_id)
                for plugin in catalog.plugins
                if plugin.source is PluginSource.EXTERNAL
            }
        if self._profile_model_choice is not None:
            spec_options["model_choices"] = {
                plugin.plugin_id: self._profile_model_choice(plugin.plugin_id) or ""
                for plugin in catalog.plugins
                if plugin.source is PluginSource.EXTERNAL
            }
        if self._settings_store is not None:
            spec_options["configurations"] = {
                plugin.plugin_id: self.stored_configuration(plugin.plugin_id) or {}
                for plugin in catalog.plugins
                if plugin.source is PluginSource.EXTERNAL
            }
        # Off the loop: building a spec resolves every artifact the plugin
        # declares, and resolving one reads and hashes the whole file. On this
        # desk that is twenty-three gigabytes of models, and doing it here
        # left the event loop unable to accept a control connection or publish
        # a snapshot for over a minute after a restart — while systemd had
        # already been told the service was ready.
        specs = await _run_off_loop(
            functools.partial(external_worker_specs, catalog.plugins, **spec_options)
        )
        workers = await (
            self._supervisor.register(specs) if self._on_demand else self._supervisor.start(specs)
        )
        self._snapshot = InstalledPluginSnapshot(catalog, workers)
        if isinstance(self._supervisor, WorkerRevoker):
            self._grant_monitor = asyncio.create_task(
                self._monitor_permission_grants(),
                name="omnitensor-plugin-grant-monitor",
            )
        return self._snapshot

    async def stop(self) -> tuple[WorkerStatus, ...]:
        exits = tuple(self._idle_exits.values())
        self._idle_exits.clear()
        self._inflight.clear()
        for pending in exits:
            pending.cancel()
        if exits:
            await asyncio.gather(*exits, return_exceptions=True)
        # A shielded stop_worker survives those cancels; the supervisor must
        # not be stopped while one is still stopping a worker of its own.
        stops = tuple(self._idle_stops.values())
        if stops:
            await asyncio.gather(*stops, return_exceptions=True)
        monitor = self._grant_monitor
        self._grant_monitor = None
        if monitor is not None:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
        workers = await self._supervisor.stop()
        self._snapshot = InstalledPluginSnapshot(self._snapshot.catalog, workers)
        return workers

    async def reload_accelerator_devices(self) -> InstalledPluginSnapshot:
        """Restart workers so a changed profile lease drops the old device."""
        await self.stop()
        return await self.start()


def _stage_selected_sources(
    root: Path,
    plugin_id: str,
    job_id: str,
    payload: Mapping[str, object],
) -> tuple[dict, Path]:
    return _staging.stage_selected_sources(
        root,
        plugin_id,
        job_id,
        payload,
        copier=_copy_selected_source,
        maximum_sources=MAX_SELECTED_SOURCES,
    )


def _copy_selected_source(
    source: str,
    staged: Path,
    index: int,
    observed: set[tuple[int, int]],
) -> Path:
    return _staging.copy_selected_source(
        source,
        staged,
        index,
        observed,
        change_detector=_selected_source_changed,
        maximum_bytes=MAX_SELECTED_SOURCE_BYTES,
    )


def _validate_selected_source_stat(status: os.stat_result) -> None:
    _staging.validate_selected_source_stat(
        status,
        maximum_bytes=MAX_SELECTED_SOURCE_BYTES,
    )


class _ProgressSink:
    def __init__(self, sink: Callable[[PluginProgress], None] | None) -> None:
        self._sink = sink

    async def report(self, progress: PluginProgress) -> None:
        if self._sink is not None:
            self._sink(progress)
