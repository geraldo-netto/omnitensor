"""Turn identity-validated installed plugins into supervised workers."""

from __future__ import annotations

import asyncio
import json as json
import logging
import os
import re as re
import shutil
import stat as stat
import sys
import tempfile as tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections.abc import Collection as Collection
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
from .budgets import current_process_cgroup
from .discovery import PluginSource, discover_plugin_metadata
from .identity import PluginCatalog, ResolvedPlugin, resolve_plugin_identities
from .manifest_compatibility import resolve_plugin_compatibility
from .protocol import PluginProgress, PluginRequest, PluginResultStatus
from .sandbox import SELECTED_FILES_PERMISSION
from .sandbox import FilesystemSandbox as FilesystemSandbox
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
OFF_LOOP_POLL_SECONDS = 0.001


async def _run_off_loop(callback: Callable[..., object], *args: object) -> object:
    """Run one blocking loading operation without asyncio's shared executor."""
    completed = threading.Event()
    outcome: list[object] = []
    errors: list[BaseException] = []

    def invoke() -> None:
        try:
            outcome.append(callback(*args))
        except BaseException as caught:
            errors.append(caught)
        finally:
            completed.set()

    worker = threading.Thread(
        target=invoke,
        name="omnitensor-loading",
        daemon=True,
    )
    worker.start()
    try:
        while not completed.is_set():
            await asyncio.sleep(OFF_LOOP_POLL_SECONDS)
    finally:
        if not worker.is_alive():
            worker.join()
    if errors:
        raise errors[0]
    return outcome[0]


async def _cleanup_abandoned_staging(staging: asyncio.Task) -> None:
    try:
        _worker_payload, abandoned = await asyncio.shield(staging)
    except BaseException:
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
        require_worker_cgroup: bool = True,
    ) -> None:
        self._bundled_root = Path(bundled_root)
        cgroup_parent = (
            current_process_cgroup() if os.environ.get("INVOCATION_ID") else None
        )
        self._supervisor = supervisor or PluginWorkerSupervisor(
            AsyncioSubprocessLauncher(
                cgroup_parent=cgroup_parent,
                require_cgroup=require_worker_cgroup,
            )
        )
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
        self._granted: dict[str, frozenset[str]] = {}
        self._revoked_workers: set[str] = set()
        self._snapshot = InstalledPluginSnapshot(PluginCatalog((), ()), ())
        self._grant_monitor: asyncio.Task | None = None
        self._grant_changes: dict[str, set[asyncio.Event]] = {}

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
        if status is None or status.state is not WorkerState.READY:
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
        worker_payload, staged = await self._stage_payload(job_id, plugin_id, payload)
        try:
            result = await self._execute_with_live_grants(
                PluginRequest(job_id, plugin_id, "manual", worker_payload, self._clock_ms(), None)
            )
        finally:
            if staged is not None:
                await _run_off_loop(shutil.rmtree, staged, True)
        if result.status is PluginResultStatus.SUCCEEDED:
            return dict(result.output)
        if result.status is PluginResultStatus.CANCELLED:
            raise asyncio.CancelledError(result.detail)
        raise PluginWorkerError("plugin-failed", result.detail or "plugin request failed")

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
        except BaseException:
            cleanup = asyncio.create_task(_cleanup_abandoned_staging(staging))
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
            await cleanup
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
        if not execution.done() and self._grant_monitor is None and isinstance(
            self._grant_source, ReloadablePermissionGrantSource
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
        workers = await self._supervisor.start(
            external_worker_specs(catalog.plugins, **spec_options)
        )
        self._snapshot = InstalledPluginSnapshot(catalog, workers)
        if isinstance(self._supervisor, WorkerRevoker):
            self._grant_monitor = asyncio.create_task(
                self._monitor_permission_grants(),
                name="omnitensor-plugin-grant-monitor",
            )
        return self._snapshot

    async def stop(self) -> tuple[WorkerStatus, ...]:
        monitor = self._grant_monitor
        self._grant_monitor = None
        if monitor is not None:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
        workers = await self._supervisor.stop()
        self._snapshot = InstalledPluginSnapshot(self._snapshot.catalog, workers)
        return workers


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
