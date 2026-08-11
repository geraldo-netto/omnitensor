"""Turn identity-validated installed plugins into supervised workers."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import stat
import sys
import tempfile
import time
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Protocol

import jsonschema

from .discovery import PluginSource, discover_plugin_metadata
from .identity import PluginCatalog, ResolvedPlugin, resolve_plugin_identities
from .manifest_compatibility import resolve_plugin_compatibility
from .protocol import PluginProgress, PluginRequest, PluginResultStatus
from .sandbox import SELECTED_FILES_PERMISSION, FilesystemSandbox
from .supervisor import (
    PluginWorkerError,
    PluginWorkerSupervisor,
    WorkerSpec,
    WorkerState,
    WorkerStatus,
)

MAX_WORKER_IMPORT_PATHS = 16
WORKER_CAPABILITIES = frozenset({"cancel", "execute", "health", "progress"})
MAX_SELECTED_SOURCES = 32
MAX_SELECTED_SOURCE_BYTES = 128 * 1024 * 1024
_STAGING_COMPONENT = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class PermissionGrantSource(Protocol):
    def active_permissions(
        self,
        plugin_id: str,
        declared_permissions: set[str],
    ) -> frozenset[str]: ...


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
    ) -> None:
        self._bundled_root = Path(bundled_root)
        self._supervisor = supervisor or PluginWorkerSupervisor()
        self._entry_points_provider = entry_points_provider
        self._python_executable = _executable(python_executable)
        self._worker_import_paths = _import_paths(worker_import_paths)
        self._grant_source = grant_source or _DenyAllGrants()
        self._progress_sink = progress_sink
        self._clock_ms = clock_ms or (lambda: max(1, int(time.time() * 1000)))
        self._selected_files_root = _prepare_selected_files_root(selected_files_root)
        self._granted: dict[str, frozenset[str]] = {}
        self._snapshot = InstalledPluginSnapshot(PluginCatalog((), ()), ())

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
        from ..jobs import JobDispatchError  # noqa: PLC0415 - avoids jobs/protocol cycle

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
        declared = set(plugin.manifest["plugin"]["permissions"])
        if declared - self.granted_permissions(plugin_id):
            raise JobDispatchError("consent-missing", "Plugin permissions are not granted")
        if not isinstance(payload, dict):
            raise JobDispatchError("payload-invalid", "Plugin payload must be an object")
        try:
            schema = plugin.manifest["plugin"]["schemas"]["input"]
            jsonschema.Draft202012Validator.check_schema(schema)
            violations = tuple(
                jsonschema.Draft202012Validator(schema).iter_errors(payload)
            )
        except jsonschema.SchemaError as error:
            raise JobDispatchError(
                "plugin-contract-invalid", "Plugin input schema is invalid"
            ) from error
        if violations:
            raise JobDispatchError("payload-invalid", "Payload violates the plugin input contract")

    def dispatch(self, job_id: str, plugin_id: str, payload: dict):
        self.admit(plugin_id, payload)
        return self._dispatch(job_id, plugin_id, payload)

    async def _dispatch(self, job_id: str, plugin_id: str, payload: dict) -> dict:
        worker_payload = dict(payload)
        staged: Path | None = None
        if SELECTED_FILES_PERMISSION in self.granted_permissions(plugin_id):
            if self._selected_files_root is None:
                raise PluginWorkerError(
                    "selected-files-unavailable", "selected-file broker is not configured"
                )
            worker_payload, staged = await asyncio.to_thread(
                _stage_selected_sources,
                self._selected_files_root,
                plugin_id,
                job_id,
                payload,
            )
        try:
            result = await self._supervisor.execute(
                PluginRequest(
                    job_id,
                    plugin_id,
                    "manual",
                    worker_payload,
                    self._clock_ms(),
                    None,
                ),
                _ProgressSink(self._progress_sink),
            )
        finally:
            if staged is not None:
                await asyncio.to_thread(shutil.rmtree, staged, True)
        if result.status is PluginResultStatus.SUCCEEDED:
            return dict(result.output)
        if result.status is PluginResultStatus.CANCELLED:
            raise asyncio.CancelledError(result.detail)
        raise PluginWorkerError("plugin-failed", result.detail or "plugin request failed")

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
        # Retained rather than discarded after launch: the inventory reports
        # what each plugin was granted, and recomputing it there would let the
        # answer drift from what the workers actually run with.
        self._granted = dict(granted_permissions)
        # Publish the accepted identities before worker statuses can become
        # observable.  The supervisor records STARTING synchronously during
        # ``start``; without this assignment, a concurrent inventory read can
        # pair new worker states with the previous (often empty) catalog.
        self._snapshot = InstalledPluginSnapshot(catalog, ())
        workers = await self._supervisor.start(
            external_worker_specs(
                catalog.plugins,
                python_executable=self._python_executable,
                worker_import_paths=self._worker_import_paths,
                granted_permissions=granted_permissions,
                selected_files_root=self._selected_files_root,
            )
        )
        self._snapshot = InstalledPluginSnapshot(catalog, workers)
        return self._snapshot

    async def stop(self) -> tuple[WorkerStatus, ...]:
        workers = await self._supervisor.stop()
        self._snapshot = InstalledPluginSnapshot(self._snapshot.catalog, workers)
        return workers


def external_worker_specs(
    plugins: Sequence[ResolvedPlugin],
    *,
    python_executable: str | Path = sys.executable,
    worker_import_paths: Sequence[Path] = (),
    granted_permissions: Mapping[str, Collection[str]] | None = None,
    selected_files_root: Path | None = None,
) -> tuple[WorkerSpec, ...]:
    """Build deterministic argv without importing plugin code in the service."""
    executable = _executable(python_executable)
    import_paths = _import_paths(worker_import_paths)
    package_root = Path(__file__).resolve().parents[2]
    permissions_by_plugin = granted_permissions or {}
    specs = []
    for plugin in plugins:
        if plugin.source is not PluginSource.EXTERNAL:
            continue
        protocol = plugin.manifest["plugin"]["protocol"]
        declared = frozenset(plugin.manifest["plugin"]["permissions"])
        granted = frozenset(permissions_by_plugin.get(plugin.plugin_id, ()))
        argv = [
            executable,
            "-m",
            "omnitensor.plugins.worker",
            "--plugin-id",
            plugin.plugin_id,
            "--entry-point",
            plugin.entry_point_name,
            "--target",
            plugin.entry_point_value,
            "--distribution",
            plugin.distribution_name,
        ]
        for path in import_paths:
            argv.extend(("--import-path", path))
        for permission in sorted(granted):
            argv.extend(("--permission", permission))
        specs.append(
            WorkerSpec(
                plugin.plugin_id,
                tuple(argv),
                minimum_protocol=protocol["minimum"],
                maximum_protocol=protocol["maximum"],
                capabilities=(
                    frozenset(protocol.get("capabilities", ())) & WORKER_CAPABILITIES
                ),
                sandbox=FilesystemSandbox.from_permissions(
                    declared,
                    granted,
                    runtime_paths=_trusted_runtime_paths(import_paths),
                    python_path=package_root,
                    selected_files_root=selected_files_root,
                ),
            )
        )
    return tuple(specs)


def _prepare_selected_files_root(root: Path | None) -> Path | None:
    if root is None:
        return None
    candidate = Path(root)
    if not candidate.is_absolute():
        raise ValueError("selected_files_root must be absolute")
    candidate.mkdir(parents=True, exist_ok=True, mode=0o700)
    candidate.chmod(0o700)
    return candidate.resolve()


def _stage_selected_sources(
    root: Path,
    plugin_id: str,
    job_id: str,
    payload: Mapping[str, object],
) -> tuple[dict, Path]:
    sources = payload.get("sources")
    if (
        not isinstance(sources, list)
        or not 1 <= len(sources) <= MAX_SELECTED_SOURCES
        or not all(isinstance(source, str) and source for source in sources)
    ):
        raise PluginWorkerError(
            "selected-files-invalid", "sources must name 1-32 selected files"
        )
    if (
        _STAGING_COMPONENT.fullmatch(plugin_id) is None
        or _STAGING_COMPONENT.fullmatch(job_id) is None
    ):
        raise PluginWorkerError("selected-files-invalid", "request identity is invalid")
    plugin_root = root / plugin_id
    plugin_root.mkdir(mode=0o700, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f"{job_id}-", dir=plugin_root))
    observed: set[tuple[int, int]] = set()
    try:
        staged_sources = [
            str(_copy_selected_source(source, staged, index, observed))
            for index, source in enumerate(sources)
        ]
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    rewritten = dict(payload)
    rewritten["sources"] = staged_sources
    return rewritten, staged


def _copy_selected_source(
    source: str,
    staged: Path,
    index: int,
    observed: set[tuple[int, int]],
) -> Path:
    candidate = Path(source)
    resolved = _canonical_selected_source(candidate)
    descriptor = _open_selected_source(resolved)
    suffix = candidate.suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
        suffix = ".bin"
    destination = staged / f"{index:02d}{suffix}"
    try:
        with os.fdopen(descriptor, "rb") as reader:
            before = os.fstat(reader.fileno())
            identity = (before.st_dev, before.st_ino)
            _validate_selected_source_stat(before)
            if identity in observed:
                raise PluginWorkerError(
                    "selected-file-invalid", "the same selected file appears more than once"
                )
            observed.add(identity)
            with destination.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
            after = os.fstat(reader.fileno())
            if _selected_source_changed(before, after, destination):
                raise PluginWorkerError(
                    "selected-file-changed", "selected source changed while it was copied"
                )
    except OSError as error:
        raise PluginWorkerError(
            "selected-file-unavailable", "selected source cannot be copied"
        ) from error
    return destination


def _canonical_selected_source(candidate: Path) -> Path:
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise PluginWorkerError(
            "selected-file-unavailable", "selected source cannot be opened"
        ) from error
    if not candidate.is_absolute() or resolved != candidate:
        raise PluginWorkerError(
            "selected-file-invalid", "selected source must be a canonical absolute path"
        )
    return resolved


def _open_selected_source(candidate: Path) -> int:
    try:
        return os.open(candidate, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise PluginWorkerError(
            "selected-file-unavailable", "selected source cannot be opened"
        ) from error


def _validate_selected_source_stat(status: os.stat_result) -> None:
    if (
        not stat.S_ISREG(status.st_mode)
        or not 0 < status.st_size <= MAX_SELECTED_SOURCE_BYTES
    ):
        raise PluginWorkerError(
            "selected-file-invalid", "selected source must be a bounded regular file"
        )


def _selected_source_changed(
    before: os.stat_result,
    after: os.stat_result,
    destination: Path,
) -> bool:
    return (
        (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or destination.stat().st_size != before.st_size
    )


class _ProgressSink:
    def __init__(self, sink: Callable[[PluginProgress], None] | None) -> None:
        self._sink = sink

    async def report(self, progress: PluginProgress) -> None:
        if self._sink is not None:
            self._sink(progress)


def entry_points_from_distributions(paths: Sequence[Path]) -> Callable[..., tuple]:
    """Create a metadata provider for an isolated installation root."""
    roots = _import_paths(paths)

    def provider(**selection) -> tuple:
        group = selection.get("group")
        entries = (
            entry_point
            for distribution in metadata.distributions(path=list(roots))
            for entry_point in distribution.entry_points
        )
        if group is not None:
            entries = (entry_point for entry_point in entries if entry_point.group == group)
        return tuple(entries)

    return provider


def _executable(value: str | Path) -> str:
    executable = str(value)
    if not executable or "\0" in executable:
        raise ValueError("python_executable must be a non-empty path")
    return executable


def _import_paths(paths: Sequence[Path]) -> tuple[str, ...]:
    if len(paths) > MAX_WORKER_IMPORT_PATHS:
        raise ValueError(f"at most {MAX_WORKER_IMPORT_PATHS} worker import paths are allowed")
    resolved = []
    for path in paths:
        candidate = Path(path)
        if not candidate.is_absolute() or not candidate.is_dir():
            raise ValueError("worker import paths must be absolute directories")
        resolved.append(str(candidate.resolve()))
    if len(set(resolved)) != len(resolved):
        raise ValueError("worker import paths must be unique")
    return tuple(resolved)


def _trusted_runtime_paths(import_paths: Sequence[str]) -> tuple[Path | str, ...]:
    return (
        Path(sys.prefix),
        Path(__file__).resolve().parents[2],
        *import_paths,
    )
