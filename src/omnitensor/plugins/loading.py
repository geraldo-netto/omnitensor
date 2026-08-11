"""Turn identity-validated installed plugins into supervised workers."""

from __future__ import annotations

import sys
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Protocol

from .discovery import PluginSource, discover_plugin_metadata
from .identity import PluginCatalog, ResolvedPlugin, resolve_plugin_identities
from .manifest_compatibility import resolve_plugin_compatibility
from .sandbox import FilesystemSandbox
from .supervisor import PluginWorkerSupervisor, WorkerSpec, WorkerStatus

MAX_WORKER_IMPORT_PATHS = 16
WORKER_CAPABILITIES = frozenset({"cancel", "health"})


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
    ) -> None:
        self._bundled_root = Path(bundled_root)
        self._supervisor = supervisor or PluginWorkerSupervisor()
        self._entry_points_provider = entry_points_provider
        self._python_executable = _executable(python_executable)
        self._worker_import_paths = _import_paths(worker_import_paths)
        self._grant_source = grant_source or _DenyAllGrants()
        self._granted: dict[str, frozenset[str]] = {}
        self._snapshot = InstalledPluginSnapshot(PluginCatalog((), ()), ())

    @property
    def snapshot(self) -> InstalledPluginSnapshot:
        return self._snapshot

    def granted_permissions(self, plugin_id: str) -> frozenset[str]:
        """What this plugin was actually granted when its worker started."""
        return self._granted.get(plugin_id, frozenset())

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
        workers = await self._supervisor.start(
            external_worker_specs(
                catalog.plugins,
                python_executable=self._python_executable,
                worker_import_paths=self._worker_import_paths,
                granted_permissions=granted_permissions,
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
                ),
            )
        )
    return tuple(specs)


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
