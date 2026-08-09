"""Turn identity-validated installed plugins into supervised workers."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from .discovery import PluginSource, discover_plugin_metadata
from .identity import PluginCatalog, ResolvedPlugin, resolve_plugin_identities
from .supervisor import PluginWorkerSupervisor, WorkerSpec, WorkerStatus

MAX_WORKER_IMPORT_PATHS = 16
WORKER_CAPABILITIES = frozenset({"cancel", "health"})


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
    ) -> None:
        self._bundled_root = Path(bundled_root)
        self._supervisor = supervisor or PluginWorkerSupervisor()
        self._entry_points_provider = entry_points_provider
        self._python_executable = _executable(python_executable)
        self._worker_import_paths = _import_paths(worker_import_paths)
        self._snapshot = InstalledPluginSnapshot(PluginCatalog((), ()), ())

    @property
    def snapshot(self) -> InstalledPluginSnapshot:
        return self._snapshot

    async def start(self) -> InstalledPluginSnapshot:
        candidates = discover_plugin_metadata(
            bundled_root=self._bundled_root,
            entry_points_provider=self._entry_points_provider,
        )
        catalog = resolve_plugin_identities(candidates)
        workers = await self._supervisor.start(
            external_worker_specs(
                catalog.plugins,
                python_executable=self._python_executable,
                worker_import_paths=self._worker_import_paths,
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
) -> tuple[WorkerSpec, ...]:
    """Build deterministic argv without importing plugin code in the service."""
    executable = _executable(python_executable)
    import_paths = _import_paths(worker_import_paths)
    specs = []
    for plugin in plugins:
        if plugin.source is not PluginSource.EXTERNAL:
            continue
        protocol = plugin.manifest["plugin"]["protocol"]
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
        specs.append(
            WorkerSpec(
                plugin.plugin_id,
                tuple(argv),
                minimum_protocol=protocol["minimum"],
                maximum_protocol=protocol["maximum"],
                capabilities=WORKER_CAPABILITIES,
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
