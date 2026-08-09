"""Metadata-only discovery for bundled and separately packaged plugins."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from importlib import metadata
from pathlib import Path
from typing import Protocol

from ..registry import bundled_workloads_path, load_workloads

PLUGIN_ENTRY_POINT_GROUP = "omnitensor.workloads"
PLUGIN_MANIFEST_FILENAME = "omnitensor-plugin.json"
MAX_PLUGIN_CANDIDATES = 256
MAX_DISTRIBUTION_FILES = 10000


class PluginSource(StrEnum):
    BUNDLED = "bundled"
    EXTERNAL = "external"


class PluginDiscoveryError(RuntimeError):
    """Discovery metadata exceeded a process boundary or catalog bound."""


@dataclass(frozen=True, slots=True)
class PluginMetadata:
    """Load-free plugin descriptor consumed by later identity validation."""

    source: PluginSource
    entry_point_name: str
    entry_point_value: str | None
    distribution_name: str | None
    distribution_version: str | None
    manifest_paths: tuple[Path, ...]
    bundled_manifest: dict | None
    metadata_error: str


class _Distribution(Protocol):
    name: str
    version: str
    files: Iterable[Path] | None

    def locate_file(self, path: Path) -> Path: ...


class _EntryPoint(Protocol):
    name: str
    value: str
    dist: _Distribution | None


def discover_plugin_metadata(
    *,
    bundled_root: Path | None = None,
    entry_points_provider: Callable[..., Iterable[_EntryPoint]] = metadata.entry_points,
) -> tuple[PluginMetadata, ...]:
    """Enumerate candidates deterministically without calling entry-point load()."""
    root = bundled_root or bundled_workloads_path()
    bundled = [
        PluginMetadata(
            PluginSource.BUNDLED,
            workload.manifest.get("plugin", {}).get("entryPoint", workload.id),
            None,
            "omnitensor",
            None,
            (root / workload.id / "manifest.json",),
            workload.manifest,
            "",
        )
        for workload in load_workloads(root).values()
    ]
    try:
        entry_points = list(entry_points_provider(group=PLUGIN_ENTRY_POINT_GROUP))
    except Exception as error:
        raise PluginDiscoveryError(
            f"entry-point enumeration failed: {type(error).__name__}"
        ) from error
    if len(bundled) + len(entry_points) > MAX_PLUGIN_CANDIDATES:
        raise PluginDiscoveryError(f"more than {MAX_PLUGIN_CANDIDATES} plugin candidates")
    external = [_external_metadata(entry_point) for entry_point in entry_points]
    bundled.sort(key=_sort_key)
    external.sort(key=_sort_key)
    return tuple([*bundled, *external])


def _external_metadata(entry_point: _EntryPoint) -> PluginMetadata:
    distribution_name = None
    distribution_version = None
    manifest_paths: tuple[Path, ...] = ()
    problem = ""
    try:
        distribution = entry_point.dist
        if distribution is None:
            problem = "entry point has no distribution"
        else:
            distribution_name = distribution.name
            distribution_version = distribution.version
            manifest_paths = _distribution_manifests(distribution)
    except Exception as error:
        problem = f"distribution metadata unavailable: {type(error).__name__}"
    return PluginMetadata(
        PluginSource.EXTERNAL,
        entry_point.name,
        entry_point.value,
        distribution_name,
        distribution_version,
        manifest_paths,
        None,
        problem,
    )


def _distribution_manifests(distribution: _Distribution) -> tuple[Path, ...]:
    files = distribution.files
    if files is None:
        return ()
    manifests = []
    for index, path in enumerate(files):
        if index >= MAX_DISTRIBUTION_FILES:
            raise PluginDiscoveryError(
                f"distribution lists more than {MAX_DISTRIBUTION_FILES} files"
            )
        if Path(path).name == PLUGIN_MANIFEST_FILENAME:
            manifests.append(Path(distribution.locate_file(path)).resolve())
    return tuple(sorted(manifests))


def _sort_key(candidate: PluginMetadata) -> tuple[str, str, str]:
    return (
        candidate.entry_point_name,
        candidate.distribution_name or "",
        candidate.entry_point_value or "",
    )
