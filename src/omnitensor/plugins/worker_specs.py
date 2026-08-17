"""Deterministic isolated-worker specifications for validated external plugins."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Collection, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Protocol

from .artifacts import ArtifactReference, ArtifactResolution
from .discovery import PluginSource
from .identity import ResolvedPlugin
from .loading_accelerator import (
    VULKAN_METADATA_ROOTS,
    accelerator_lease_path,
    accelerator_paths,
    vulkan_sysfs_resources,
)
from .loading_staging import _STAGING_COMPONENT
from .sandbox import FilesystemSandbox
from .supervisor_process import WorkerSpec
from .worker_budgets import limits_for

MAX_WORKER_IMPORT_PATHS = 16
WORKER_CAPABILITIES = frozenset({"cancel", "execute", "health", "progress"})
TIMEZONE_METADATA_ROOT = Path("/usr/share/zoneinfo")
_LEGACY_MODULE = "omnitensor.plugins.loading"


class ArtifactProvider(Protocol):
    def __call__(self, reference: ArtifactReference) -> ArtifactResolution: ...


ArtifactProvider.__module__ = _LEGACY_MODULE


def external_worker_specs(
    plugins: Sequence[ResolvedPlugin],
    *,
    python_executable: str | Path = sys.executable,
    worker_import_paths: Sequence[Path] = (),
    granted_permissions: Mapping[str, Collection[str]] | None = None,
    selected_files_root: Path | None = None,
    worker_state_root: Path | None = None,
    resolve_artifact: ArtifactProvider | None = None,
    accelerator_devices: Mapping[str, Path] | None = None,
    accelerator_devices_by_plugin: Mapping[str, Mapping[str, Path]] | None = None,
    model_choices: Mapping[str, str] | None = None,
) -> tuple[WorkerSpec, ...]:
    """Build deterministic argv without importing plugin code in the service."""
    executable = executable_path(python_executable)
    import_paths = worker_import_paths_tuple(worker_import_paths)
    permissions_by_plugin = granted_permissions or {}
    device_map = dict(accelerator_devices or {})
    per_plugin = accelerator_devices_by_plugin or {}
    chosen_models = model_choices or {}
    specs = []
    for plugin in plugins:
        if plugin.source is not PluginSource.EXTERNAL:
            continue
        try:
            spec = external_worker_spec(
                plugin,
                executable=executable,
                import_paths=import_paths,
                granted=frozenset(permissions_by_plugin.get(plugin.plugin_id, ())),
                selected_files_root=selected_files_root,
                worker_state_root=worker_state_root,
                resolve_artifact=resolve_artifact,
                accelerator_devices=dict(per_plugin.get(plugin.plugin_id, device_map)),
                model_choice=str(chosen_models.get(plugin.plugin_id, "")),
            )
        except ValueError as error:
            if plugin.plugin_id in per_plugin and str(error).startswith(
                "granted accelerator device is unavailable:"
            ):
                # A saved per-profile choice remains saved when the GPU is
                # unplugged, but no worker may be built on another GPU.
                continue
            raise
        specs.append(spec)
    return tuple(specs)


def external_worker_spec(
    plugin: ResolvedPlugin,
    *,
    executable: str,
    import_paths: tuple[str, ...],
    granted: frozenset[str],
    selected_files_root: Path | None,
    worker_state_root: Path | None,
    resolve_artifact: ArtifactProvider | None,
    accelerator_devices: Mapping[str, Path],
    model_choice: str = "",
) -> WorkerSpec:
    protocol = plugin.manifest["plugin"]["protocol"]
    declared = frozenset(plugin.manifest["plugin"]["permissions"])
    argv = worker_argv(plugin, executable, import_paths, granted)
    resolved = resolved_artifacts(plugin, resolve_artifact)
    for reference, resolution in resolved:
        argv.extend(("--artifact", artifact_bootstrap(reference, resolution)))
    state_path = plugin_state_path(worker_state_root, plugin.plugin_id)
    if state_path is not None:
        argv.extend(("--state-path", str(state_path)))
    device_paths = accelerator_paths(declared, granted, accelerator_devices)
    lease_path = accelerator_lease_path(worker_state_root, declared, granted)
    if lease_path is not None:
        argv.extend(("--accelerator-lease-path", str(lease_path)))
    if model_choice:
        argv.extend(("--model-choice", model_choice))
    runtime_paths = [
        *trusted_runtime_paths(import_paths),
        *((TIMEZONE_METADATA_ROOT,) if TIMEZONE_METADATA_ROOT.is_dir() else ()),
        *(
            path
            for reference, resolution in resolved
            for path in artifact_paths(reference, resolution)
        ),
    ]
    sysfs_paths: tuple[Path, ...] = ()
    sysfs_links: tuple[tuple[str, Path], ...] = ()
    if device_paths:
        runtime_paths.extend(path for path in VULKAN_METADATA_ROOTS if path.is_dir())
        sysfs_paths, sysfs_links = vulkan_sysfs_resources(device_paths)
        runtime_paths.extend(sysfs_paths)
    return WorkerSpec(
        plugin.plugin_id,
        tuple(argv),
        budget_limits=limits_for(plugin.manifest),
        minimum_protocol=protocol["minimum"],
        maximum_protocol=protocol["maximum"],
        capabilities=frozenset(protocol.get("capabilities", ())) & WORKER_CAPABILITIES,
        sandbox=FilesystemSandbox.from_permissions(
            declared,
            granted,
            runtime_paths=runtime_paths,
            trusted_write_paths=tuple(
                path for path in (state_path, lease_path) if path is not None
            ),
            accelerator_devices=device_paths,
            trusted_symlinks=sysfs_links,
            python_path=Path(__file__).resolve().parents[2],
            selected_files_root=selected_files_root,
        ),
    )


def worker_argv(
    plugin: ResolvedPlugin,
    executable: str,
    import_paths: tuple[str, ...],
    granted: frozenset[str],
) -> list[str]:
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
    return argv


def artifact_bootstrap(
    reference: ArtifactReference,
    resolution: ArtifactResolution,
) -> str:
    return json.dumps(
        {
            "id": reference.id,
            "version": reference.version,
            "format": reference.format,
            "sha256": reference.sha256,
            "companions": reference.declared_companions,
            "path": str(resolution.path),
            "sourceUri": reference.source_uri,
            "licenseSpdx": reference.license_spdx,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def artifact_paths(
    reference: ArtifactReference,
    resolution: ArtifactResolution,
) -> tuple[Path, ...]:
    paths = [resolution.path]
    paths.extend(resolution.path.parent / name for name in reference.declared_companions)
    if any(not path.is_file() for path in paths):
        raise ValueError("resolved plugin artifact companions are unavailable")
    return tuple(paths)


def prepare_worker_state_root(root: Path | None) -> Path | None:
    if root is None:
        return None
    candidate = Path(root)
    if not candidate.is_absolute():
        raise ValueError("worker_state_root must be absolute")
    candidate.mkdir(parents=True, exist_ok=True, mode=0o700)
    candidate.chmod(0o700)
    return candidate.resolve()


def plugin_state_path(root: Path | None, plugin_id: str) -> Path | None:
    if root is None:
        return None
    if _STAGING_COMPONENT.fullmatch(plugin_id) is None:
        raise ValueError("plugin identity is invalid for state storage")
    path = root / plugin_id
    path.mkdir(exist_ok=True)
    path.chmod(0o700)
    return path.resolve()


def resolved_artifacts(
    plugin: ResolvedPlugin,
    resolver: ArtifactProvider | None,
) -> tuple[tuple[ArtifactReference, ArtifactResolution], ...]:
    resolved = []
    for declared in plugin.manifest["plugin"]["artifacts"]:
        reference = ArtifactReference(
            declared["id"],
            declared["version"],
            declared["format"],
            declared["sha256"],
            tuple(sorted((declared.get("companions") or {}).items())),
            declared.get("sourceUri", ""),
            declared.get("licenseSpdx", ""),
        )
        resolution = (
            resolver(reference)
            if resolver is not None
            else ArtifactResolution(False, None, "no artifact resolver is configured", 0)
        )
        if not resolution.ready or resolution.path is None:
            continue
        resolved.append((reference, resolution))
    return tuple(resolved)


def entry_points_from_distributions(paths: Sequence[Path]) -> Callable[..., tuple]:
    """Create a metadata provider for an isolated installation root."""
    roots = worker_import_paths_tuple(paths)

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


def executable_path(value: str | Path) -> str:
    executable = str(value)
    if not executable or "\0" in executable:
        raise ValueError("python_executable must be a non-empty path")
    return executable


def worker_import_paths_tuple(paths: Sequence[Path]) -> tuple[str, ...]:
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


def trusted_runtime_paths(import_paths: Sequence[str]) -> tuple[Path | str, ...]:
    return (
        Path(sys.prefix),
        Path(__file__).resolve().parents[2],
        *import_paths,
    )


__all__ = ["ArtifactProvider", "entry_points_from_distributions", "external_worker_specs"]
