"""Artifact readiness, declared-reference lookup, and resolution caching."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from .dispatch import declared_artifact_reference
from .plugins.artifacts import ArtifactReference, ArtifactResolution, artifact_filename
from .registry import Workload
from .scheduler import runnable_model, select_backend


def profile_artifact_ready(
    workload: Workload,
    executors: dict,
    resolve_artifact: Callable[[str], ArtifactResolution],
    selected_backend_of: Callable[[Workload], str] | None = None,
) -> tuple[bool, str]:
    if not workload.models:
        return True, ""
    backend = (
        selected_backend_of(workload)
        if selected_backend_of is not None
        else selected_backend(workload, executors)
    )
    model = runnable_model(workload, backend, executors)
    if model is None:
        return True, ""
    if not model.get("sha256"):
        return False, (
            "the manifest declares this model without a sha256, so the runtime "
            "cannot tell which file it means"
        )
    resolution = resolve_artifact(model["id"])
    if getattr(resolution, "ready", False):
        return True, ""
    return False, getattr(resolution, "reason", "") or "model artifact is not installed"


def selected_backend(workload: Workload, executors: dict) -> str:
    choice = select_backend(workload, executors)
    return choice.backend or (workload.preference[0] if workload.preference else "")


def resolve_artifact(
    artifact_id: str,
    artifact_store,
    declared_reference: Callable[[str], ArtifactReference | None],
    cached_resolution: Callable[[str, ArtifactReference], ArtifactResolution],
) -> ArtifactResolution:
    if artifact_store is None:
        return ArtifactResolution(
            False, None, "no artifact store is configured for this service", 0
        )
    reference = declared_reference(artifact_id)
    if reference is None:
        return ArtifactResolution(False, None, "no manifest declares a sha256 for this artifact", 0)
    return cached_resolution(artifact_id, reference)


def resolve_plugin_artifact(reference: ArtifactReference, artifact_store) -> ArtifactResolution:
    if artifact_store is None:
        return ArtifactResolution(
            False, None, "no artifact store is configured for this service", 0
        )
    return artifact_store.resolve(reference)


def cached_resolution(
    artifact_id: str,
    reference: ArtifactReference,
    artifact_store,
    resolutions: dict[str, tuple],
    stamp: Callable[[str, ArtifactReference], tuple | None],
) -> ArtifactResolution:
    current = stamp(artifact_id, reference)
    cached = resolutions.get(artifact_id)
    if cached is not None and cached[0] == current:
        return cached[1]
    resolution = artifact_store.resolve(reference)
    resolutions[artifact_id] = (current, resolution)
    return resolution


def artifact_stamp(
    artifact_root: Path | str | None,
    artifact_id: str,
    reference: ArtifactReference,
) -> tuple | None:
    try:
        status = (
            Path(artifact_root)
            / artifact_id
            / reference.version
            / artifact_filename(reference.format)
        ).stat()
    except (OSError, ValueError):
        return None
    return (reference, status.st_size, status.st_mtime_ns)


def declared_reference(
    artifact_id: str,
    workloads: Mapping[str, Workload],
    plugins_provider: Callable[[], Sequence[object]],
) -> ArtifactReference | None:
    for workload in workloads.values():
        for model in workload.models:
            if model["id"] != artifact_id:
                continue
            reference = declared_artifact_reference(workload, model)
            if reference is not None:
                return reference
    for plugin in plugins_provider():
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


def plugin_accelerator_devices(devices: Sequence[object]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for device in devices:
        if device.backend == "gpu" and device.kind == "dri":
            paths.setdefault("gpu", Path("/dev/dri") / device.id.removeprefix("gpu-"))
        elif device.backend == "npu" and device.kind == "accel":
            paths.setdefault("npu", Path("/dev/accel") / device.id.removeprefix("npu-"))
    return paths


__all__ = [
    "artifact_stamp",
    "cached_resolution",
    "declared_reference",
    "plugin_accelerator_devices",
    "profile_artifact_ready",
    "resolve_artifact",
    "resolve_plugin_artifact",
    "selected_backend",
]
