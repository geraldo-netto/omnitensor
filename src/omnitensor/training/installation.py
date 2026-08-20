"""Compile one portable fit into native lanes and bind it to a profile."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from omnitensor.preparation import (
    PreparedArtifact,
    install_prepared,
    prepare_artifact,
    uninstall_prepared,
)

from ..atomicio import write_json_atomic
from ..registry import (
    bundled_workloads_path,
    load_workloads,
    validate_workload_document,
)
from .binding import binding_manifest, model_fragment, publish_binding
from .compilers import (
    TARGET_ORDER,
    CompilationRequest,
    CompiledTarget,
    CompilerError,
    TargetCompiler,
    compatible_available_targets,
    compiler_catalog,
    default_target_compilers,
    resolve_compiler_tool,
)
from .contracts import TrainingError, TrainingReport

LOGGER = logging.getLogger(__name__)

SUPPORTED_TARGETS = TARGET_ORDER


@dataclass(frozen=True, slots=True)
class InstalledVariant:
    accelerator: str
    artifact_id: str
    model_format: str
    installed_at: Path
    manifest_fragment: dict


@dataclass(frozen=True, slots=True)
class InstalledTraining:
    profile_id: str
    binding_path: Path
    variants: tuple[InstalledVariant, ...]

    def document(self) -> dict:
        return {
            "profileId": self.profile_id,
            "bindingPath": str(self.binding_path),
            "variants": [
                {
                    "accelerator": variant.accelerator,
                    "artifactId": variant.artifact_id,
                    "format": variant.model_format,
                    "installedAt": str(variant.installed_at),
                }
                for variant in self.variants
            ],
            "restartRequired": True,
        }


def available_targets(
    *,
    source_format: str = "onnx",
    fully_quantized: bool = False,
    compilers: Sequence[TargetCompiler] | None = None,
) -> tuple[str, ...]:
    """Available native lanes compatible with the stated portable source."""
    providers = tuple(compilers) if compilers is not None else default_target_compilers(_tool)
    try:
        return compatible_available_targets(source_format, fully_quantized, providers)
    except CompilerError as error:
        raise TrainingError(error.code, error.detail) from error


def install_training(
    report_path: Path | str,
    *,
    targets: Sequence[str],
    artifact_root: Path | str,
    bindings_root: Path | str,
    build_root: Path | str | None = None,
    bundled_root: Path | None = None,
    compilers: Sequence[TargetCompiler] | None = None,
) -> InstalledTraining:
    """Compile, digest-install, then atomically publish a restricted binding."""
    report_file = Path(report_path)
    report = TrainingReport.load(report_file)
    selected = validated_targets(targets)
    providers = tuple(compilers) if compilers is not None else default_target_compilers(_tool)
    try:
        catalog = compiler_catalog(providers)
    except CompilerError as error:
        raise TrainingError(error.code, error.detail) from error
    output = Path(build_root) if build_root is not None else report_file.parent / "compiled"
    portable = report_file.parent / "model.onnx"
    request = CompilationRequest(portable, "onnx", (1, report.spec.input_width), False)
    prepared = []
    for target in selected:
        compiler = catalog.get(target)
        if compiler is None:
            raise TrainingError("compiler-missing", f"no compiler provider for {target}")
        prepared.append(_compile_and_prepare(report, request, compiler, output))
    installed: list[InstalledVariant] = []
    landed_references = []
    try:
        for compiled, artifact in prepared:
            landed = install_prepared(artifact, artifact_root)
            landed_references.append(artifact.reference)
            fragment = _model_fragment(report, artifact, compiled.fully_quantized)
            installed.append(
                InstalledVariant(
                    compiled.accelerator,
                    artifact.reference.id,
                    artifact.reference.format,
                    landed,
                    fragment,
                )
            )
        binding = _binding_manifest(
            report,
            {variant.accelerator: variant.manifest_fragment for variant in installed},
            bundled_root=bundled_root,
        )
        binding_path = Path(bindings_root) / report.spec.profile_id / "manifest.json"
        publish_binding(
            binding_path,
            binding,
            prefix=".model-binding-",
            writer=write_json_atomic,
        )
    except BaseException:
        # All the variants or none: failing on the second of three left the
        # first in the store with no binding naming it, an orphan nothing
        # cleaned and a retry installed beside.
        remove_landed_artifacts(landed_references, artifact_root)
        raise
    return InstalledTraining(report.spec.profile_id, binding_path, tuple(installed))


def remove_landed_artifacts(references, artifact_root) -> None:
    """Undo a part-finished publication, reporting nothing it cannot undo."""
    for reference in reversed(list(references)):
        try:
            uninstall_prepared(reference, artifact_root)
        except Exception:  # noqa: BLE001 - cleanup must not mask the real failure
            LOGGER.exception("could not remove partially installed %s", reference.id)


def validated_targets(targets: Sequence[str]) -> tuple[str, ...]:
    if isinstance(targets, (str, bytes)) or not targets:
        raise TrainingError("targets-invalid", "at least one target is required")
    selected = tuple(targets)
    if len(set(selected)) != len(selected):
        raise TrainingError("targets-invalid", "targets must be unique")
    unsupported = [target for target in selected if target not in SUPPORTED_TARGETS]
    if unsupported:
        detail = f"unsupported target: {unsupported[0]}"
        raise TrainingError("targets-invalid", detail)
    return tuple(target for target in TARGET_ORDER if target in selected)


def _compile_and_prepare(
    report: TrainingReport,
    request: CompilationRequest,
    compiler: TargetCompiler,
    output: Path,
) -> tuple[CompiledTarget, PreparedArtifact]:
    destination = output / compiler.accelerator
    try:
        compiled = compiler.compile(request, destination)
    except CompilerError as error:
        raise TrainingError(error.code, error.detail) from error
    artifact = prepare_artifact(
        compiled.primary,
        artifact_id=f"{report.spec.artifact_id}-{compiled.accelerator}",
        version=report.spec.artifact_version,
        model_format=compiled.model_format,
    )
    return compiled, artifact


def _model_fragment(
    report: TrainingReport, artifact: PreparedArtifact, fully_quantized: bool
) -> dict:
    return model_fragment(
        artifact.manifest_fragment(),
        fully_quantized=fully_quantized,
        minimum_compiler_version="0.0.0",
        minimum_runtime_version="0.0.0",
        tensor_contract=report.spec.tensor_contract,
        feature_contract=report.spec.feature_contract,
        output_contract=report.spec.output_contract,
    )


def _binding_manifest(
    report: TrainingReport,
    variants: Mapping[str, dict],
    *,
    bundled_root: Path | None,
) -> dict:
    return binding_manifest(
        report.spec.profile_id,
        variants,
        lane_order=TARGET_ORDER,
        plural=True,
        root=bundled_root or bundled_workloads_path(),
        load=load_workloads,
        validate=validate_workload_document,
        error_type=TrainingError,
    )


def _tool(name: str) -> Path | None:
    """Compatibility seam for callers patching installation host discovery."""
    return resolve_compiler_tool(
        name,
        python_executable=sys.executable,
        executable_check=os.access,
        path_search=shutil.which,
    )
