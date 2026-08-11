"""Compile one portable fit into native lanes and bind it to a profile."""

from __future__ import annotations

import copy
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..atomicio import write_json_atomic
from ..conversion import OvcConverter, PnnxConverter, convert_to_ncnn, convert_to_openvino
from ..preparation import PreparedArtifact, install_prepared, prepare_artifact
from ..registry import (
    bundled_workloads_path,
    load_workloads,
    validate_workload_document,
)
from .contracts import TrainingError, TrainingReport

SUPPORTED_TARGETS = ("npu", "gpu")
TARGET_ORDER = ("tpu", "npu", "gpu")


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


def available_targets() -> tuple[str, ...]:
    """Native lanes whose offline compiler exists in this producer environment."""
    found = []
    if _tool("ovc") is not None:
        found.append("npu")
    if _tool("pnnx") is not None:
        found.append("gpu")
    return tuple(found)


def install_training(
    report_path: Path | str,
    *,
    targets: Sequence[str],
    artifact_root: Path | str,
    bindings_root: Path | str,
    build_root: Path | str | None = None,
    bundled_root: Path | None = None,
) -> InstalledTraining:
    """Compile, digest-install, then atomically publish a restricted binding."""
    report_file = Path(report_path)
    report = TrainingReport.load(report_file)
    selected = _validated_targets(targets)
    output = Path(build_root) if build_root is not None else report_file.parent / "compiled"
    output.mkdir(parents=True, exist_ok=True)
    prepared = [
        _compile_and_prepare(report, report_file.parent / "model.onnx", target, output)
        for target in selected
    ]
    installed = []
    for target, artifact in prepared:
        landed = install_prepared(artifact, artifact_root)
        fragment = _model_fragment(report, artifact)
        installed.append(
            InstalledVariant(
                target,
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
    write_json_atomic(binding_path, binding, prefix=".model-binding-")
    return InstalledTraining(report.spec.profile_id, binding_path, tuple(installed))


def _validated_targets(targets: Sequence[str]) -> tuple[str, ...]:
    if isinstance(targets, (str, bytes)) or not targets:
        raise TrainingError("targets-invalid", "at least one target is required")
    selected = tuple(targets)
    if len(set(selected)) != len(selected):
        raise TrainingError("targets-invalid", "targets must be unique")
    unsupported = [target for target in selected if target not in SUPPORTED_TARGETS]
    if unsupported:
        detail = (
            "TPU export needs a matching fully-int8 TFLite graph and Edge TPU compiler"
            if unsupported[0] == "tpu"
            else f"unsupported target: {unsupported[0]}"
        )
        raise TrainingError("targets-invalid", detail)
    return tuple(target for target in TARGET_ORDER if target in selected)


def _compile_and_prepare(
    report: TrainingReport,
    portable: Path,
    target: str,
    output: Path,
) -> tuple[str, PreparedArtifact]:
    shape = f"[1,{report.spec.input_width}]"
    destination = output / target
    if target == "gpu":
        executable = _tool("pnnx")
        if executable is None:
            raise TrainingError("compiler-missing", "pnnx is required for the GPU target")
        converted = convert_to_ncnn(
            portable,
            shape,
            output_dir=destination,
            converter=PnnxConverter(str(executable)),
        )
        primary, model_format = converted.param, "ncnn"
    else:
        executable = _tool("ovc")
        if executable is None:
            raise TrainingError("compiler-missing", "ovc is required for the NPU target")
        converted = convert_to_openvino(
            portable,
            shape,
            output_dir=destination,
            converter=OvcConverter(str(executable)),
        )
        primary, model_format = converted.xml, "openvino"
    artifact = prepare_artifact(
        primary,
        artifact_id=f"{report.spec.artifact_id}-{target}",
        version=report.spec.artifact_version,
        model_format=model_format,
    )
    return target, artifact


def _model_fragment(report: TrainingReport, artifact: PreparedArtifact) -> dict:
    return {
        **artifact.manifest_fragment(),
        "fullyQuantized": False,
        "minimumCompilerVersion": "0.0.0",
        "minimumRuntimeVersion": "0.0.0",
        "tensorContract": report.spec.tensor_contract,
        "featureContract": report.spec.feature_contract,
        "outputContract": report.spec.output_contract,
    }


def _binding_manifest(
    report: TrainingReport,
    variants: Mapping[str, dict],
    *,
    bundled_root: Path | None,
) -> dict:
    workloads = load_workloads(bundled_root or bundled_workloads_path())
    workload = workloads.get(report.spec.profile_id)
    if workload is None:
        raise TrainingError(
            "profile-unknown", f"no bundled profile is named {report.spec.profile_id}"
        )
    lanes = [target for target in TARGET_ORDER if target in variants]
    manifest = copy.deepcopy(workload.manifest)
    requirements = manifest["requirements"]
    requirements.pop("model", None)
    requirements["accelerator"] = lanes[0]
    requirements["acceleratorPreference"] = lanes
    requirements["models"] = [copy.deepcopy(variants[target]) for target in lanes]
    violations = validate_workload_document(manifest)
    if violations:
        raise TrainingError("binding-invalid", "; ".join(violations))
    return manifest


def _tool(name: str) -> Path | None:
    sibling = Path(sys.executable).with_name(name)
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return sibling
    found = shutil.which(name)
    return Path(found) if found else None
