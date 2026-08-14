"""Evidence-gated promotion of portable numeric training reports.

Compilation is deliberately not acceptance.  This producer boundary requires
an injected native parity runner and an offline publisher signature before it
installs a GPU or NPU variant.  Named-device profile acceptance remains a
separate consumer gate, and ONNX reports cannot enter the TPU lane.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..atomicio import JsonTooLargeError, read_json_bounded, write_json_atomic
from ..plugins.artifact_installation import ArtifactInstaller
from ..plugins.artifact_trust import ArtifactProvenance, ArtifactTrustVerifier
from ..plugins.artifacts import ArtifactReference
from ..preparation import PreparedArtifact, file_digest, prepare_artifact
from ..registry import bundled_workloads_path, load_workloads, validate_workload_document
from .compilers import (
    CompilationRequest,
    CompiledTarget,
    CompilerError,
    TargetCompiler,
    compiler_catalog,
)
from .contracts import TrainingError, validate_training_report_document
from .installation import InstalledTraining, InstalledVariant, _validated_targets

MAX_NUMERIC_REPORT_BYTES = 1024 * 1024
MAX_PARITY_ERROR = 1e-4
NUMERIC_RECIPES = {
    "backblaze-smart-risk-v1": "storage-intelligence",
    "aggregate-reconstruction-v1": "network-peripherals",
    "metadata-risk-ranking-v1": "build-advisor",
    "labelled-sensor-window-v1": "hardware-health",
    "confirmed-layout-suggestion-v1": "desktop-context",
}
_SEMANTIC_FIELDS = (
    "featureContract",
    "features",
    "label",
    "outputs",
    "resultContract",
    "split",
)


@dataclass(frozen=True, slots=True)
class NumericTrainingReport:
    """Verified common boundary shared by five profile-specific reports."""

    profile_id: str
    recipe: str
    report_path: Path
    model_path: Path
    model_sha256: str
    report_sha256: str
    semantics_sha256: str
    tensor_contract: dict
    output_contract: dict

    @classmethod
    def load(cls, path: Path | str) -> NumericTrainingReport:
        report_path = Path(path)
        document = _load_numeric_document(report_path)
        recipe = document["recipe"]
        profile_id = NUMERIC_RECIPES[recipe]
        model_path, observed_model_digest = _portable_model_identity(report_path, document)
        tensor_contract = document["tensorContract"]
        output_contract = document["outputContract"]
        report_digest = file_digest(report_path)
        semantics = {field: document[field] for field in _SEMANTIC_FIELDS if field in document}
        semantics_digest = _document_digest(semantics)
        return cls(
            profile_id,
            recipe,
            report_path,
            model_path,
            observed_model_digest,
            report_digest,
            semantics_digest,
            copy.deepcopy(tensor_contract),
            copy.deepcopy(output_contract),
        )

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(self.tensor_contract["inputs"][0]["shape"])

    @property
    def training_contract(self) -> dict:
        return {
            "version": 1,
            "profileId": self.profile_id,
            "recipe": self.recipe,
            "reportSha256": self.report_sha256,
            "taskSemanticsSha256": self.semantics_sha256,
        }


def _load_numeric_document(report_path: Path) -> dict:
    """Read one bounded versioned report document."""
    try:
        document = read_json_bounded(report_path, MAX_NUMERIC_REPORT_BYTES)
    except (OSError, ValueError, JsonTooLargeError) as error:
        raise TrainingError("report-invalid", f"cannot read numeric report: {error}") from error
    if not isinstance(document, dict) or document.get("version") != 1:
        raise TrainingError("report-invalid", "numeric report version is unsupported")
    if document.get("recipe") not in NUMERIC_RECIPES:
        raise TrainingError("report-invalid", "numeric report recipe is unsupported")
    model = document.get("model")
    if (
        not isinstance(model, dict)
        or any(name not in model for name in ("format", "filename", "sha256"))
        or not isinstance(model.get("sha256"), str)
    ):
        raise TrainingError("report-invalid", "portable model declaration is invalid")
    if model.get("format") != "onnx" or model.get("filename") != "model.onnx":
        raise TrainingError("report-invalid", "numeric report must declare model.onnx")
    _validate_numeric_contract(document.get("tensorContract"), document.get("outputContract"))
    if document.get("targets") != {
        "tpu": "uncompiled",
        "npu": "uncompiled",
        "gpu": "uncompiled",
    }:
        raise TrainingError("report-invalid", "numeric report must not claim native targets")
    validate_training_report_document(document, numeric=True)
    return document


def _portable_model_identity(report_path: Path, document: dict) -> tuple[Path, str]:
    """Bind one exact local ONNX file to its report declaration."""
    model = document.get("model")
    if (
        not isinstance(model, dict)
        or any(name not in model for name in ("format", "filename", "sha256"))
        or not isinstance(model.get("sha256"), str)
    ):
        raise TrainingError("report-invalid", "portable model declaration is invalid")
    if model["format"] != "onnx" or model["filename"] != "model.onnx":
        raise TrainingError("report-invalid", "numeric report must declare model.onnx")
    model_path = report_path.parent / "model.onnx"
    try:
        observed_model_digest = file_digest(model_path)
    except OSError as error:
        raise TrainingError("model-invalid", f"cannot read portable model: {error}") from error
    if observed_model_digest != model["sha256"]:
        raise TrainingError("model-invalid", "portable model does not match numeric report")
    return model_path, observed_model_digest


@dataclass(frozen=True, slots=True)
class NativeParityEvidence:
    """Portable/native comparison supplied by a producer-side lane runner."""

    samples: int
    maximum_absolute_error: float
    portable_sha256: str
    native_sha256: str


class NativeParityVerifier(Protocol):
    """Compare portable and compiled outputs without changing runtime policy."""

    def verify(
        self,
        report: NumericTrainingReport,
        compiled: CompiledTarget,
    ) -> NativeParityEvidence: ...


ArtifactSigner = Callable[[ArtifactReference, str], ArtifactProvenance]


def promote_numeric_training(
    report_path: Path | str,
    *,
    artifact_id: str,
    variant_versions: Mapping[str, str],
    targets: Sequence[str],
    artifact_root: Path | str,
    bindings_root: Path | str,
    compilers: Sequence[TargetCompiler],
    parity_verifier: NativeParityVerifier,
    signer: ArtifactSigner,
    trust_verifier: ArtifactTrustVerifier,
    build_root: Path | str | None = None,
    bundled_root: Path | None = None,
) -> InstalledTraining:
    """Compile, parity-gate, sign, install, and bind one numeric report."""
    report = NumericTrainingReport.load(report_path)
    selected = _validated_targets(targets)
    if "tpu" in selected:
        raise TrainingError(
            "source-incompatible",
            "numeric ONNX reports need a separate representative fully-int8 TPU export",
        )
    if set(variant_versions) != set(selected):
        raise TrainingError("versions-invalid", "variant versions must exactly match targets")
    try:
        catalog = compiler_catalog(compilers)
    except CompilerError as error:
        raise TrainingError(error.code, error.detail) from error
    output = Path(build_root) if build_root is not None else report.report_path.parent / "compiled"
    request = CompilationRequest(report.model_path, "onnx", report.input_shape, False)
    prepared: list[tuple[CompiledTarget, PreparedArtifact, NativeParityEvidence]] = []
    for target in selected:
        compiler = catalog.get(target)
        if compiler is None:
            raise TrainingError("compiler-missing", f"no compiler provider for {target}")
        try:
            compiled = compiler.compile(request, output / target)
        except CompilerError as error:
            raise TrainingError(error.code, error.detail) from error
        artifact = prepare_artifact(
            compiled.primary,
            artifact_id=f"{artifact_id}-{target}",
            version=variant_versions[target],
            model_format=compiled.model_format,
        )
        evidence = parity_verifier.verify(report, compiled)
        _validate_parity(report, artifact, evidence)
        prepared.append((compiled, artifact, evidence))
    installer = ArtifactInstaller(Path(artifact_root), trust_verifier=trust_verifier)
    installed = []
    for compiled, artifact, evidence in prepared:
        provenance = signer(artifact.reference, report.report_sha256)
        landed = installer.install(
            artifact.reference,
            artifact.source,
            companions=artifact.companions,
            provenance=provenance,
        ).path
        fragment = _numeric_model_fragment(report, artifact, compiled, evidence)
        installed.append(
            InstalledVariant(
                compiled.accelerator,
                artifact.reference.id,
                artifact.reference.format,
                landed,
                fragment,
            )
        )
    binding = _numeric_binding_manifest(
        report,
        {variant.accelerator: variant.manifest_fragment for variant in installed},
        bundled_root=bundled_root,
    )
    binding_path = Path(bindings_root) / report.profile_id / "manifest.json"
    write_json_atomic(binding_path, binding, prefix=".numeric-model-binding-")
    return InstalledTraining(report.profile_id, binding_path, tuple(installed))


def _validate_numeric_contract(tensor_contract: object, output_contract: object) -> None:
    if not isinstance(output_contract, dict) or output_contract.get("kind") != "raw":
        raise TrainingError("report-invalid", "numeric report output contract must be raw")
    if not isinstance(tensor_contract, dict) or "inputs" not in tensor_contract:
        raise TrainingError("report-invalid", "numeric tensor contract is invalid")
    inputs = tensor_contract["inputs"]
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise TrainingError("report-invalid", "numeric report requires one input")
    item = inputs[0]
    if not isinstance(item, dict) or item.get("dtype") != "float32" or item.get("layout") != "NC":
        raise TrainingError("report-invalid", "numeric input must be float32 NC")
    shape = item.get("shape")
    if (
        not isinstance(shape, list)
        or not 2 <= len(shape) <= 6
        or shape[0] != 1
        or any(type(value) is not int or not 1 <= value <= 65536 for value in shape)
    ):
        raise TrainingError("report-invalid", "numeric input shape is invalid")


def _validate_parity(
    report: NumericTrainingReport,
    artifact: PreparedArtifact,
    evidence: NativeParityEvidence,
) -> None:
    if not isinstance(evidence, NativeParityEvidence):
        raise TrainingError("parity-invalid", "native parity evidence is not typed")
    if type(evidence.samples) is not int or evidence.samples < 1:
        raise TrainingError("parity-invalid", "native parity samples must be positive")
    error = evidence.maximum_absolute_error
    if not isinstance(error, (int, float)) or isinstance(error, bool) or not math.isfinite(error):
        raise TrainingError("parity-invalid", "native parity error must be finite")
    if error < 0 or error > MAX_PARITY_ERROR:
        raise TrainingError("parity-failed", "native output exceeds portable numerical tolerance")
    if evidence.portable_sha256 != report.model_sha256:
        raise TrainingError("parity-invalid", "parity evidence names another portable model")
    if evidence.native_sha256 != artifact.reference.sha256:
        raise TrainingError("parity-invalid", "parity evidence names another native model")


def _numeric_model_fragment(
    report: NumericTrainingReport,
    artifact: PreparedArtifact,
    compiled: CompiledTarget,
    evidence: NativeParityEvidence,
) -> dict:
    compiler_report_sha256 = (
        file_digest(compiled.compiler_report) if compiled.compiler_report is not None else None
    )
    return {
        **artifact.manifest_fragment(),
        "fullyQuantized": compiled.fully_quantized,
        "minimumCompilerVersion": "0.0.0",
        "minimumRuntimeVersion": "0.0.0",
        "tensorContract": copy.deepcopy(report.tensor_contract),
        "outputContract": copy.deepcopy(report.output_contract),
        "trainingContract": report.training_contract,
        "nativeEvidence": {
            "portableSha256": evidence.portable_sha256,
            "nativeSha256": evidence.native_sha256,
            "reportSha256": report.report_sha256,
            "samples": evidence.samples,
            "maximumAbsoluteError": float(evidence.maximum_absolute_error),
            "tolerance": MAX_PARITY_ERROR,
            "compilerReportSha256": compiler_report_sha256,
            "namedDeviceAccepted": False,
        },
    }


def _numeric_binding_manifest(
    report: NumericTrainingReport,
    variants: Mapping[str, dict],
    *,
    bundled_root: Path | None,
) -> dict:
    workloads = load_workloads(bundled_root or bundled_workloads_path())
    workload = workloads.get(report.profile_id)
    if workload is None:
        raise TrainingError("profile-unknown", f"no bundled profile is named {report.profile_id}")
    lanes = [target for target in ("npu", "gpu") if target in variants]
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


def _document_digest(document: object) -> str:
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
