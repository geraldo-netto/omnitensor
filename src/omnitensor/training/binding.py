"""Shared construction and publication for accelerator-bound training models."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from ..atomicio import write_json_atomic
from ..registry import Workload

ManifestLoader = Callable[[Path], Mapping[str, Workload]]
ManifestValidator = Callable[[dict], list[str]]
BindingErrorType = type[Exception]
Variants = Mapping[str, Mapping[str, object]]
VariantSource = Variants | Callable[[], Variants]


def native_evidence(
    *,
    portable_sha256: str,
    native_sha256: str,
    report_sha256: str,
    samples: int,
    maximum_absolute_error: int | float,
    tolerance: int | float,
    compiler_report_sha256: str | None,
    named_device_accepted: bool,
) -> dict:
    """Build the common portable/native proof block in canonical field order."""
    return {
        "portableSha256": portable_sha256,
        "nativeSha256": native_sha256,
        "reportSha256": report_sha256,
        "samples": samples,
        "maximumAbsoluteError": maximum_absolute_error,
        "tolerance": tolerance,
        "compilerReportSha256": compiler_report_sha256,
        "namedDeviceAccepted": named_device_accepted,
    }


def model_fragment(
    artifact: Mapping[str, object],
    *,
    fully_quantized: bool,
    minimum_compiler_version: str,
    minimum_runtime_version: str,
    tensor_contract: Mapping[str, object],
    output_contract: Mapping[str, object],
    feature_contract: Mapping[str, object] | None = None,
    training_contract: Mapping[str, object] | None = None,
    evidence: Mapping[str, object] | None = None,
) -> dict:
    """Combine an immutable artifact identity with one family's contracts."""
    fragment = {
        **copy.deepcopy(artifact),
        "fullyQuantized": fully_quantized,
        "minimumCompilerVersion": minimum_compiler_version,
        "minimumRuntimeVersion": minimum_runtime_version,
        "tensorContract": copy.deepcopy(tensor_contract),
    }
    if feature_contract is not None:
        fragment["featureContract"] = copy.deepcopy(feature_contract)
    fragment["outputContract"] = copy.deepcopy(output_contract)
    if training_contract is not None:
        fragment["trainingContract"] = copy.deepcopy(training_contract)
    if evidence is not None:
        fragment["nativeEvidence"] = copy.deepcopy(evidence)
    return fragment


def binding_manifest(
    profile_id: str,
    variants: VariantSource,
    *,
    lane_order: Sequence[str],
    plural: bool,
    root: Path,
    load: ManifestLoader,
    validate: ManifestValidator,
    error_type: BindingErrorType,
) -> dict:
    """Restrict a copied profile to available variants in caller-defined order."""
    workload = load(root).get(profile_id)
    if workload is None:
        raise error_type("profile-unknown", f"no bundled profile is named {profile_id}")
    resolved = variants() if callable(variants) else variants
    lanes = [lane for lane in lane_order if lane in resolved]
    if not lanes:
        raise error_type(
            "binding-invalid",
            f"{profile_id} has no variant for any of the requested lanes "
            f"{', '.join(lane_order) or '(none)'}",
        )
    manifest = copy.deepcopy(workload.manifest)
    requirements = manifest["requirements"]
    if plural:
        requirements.pop("model", None)
        requirements["accelerator"] = lanes[0]
        requirements["acceleratorPreference"] = lanes
        requirements["models"] = [copy.deepcopy(resolved[lane]) for lane in lanes]
    else:
        requirements["accelerator"] = lanes[0]
        requirements["acceleratorPreference"] = lanes
        requirements["model"] = copy.deepcopy(resolved[lanes[0]])
        requirements.pop("models", None)
    violations = validate(manifest)
    if violations:
        raise error_type("binding-invalid", "; ".join(violations))
    return manifest


def publish_binding(
    path: Path | str,
    document: dict,
    *,
    prefix: str,
    writer: Callable = write_json_atomic,
) -> None:
    """Atomically publish one already validated binding without rewriting it."""
    writer(Path(path), document, prefix=prefix)
