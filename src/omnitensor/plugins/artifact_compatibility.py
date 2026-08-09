"""Artifact compatibility gates for accelerator routing readiness."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .artifacts import ArtifactReference, ArtifactResolution

_ACCELERATOR_FORMATS = {
    "tpu": frozenset({"tflite-edgetpu"}),
    "npu": frozenset({"onnx", "openvino"}),
    "gpu": frozenset({"onnx", "ncnn"}),
}
_DTYPES = frozenset({"bool", "float16", "float32", "int8", "int16", "int32", "int64", "uint8"})
_SEMANTIC_VERSION = re.compile(
    r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
)
_TENSOR_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.:/-]{0,127}")
_MAX_TENSORS_PER_DIRECTION = 64
_MAX_TENSOR_RANK = 8
_MAX_TENSOR_DIMENSION = 2**31 - 1


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """One exact runtime tensor declaration."""

    name: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TensorContract:
    """Ordered model inputs and outputs."""

    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]


@dataclass(frozen=True, slots=True)
class ArtifactCompatibility:
    """Publisher-declared requirements for one model artifact."""

    model_format: str
    target_accelerator: str
    fully_quantized: bool
    minimum_runtime_version: str
    minimum_compiler_version: str
    tensor_contract: TensorContract


@dataclass(frozen=True, slots=True)
class AcceleratorRuntime:
    """Locally observed accelerator runtime and compiler capability."""

    accelerator: str
    runtime_version: str
    compiler_version: str
    supported_formats: frozenset[str]


@dataclass(frozen=True, slots=True)
class ArtifactCompatibilityDecision:
    """Stable compatibility result used by artifact readiness."""

    compatible: bool
    reason: str


class ArtifactResolutionPort(Protocol):
    """Minimum immutable artifact resolver surface."""

    def resolve(self, reference: ArtifactReference) -> ArtifactResolution:
        """Resolve one immutable reference."""


class ArtifactCompatibilityChecker:
    """Fail closed on malformed or unsupported model declarations."""

    def check(
        self,
        reference: ArtifactReference,
        compatibility: ArtifactCompatibility,
        runtime: AcceleratorRuntime,
    ) -> ArtifactCompatibilityDecision:
        """Check declared model metadata against one observed runtime."""
        type_error = _compatibility_type_error(compatibility, runtime)
        if type_error:
            return ArtifactCompatibilityDecision(False, type_error)
        declaration_error = _declaration_error(reference, compatibility)
        if declaration_error:
            return ArtifactCompatibilityDecision(False, declaration_error)
        runtime_error = _runtime_error(runtime)
        if runtime_error:
            return ArtifactCompatibilityDecision(False, runtime_error)
        if runtime.accelerator != compatibility.target_accelerator:
            return ArtifactCompatibilityDecision(
                False,
                "artifact target accelerator does not match runtime",
            )
        if reference.format not in runtime.supported_formats:
            return ArtifactCompatibilityDecision(
                False,
                f"runtime does not support artifact format: {reference.format}",
            )
        if _version_key(runtime.runtime_version) < _version_key(
            compatibility.minimum_runtime_version
        ):
            return ArtifactCompatibilityDecision(
                False,
                "artifact requires a newer accelerator runtime",
            )
        if _version_key(runtime.compiler_version) < _version_key(
            compatibility.minimum_compiler_version
        ):
            return ArtifactCompatibilityDecision(
                False,
                "artifact requires a newer accelerator compiler",
            )
        return ArtifactCompatibilityDecision(True, "")


class CompatibleArtifactResolver:
    """Report ready only after immutable and compatibility checks pass."""

    def __init__(
        self,
        resolver: ArtifactResolutionPort,
        *,
        checker: ArtifactCompatibilityChecker | None = None,
    ):
        self._resolver = resolver
        self._checker = checker or ArtifactCompatibilityChecker()

    def resolve(
        self,
        reference: ArtifactReference,
        compatibility: ArtifactCompatibility,
        runtime: AcceleratorRuntime,
    ) -> ArtifactResolution:
        """Preserve immutable failures; otherwise apply compatibility gate."""
        resolution = self._resolver.resolve(reference)
        if not resolution.ready:
            return resolution
        decision = self._checker.check(reference, compatibility, runtime)
        if not decision.compatible:
            return ArtifactResolution(False, None, decision.reason, resolution.size_bytes)
        return resolution


def _declaration_error(
    reference: ArtifactReference,
    compatibility: ArtifactCompatibility,
) -> str:
    if not isinstance(compatibility.model_format, str) or (
        compatibility.model_format != reference.format
    ):
        return "declared model format does not match artifact"
    accelerator_error = _accelerator_declaration_error(reference, compatibility)
    if accelerator_error:
        return accelerator_error
    if _version_error(compatibility.minimum_runtime_version):
        return "invalid minimum runtime version"
    if _version_error(compatibility.minimum_compiler_version):
        return "invalid minimum compiler version"
    return _tensor_contract_error(compatibility.tensor_contract)


def _accelerator_declaration_error(
    reference: ArtifactReference,
    compatibility: ArtifactCompatibility,
) -> str:
    if not isinstance(compatibility.target_accelerator, str):
        return f"unsupported target accelerator: {compatibility.target_accelerator}"
    formats = _ACCELERATOR_FORMATS.get(compatibility.target_accelerator)
    if formats is None:
        return f"unsupported target accelerator: {compatibility.target_accelerator}"
    if reference.format not in formats:
        return (
            f"artifact format {reference.format} is incompatible with "
            f"{compatibility.target_accelerator}"
        )
    if type(compatibility.fully_quantized) is not bool:
        return "fully_quantized must be a boolean"
    if compatibility.target_accelerator == "tpu" and not compatibility.fully_quantized:
        return "Edge TPU artifacts must be fully quantized"
    return ""


def _compatibility_type_error(
    compatibility: object,
    runtime: object,
) -> str:
    if not isinstance(compatibility, ArtifactCompatibility):
        return "artifact compatibility declaration is invalid"
    if not isinstance(runtime, AcceleratorRuntime):
        return "accelerator runtime declaration is invalid"
    return ""


def _runtime_error(runtime: AcceleratorRuntime) -> str:
    if not isinstance(runtime.accelerator, str) or runtime.accelerator not in _ACCELERATOR_FORMATS:
        return f"unsupported runtime accelerator: {runtime.accelerator}"
    if _version_error(runtime.runtime_version):
        return "invalid accelerator runtime version"
    if _version_error(runtime.compiler_version):
        return "invalid accelerator compiler version"
    if not isinstance(runtime.supported_formats, frozenset) or not all(
        isinstance(value, str) for value in runtime.supported_formats
    ):
        return "runtime supported formats are invalid"
    return ""


def _tensor_contract_error(contract: TensorContract) -> str:
    if not isinstance(contract, TensorContract):
        return "tensor contract is invalid"
    error = _tensor_direction_error("input", contract.inputs)
    if error:
        return error
    return _tensor_direction_error("output", contract.outputs)


def _tensor_direction_error(direction: str, tensors: object) -> str:
    if not isinstance(tensors, tuple) or not 1 <= len(tensors) <= _MAX_TENSORS_PER_DIRECTION:
        return f"tensor contract must declare 1-{_MAX_TENSORS_PER_DIRECTION} {direction}s"
    names: set[str] = set()
    for tensor in tensors:
        error = _tensor_error(direction, tensor)
        if error:
            return error
        if tensor.name in names:
            return f"duplicate {direction} tensor name: {tensor.name}"
        names.add(tensor.name)
    return ""


def _tensor_error(direction: str, tensor: object) -> str:
    if not isinstance(tensor, TensorSpec):
        return f"{direction} tensor declaration is invalid"
    if not isinstance(tensor.name, str) or not _TENSOR_NAME.fullmatch(tensor.name):
        return f"{direction} tensor name is invalid"
    if not isinstance(tensor.dtype, str) or tensor.dtype not in _DTYPES:
        return f"unsupported {direction} tensor dtype: {tensor.dtype}"
    if not isinstance(tensor.shape, tuple) or not 1 <= len(tensor.shape) <= _MAX_TENSOR_RANK:
        return f"{direction} tensor rank must be 1-{_MAX_TENSOR_RANK}"
    if any(
        type(value) is not int or not 1 <= value <= _MAX_TENSOR_DIMENSION for value in tensor.shape
    ):
        return f"{direction} tensor shape must contain positive bounded dimensions"
    return ""


def _version_error(version: object) -> bool:
    if not isinstance(version, str) or len(version) > 128:
        return True
    match = _SEMANTIC_VERSION.fullmatch(version)
    if match is None:
        return True
    prerelease = match.group(4)
    return prerelease is not None and any(
        len(value) > 1 and value.startswith("0") and value.isdigit()
        for value in prerelease.split(".")
    )


def _version_key(version: str) -> tuple[int, int, int, int, tuple[tuple[int, object], ...]]:
    match = _SEMANTIC_VERSION.fullmatch(version)
    if match is None:  # guarded by metadata validation
        raise ValueError(f"invalid semantic version: {version}")
    major, minor, patch, prerelease = match.groups()
    if prerelease is None:
        return int(major), int(minor), int(patch), 1, ()
    identifiers = tuple(
        (0, int(value)) if value.isdigit() else (1, value) for value in prerelease.split(".")
    )
    return int(major), int(minor), int(patch), 0, identifiers
