from __future__ import annotations

import hashlib

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins import (
    AcceleratorRuntime,
    ArtifactCompatibility,
    ArtifactCompatibilityChecker,
    ArtifactCompatibilityDecision,
    ArtifactReference,
    ArtifactResolution,
    CompatibleArtifactResolver,
    TensorContract,
    TensorSpec,
)


def artifact(model_format: str = "onnx") -> ArtifactReference:
    return ArtifactReference(
        "sample-model",
        "1.2.3",
        model_format,
        hashlib.sha256(b"model").hexdigest(),
    )


def tensor_contract() -> TensorContract:
    return TensorContract(
        inputs=(TensorSpec("image", "float32", (1, 3, 224, 224)),),
        outputs=(TensorSpec("scores", "float32", (1, 1000)),),
    )


def compatibility(**changes) -> ArtifactCompatibility:
    values = {
        "model_format": "onnx",
        "target_accelerator": "gpu",
        "fully_quantized": False,
        "minimum_runtime_version": "1.17.0",
        "minimum_compiler_version": "1.16.0",
        "tensor_contract": tensor_contract(),
    }
    values.update(changes)
    return ArtifactCompatibility(**values)


def runtime(**changes) -> AcceleratorRuntime:
    values = {
        "accelerator": "gpu",
        "runtime_version": "1.18.0",
        "compiler_version": "1.16.1",
        "supported_formats": frozenset({"onnx", "ncnn"}),
    }
    values.update(changes)
    return AcceleratorRuntime(**values)


def test_compatible_artifact_passes_every_gate():
    decision = ArtifactCompatibilityChecker().check(artifact(), compatibility(), runtime())
    assert decision == ArtifactCompatibilityDecision(True, "")


@pytest.mark.parametrize(
    ("model_format", "target", "quantized"),
    [
        ("tflite-edgetpu", "tpu", True),
        ("onnx", "npu", False),
        ("openvino", "npu", False),
        ("onnx", "gpu", False),
        ("ncnn", "gpu", False),
    ],
)
def test_executor_format_matrix_matches_runtime_lanes(model_format, target, quantized):
    model = artifact(model_format)
    declared = compatibility(
        model_format=model_format,
        target_accelerator=target,
        fully_quantized=quantized,
    )
    observed = runtime(accelerator=target, supported_formats=frozenset({model_format}))
    assert ArtifactCompatibilityChecker().check(model, declared, observed).compatible is True


@pytest.mark.parametrize(
    ("model_format", "target", "reason"),
    [
        ("onnx", "tpu", "artifact format onnx is incompatible with tpu"),
        (
            "tflite-edgetpu",
            "gpu",
            "artifact format tflite-edgetpu is incompatible with gpu",
        ),
        ("openvino", "gpu", "artifact format openvino is incompatible with gpu"),
    ],
)
def test_incompatible_format_target_pairs_fail_closed(model_format, target, reason):
    model = artifact(model_format)
    decision = ArtifactCompatibilityChecker().check(
        model,
        compatibility(model_format=model_format, target_accelerator=target),
        runtime(accelerator=target, supported_formats=frozenset({model_format})),
    )
    assert decision == ArtifactCompatibilityDecision(False, reason)


@pytest.mark.parametrize(
    ("declared_changes", "reason"),
    [
        ({"model_format": "ncnn"}, "declared model format does not match artifact"),
        ({"target_accelerator": "cpu"}, "unsupported target accelerator: cpu"),
        ({"fully_quantized": 1}, "fully_quantized must be a boolean"),
        ({"minimum_runtime_version": "1.17"}, "invalid minimum runtime version"),
        ({"minimum_compiler_version": "latest"}, "invalid minimum compiler version"),
    ],
)
def test_invalid_artifact_declarations_have_stable_reasons(declared_changes, reason):
    decision = ArtifactCompatibilityChecker().check(
        artifact(), compatibility(**declared_changes), runtime()
    )
    assert decision == ArtifactCompatibilityDecision(False, reason)


def test_checker_rejects_wrong_declaration_types():
    checker = ArtifactCompatibilityChecker()
    assert checker.check(artifact(), None, runtime()) == ArtifactCompatibilityDecision(
        False, "artifact compatibility declaration is invalid"
    )
    assert checker.check(artifact(), compatibility(), None) == ArtifactCompatibilityDecision(
        False, "accelerator runtime declaration is invalid"
    )


def test_edge_tpu_requires_explicit_full_quantization():
    model = artifact("tflite-edgetpu")
    decision = ArtifactCompatibilityChecker().check(
        model,
        compatibility(
            model_format="tflite-edgetpu",
            target_accelerator="tpu",
            fully_quantized=False,
        ),
        runtime(accelerator="tpu", supported_formats=frozenset({"tflite-edgetpu"})),
    )
    assert decision.reason == "Edge TPU artifacts must be fully quantized"


@pytest.mark.parametrize(
    ("runtime_changes", "reason"),
    [
        ({"accelerator": "cpu"}, "unsupported runtime accelerator: cpu"),
        ({"runtime_version": "1.18"}, "invalid accelerator runtime version"),
        ({"runtime_version": "1.0.0-01"}, "invalid accelerator runtime version"),
        ({"compiler_version": "current"}, "invalid accelerator compiler version"),
        ({"supported_formats": {"onnx"}}, "runtime supported formats are invalid"),
        ({"accelerator": "npu"}, "artifact target accelerator does not match runtime"),
        (
            {"supported_formats": frozenset({"ncnn"})},
            "runtime does not support artifact format: onnx",
        ),
        ({"runtime_version": "1.16.9"}, "artifact requires a newer accelerator runtime"),
        ({"compiler_version": "1.15.9"}, "artifact requires a newer accelerator compiler"),
    ],
)
def test_runtime_mismatches_have_stable_reasons(runtime_changes, reason):
    decision = ArtifactCompatibilityChecker().check(
        artifact(), compatibility(), runtime(**runtime_changes)
    )
    assert decision == ArtifactCompatibilityDecision(False, reason)


def test_semantic_version_prerelease_order_is_enforced():
    checker = ArtifactCompatibilityChecker()
    required = compatibility(
        minimum_runtime_version="2.0.0-rc.2",
        minimum_compiler_version="3.0.0-beta",
    )
    assert checker.check(
        artifact(),
        required,
        runtime(runtime_version="2.0.0-rc.10", compiler_version="3.0.0"),
    ).compatible
    assert (
        checker.check(
            artifact(),
            required,
            runtime(runtime_version="2.0.0-rc.1", compiler_version="3.0.0"),
        ).reason
        == "artifact requires a newer accelerator runtime"
    )


@pytest.mark.parametrize(
    ("contract", "reason"),
    [
        (None, "tensor contract is invalid"),
        (
            TensorContract((), (TensorSpec("out", "float32", (1,)),)),
            "tensor contract must declare 1-64 inputs",
        ),
        (
            TensorContract((TensorSpec("in", "float32", (1,)),), ()),
            "tensor contract must declare 1-64 outputs",
        ),
        (
            TensorContract(("bad",), (TensorSpec("out", "float32", (1,)),)),
            "input tensor declaration is invalid",
        ),
        (
            TensorContract(
                (TensorSpec("bad name", "float32", (1,)),),
                (TensorSpec("out", "float32", (1,)),),
            ),
            "input tensor name is invalid",
        ),
        (
            TensorContract(
                (TensorSpec("in", "complex64", (1,)),),
                (TensorSpec("out", "float32", (1,)),),
            ),
            "unsupported input tensor dtype: complex64",
        ),
        (
            TensorContract(
                (TensorSpec("in", "float32", ()),),
                (TensorSpec("out", "float32", (1,)),),
            ),
            "input tensor rank must be 1-8",
        ),
        (
            TensorContract(
                (TensorSpec("in", "float32", (1, 0)),),
                (TensorSpec("out", "float32", (1,)),),
            ),
            "input tensor shape must contain positive bounded dimensions",
        ),
        (
            TensorContract(
                (TensorSpec("in", "float32", (1,)), TensorSpec("in", "int8", (1,))),
                (TensorSpec("out", "float32", (1,)),),
            ),
            "duplicate input tensor name: in",
        ),
    ],
)
def test_tensor_contract_rejects_malformed_metadata(contract, reason):
    decision = ArtifactCompatibilityChecker().check(
        artifact(), compatibility(tensor_contract=contract), runtime()
    )
    assert decision == ArtifactCompatibilityDecision(False, reason)


def test_compatible_resolver_never_reports_incompatible_artifact_ready(tmp_path):
    class Resolver:
        def resolve(self, _reference):
            return ArtifactResolution(True, tmp_path / "model.onnx", "", 5)

    resolution = CompatibleArtifactResolver(Resolver()).resolve(
        artifact(), compatibility(target_accelerator="cpu"), runtime()
    )
    assert resolution == ArtifactResolution(
        False,
        None,
        "unsupported target accelerator: cpu",
        5,
    )


def test_compatible_resolver_preserves_immutable_resolution_failures():
    missing = ArtifactResolution(False, None, "artifact not installed", 0)

    class Resolver:
        def resolve(self, _reference):
            return missing

    checker = ArtifactCompatibilityChecker()
    resolver = CompatibleArtifactResolver(Resolver(), checker=checker)
    assert resolver._resolver.resolve(artifact()) is missing
    assert resolver._checker is checker
    assert resolver.resolve(artifact(), compatibility(), runtime()) is missing


@given(
    name=st.one_of(st.none(), st.integers(), st.text(max_size=140)),
    dtype=st.one_of(st.none(), st.integers(), st.text(max_size=30)),
    shape=st.one_of(
        st.none(),
        st.lists(st.one_of(st.none(), st.booleans(), st.integers()), max_size=10).map(tuple),
    ),
)
def test_arbitrary_tensor_metadata_never_raises(name, dtype, shape):
    contract = TensorContract(
        (TensorSpec(name, dtype, shape),),
        (TensorSpec("out", "float32", (1,)),),
    )
    decision = ArtifactCompatibilityChecker().check(
        artifact(), compatibility(tensor_contract=contract), runtime()
    )
    assert isinstance(decision.compatible, bool)
    assert isinstance(decision.reason, str)
