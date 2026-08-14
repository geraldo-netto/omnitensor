from __future__ import annotations

import base64
import dataclasses
import json

import pytest
from conftest import sample_manifest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from hypothesis import given
from hypothesis import strategies as st

import omnitensor.training.numeric_promotion as numeric_promotion
from omnitensor.plugins.artifact_trust import (
    ArtifactProvenance,
    ArtifactTrustRoot,
    ArtifactTrustVerifier,
    artifact_signature_payload,
)
from omnitensor.preparation import file_digest, prepare_artifact
from omnitensor.registry import Workload
from omnitensor.training.compilers import CompilationRequest, CompiledTarget, CompilerCapability
from omnitensor.training.contracts import TrainingError
from omnitensor.training.numeric_promotion import (
    MAX_PARITY_ERROR,
    NUMERIC_RECIPES,
    NativeParityEvidence,
    NumericTrainingReport,
    _document_digest,
    _numeric_binding_manifest,
    _portable_model_identity,
    _validate_numeric_contract,
    _validate_parity,
    promote_numeric_training,
)


def write_report(root, recipe="backblaze-smart-risk-v1", **changes):
    root.mkdir(parents=True, exist_ok=True)
    model = root / "model.onnx"
    model.write_bytes(b"portable")
    report = {
        "version": 1,
        "recipe": recipe,
        "features": ["first", "second"],
        "quality": {"accepted": True},
        "model": {
            "format": "onnx",
            "filename": "model.onnx",
            "sha256": file_digest(model),
        },
        "tensorContract": {
            "inputs": [{"shape": [1, 2], "dtype": "float32", "layout": "NC"}]
        },
        "outputContract": {"kind": "raw"},
        "targets": {"tpu": "uncompiled", "npu": "uncompiled", "gpu": "uncompiled"},
    }
    report.update(changes)
    path = root / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


class FakeCompiler:
    def __init__(self, accelerator):
        self.accelerator = accelerator
        self.requests = []
        self.outputs = []

    def capability(self):
        return CompilerCapability(self.accelerator, True, "test")

    def incompatibility(self, _source_format, _fully_quantized):
        return None

    def compile(self, request, output_dir):
        self.requests.append(request)
        self.outputs.append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".param" if self.accelerator == "gpu" else ".xml"
        primary = output_dir / f"model{suffix}"
        primary.write_bytes(f"native-{self.accelerator}".encode())
        primary.with_suffix(".bin").write_bytes(b"weights")
        return CompiledTarget(
            self.accelerator,
            primary,
            "ncnn" if self.accelerator == "gpu" else "openvino",
            False,
        )


class Parity:
    def __init__(self, error=0.0, samples=8):
        self.error = error
        self.samples = samples

    def verify(self, report, compiled):
        return NativeParityEvidence(
            self.samples,
            self.error,
            report.model_sha256,
            file_digest(compiled.primary),
        )


def signing_boundary():
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    verifier = ArtifactTrustVerifier([ArtifactTrustRoot("local.models", "release-1", public_key)])

    def sign(reference, report_sha256):
        unsigned = ArtifactProvenance(
            "local.models", "release-1", f"report:sha256:{report_sha256}", 1, ""
        )
        signature = private_key.sign(artifact_signature_payload(reference, unsigned))
        return dataclasses.replace(
            unsigned, signature=base64.b64encode(signature).decode("ascii")
        )

    return sign, verifier


@pytest.mark.parametrize(("recipe", "profile"), NUMERIC_RECIPES.items())
def test_common_loader_maps_every_numeric_recipe_and_pins_semantics(tmp_path, recipe, profile):
    path = write_report(tmp_path / recipe, recipe)
    loaded = NumericTrainingReport.load(path)

    assert loaded.profile_id == profile
    assert loaded.input_shape == (1, 2)
    assert loaded.training_contract == {
        "version": 1,
        "profileId": profile,
        "recipe": recipe,
        "reportSha256": file_digest(path),
        "taskSemanticsSha256": loaded.semantics_sha256,
    }


def test_promotion_compiles_parity_gates_signs_and_reloads_restricted_binding(
    tmp_path, monkeypatch
):
    report_path = write_report(tmp_path / "fit")
    signer, verifier = signing_boundary()
    gpu = FakeCompiler("gpu")
    npu = FakeCompiler("npu")
    writes = []
    atomic_write = numeric_promotion.write_json_atomic

    def write(path, document, *, prefix):
        writes.append((path, document, prefix))
        atomic_write(path, document, prefix=prefix)

    monkeypatch.setattr(numeric_promotion, "write_json_atomic", write)

    installed = promote_numeric_training(
        report_path,
        artifact_id="local-storage-risk",
        variant_versions={"npu": "2.0.0", "gpu": "3.0.0"},
        targets=("gpu", "npu"),
        artifact_root=tmp_path / "artifacts",
        bindings_root=tmp_path / "bindings",
        compilers=(gpu, npu),
        parity_verifier=Parity(),
        signer=signer,
        trust_verifier=verifier,
    )

    assert [variant.accelerator for variant in installed.variants] == ["npu", "gpu"]
    assert [variant.artifact_id for variant in installed.variants] == [
        "local-storage-risk-npu",
        "local-storage-risk-gpu",
    ]
    assert [variant.model_format for variant in installed.variants] == ["openvino", "ncnn"]
    assert gpu.requests == [
        CompilationRequest(report_path.parent / "model.onnx", "onnx", (1, 2), False)
    ]
    assert gpu.outputs == [report_path.parent / "compiled/gpu"]
    assert npu.outputs == [report_path.parent / "compiled/npu"]
    manifest = json.loads(installed.binding_path.read_text())
    models = manifest["requirements"]["models"]
    assert writes == [(installed.binding_path, manifest, ".numeric-model-binding-")]
    assert manifest["requirements"]["acceleratorPreference"] == ["npu", "gpu"]
    assert [model["version"] for model in models] == ["2.0.0", "3.0.0"]
    assert all(model["minimumCompilerVersion"] == "0.0.0" for model in models)
    assert all(model["minimumRuntimeVersion"] == "0.0.0" for model in models)
    assert all(
        tuple(model)[-7:]
        == (
            "fullyQuantized",
            "minimumCompilerVersion",
            "minimumRuntimeVersion",
            "tensorContract",
            "outputContract",
            "trainingContract",
            "nativeEvidence",
        )
        for model in models
    )
    assert models[0]["trainingContract"] == models[1]["trainingContract"]
    assert all(model["nativeEvidence"]["namedDeviceAccepted"] is False for model in models)
    assert all(model["nativeEvidence"]["samples"] == 8 for model in models)
    for variant in installed.variants:
        provenance = variant.installed_at.parent / "provenance.json"
        assert json.loads(provenance.read_text())["provenance"]["source"].startswith(
            "report:sha256:"
        )


@pytest.mark.parametrize(
    ("changes", "detail"),
    [
        ({"version": 2}, "numeric report version is unsupported"),
        ({"recipe": "forecast-v1"}, "numeric report recipe is unsupported"),
        ({"model": {}}, "portable model declaration is invalid"),
        ({"outputContract": {"kind": "embedding"}}, "numeric report output contract must be raw"),
        ({"tensorContract": {"inputs": []}}, "numeric report requires one input"),
        ({"targets": {}}, "numeric report must not claim native targets"),
    ],
)
def test_numeric_loader_fails_closed_on_contract_drift(tmp_path, changes, detail):
    path = write_report(tmp_path / "fit", **changes)
    with pytest.raises(TrainingError) as captured:
        NumericTrainingReport.load(path)
    assert captured.value.code == "report-invalid"
    assert captured.value.detail == detail


def test_numeric_loader_rechecks_model_bytes(tmp_path):
    path = write_report(tmp_path / "fit")
    (path.parent / "model.onnx").write_bytes(b"changed")
    with pytest.raises(TrainingError) as captured:
        NumericTrainingReport.load(path)
    assert (captured.value.code, captured.value.detail) == (
        "model-invalid",
        "portable model does not match numeric report",
    )


def test_numeric_loader_reports_unreadable_report_and_model(tmp_path):
    with pytest.raises(TrainingError) as report_error:
        NumericTrainingReport.load(tmp_path / "missing.json")
    assert report_error.value.code == "report-invalid"
    assert report_error.value.detail.startswith("cannot read numeric report:")
    path = write_report(tmp_path / "fit")
    document = json.loads(path.read_text())
    (path.parent / "model.onnx").unlink()
    with pytest.raises(TrainingError) as model_error:
        _portable_model_identity(path, document)
    assert model_error.value.code == "model-invalid"
    assert model_error.value.detail.startswith("cannot read portable model:")


def test_numeric_loader_requires_exact_onnx_filename(tmp_path):
    path = write_report(tmp_path / "fit")
    document = json.loads(path.read_text())
    document["model"]["filename"] = "other.onnx"
    with pytest.raises(TrainingError) as captured:
        _portable_model_identity(path, document)
    assert (captured.value.code, captured.value.detail) == (
        "report-invalid",
        "numeric report must declare model.onnx",
    )


@pytest.mark.parametrize(
    "evidence",
    [Parity(error=MAX_PARITY_ERROR * 2), Parity(error=float("nan")), Parity(samples=0)],
)
def test_promotion_rejects_missing_or_failed_parity_before_install(tmp_path, evidence):
    path = write_report(tmp_path / "fit")
    signer, verifier = signing_boundary()
    with pytest.raises(TrainingError, match="parity"):
        promote_numeric_training(
            path,
            artifact_id="local-storage-risk",
            variant_versions={"gpu": "1.0.0"},
            targets=("gpu",),
            artifact_root=tmp_path / "artifacts",
            bindings_root=tmp_path / "bindings",
            compilers=(FakeCompiler("gpu"),),
            parity_verifier=evidence,
            signer=signer,
            trust_verifier=verifier,
        )
    assert not (tmp_path / "bindings").exists()


def test_parity_evidence_must_name_both_exact_artifacts(tmp_path):
    report = NumericTrainingReport.load(write_report(tmp_path / "fit"))
    native = tmp_path / "model.param"
    native.write_bytes(b"native")
    artifact = prepare_artifact(
        native, artifact_id="model-gpu", version="1.0.0", model_format="onnx"
    )
    valid = NativeParityEvidence(1, 0.0, report.model_sha256, artifact.reference.sha256)

    with pytest.raises(TrainingError) as untyped:
        _validate_parity(report, artifact, object())
    assert (untyped.value.code, untyped.value.detail) == (
        "parity-invalid",
        "native parity evidence is not typed",
    )
    with pytest.raises(TrainingError) as portable:
        _validate_parity(report, artifact, dataclasses.replace(valid, portable_sha256="0" * 64))
    assert (portable.value.code, portable.value.detail) == (
        "parity-invalid",
        "parity evidence names another portable model",
    )
    with pytest.raises(TrainingError) as native_error:
        _validate_parity(report, artifact, dataclasses.replace(valid, native_sha256="0" * 64))
    assert (native_error.value.code, native_error.value.detail) == (
        "parity-invalid",
        "parity evidence names another native model",
    )
    with pytest.raises(TrainingError) as negative:
        _validate_parity(report, artifact, dataclasses.replace(valid, maximum_absolute_error=-1))
    assert (negative.value.code, negative.value.detail) == (
        "parity-failed",
        "native output exceeds portable numerical tolerance",
    )
    _validate_parity(
        report, artifact, dataclasses.replace(valid, maximum_absolute_error=MAX_PARITY_ERROR)
    )


@pytest.mark.parametrize(
    ("evidence", "detail"),
    [
        (NativeParityEvidence(0, 0, "0" * 64, "0" * 64), "native parity samples must be positive"),
        (NativeParityEvidence(1, True, "0" * 64, "0" * 64), "native parity error must be finite"),
    ],
)
def test_parity_evidence_reports_exact_boundary_failure(tmp_path, evidence, detail):
    report = NumericTrainingReport.load(write_report(tmp_path / "fit"))
    native = tmp_path / "model.onnx"
    native.write_bytes(b"native")
    artifact = prepare_artifact(
        native, artifact_id="model-gpu", version="1.0.0", model_format="onnx"
    )
    with pytest.raises(TrainingError) as captured:
        _validate_parity(report, artifact, evidence)
    assert (captured.value.code, captured.value.detail) == ("parity-invalid", detail)


def test_promotion_refuses_tpu_and_requires_exact_independent_versions(tmp_path):
    path = write_report(tmp_path / "fit")
    signer, verifier = signing_boundary()
    common = {
        "report_path": path,
        "artifact_id": "local-storage-risk",
        "artifact_root": tmp_path / "artifacts",
        "bindings_root": tmp_path / "bindings",
        "compilers": (FakeCompiler("gpu"),),
        "parity_verifier": Parity(),
        "signer": signer,
        "trust_verifier": verifier,
    }
    with pytest.raises(TrainingError) as tpu_error:
        promote_numeric_training(
            **common, variant_versions={"tpu": "1.0.0"}, targets=("tpu",)
        )
    assert (tpu_error.value.code, tpu_error.value.detail) == (
        "source-incompatible",
        "numeric ONNX reports need a separate representative fully-int8 TPU export",
    )
    with pytest.raises(TrainingError) as version_error:
        promote_numeric_training(
            **common, variant_versions={"npu": "1.0.0"}, targets=("gpu",)
        )
    assert (version_error.value.code, version_error.value.detail) == (
        "versions-invalid",
        "variant versions must exactly match targets",
    )


def test_promotion_preserves_compiler_errors_and_missing_provider(tmp_path):
    path = write_report(tmp_path / "fit")
    signer, verifier = signing_boundary()
    arguments = {
        "report_path": path,
        "artifact_id": "local-storage-risk",
        "variant_versions": {"gpu": "1.0.0"},
        "targets": ("gpu",),
        "artifact_root": tmp_path / "artifacts",
        "bindings_root": tmp_path / "bindings",
        "parity_verifier": Parity(),
        "signer": signer,
        "trust_verifier": verifier,
    }
    with pytest.raises(TrainingError) as missing:
        promote_numeric_training(**arguments, compilers=())
    assert (missing.value.code, missing.value.detail) == (
        "compiler-missing",
        "no compiler provider for gpu",
    )
    with pytest.raises(TrainingError) as duplicate:
        promote_numeric_training(
            **arguments, compilers=(FakeCompiler("gpu"), FakeCompiler("gpu"))
        )
    assert (duplicate.value.code, duplicate.value.detail) == (
        "compiler-invalid",
        "duplicate compiler target: gpu",
    )


def test_promotion_rejects_unsigned_artifact_before_binding(tmp_path):
    path = write_report(tmp_path / "fit")
    _signer, verifier = signing_boundary()
    with pytest.raises(Exception, match="artifact is unsigned"):
        promote_numeric_training(
            path,
            artifact_id="local-storage-risk",
            variant_versions={"gpu": "1.0.0"},
            targets=("gpu",),
            artifact_root=tmp_path / "artifacts",
            bindings_root=tmp_path / "bindings",
            compilers=(FakeCompiler("gpu"),),
            parity_verifier=Parity(),
            signer=lambda _reference, _report: None,
            trust_verifier=verifier,
        )
    assert not (tmp_path / "bindings").exists()


@given(width=st.integers(min_value=1, max_value=65536))
def test_numeric_input_contract_accepts_every_bounded_nc_width(width):
    contract = {"inputs": [{"shape": [1, width], "dtype": "float32", "layout": "NC"}]}
    _validate_numeric_contract(contract, {"kind": "raw"})


def test_task_semantics_digest_uses_frozen_canonical_json():
    expected = "e32a6bf3d33f1321b4b34814fc03958e8c4a47088eb4e4ba05144c53a7827930"

    assert _document_digest({"b": 2, "a": ["x"]}) == expected
    assert _document_digest({"a": ["x"], "b": 2}) == expected


@pytest.mark.parametrize(
    ("contract", "detail"),
    [
        (None, "numeric tensor contract is invalid"),
        ({}, "numeric tensor contract is invalid"),
        ({"inputs": "bad"}, "numeric report requires one input"),
        (
            {"inputs": [{"shape": [1, 2], "dtype": "float64", "layout": "NC"}]},
            "numeric input must be float32 NC",
        ),
        (
            {"inputs": [{"shape": [2, 2], "dtype": "float32", "layout": "NC"}]},
            "numeric input shape is invalid",
        ),
        (
            {"inputs": [{"shape": [1, True], "dtype": "float32", "layout": "NC"}]},
            "numeric input shape is invalid",
        ),
        (
            {"inputs": [{"shape": [1] * 7, "dtype": "float32", "layout": "NC"}]},
            "numeric input shape is invalid",
        ),
        (
            {"inputs": [{"shape": [1, 65537], "dtype": "float32", "layout": "NC"}]},
            "numeric input shape is invalid",
        ),
    ],
)
def test_numeric_input_contract_rejects_ambiguous_boundaries(contract, detail):
    with pytest.raises(TrainingError) as captured:
        _validate_numeric_contract(contract, {"kind": "raw"})
    assert (captured.value.code, captured.value.detail) == ("report-invalid", detail)


def test_numeric_binding_reports_profile_and_schema_failures_exactly(tmp_path, monkeypatch):
    report = NumericTrainingReport.load(write_report(tmp_path / "fit"))
    monkeypatch.setattr("omnitensor.training.numeric_promotion.load_workloads", lambda _root: {})
    with pytest.raises(TrainingError) as unknown:
        _numeric_binding_manifest(report, {"gpu": {}}, bundled_root=tmp_path)
    assert (unknown.value.code, unknown.value.detail) == (
        "profile-unknown",
        "no bundled profile is named storage-intelligence",
    )
    base = sample_manifest("storage-intelligence")
    monkeypatch.setattr(
        "omnitensor.training.numeric_promotion.load_workloads",
        lambda _root: {"storage-intelligence": Workload("storage-intelligence", base)},
    )
    monkeypatch.setattr(
        "omnitensor.training.numeric_promotion.validate_workload_document",
        lambda _manifest: ["first", "second"],
    )
    with pytest.raises(TrainingError) as invalid:
        _numeric_binding_manifest(report, {"gpu": {}}, bundled_root=tmp_path)
    assert (invalid.value.code, invalid.value.detail) == (
        "binding-invalid",
        "first; second",
    )


def test_numeric_binding_deep_copies_profile_and_variant(tmp_path, monkeypatch):
    report = NumericTrainingReport.load(write_report(tmp_path / "fit"))
    legacy = {"id": "legacy"}
    base = sample_manifest("storage-intelligence", model=legacy)
    variant = {"trainingContract": {"recipe": report.recipe}}
    monkeypatch.setattr(
        "omnitensor.training.numeric_promotion.load_workloads",
        lambda _root: {"storage-intelligence": Workload("storage-intelligence", base)},
    )
    monkeypatch.setattr(
        "omnitensor.training.numeric_promotion.validate_workload_document", lambda _manifest: []
    )

    binding = _numeric_binding_manifest(report, {"gpu": variant}, bundled_root=tmp_path)
    variant["trainingContract"]["recipe"] = "changed"
    base["requirements"]["accelerator"] = "tpu"

    assert "model" not in binding["requirements"]
    assert binding["requirements"]["accelerator"] == "gpu"
    assert binding["requirements"]["models"] == [
        {"trainingContract": {"recipe": report.recipe}}
    ]
