from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import sample_manifest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.registry import Workload
from omnitensor.training.binding import (
    binding_manifest,
    model_fragment,
    native_evidence,
    publish_binding,
)
from omnitensor.training.contracts import TrainingError
from omnitensor.training.document_model import (
    DocumentModelError,
    DocumentModelEvidence,
    _binding_document,
)


def _loader(profile_id: str, manifest: dict):
    return lambda _root: {profile_id: Workload(profile_id, manifest)}


def _variants(lanes=("tpu", "npu", "gpu")) -> dict:
    return {lane: {"id": f"model-{lane}", "nested": {"lane": lane}} for lane in reversed(lanes)}


def test_plural_and_singular_bindings_preserve_exact_golden_order():
    plural_base = sample_manifest(
        "resource-scheduler",
        model={"id": "legacy-singular"},
        models=[{"id": "legacy-plural"}],
    )
    plural_original = copy.deepcopy(plural_base)
    plural = binding_manifest(
        "resource-scheduler",
        _variants(),
        lane_order=("tpu", "npu", "gpu"),
        plural=True,
        root=Path("catalog"),
        load=_loader("resource-scheduler", plural_base),
        validate=lambda _manifest: [],
        error_type=TrainingError,
    )
    assert plural["requirements"] == {
        "runtimeApi": 1,
        "accelerator": "tpu",
        "acceleratorPreference": ["tpu", "npu", "gpu"],
        "minimumDevices": 1,
        "models": [
            {"id": "model-tpu", "nested": {"lane": "tpu"}},
            {"id": "model-npu", "nested": {"lane": "npu"}},
            {"id": "model-gpu", "nested": {"lane": "gpu"}},
        ],
    }
    assert tuple(plural["requirements"]) == (
        "runtimeApi",
        "accelerator",
        "acceleratorPreference",
        "minimumDevices",
        "models",
    )
    assert plural_base == plural_original
    assert tuple(plural) == tuple(plural_base)

    singular_base = sample_manifest(
        "document-intelligence",
        model={"id": "legacy-singular"},
        models=[{"id": "legacy-plural"}],
    )
    singular_original = copy.deepcopy(singular_base)
    singular = binding_manifest(
        "document-intelligence",
        _variants(("gpu",)),
        lane_order=("gpu",),
        plural=False,
        root=Path("catalog"),
        load=_loader("document-intelligence", singular_base),
        validate=lambda _manifest: [],
        error_type=DocumentModelError,
    )
    assert singular["requirements"] == {
        "runtimeApi": 1,
        "accelerator": "gpu",
        "acceleratorPreference": ["gpu"],
        "minimumDevices": 1,
        "model": {"id": "model-gpu", "nested": {"lane": "gpu"}},
    }
    assert tuple(singular["requirements"]) == (
        "runtimeApi",
        "accelerator",
        "acceleratorPreference",
        "minimumDevices",
        "model",
    )
    assert singular_base == singular_original
    assert tuple(singular) == tuple(singular_base)


@given(
    st.lists(
        st.sampled_from(("tpu", "npu", "gpu")),
        min_size=1,
        max_size=3,
        unique=True,
    )
)
def test_plural_binding_orders_every_lane_subset_by_policy(selected):
    base = sample_manifest("resource-scheduler")
    variants = _variants(tuple(selected))
    binding = binding_manifest(
        "resource-scheduler",
        variants,
        lane_order=("tpu", "npu", "gpu"),
        plural=True,
        root=Path("catalog"),
        load=_loader("resource-scheduler", base),
        validate=lambda _manifest: [],
        error_type=TrainingError,
    )
    expected = [lane for lane in ("tpu", "npu", "gpu") if lane in selected]
    assert binding["requirements"]["accelerator"] == expected[0]
    assert binding["requirements"]["acceleratorPreference"] == expected
    assert [item["id"] for item in binding["requirements"]["models"]] == [
        f"model-{lane}" for lane in expected
    ]


def test_binding_and_model_fragments_deep_copy_every_input():
    artifact = {
        "id": "model",
        "version": "1.0.0",
        "format": "ncnn",
        "sha256": "a" * 64,
        "companions": {"model.bin": "b" * 64},
    }
    tensor = {"inputs": [{"shape": [1, 2], "dtype": "float32", "layout": "NC"}]}
    output = {"kind": "raw"}
    feature = {"featureNames": ["load", "queue"]}
    training = {"version": 1, "recipe": "forecast-v1"}
    evidence = {"samples": 8, "nested": {"accepted": False}}
    fragment = model_fragment(
        artifact,
        fully_quantized=False,
        minimum_compiler_version="0.0.0",
        minimum_runtime_version="0.0.0",
        tensor_contract=tensor,
        feature_contract=feature,
        output_contract=output,
        training_contract=training,
        evidence=evidence,
    )
    assert tuple(fragment) == (
        "id",
        "version",
        "format",
        "sha256",
        "companions",
        "fullyQuantized",
        "minimumCompilerVersion",
        "minimumRuntimeVersion",
        "tensorContract",
        "featureContract",
        "outputContract",
        "trainingContract",
        "nativeEvidence",
    )

    base = sample_manifest("resource-scheduler")
    variants = {"gpu": fragment}
    binding = binding_manifest(
        "resource-scheduler",
        variants,
        lane_order=("tpu", "npu", "gpu"),
        plural=True,
        root=Path("catalog"),
        load=_loader("resource-scheduler", base),
        validate=lambda _manifest: [],
        error_type=TrainingError,
    )
    frozen = copy.deepcopy(binding)
    artifact["companions"]["model.bin"] = "changed"
    tensor["inputs"][0]["shape"].append(3)
    feature["featureNames"].append("memory")
    training["recipe"] = "changed"
    evidence["nested"]["accepted"] = True
    variants["gpu"]["nativeEvidence"]["samples"] = 0
    base["requirements"]["accelerator"] = "npu"
    assert binding == frozen


def test_native_evidence_preserves_family_values_and_field_order():
    numeric = native_evidence(
        portable_sha256="a" * 64,
        native_sha256="b" * 64,
        report_sha256="c" * 64,
        samples=8,
        maximum_absolute_error=0.00001,
        tolerance=0.0001,
        compiler_report_sha256="d" * 64,
        named_device_accepted=False,
    )
    document = native_evidence(
        portable_sha256="e" * 64,
        native_sha256="f" * 64,
        report_sha256="0" * 64,
        samples=28,
        maximum_absolute_error=0.0002,
        tolerance=0.001,
        compiler_report_sha256=None,
        named_device_accepted=False,
    )
    assert (
        tuple(numeric)
        == tuple(document)
        == (
            "portableSha256",
            "nativeSha256",
            "reportSha256",
            "samples",
            "maximumAbsoluteError",
            "tolerance",
            "compilerReportSha256",
            "namedDeviceAccepted",
        )
    )
    assert numeric["compilerReportSha256"] == "d" * 64
    assert document["compilerReportSha256"] is None
    assert numeric["tolerance"] == 0.0001
    assert document["tolerance"] == 0.001


def test_binding_failures_keep_error_type_details_and_never_publish():
    writes = []
    base = sample_manifest("resource-scheduler")

    def build_then_publish(load, validate):
        document = binding_manifest(
            "resource-scheduler",
            _variants(("gpu",)),
            lane_order=("tpu", "npu", "gpu"),
            plural=True,
            root=Path("catalog"),
            load=load,
            validate=validate,
            error_type=TrainingError,
        )
        publish_binding(
            "bindings/resource-scheduler/manifest.json",
            document,
            prefix=".model-binding-",
            writer=lambda *args, **kwargs: writes.append((args, kwargs)),
        )

    with pytest.raises(TrainingError) as unknown:
        build_then_publish(lambda _root: {}, lambda _manifest: [])
    assert (unknown.value.code, unknown.value.detail) == (
        "profile-unknown",
        "no bundled profile is named resource-scheduler",
    )
    with pytest.raises(TrainingError) as invalid:
        build_then_publish(_loader("resource-scheduler", base), lambda _manifest: ["a", "b"])
    assert (invalid.value.code, invalid.value.detail) == (
        "binding-invalid",
        "a; b",
    )
    assert writes == []


def test_publication_preserves_path_document_prefix_and_writer_seam(tmp_path):
    observed = []
    document = {"manifestVersion": 1}
    target = tmp_path / "profile/manifest.json"

    def writer(path, payload, *, prefix):
        observed.append((path, payload, prefix))

    publish_binding(target, document, prefix=".numeric-model-binding-", writer=writer)
    assert observed == [(target, document, ".numeric-model-binding-")]
    assert observed[0][1] is document


def test_document_private_facade_keeps_error_mapping_and_patch_seams(monkeypatch, tmp_path):
    malformed_source = SimpleNamespace(recipe=SimpleNamespace())
    evidence = DocumentModelEvidence(0, "GPU", 28, 0.9999, 1.0, 0.0002, 2)
    monkeypatch.setattr("omnitensor.training.document_model.load_workloads", lambda _root: {})
    with pytest.raises(DocumentModelError) as unknown:
        _binding_document(malformed_source, {}, "c" * 64, "d" * 64, evidence, tmp_path)
    assert (unknown.value.code, unknown.value.detail) == (
        "profile-unknown",
        "no bundled profile is named document-intelligence",
    )

    base = sample_manifest("document-intelligence")
    source = SimpleNamespace(
        recipe=SimpleNamespace(
            document_sha256="a" * 64,
            tensor_contract={
                "inputs": [
                    {"shape": [1, 128], "dtype": "int64", "layout": "NC"},
                    {"shape": [1, 128], "dtype": "int64", "layout": "NC"},
                    {"shape": [1, 128], "dtype": "int64", "layout": "NC"},
                ]
            },
            producer={"inputNames": ["input_ids", "attention_mask", "token_type_ids"]},
        )
    )
    artifact = {"sha256": "b" * 64}
    monkeypatch.setattr(
        "omnitensor.training.document_model.load_workloads",
        _loader("document-intelligence", base),
    )
    monkeypatch.setattr(
        "omnitensor.training.document_model.validate_workload_document",
        lambda _manifest: ["first", "second"],
    )
    with pytest.raises(DocumentModelError) as invalid:
        _binding_document(source, artifact, "c" * 64, "d" * 64, evidence, tmp_path)
    assert (invalid.value.code, invalid.value.detail) == (
        "binding-invalid",
        "first; second",
    )


@pytest.mark.parametrize("plural", [True, False])
def test_binding_without_any_matching_lane_raises_the_injected_error(plural):
    base = sample_manifest("resource-scheduler", model={"id": "legacy"})
    with pytest.raises(TrainingError) as failure:
        binding_manifest(
            "resource-scheduler",
            {"cpu": {"id": "model-cpu"}},
            lane_order=("tpu", "npu", "gpu"),
            plural=plural,
            root=Path("catalog"),
            load=_loader("resource-scheduler", base),
            validate=lambda _manifest: [],
            error_type=TrainingError,
        )
    assert failure.value.code == "binding-invalid"
    assert "tpu, npu, gpu" in str(failure.value)
