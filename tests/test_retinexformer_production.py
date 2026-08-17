from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest
from hypothesis import given
from hypothesis import strategies as st
from onnx import TensorProto, helper

import omnitensor.training.retinexformer_production as production
from omnitensor.training.recipes import FetchedModelSource, ModelRecipeError, load_model_recipe
from omnitensor.training.retinexformer_production import (
    RETINEXFORMER_IMAGE_COMPONENTS,
    PinnedRetinexformerLoader,
    ProducedRetinexformerSource,
    RetinexformerGateEvidence,
    RetinexformerHoldout,
    TorchRetinexformerOnnxExporter,
    _source_path,
    _validate_retinexformer_graph,
    evaluate_retinexformer_gate,
    produce_retinexformer_source,
    structural_similarity,
)


def _holdout() -> RetinexformerHoldout:
    return RetinexformerHoldout(
        tuple(f"input-{index}" for index in range(4)),
        tuple(f"target-{index}" for index in range(4)),
        "reviewed-lol-fixture",
        "b" * 64,
    )


def _fetched(tmp_path: Path) -> FetchedModelSource:
    recipe = load_model_recipe("model-recipes/retinexformer-lol-v1.json")
    root = tmp_path / "source"
    root.mkdir()
    for source in recipe.sources:
        (root / source.filename).write_bytes(source.role.encode())
    receipt = root / "source-receipt.json"
    receipt.write_bytes(b'{"verified":true}')
    return FetchedModelSource(recipe, root, receipt)


def _onnx_model(shape=(1, 3, 256, 256)):
    graph = helper.make_graph(
        [helper.make_node("Identity", ["image"], ["enhanced"])],
        "retinexformer",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("enhanced", TensorProto.FLOAT, shape)],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


class _ImageRunner:
    def __init__(self, values):
        self.values = values

    def enhance(self, image_ref):
        return self.values


class _TargetLoader:
    def __init__(self, values):
        self.values = values

    def load(self, image_ref):
        return self.values


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"input_refs": ("one",)}, "Retinexformer holdout requires 4..128 pairs"),
        ({"target_refs": ("one",)}, "Retinexformer holdout arrays must have equal lengths"),
        (
            {"input_refs": ("", "b", "c", "d")},
            "Retinexformer holdout references must be non-empty strings",
        ),
        (
            {"target_refs": ("same",) * 4},
            "Retinexformer holdout references must be unique within each side",
        ),
        ({"license_id": " "}, "Retinexformer holdout license id is required"),
        (
            {"corpus_sha256": "B" * 64},
            "Retinexformer corpus digest must be lowercase SHA-256",
        ),
    ],
)
def test_retinexformer_holdout_validation(change, message):
    values = {
        "input_refs": _holdout().input_refs,
        "target_refs": _holdout().target_refs,
        "license_id": "reviewed-lol-fixture",
        "corpus_sha256": "b" * 64,
    }
    values.update(change)
    with pytest.raises(ValueError, match=f"^{message}$"):
        RetinexformerHoldout(**values)


@given(
    values=st.lists(
        st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=64,
    )
)
def test_structural_similarity_property_identical_is_one(values):
    assert structural_similarity(values, values) == pytest.approx(1.0)


def test_structural_similarity_rejects_shapes_and_distinguishes_images():
    assert structural_similarity((0.0,) * 8, (1.0,) * 8) < 0.001
    left = (0.0, 0.5, 1.0, 0.25)
    right = (0.1, 0.4, 0.9, 0.3)
    assert structural_similarity(left, right) == pytest.approx(0.9640984916329907)
    assert structural_similarity(right, left) == pytest.approx(0.9640984916329907)
    with pytest.raises(ValueError, match="^SSIM vectors must be non-empty and equal length$"):
        structural_similarity((), ())
    with pytest.raises(ValueError, match="^SSIM vectors must be non-empty and equal length$"):
        structural_similarity((1.0,), (1.0, 2.0))


def test_retinexformer_gate_accepts_parity_and_rejects_range(monkeypatch):
    monkeypatch.setattr(production, "RETINEXFORMER_IMAGE_COMPONENTS", 12)
    pixels = (0.5,) * 12
    accepted = evaluate_retinexformer_gate(
        _holdout(), _ImageRunner(pixels), _ImageRunner(pixels), _TargetLoader(pixels)
    )
    assert accepted == RetinexformerGateEvidence(1.0, 1.0, 0.0, 4, True)

    invalid = (2.0,) + pixels[1:]
    rejected = evaluate_retinexformer_gate(
        _holdout(), _ImageRunner(pixels), _ImageRunner(invalid), _TargetLoader(pixels)
    )
    assert rejected.output_range_violation_rate == pytest.approx(1 / 12)
    assert not rejected.accepted


def test_retinexformer_gate_refuses_bad_source_and_component_contract(monkeypatch):
    monkeypatch.setattr(production, "RETINEXFORMER_IMAGE_COMPONENTS", 12)
    pixels = (0.5,) * 12
    with pytest.raises(ValueError, match="source image must be finite and within"):
        evaluate_retinexformer_gate(
            _holdout(),
            _ImageRunner((math.nan,) + pixels[1:]),
            _ImageRunner(pixels),
            _TargetLoader(pixels),
        )
    with pytest.raises(ValueError, match="^Retinexformer image must contain 12 components$"):
        evaluate_retinexformer_gate(
            _holdout(), _ImageRunner((0.5,)), _ImageRunner(pixels), _TargetLoader(pixels)
        )
    with pytest.raises(ValueError, match="^Retinexformer image components must be numeric$"):
        evaluate_retinexformer_gate(
            _holdout(),
            _ImageRunner(("bad",) + pixels[1:]),
            _ImageRunner(pixels),
            _TargetLoader(pixels),
        )


def test_pinned_loader_executes_exact_architecture_and_strict_checkpoint(tmp_path, monkeypatch):
    fetched = _fetched(tmp_path)
    calls = {}

    class Candidate:
        def __init__(self, **kwargs):
            calls["constructor"] = kwargs

        def load_state_dict(self, params, *, strict):
            calls["state"] = (params, strict)

        def eval(self):
            calls["eval"] = True
            return self

    module = SimpleNamespace()

    class SpecLoader:
        def exec_module(self, target):
            assert target is module
            target.RetinexFormer = Candidate

    spec = SimpleNamespace(loader=SpecLoader())

    def spec_from_file_location(name, path):
        calls["spec"] = (name, path)
        return spec

    monkeypatch.setattr(importlib.util, "spec_from_file_location", spec_from_file_location)
    monkeypatch.setattr(importlib.util, "module_from_spec", lambda value: module)
    params = {"body.weight": "tensor"}

    def load(path, **kwargs):
        calls["load"] = (path, kwargs)
        return {"params": params}

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=load))
    model = PinnedRetinexformerLoader()(fetched)

    assert isinstance(model, Candidate)
    assert calls["constructor"] == {
        "in_channels": 3,
        "out_channels": 3,
        "n_feat": 40,
        "stage": 1,
        "num_blocks": [1, 2, 2],
    }
    architecture = fetched.root / "RetinexFormer_arch.py"
    assert calls["spec"] == (
        f"_omnitensor_retinexformer_{fetched.recipe.document_sha256[:16]}",
        architecture,
    )
    assert calls["load"] == (
        fetched.root / "LOL_v1.pth",
        {"map_location": "cpu", "weights_only": True},
    )
    assert calls["state"] == (params, True)
    assert calls["eval"] is True


def test_pinned_loader_refuses_wrong_recipe_and_malformed_checkpoint(tmp_path, monkeypatch):
    fetched = _fetched(tmp_path)
    wrong = replace(fetched, recipe=replace(fetched.recipe, id="wrong"))
    with pytest.raises(ModelRecipeError) as caught:
        PinnedRetinexformerLoader()(wrong)
    assert (caught.value.code, caught.value.detail) == (
        "producer-incompatible",
        "recipe is not pinned Retinexformer",
    )

    module = SimpleNamespace(RetinexFormer=lambda **kwargs: SimpleNamespace())
    loader = SimpleNamespace(exec_module=lambda target: None)
    monkeypatch.setattr(
        importlib.util, "spec_from_file_location", lambda name, path: SimpleNamespace(loader=loader)
    )
    monkeypatch.setattr(importlib.util, "module_from_spec", lambda spec: module)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=lambda *args, **kwargs: {}))
    with pytest.raises(ModelRecipeError) as caught:
        PinnedRetinexformerLoader()(fetched)
    assert (caught.value.code, caught.value.detail) == (
        "source-invalid",
        "Retinexformer checkpoint must contain a params mapping",
    )


def test_pinned_loader_refuses_unloadable_architecture(tmp_path, monkeypatch):
    fetched = _fetched(tmp_path)
    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda name, path: None)
    with pytest.raises(ModelRecipeError) as caught:
        PinnedRetinexformerLoader()(fetched)
    assert (caught.value.code, caught.value.detail) == (
        "source-invalid",
        "cannot load pinned Retinexformer architecture",
    )


def test_source_role_lookup_refuses_incomplete_recipe(tmp_path):
    fetched = _fetched(tmp_path)
    incomplete = replace(
        fetched,
        recipe=replace(
            fetched.recipe,
            sources=tuple(item for item in fetched.recipe.sources if item.role != "architecture"),
        ),
    )
    with pytest.raises(ModelRecipeError) as caught:
        _source_path(incomplete, "architecture")
    assert (caught.value.code, caught.value.detail) == (
        "source-invalid",
        "Retinexformer source lacks architecture",
    )


def test_torch_exporter_freezes_clamped_image_contract(tmp_path, monkeypatch):
    fetched = _fetched(tmp_path)
    calls = {}

    class FakeModule:
        def eval(self):
            return self

    class Candidate(FakeModule):
        def __call__(self, image):
            calls["candidate_image"] = image
            return "raw"

    candidate = Candidate()

    def export(wrapper, dummy, destination, **kwargs):
        calls.update({"wrapper": wrapper, "dummy": dummy, **kwargs})
        onnx.save_model(_onnx_model(), destination)

    def clamp(value, **kwargs):
        calls["clamp"] = (value, kwargs)
        return "clamped"

    fake_torch = SimpleNamespace(
        nn=SimpleNamespace(Module=FakeModule),
        onnx=SimpleNamespace(export=export),
        zeros=lambda shape, dtype: (shape, dtype),
        float32="float32",
        clamp=clamp,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    destination = tmp_path / "out" / "retinexformer.onnx"
    real_mkstemp = tempfile.mkstemp

    def checked_mkstemp(*, prefix, suffix, dir):
        assert (prefix, suffix, dir) == (".retinexformer-", ".onnx", destination.parent)
        return real_mkstemp(prefix=prefix, suffix=suffix, dir=dir)

    monkeypatch.setattr(tempfile, "mkstemp", checked_mkstemp)
    real_onnx_load = onnx.load

    def checked_onnx_load(path, *, load_external_data):
        calls["onnx_load"] = (path, load_external_data)
        return real_onnx_load(path, load_external_data=load_external_data)

    monkeypatch.setattr(onnx, "load", checked_onnx_load)

    def load(source):
        assert source is fetched
        return candidate

    TorchRetinexformerOnnxExporter(load).export(fetched, destination)

    assert destination.is_file()
    assert calls["dummy"] == ((1, 3, 256, 256), "float32")
    assert calls["input_names"] == ["image"]
    assert calls["output_names"] == ["enhanced"]
    assert calls["opset_version"] == 17
    assert calls["do_constant_folding"] is True
    assert calls["dynamo"] is False
    assert calls["onnx_load"][1] is False
    assert calls["wrapper"].forward("pixels") == "clamped"
    assert calls["candidate_image"] == "pixels"
    assert calls["clamp"] == ("raw", {"min": 0.0, "max": 1.0})


def test_torch_exporter_refuses_wrong_recipe_and_loader(tmp_path):
    fetched = _fetched(tmp_path)
    wrong = replace(fetched, recipe=replace(fetched.recipe, id="wrong"))
    with pytest.raises(ModelRecipeError) as caught:
        TorchRetinexformerOnnxExporter(lambda source: object()).export(
            wrong, tmp_path / "wrong.onnx"
        )
    assert (caught.value.code, caught.value.detail) == (
        "producer-incompatible",
        "recipe is not pinned Retinexformer",
    )

    with pytest.raises(ModelRecipeError) as caught:
        TorchRetinexformerOnnxExporter(lambda source: object()).export(
            fetched, tmp_path / "loader.onnx"
        )
    assert (caught.value.code, caught.value.detail) == (
        "producer-incompatible",
        "Retinexformer loader returned no module",
    )


def _wrong_shape(model):
    model.graph.output[0].type.tensor_type.shape.dim[3].dim_value = 1


@pytest.mark.parametrize(
    ("mutate", "detail"),
    [
        (
            lambda model: setattr(model.graph.input[0], "name", "wrong"),
            "Retinexformer ONNX needs one image input",
        ),
        (
            lambda model: setattr(model.graph.output[0], "name", "wrong"),
            "Retinexformer ONNX needs one enhanced output",
        ),
        (_wrong_shape, "Retinexformer ONNX shapes disagree with recipe"),
    ],
)
def test_retinexformer_graph_rejects_contract_drift(mutate, detail):
    model = _onnx_model()
    mutate(model)
    with pytest.raises(ModelRecipeError) as caught:
        _validate_retinexformer_graph(model)
    assert (caught.value.code, caught.value.detail) == ("producer-invalid", detail)


def test_produce_retinexformer_emits_private_evidence_report(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "RETINEXFORMER_IMAGE_COMPONENTS", 12)
    fetched = _fetched(tmp_path)
    monkeypatch.setattr(production, "open_fetched_model_source", lambda recipe, root: fetched)

    class Exporter:
        def export(self, source, destination):
            assert source is fetched
            destination.write_bytes(b"portable-retinexformer")

    pixels = (0.5,) * 12
    produced = produce_retinexformer_source(
        "recipe",
        "sources",
        tmp_path / "nested" / "output",
        _holdout(),
        lambda source: _ImageRunner(pixels),
        lambda path: _ImageRunner(pixels),
        _TargetLoader(pixels),
        Exporter(),
    )
    evidence = RetinexformerGateEvidence(1.0, 1.0, 0.0, 4, True)
    assert produced == ProducedRetinexformerSource(
        produced.model_path, produced.report_path, evidence
    )
    report = json.loads(produced.report_path.read_text())
    assert produced.model_path.read_bytes() == b"portable-retinexformer"
    assert produced.report_path.read_bytes() == json.dumps(report, separators=(",", ":")).encode()
    assert tuple(report) == (
        "reportVersion",
        "kind",
        "recipeId",
        "recipeVersion",
        "recipeSha256",
        "sourceReceiptSha256",
        "sourceDigests",
        "portableModel",
        "holdout",
        "portableSourceGate",
        "nativeTargets",
        "cpuFallback",
    )
    assert tuple(report["portableModel"]) == (
        "format",
        "sha256",
        "tensorContract",
        "outputContract",
        "producer",
    )
    assert tuple(report["holdout"]) == ("licenseId", "corpusSha256", "pairCount")
    assert tuple(report["portableSourceGate"]) == (
        "minimumPortableSourceSsim",
        "minimumPairedHoldoutSsim",
        "outputRangeViolationRate",
        "accepted",
    )
    assert tuple(report["nativeTargets"]) == ("gpu", "npu", "tpu")
    assert all(tuple(claim) == ("status", "reason") for claim in report["nativeTargets"].values())
    generic = {
        "status": "unqualified",
        "reason": "no target compiler, native parity, or named-device evidence was run",
    }
    assert report == {
        "reportVersion": 1,
        "kind": "retinexformer-source-production",
        "recipeId": fetched.recipe.id,
        "recipeVersion": fetched.recipe.version,
        "recipeSha256": fetched.recipe.document_sha256,
        "sourceReceiptSha256": hashlib.sha256(b'{"verified":true}').hexdigest(),
        "sourceDigests": {item.role: item.sha256 for item in fetched.recipe.sources},
        "portableModel": {
            "format": "onnx",
            "sha256": hashlib.sha256(b"portable-retinexformer").hexdigest(),
            "tensorContract": fetched.recipe.tensor_contract,
            "outputContract": fetched.recipe.output_contract,
            "producer": fetched.recipe.producer,
        },
        "holdout": {
            "licenseId": "reviewed-lol-fixture",
            "corpusSha256": "b" * 64,
            "pairCount": 4,
        },
        "portableSourceGate": {
            "minimumPortableSourceSsim": 1.0,
            "minimumPairedHoldoutSsim": 1.0,
            "outputRangeViolationRate": 0.0,
            "accepted": True,
        },
        "nativeTargets": {
            "gpu": generic,
            "npu": generic,
            "tpu": {
                "status": "unqualified",
                "reason": (
                    "needs representative fully-int8 calibration, full Edge TPU mapping, "
                    "portable/native fidelity, and named Coral execution evidence"
                ),
            },
        },
        "cpuFallback": "forbidden",
    }
    serialized = json.dumps(report)
    assert "input-0" not in serialized
    assert "target-0" not in serialized


def test_production_refuses_wrong_recipe_conflict_and_failed_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "RETINEXFORMER_IMAGE_COMPONENTS", 12)
    fetched = _fetched(tmp_path)

    class Exporter:
        def export(self, source, destination):
            destination.write_bytes(b"portable")

    pixels = (0.5,) * 12
    monkeypatch.setattr(
        production,
        "open_fetched_model_source",
        lambda recipe, root: replace(fetched, recipe=replace(fetched.recipe, id="wrong")),
    )
    with pytest.raises(ModelRecipeError) as caught:
        produce_retinexformer_source(
            "recipe",
            "sources",
            tmp_path / "wrong",
            _holdout(),
            lambda source: _ImageRunner(pixels),
            lambda path: _ImageRunner(pixels),
            _TargetLoader(pixels),
            Exporter(),
        )
    assert (caught.value.code, caught.value.detail) == (
        "producer-incompatible",
        "recipe is not pinned Retinexformer",
    )

    monkeypatch.setattr(production, "open_fetched_model_source", lambda recipe, root: fetched)
    conflict = tmp_path / "conflict"
    conflict.mkdir()
    (conflict / "retinexformer-lol-v1.onnx").write_bytes(b"existing")
    with pytest.raises(ModelRecipeError) as caught:
        produce_retinexformer_source(
            "recipe",
            "sources",
            conflict,
            _holdout(),
            lambda source: _ImageRunner(pixels),
            lambda path: _ImageRunner(pixels),
            _TargetLoader(pixels),
            Exporter(),
        )
    assert (caught.value.code, caught.value.detail) == (
        "producer-conflict",
        "Retinexformer production output exists",
    )
    assert (conflict / "retinexformer-lol-v1.onnx").read_bytes() == b"existing"

    failed = tmp_path / "failed"
    with pytest.raises(ModelRecipeError) as caught:
        produce_retinexformer_source(
            "recipe",
            "sources",
            failed,
            _holdout(),
            lambda source: _ImageRunner(pixels),
            lambda path: _ImageRunner((2.0,) + pixels[1:]),
            _TargetLoader(pixels),
            Exporter(),
        )
    assert (caught.value.code, caught.value.detail) == (
        "quality-gate-failed",
        "Retinexformer did not pass gates",
    )
    assert list(failed.iterdir()) == []


def test_fixed_component_count_matches_recipe():
    assert RETINEXFORMER_IMAGE_COMPONENTS == 196_608
