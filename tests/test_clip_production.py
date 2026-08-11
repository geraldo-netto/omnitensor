from __future__ import annotations

import hashlib
import json
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

from omnitensor.training.clip_production import (
    CLIP_EMBEDDING_WIDTH,
    CLIP_IMAGE_SIZE,
    ClipGateEvidence,
    ClipHoldout,
    TorchScriptClipOnnxExporter,
    _validate_clip_graph,
    clip_resize_geometry,
    evaluate_clip_gate,
    normalize_clip_rgb,
    produce_clip_source,
)
from omnitensor.training.recipes import FetchedModelSource, ModelRecipeError, load_model_recipe


def _basis(index: int, *, width: int = CLIP_EMBEDDING_WIDTH) -> tuple[float, ...]:
    return tuple(1.0 if position == index else 0.0 for position in range(width))


def _holdout() -> ClipHoldout:
    return ClipHoldout(
        tuple(f"image-{index}" for index in range(10)),
        (_basis(0), _basis(1)),
        "reviewed-fixture",
        "c" * 64,
    )


class _ClipRunner:
    def __init__(self, drift: bool = False):
        self._drift = drift

    def embed_image(self, image_ref):
        index = int(image_ref.rsplit("-", 1)[1])
        if index == 0:
            a, b = ((0.999, 1.001) if self._drift else (1.001, 0.999))
            return (a, b) + (0.0,) * (CLIP_EMBEDDING_WIDTH - 2)
        return _basis(index % 2)


def _fetched(tmp_path: Path) -> FetchedModelSource:
    recipe = load_model_recipe("model-recipes/clip-vit-b-32-image.json")
    root = tmp_path / "source"
    root.mkdir()
    (root / recipe.model_source.filename).write_bytes(b"torchscript")
    receipt = root / "source-receipt.json"
    receipt.write_bytes(b'{"verified":true}')
    return FetchedModelSource(recipe, root, receipt)


def _onnx_model(input_shape=(1, 3, 224, 224), output_shape=(1, 512)):
    graph = helper.make_graph(
        [helper.make_node("Flatten", ["image"], ["embedding"], axis=1)],
        "clip",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, output_shape)],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=8,
    )


def _wrong_input_shape(model):
    model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 1


@given(
    width=st.integers(min_value=1, max_value=16_384),
    height=st.integers(min_value=1, max_value=16_384),
)
def test_clip_resize_geometry_property(width, height):
    resized_width, resized_height, left, top = clip_resize_geometry(width, height)
    assert min(resized_width, resized_height) == CLIP_IMAGE_SIZE
    assert resized_width >= CLIP_IMAGE_SIZE
    assert resized_height >= CLIP_IMAGE_SIZE
    assert left == (resized_width - CLIP_IMAGE_SIZE) // 2
    assert top == (resized_height - CLIP_IMAGE_SIZE) // 2


def test_clip_resize_geometry_exact_examples_and_rejections():
    assert clip_resize_geometry(400, 200) == (448, 224, 112, 0)
    assert clip_resize_geometry(200, 400) == (224, 448, 0, 112)
    assert clip_resize_geometry(225, 224) == (225, 224, 0, 0)
    for dimensions in ((True, 1), (1.0, 1), (0, 1), (1, 16_385)):
        with pytest.raises(ValueError, match="^CLIP image dimensions"):
            clip_resize_geometry(*dimensions)


def test_clip_normalization_is_exact_chw_and_bounded():
    pixels = bytes((0, 127, 255)) * (CLIP_IMAGE_SIZE * CLIP_IMAGE_SIZE)
    result = normalize_clip_rgb(pixels)
    plane = CLIP_IMAGE_SIZE * CLIP_IMAGE_SIZE
    assert len(result) == plane * 3
    assert result[0] == pytest.approx((0 - 122.7709383) * 0.01459842661924292)
    assert result[plane] == pytest.approx((127 - 116.7460125) * 0.015007768493717056)
    assert result[plane * 2] == pytest.approx((255 - 104.09373615) * 0.014220065717024088)
    assert result[-1] == result[plane * 2]
    with pytest.raises(ValueError, match="^CLIP RGB crop must contain exactly 150528 bytes$"):
        normalize_clip_rgb(b"short")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"image_refs": ("one",)}, "CLIP holdout requires 10..256 images"),
        (
            {"image_refs": ("same",) * 10},
            "CLIP holdout image references must be unique",
        ),
        (
            {"image_refs": ("",) + tuple(f"x{i}" for i in range(9))},
            "CLIP holdout image references must be non-empty strings",
        ),
        ({"label_embeddings": (_basis(0),)}, "CLIP holdout requires at least two label"),
        (
            {"label_embeddings": (_basis(0, width=511), _basis(1))},
            "CLIP label embedding width must be 512",
        ),
        ({"license_id": " "}, "CLIP holdout license id is required"),
        (
            {"corpus_sha256": "C" * 64},
            "CLIP holdout corpus digest must be lowercase SHA-256",
        ),
    ],
)
def test_clip_holdout_validation(change, message):
    values = {
        "image_refs": tuple(f"image-{index}" for index in range(10)),
        "label_embeddings": (_basis(0), _basis(1)),
        "license_id": "reviewed",
        "corpus_sha256": "c" * 64,
    }
    values.update(change)
    with pytest.raises(ValueError, match=f"^{message}"):
        ClipHoldout(**values)


def test_clip_gate_accepts_parity_and_rejects_near_boundary_label_drift():
    accepted = evaluate_clip_gate(_holdout(), _ClipRunner(), _ClipRunner())
    rejected = evaluate_clip_gate(_holdout(), _ClipRunner(), _ClipRunner(True))
    assert accepted == ClipGateEvidence(1.0, 1.0, 0.0, 10, True)
    assert rejected.minimum_cosine_similarity > 0.999
    assert rejected.zero_shot_top1_agreement == 0.9
    assert not rejected.accepted


def test_clip_gate_rejects_nonfinite_or_wrong_width():
    class BadRunner:
        def __init__(self, value):
            self.value = value

        def embed_image(self, image_ref):
            return self.value

    with pytest.raises(ValueError, match="^CLIP runner output width must be 512$"):
        evaluate_clip_gate(_holdout(), BadRunner((1.0,)), _ClipRunner())
    with pytest.raises(ValueError, match="must be finite"):
        evaluate_clip_gate(_holdout(), BadRunner((float("nan"),) * 512), _ClipRunner())


def test_torchscript_exporter_uses_fixed_image_only_contract(  # noqa: C901
    tmp_path, monkeypatch
):
    fetched = _fetched(tmp_path)
    calls = {}

    class FakeModule:
        def __init__(self):
            pass

        def eval(self):
            return self

    class FakeEncoder(FakeModule):
        def encode_image(self, image):
            return image

    def export(wrapper, dummy, destination, **kwargs):
        calls.update({"wrapper": wrapper, "dummy": dummy, **kwargs})
        onnx.save_model(_onnx_model(), destination)

    def normalize(value, **kwargs):
        calls["normalize"] = (value, kwargs)
        return value

    def load(path, map_location):
        calls["load"] = (path, map_location)
        return FakeEncoder()

    fake_torch = SimpleNamespace(
        jit=SimpleNamespace(load=load),
        nn=SimpleNamespace(
            Module=FakeModule,
            functional=SimpleNamespace(normalize=normalize),
        ),
        onnx=SimpleNamespace(export=export),
        zeros=lambda shape, dtype: (shape, dtype),
        float32="float32",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    destination = tmp_path / "nested" / "clip.onnx"
    real_mkstemp = tempfile.mkstemp

    def checked_mkstemp(*, prefix, suffix, dir):
        assert prefix == ".clip-image-"
        assert suffix == ".onnx"
        assert dir == destination.parent
        return real_mkstemp(prefix=prefix, suffix=suffix, dir=dir)

    monkeypatch.setattr(tempfile, "mkstemp", checked_mkstemp)
    real_onnx_load = onnx.load

    def checked_onnx_load(path, *, load_external_data):
        calls["onnx_load"] = (path, load_external_data)
        return real_onnx_load(path, load_external_data=load_external_data)

    monkeypatch.setattr(onnx, "load", checked_onnx_load)

    TorchScriptClipOnnxExporter().export(fetched, destination)

    assert destination.is_file()
    assert calls["load"] == (str(fetched.root / "ViT-B-32.pt"), "cpu")
    assert calls["dummy"] == ((1, 3, 224, 224), "float32")
    assert calls["input_names"] == ["image"]
    assert calls["output_names"] == ["embedding"]
    assert calls["opset_version"] == 17
    assert calls["do_constant_folding"] is True
    assert calls["dynamo"] is False
    assert calls["onnx_load"][1] is False
    assert calls["onnx_load"][0].parent == destination.parent
    assert calls["onnx_load"][0].name.startswith(".clip-image-")
    assert calls["wrapper"].forward("pixels") == "pixels"
    assert calls["normalize"] == ("pixels", {"p": 2, "dim": 1, "eps": 1e-12})


def test_torchscript_exporter_refuses_wrong_recipe(tmp_path):
    fetched = _fetched(tmp_path)
    fetched = replace(fetched, recipe=replace(fetched.recipe, id="wrong"))
    with pytest.raises(ModelRecipeError) as caught:
        TorchScriptClipOnnxExporter().export(fetched, tmp_path / "clip.onnx")
    assert caught.value.code == "producer-incompatible"
    assert caught.value.detail == "recipe is not the pinned CLIP image model"


@pytest.mark.parametrize(
    ("mutate", "detail"),
    [
        (lambda model: setattr(model.graph.input[0], "name", "wrong"), "one image input"),
        (lambda model: setattr(model.graph.output[0], "name", "wrong"), "one embedding output"),
        (_wrong_input_shape, "shapes disagree"),
    ],
)
def test_clip_graph_validation_rejects_contract_drift(mutate, detail):
    model = _onnx_model()
    mutate(model)
    with pytest.raises(ModelRecipeError) as caught:
        _validate_clip_graph(model)
    assert caught.value.code == "producer-invalid"
    expected = {
        "one image input": "CLIP ONNX must have one image input",
        "one embedding output": "CLIP ONNX must have one embedding output",
        "shapes disagree": "CLIP ONNX shapes disagree with recipe",
    }
    assert caught.value.detail == expected[detail]


def test_produce_clip_source_emits_private_safe_report(tmp_path, monkeypatch):
    fetched = _fetched(tmp_path)
    open_calls = []

    def open_source(recipe, root):
        open_calls.append((recipe, root))
        return fetched

    monkeypatch.setattr(
        "omnitensor.training.clip_production.open_fetched_model_source",
        open_source,
    )

    class FakeExporter:
        def export(self, source, destination):
            assert source is fetched
            destination.write_bytes(b"portable-clip")

    source_calls = []
    portable_calls = []

    def source_factory(source):
        source_calls.append(source)
        return _ClipRunner()

    def portable_factory(model):
        portable_calls.append(model)
        return _ClipRunner()

    produced = produce_clip_source(
        "recipe",
        "source",
        tmp_path / "output",
        _holdout(),
        source_factory,
        portable_factory,
        exporter=FakeExporter(),
    )
    report = json.loads(produced.report_path.read_text())
    assert produced.evidence.accepted
    assert open_calls == [("recipe", "source")]
    assert source_calls == [fetched]
    assert portable_calls == [produced.model_path]
    native_claim = {
        "status": "unqualified",
        "reason": "no target compiler, native parity, or named-device evidence was run",
    }
    assert report == {
        "reportVersion": 1,
        "kind": "clip-image-source-production",
        "recipeId": fetched.recipe.id,
        "recipeVersion": fetched.recipe.version,
        "recipeSha256": fetched.recipe.document_sha256,
        "sourceReceiptSha256": hashlib.sha256(b'{"verified":true}').hexdigest(),
        "sourceModelSha256": fetched.recipe.model_source.sha256,
        "portableModel": {
            "format": "onnx",
            "sha256": hashlib.sha256(b"portable-clip").hexdigest(),
            "tensorContract": fetched.recipe.tensor_contract,
            "outputContract": fetched.recipe.output_contract,
            "producer": fetched.recipe.producer,
        },
        "holdout": {
            "licenseId": "reviewed-fixture",
            "corpusSha256": "c" * 64,
            "imageCount": 10,
            "labelCount": 2,
        },
        "portableSourceGate": {
            "minimumCosineSimilarity": 1.0,
            "zeroShotTop1Agreement": 1.0,
            "nonfiniteOutputRate": 0.0,
            "accepted": True,
        },
        "nativeTargets": {"gpu": native_claim, "npu": native_claim, "tpu": native_claim},
        "cpuFallback": "forbidden",
    }
    assert not any(ref in json.dumps(report) for ref in _holdout().image_refs)
    assert {claim["status"] for claim in report["nativeTargets"].values()} == {"unqualified"}
    assert report["cpuFallback"] == "forbidden"


def test_produce_clip_source_refuses_wrong_recipe_and_single_conflict(tmp_path, monkeypatch):
    fetched = _fetched(tmp_path)
    wrong = replace(fetched, recipe=replace(fetched.recipe, id="wrong"))
    monkeypatch.setattr(
        "omnitensor.training.clip_production.open_fetched_model_source",
        lambda recipe, root: wrong,
    )
    with pytest.raises(ModelRecipeError) as caught:
        produce_clip_source(
            "recipe",
            "source",
            tmp_path / "wrong",
            _holdout(),
            lambda source: _ClipRunner(),
            lambda model: _ClipRunner(),
        )
    assert caught.value.code == "producer-incompatible"
    assert caught.value.detail == "recipe is not the pinned CLIP image model"

    monkeypatch.setattr(
        "omnitensor.training.clip_production.open_fetched_model_source",
        lambda recipe, root: fetched,
    )
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "clip-vit-b-32-image-production-report.json").write_text("occupied")
    with pytest.raises(ModelRecipeError) as caught:
        produce_clip_source(
            "recipe",
            "source",
            output,
            _holdout(),
            lambda source: _ClipRunner(),
            lambda model: _ClipRunner(),
        )
    assert caught.value.code == "producer-conflict"
    assert caught.value.detail == "CLIP production output already exists"


def test_produce_clip_source_cleans_rejected_graph(tmp_path, monkeypatch):
    fetched = _fetched(tmp_path)
    monkeypatch.setattr(
        "omnitensor.training.clip_production.open_fetched_model_source",
        lambda recipe, root: fetched,
    )

    class FakeExporter:
        def export(self, source, destination):
            destination.write_bytes(b"rejected")

    monkeypatch.setattr(
        "omnitensor.training.clip_production.evaluate_clip_gate",
        lambda *args: ClipGateEvidence(0.5, 0.5, 0.0, 10, False),
    )
    output = tmp_path / "output"
    with pytest.raises(ModelRecipeError) as caught:
        produce_clip_source(
            "recipe",
            "source",
            output,
            _holdout(),
            lambda source: _ClipRunner(),
            lambda model: _ClipRunner(),
            exporter=FakeExporter(),
        )
    assert caught.value.code == "portable-parity-failed"
    assert caught.value.detail == "CLIP portable-source gate failed"
    assert not list(output.iterdir())
