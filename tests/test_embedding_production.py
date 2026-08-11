from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import onnx
import pytest
from hypothesis import given
from hypothesis import strategies as st
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

from omnitensor.training.embedding_production import (
    EMBEDDING_WIDTH,
    EmbeddingGateEvidence,
    EmbeddingHoldout,
    SentenceEmbeddingOnnxExporter,
    _dot,
    _save_onnx_atomic,
    _top_k,
    evaluate_embedding_gate,
    l2_normalize,
    pool_sentence_embedding,
    produce_sentence_embedding,
)
from omnitensor.training.recipes import (
    FetchedModelSource,
    ModelRecipeError,
    fetch_model_sources,
    load_model_recipe,
    open_fetched_model_source,
)


def _recipe(identifier: str = "all-minilm-l6-v2"):
    return load_model_recipe(Path("model-recipes") / f"{identifier}.json")


def _holdout(document_count: int = 12) -> EmbeddingHoldout:
    return EmbeddingHoldout(
        ("q0", "q1"),
        tuple(f"d{index}" for index in range(document_count)),
        "operator-reviewed-test-fixture",
        "a" * 64,
    )


class _Runner:
    def __init__(self, reverse_documents: bool = False):
        self._reverse_documents = reverse_documents

    def embed(self, text: str):
        values = [0.0] * EMBEDDING_WIDTH
        if text.startswith("q"):
            values[1] = 1.0
        else:
            index = int(text[1:])
            values[0] = 1.0
            values[1] = (11 - index if self._reverse_documents else index) * 0.001
        return values


class _PayloadTransport:
    def __init__(self, payloads: dict[str, bytes]):
        self._payloads = payloads

    def chunks(self, uri: str, maximum_bytes: int):
        assert maximum_bytes >= len(self._payloads[uri])
        yield self._payloads[uri]


def _write_tiny_recipe(tmp_path: Path, payload: bytes) -> Path:
    document = json.loads(Path("model-recipes/all-minilm-l6-v2.json").read_text())
    revision = "0123456789abcdef0123456789abcdef01234567"
    document["sources"] = [
        {
            "role": "model",
            "uri": f"https://models.example/{revision}/model.onnx",
            "revision": revision,
            "filename": "model.onnx",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "sizeBytes": len(payload),
        }
    ]
    document["preprocessing"].pop("artifacts")
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _source_model(path: Path, *, opset: int = 13, output_name: str = "last_hidden_state"):
    inputs = [
        helper.make_tensor_value_info(name, TensorProto.INT64, ["batch", "tokens"])
        for name in ("input_ids", "attention_mask", "token_type_ids")
    ]
    base = helper.make_tensor(
        "base",
        TensorProto.FLOAT,
        [1, 1, EMBEDDING_WIDTH],
        [float(index + 1) for index in range(EMBEDDING_WIDTH)],
    )
    shape = helper.make_tensor("shape", TensorProto.INT64, [3], [1, 128, EMBEDDING_WIDTH])
    graph = helper.make_graph(
        [helper.make_node("Expand", ["base", "shape"], [output_name])],
        "encoder",
        inputs,
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 128, 384])],
        [base, shape],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=8,
    )
    onnx.save_model(model, path)


def _fetched(tmp_path: Path, identifier: str = "all-minilm-l6-v2") -> FetchedModelSource:
    recipe = _recipe(identifier)
    root = tmp_path / "source"
    root.mkdir()
    _source_model(root / recipe.model_source.filename)
    receipt = root / "source-receipt.json"
    receipt.write_text('{"verified":true}', encoding="utf-8")
    return FetchedModelSource(recipe, root, receipt)


def test_open_fetched_source_revalidates_pinned_bytes(tmp_path):
    payload = b"verified"
    recipe_path = _write_tiny_recipe(tmp_path, payload)
    recipe = load_model_recipe(recipe_path)
    source_root = tmp_path / "sources"
    fetched = fetch_model_sources(
        recipe_path,
        source_root,
        accepted_license="Apache-2.0",
        transport=_PayloadTransport({recipe.model_source.uri: payload}),
    )

    reopened = open_fetched_model_source(recipe_path, source_root)
    assert reopened.root == fetched.root
    (reopened.root / "model.onnx").write_bytes(b"tampered")
    with pytest.raises(ModelRecipeError, match="source-invalid"):
        open_fetched_model_source(recipe_path, source_root)


@given(
    values=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=32,
    ).filter(lambda values: any(abs(value) > 1e-9 for value in values))
)
def test_l2_normalize_property(values):
    result = l2_normalize(values)
    assert math.sqrt(sum(value * value for value in result)) == pytest.approx(1.0)
    assert all(math.isfinite(value) for value in result)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ((), "embedding vector must not be empty"),
        ((0.0, 0.0), "embedding vector norm is zero"),
        ((1e-12,), "embedding vector norm is zero"),
        ((math.nan,), "embedding vector must be finite"),
        ((math.inf,), "embedding vector must be finite"),
        (("bad",), "embedding vector must be numeric"),
    ],
)
def test_l2_normalize_rejects_invalid_vectors(value, message):
    with pytest.raises(ValueError, match=f"^{message}$"):
        l2_normalize(value)


def test_pool_sentence_embedding_matches_both_recipe_methods():
    hidden = ((3.0, 4.0), (0.0, 5.0), (100.0, 100.0))
    assert pool_sentence_embedding(hidden, (1, 1, 0), "cls-token-l2") == pytest.approx(
        (0.6, 0.8)
    )
    assert pool_sentence_embedding(
        hidden, (1, 1, 0), "attention-mask-mean-pool-l2"
    ) == pytest.approx(l2_normalize((1.5, 4.5)))


@pytest.mark.parametrize(
    ("hidden", "mask", "method", "message"),
    [
        ((), (), "cls-token-l2", "hidden states must be a non-empty rectangular matrix"),
        (
            ((1.0,), (1.0, 2.0)),
            (1, 1),
            "cls-token-l2",
            "hidden states must be a non-empty rectangular matrix",
        ),
        (
            ((1.0,),),
            (),
            "cls-token-l2",
            "attention mask must contain one binary value per token",
        ),
        (
            ((1.0,),),
            (2,),
            "cls-token-l2",
            "attention mask must contain one binary value per token",
        ),
        (
            ((1.0,),),
            (0,),
            "attention-mask-mean-pool-l2",
            "mean pooling requires at least one unmasked token",
        ),
        (((1.0,),), (1,), "unknown", "unsupported embedding pooling method: unknown"),
    ],
)
def test_pool_sentence_embedding_rejects_malformed_contract(hidden, mask, method, message):
    with pytest.raises(ValueError, match=f"^{message}$"):
        pool_sentence_embedding(hidden, mask, method)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"queries": ()}, "embedding holdout requires at least one query"),
        (
            {"documents": tuple(f"d{i}" for i in range(10))},
            "embedding holdout requires at least 11 documents",
        ),
        ({"license_id": " "}, "embedding holdout license id is required"),
        (
            {"corpus_sha256": "A" * 64},
            "embedding holdout corpus digest must be lowercase SHA-256",
        ),
        ({"queries": ("",)}, "embedding holdout texts must be non-empty strings"),
        ({"queries": ["query"]}, "embedding holdout texts must be immutable tuples"),
    ],
)
def test_embedding_holdout_refuses_unreviewed_or_unbounded_inputs(kwargs, message):
    values = {
        "queries": ("query",),
        "documents": tuple(f"d{i}" for i in range(11)),
        "license_id": "reviewed",
        "corpus_sha256": "b" * 64,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=f"^{message}$"):
        EmbeddingHoldout(**values)


def test_embedding_holdout_refuses_count_and_byte_bounds():
    documents = tuple(f"d{i}" for i in range(11))
    with pytest.raises(ValueError, match="^embedding holdout contains too many texts$"):
        EmbeddingHoldout(("q",) * 502, documents, "reviewed", "b" * 64)
    with pytest.raises(ValueError, match="^embedding holdout text exceeds its byte bound$"):
        EmbeddingHoldout(("x" * (2 * 1024 * 1024 + 1),), documents, "reviewed", "b" * 64)
    EmbeddingHoldout(("q",) * 501, documents, "reviewed", "b" * 64)
    EmbeddingHoldout(("x" * (2 * 1024 * 1024 - 23),), documents, "reviewed", "b" * 64)


def test_embedding_gate_accepts_equivalent_runners_and_rejects_ranking_drift():
    accepted = evaluate_embedding_gate(_holdout(), _Runner(), _Runner())
    rejected = evaluate_embedding_gate(_holdout(), _Runner(), _Runner(True))

    assert accepted.accepted
    assert accepted.minimum_cosine_similarity == pytest.approx(1.0)
    assert accepted.minimum_top10_overlap == 1.0
    assert accepted.vector_count == 28
    assert not rejected.accepted
    assert rejected.minimum_cosine_similarity > 0.999
    assert rejected.minimum_top10_overlap == 0.8


def test_embedding_gate_rejects_wrong_runner_width():
    class ShortRunner:
        def embed(self, text):
            return [1.0]

    with pytest.raises(ValueError, match="384"):
        evaluate_embedding_gate(_holdout(), ShortRunner(), _Runner())


def test_retrieval_helpers_refuse_disagreeing_or_small_inputs():
    with pytest.raises(ValueError, match="^embedding widths disagree$"):
        _dot((1.0,), (1.0, 2.0))
    with pytest.raises(ValueError, match="candidate count"):
        _top_k((1.0,), ((1.0,),), 0)
    with pytest.raises(ValueError, match="candidate count"):
        _top_k((1.0,), ((1.0,),), 2)


@pytest.mark.parametrize("identifier", ["all-minilm-l6-v2", "bge-small-en-v1-5"])
def test_onnx_exporter_freezes_inputs_and_executes_recipe_wrapper(tmp_path, identifier):
    fetched = _fetched(tmp_path, identifier)
    destination = tmp_path / "portable.onnx"

    SentenceEmbeddingOnnxExporter().export(fetched, destination)

    model = onnx.load(destination)
    onnx.checker.check_model(model)
    input_shape = model.graph.input[0].type.tensor_type.shape.dim
    assert [dimension.dim_value for dimension in input_shape] == [1, 128]
    assert model.graph.output[0].name == "embedding"
    output_shape = model.graph.output[0].type.tensor_type.shape.dim
    assert [dimension.dim_value for dimension in output_shape] == [1, EMBEDDING_WIDTH]
    zeros = np.zeros((1, 128), dtype=np.int64)
    mask = np.array([[1] * 16 + [0] * 112], dtype=np.int64)
    result = ReferenceEvaluator(model).run(
        None,
        {"input_ids": zeros, "attention_mask": mask, "token_type_ids": zeros},
    )[0]
    assert result.shape == (1, EMBEDDING_WIDTH)
    assert math.sqrt(sum(float(value) ** 2 for value in result[0])) == pytest.approx(1.0)


def test_bge_exporter_accepts_the_pinned_source_opset(tmp_path):
    fetched = _fetched(tmp_path, "bge-small-en-v1-5")
    _source_model(fetched.root / "model.onnx", opset=11)

    SentenceEmbeddingOnnxExporter().export(fetched, tmp_path / "bge.onnx")

    assert onnx.load(tmp_path / "bge.onnx").opset_import[0].version == 11


@pytest.mark.parametrize(
    ("opset", "output_name", "detail"),
    [
        (12, "last_hidden_state", "embedding export requires ONNX opset 13+"),
        (13, "wrong", "ONNX encoder output disagrees with recipe"),
    ],
)
def test_onnx_exporter_rejects_incompatible_source_graph(tmp_path, opset, output_name, detail):
    fetched = _fetched(tmp_path)
    _source_model(fetched.root / "model.onnx", opset=opset, output_name=output_name)
    with pytest.raises(ModelRecipeError) as caught:
        SentenceEmbeddingOnnxExporter().export(fetched, tmp_path / "output.onnx")
    assert caught.value.code == "producer-incompatible"
    assert caught.value.detail == detail


def test_onnx_exporter_rejects_unsupported_recipe_and_input_contract(tmp_path):
    fetched = _fetched(tmp_path)
    unsupported = replace(fetched, recipe=replace(fetched.recipe, id="unsupported"))
    with pytest.raises(ModelRecipeError) as caught:
        SentenceEmbeddingOnnxExporter().export(unsupported, tmp_path / "unsupported.onnx")
    assert caught.value.code == "producer-incompatible"
    assert caught.value.detail == "recipe is not a supported embedding"

    model = onnx.load(fetched.root / "model.onnx")
    model.graph.input[0].name = "wrong"
    onnx.save_model(model, fetched.root / "model.onnx")
    with pytest.raises(ModelRecipeError) as caught:
        SentenceEmbeddingOnnxExporter().export(fetched, tmp_path / "wrong-input.onnx")
    assert caught.value.code == "producer-incompatible"
    assert caught.value.detail == "ONNX encoder inputs disagree with recipe"

    _source_model(fetched.root / "model.onnx")
    model = onnx.load(fetched.root / "model.onnx")
    del model.graph.input[0].type.tensor_type.shape.dim[-1]
    onnx.save_model(model, fetched.root / "model.onnx")
    with pytest.raises(ModelRecipeError) as caught:
        SentenceEmbeddingOnnxExporter().export(fetched, tmp_path / "wrong-rank.onnx")
    assert caught.value.code == "producer-incompatible"
    assert caught.value.detail == "ONNX encoder input rank disagrees"


def test_onnx_exporter_rejects_wrapper_name_collision(tmp_path):
    fetched = _fetched(tmp_path)
    model = onnx.load(fetched.root / "model.onnx")
    model.graph.node[0].name = "omnitensor_embedding_collision"
    model.graph.node[0].output[0] = "omnitensor_embedding_collision"
    model.graph.output[0].name = "omnitensor_embedding_collision"
    producer = dict(fetched.recipe.producer)
    producer["sourceOutputName"] = "omnitensor_embedding_collision"
    fetched = replace(fetched, recipe=replace(fetched.recipe, producer=producer))
    onnx.save_model(model, fetched.root / "model.onnx")
    with pytest.raises(ModelRecipeError) as caught:
        SentenceEmbeddingOnnxExporter().export(fetched, tmp_path / "collision.onnx")
    assert caught.value.code == "producer-incompatible"
    assert caught.value.detail == "ONNX graph collides with wrapper names"


def test_atomic_onnx_save_uses_private_sibling_and_disables_external_data(
    tmp_path, monkeypatch
):
    destination = tmp_path / "nested" / "models" / "portable.onnx"
    calls = []

    class FakeOnnx:
        @staticmethod
        def save_model(model, path, *, save_as_external_data):
            calls.append((model, path, save_as_external_data))
            path.write_bytes(b"model")

    real_mkstemp = tempfile.mkstemp

    def checked_mkstemp(*, prefix, suffix, dir):
        assert prefix == ".embedding-model-"
        assert suffix == ".onnx"
        assert dir == destination.parent
        return real_mkstemp(prefix=prefix, suffix=suffix, dir=dir)

    monkeypatch.setattr(tempfile, "mkstemp", checked_mkstemp)
    _save_onnx_atomic("graph", destination, FakeOnnx())
    assert destination.read_bytes() == b"model"
    assert calls == [("graph", calls[0][1], False)]
    assert calls[0][1].parent == destination.parent
    assert calls[0][1].name.startswith(".embedding-model-")
    assert calls[0][1].suffix == ".onnx"
    assert not calls[0][1].exists()


def test_produce_sentence_embedding_emits_provenance_without_native_claims(
    tmp_path, monkeypatch
):
    fetched = _fetched(tmp_path)
    opened = []

    def open_source(recipe_path, source_root):
        opened.append((recipe_path, source_root))
        return fetched

    monkeypatch.setattr(
        "omnitensor.training.embedding_production.open_fetched_model_source",
        open_source,
    )

    class FakeExporter:
        def export(self, source, destination):
            assert source is fetched
            destination.write_bytes(b"portable")

    source_calls = []
    portable_calls = []

    def source_factory(source):
        source_calls.append(source)
        return _Runner()

    def portable_factory(model, source):
        portable_calls.append((model, source))
        return _Runner()

    output = tmp_path / "nested" / "output"
    produced = produce_sentence_embedding(
        "recipe",
        "sources",
        output,
        _holdout(),
        source_factory,
        portable_factory,
        exporter=FakeExporter(),
    )
    report = json.loads(produced.report_path.read_text())
    assert produced.evidence.accepted
    assert opened == [("recipe", "sources")]
    assert source_calls == [fetched]
    assert portable_calls == [(produced.model_path, fetched)]
    native_claim = {
        "status": "unqualified",
        "reason": "no target compiler, native parity, or named-device evidence was run",
    }
    evidence = produced.evidence
    assert report == {
        "reportVersion": 1,
        "kind": "sentence-embedding-source-production",
        "recipeId": fetched.recipe.id,
        "recipeVersion": fetched.recipe.version,
        "recipeSha256": fetched.recipe.document_sha256,
        "sourceReceiptSha256": hashlib.sha256(b'{"verified":true}').hexdigest(),
        "sourceDigests": {source.role: source.sha256 for source in fetched.recipe.sources},
        "portableModel": {
            "format": "onnx",
            "sha256": hashlib.sha256(b"portable").hexdigest(),
            "tensorContract": fetched.recipe.tensor_contract,
            "outputContract": fetched.recipe.output_contract,
            "producer": fetched.recipe.producer,
        },
        "holdout": {
            "licenseId": "operator-reviewed-test-fixture",
            "corpusSha256": "a" * 64,
            "queryCount": 2,
            "documentCount": 12,
        },
        "portableSourceGate": {
            "minimumCosineSimilarity": evidence.minimum_cosine_similarity,
            "minimumTop10Overlap": evidence.minimum_top10_overlap,
            "nonfiniteOutputRate": 0.0,
            "vectorCount": 28,
            "accepted": True,
        },
        "nativeTargets": {"gpu": native_claim, "npu": native_claim, "tpu": native_claim},
        "cpuFallback": "forbidden",
    }
    with pytest.raises(ModelRecipeError, match="already exists"):
        produce_sentence_embedding(
            "recipe",
            "sources",
            output,
            _holdout(),
            lambda source: _Runner(),
            lambda model, source: _Runner(),
            exporter=FakeExporter(),
        )


def test_produce_sentence_embedding_refuses_unsupported_and_single_output_conflict(
    tmp_path, monkeypatch
):
    fetched = _fetched(tmp_path)
    unsupported = replace(fetched, recipe=replace(fetched.recipe, id="unsupported"))
    monkeypatch.setattr(
        "omnitensor.training.embedding_production.open_fetched_model_source",
        lambda recipe_path, source_root: unsupported,
    )
    with pytest.raises(ModelRecipeError) as caught:
        produce_sentence_embedding(
            "recipe",
            "source",
            tmp_path / "unsupported",
            _holdout(),
            lambda source: _Runner(),
            lambda model, source: _Runner(),
        )
    assert caught.value.code == "producer-incompatible"
    assert caught.value.detail == "recipe is not a supported embedding"

    monkeypatch.setattr(
        "omnitensor.training.embedding_production.open_fetched_model_source",
        lambda recipe_path, source_root: fetched,
    )
    output = tmp_path / "conflict"
    output.mkdir()
    (output / f"{fetched.recipe.id}-production-report.json").write_text("occupied")
    with pytest.raises(ModelRecipeError) as caught:
        produce_sentence_embedding(
            "recipe",
            "source",
            output,
            _holdout(),
            lambda source: _Runner(),
            lambda model, source: _Runner(),
        )
    assert caught.value.code == "producer-conflict"
    assert caught.value.detail == "embedding output already exists"


def test_produce_sentence_embedding_removes_failed_output(tmp_path, monkeypatch):
    fetched = _fetched(tmp_path)
    monkeypatch.setattr(
        "omnitensor.training.embedding_production.open_fetched_model_source",
        lambda recipe_path, source_root: fetched,
    )
    failed = EmbeddingGateEvidence(0.5, 0.2, 0.0, 24, 1, 11, False)
    monkeypatch.setattr(
        "omnitensor.training.embedding_production.evaluate_embedding_gate",
        lambda *args: failed,
    )

    class FakeExporter:
        def export(self, source, destination):
            destination.write_bytes(b"rejected")

    output = tmp_path / "output"
    with pytest.raises(ModelRecipeError) as caught:
        produce_sentence_embedding(
            "recipe",
            "sources",
            output,
            _holdout(),
            lambda source: _Runner(),
            lambda model, source: _Runner(),
            exporter=FakeExporter(),
        )
    assert caught.value.code == "portable-parity-failed"
    assert caught.value.detail == "embedding gate did not pass"
    assert not list(output.iterdir())
