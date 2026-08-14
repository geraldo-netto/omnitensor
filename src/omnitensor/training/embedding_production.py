"""Fail-closed production of portable sentence-embedding ONNX sources."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..atomicio import write_json_atomic
from .production_pipeline import (
    production_report,
    run_production_pipeline,
    width_checked_dot,
)
from .recipe_fetch import open_fetched_model_source
from .recipe_model import FetchedModelSource, ModelRecipeError

EMBEDDING_RECIPE_IDS = frozenset({"all-minilm-l6-v2", "bge-small-en-v1-5"})
EMBEDDING_WIDTH = 384
MIN_RETRIEVAL_DOCUMENTS = 11
MAX_HOLDOUT_TEXTS = 512
MAX_HOLDOUT_TEXT_BYTES = 2 * 1024 * 1024
MIN_COSINE_PARITY = 0.999
MIN_TOP10_OVERLAP = 0.9
_EPSILON = 1e-12


class EmbeddingRunner(Protocol):
    """Producer-only runner for one exact source or portable graph."""

    def embed(self, text: str) -> Sequence[float]: ...


class SentenceEmbeddingExporter(Protocol):
    """Convert one verified encoder into the recipe's fixed portable graph."""

    def export(self, source: FetchedModelSource, destination: Path) -> None: ...


@dataclass(frozen=True, slots=True)
class EmbeddingHoldout:
    """Separately reviewed local text; its content never enters evidence."""

    queries: tuple[str, ...]
    documents: tuple[str, ...]
    license_id: str
    corpus_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.queries, tuple) or not isinstance(self.documents, tuple):
            raise ValueError("embedding holdout texts must be immutable tuples")
        if not self.queries:
            raise ValueError("embedding holdout requires at least one query")
        if len(self.documents) < MIN_RETRIEVAL_DOCUMENTS:
            raise ValueError(
                f"embedding holdout requires at least {MIN_RETRIEVAL_DOCUMENTS} documents"
            )
        if not self.license_id.strip():
            raise ValueError("embedding holdout license id is required")
        if len(self.corpus_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.corpus_sha256
        ):
            raise ValueError("embedding holdout corpus digest must be lowercase SHA-256")
        _validate_texts(self.queries + self.documents)


@dataclass(frozen=True, slots=True)
class EmbeddingGateEvidence:
    """Portable/source numerical and retrieval evidence, never device evidence."""

    minimum_cosine_similarity: float
    minimum_top10_overlap: float
    nonfinite_output_rate: float
    vector_count: int
    query_count: int
    document_count: int
    accepted: bool


@dataclass(frozen=True, slots=True)
class ProducedEmbeddingSource:
    """One portable model plus its private-corpus-free production report."""

    model_path: Path
    report_path: Path
    evidence: EmbeddingGateEvidence


def l2_normalize(vector: Sequence[float]) -> tuple[float, ...]:
    """Return a finite unit vector and refuse zero or malformed input."""
    if not vector:
        raise ValueError("embedding vector must not be empty")
    try:
        values = tuple(float(value) for value in vector)
    except (TypeError, ValueError) as error:
        raise ValueError("embedding vector must be numeric") from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError("embedding vector must be finite")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= _EPSILON:
        raise ValueError("embedding vector norm is zero")
    return tuple(value / norm for value in values)


def pool_sentence_embedding(
    hidden_states: Sequence[Sequence[float]],
    attention_mask: Sequence[int],
    method: str,
) -> tuple[float, ...]:
    """Reference implementation of the pooling embedded in the ONNX graph."""
    rows = tuple(tuple(float(value) for value in row) for row in hidden_states)
    if not rows or any(len(row) != len(rows[0]) for row in rows) or not rows[0]:
        raise ValueError("hidden states must be a non-empty rectangular matrix")
    if len(attention_mask) != len(rows) or any(value not in (0, 1) for value in attention_mask):
        raise ValueError("attention mask must contain one binary value per token")
    if method == "cls-token-l2":
        pooled = rows[0]
    elif method == "attention-mask-mean-pool-l2":
        count = sum(attention_mask)
        if count == 0:
            raise ValueError("mean pooling requires at least one unmasked token")
        pooled = tuple(
            sum(row[column] * mask for row, mask in zip(rows, attention_mask, strict=True))
            / count
            for column in range(len(rows[0]))
        )
    else:
        raise ValueError(f"unsupported embedding pooling method: {method}")
    return l2_normalize(pooled)


def evaluate_embedding_gate(
    holdout: EmbeddingHoldout,
    source_runner: EmbeddingRunner,
    portable_runner: EmbeddingRunner,
) -> EmbeddingGateEvidence:
    """Compare fixed portable output and retrieval ordering with its source."""
    texts = holdout.queries + holdout.documents
    source_vectors = tuple(_runner_vector(source_runner, text) for text in texts)
    portable_vectors = tuple(_runner_vector(portable_runner, text) for text in texts)
    cosines = tuple(
        _dot(source, portable)
        for source, portable in zip(source_vectors, portable_vectors, strict=True)
    )
    query_count = len(holdout.queries)
    source_queries, source_documents = (
        source_vectors[:query_count],
        source_vectors[query_count:],
    )
    portable_queries, portable_documents = (
        portable_vectors[:query_count],
        portable_vectors[query_count:],
    )
    overlaps = []
    for source_query, portable_query in zip(
        source_queries, portable_queries, strict=True
    ):
        source_top = set(_top_k(source_query, source_documents, 10))
        portable_top = set(_top_k(portable_query, portable_documents, 10))
        overlaps.append(len(source_top & portable_top) / 10)
    minimum_cosine = min(cosines)
    minimum_overlap = min(overlaps)
    accepted = minimum_cosine >= MIN_COSINE_PARITY and minimum_overlap >= MIN_TOP10_OVERLAP
    return EmbeddingGateEvidence(
        minimum_cosine,
        minimum_overlap,
        0.0,
        len(texts) * 2,
        query_count,
        len(holdout.documents),
        accepted,
    )


class SentenceEmbeddingOnnxExporter:
    """Append fixed recipe pooling and normalization to a verified ONNX encoder."""

    def export(self, source: FetchedModelSource, destination: Path) -> None:
        try:
            import onnx
            from onnx import TensorProto, checker, helper
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise ModelRecipeError(
                "producer-dependency-missing", "install OmniTensor with the train extra"
            ) from error
        recipe = source.recipe
        if recipe.id not in EMBEDDING_RECIPE_IDS or not recipe.producer:
            raise ModelRecipeError("producer-incompatible", "recipe is not a supported embedding")
        model = onnx.load(source.root / recipe.model_source.filename, load_external_data=False)
        default_opsets = [item.version for item in model.opset_import if not item.domain]
        minimum_opset = (
            13
            if recipe.producer["postprocessing"] == "attention-mask-mean-pool-l2"
            else 11
        )
        if not default_opsets or default_opsets[0] < minimum_opset:
            raise ModelRecipeError(
                "producer-incompatible",
                f"embedding export requires ONNX opset {minimum_opset}+",
            )
        _freeze_embedding_inputs(model, recipe.producer, recipe.tensor_contract)
        _append_embedding_wrapper(model, recipe.producer, TensorProto, helper)
        checker.check_model(model)
        _save_onnx_atomic(model, destination, onnx)


def produce_sentence_embedding(
    recipe_path: Path | str,
    source_root: Path | str,
    output_directory: Path | str,
    holdout: EmbeddingHoldout,
    source_runner_factory: Callable[[FetchedModelSource], EmbeddingRunner],
    portable_runner_factory: Callable[[Path, FetchedModelSource], EmbeddingRunner],
    *,
    exporter: SentenceEmbeddingExporter | None = None,
) -> ProducedEmbeddingSource:
    """Build and gate a portable source without making a native/device claim."""
    fetched = open_fetched_model_source(recipe_path, source_root)
    if fetched.recipe.id not in EMBEDDING_RECIPE_IDS:
        raise ModelRecipeError("producer-incompatible", "recipe is not a supported embedding")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / f"{fetched.recipe.id}.onnx"
    report_path = output / f"{fetched.recipe.id}-production-report.json"

    def prepare_export() -> Callable[[], None]:
        adapter = exporter or SentenceEmbeddingOnnxExporter()
        return lambda: adapter.export(fetched, model_path)

    return run_production_pipeline(
        model_path,
        report_path,
        conflict_detail="embedding output already exists",
        export=None,
        evaluate=lambda: evaluate_embedding_gate(
            holdout,
            source_runner_factory(fetched),
            portable_runner_factory(model_path, fetched),
        ),
        accepted=lambda evidence: evidence.accepted,
        rejection_code="portable-parity-failed",
        rejection_detail="embedding gate did not pass",
        report=lambda evidence: _embedding_report(fetched, model_path, holdout, evidence),
        report_prefix=".embedding-report-",
        result=ProducedEmbeddingSource,
        writer=write_json_atomic,
        prepare_export=prepare_export,
    )


def _validate_texts(texts: tuple[str, ...]) -> None:
    if len(texts) > MAX_HOLDOUT_TEXTS:
        raise ValueError("embedding holdout contains too many texts")
    size = 0
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("embedding holdout texts must be non-empty strings")
        size += len(text.encode("utf-8"))
    if size > MAX_HOLDOUT_TEXT_BYTES:
        raise ValueError("embedding holdout text exceeds its byte bound")


def _runner_vector(runner: EmbeddingRunner, text: str) -> tuple[float, ...]:
    vector = l2_normalize(runner.embed(text))
    if len(vector) != EMBEDDING_WIDTH:
        raise ValueError(f"embedding runner must return {EMBEDDING_WIDTH} values")
    return vector


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return width_checked_dot(left, right, mismatch_detail="embedding widths disagree")


def _top_k(
    query: Sequence[float], documents: Sequence[Sequence[float]], count: int
) -> tuple[int, ...]:
    if count <= 0 or len(documents) < count:
        raise ValueError("retrieval candidate count is too small")
    ranked = sorted(
        range(len(documents)),
        key=lambda index: (-_dot(query, documents[index]), index),
    )
    return tuple(ranked[:count])


def _freeze_embedding_inputs(model: object, producer: dict, tensor_contract: dict) -> None:
    graph = model.graph
    inputs = {value.name: value for value in graph.input}
    if tuple(inputs) != tuple(producer["inputNames"]):
        raise ModelRecipeError("producer-incompatible", "ONNX encoder inputs disagree with recipe")
    for name, contract in zip(producer["inputNames"], tensor_contract["inputs"], strict=True):
        value = inputs[name]
        dimensions = value.type.tensor_type.shape.dim
        if len(dimensions) != len(contract["shape"]):
            raise ModelRecipeError("producer-incompatible", "ONNX encoder input rank disagrees")
        for dimension, expected in zip(dimensions, contract["shape"], strict=True):
            dimension.ClearField("dim_param")
            dimension.dim_value = expected


def _append_embedding_wrapper(
    model: object,
    producer: dict,
    tensor_proto: object,
    helper: object,
) -> None:
    graph = model.graph
    source_name = producer["sourceOutputName"]
    if source_name not in {value.name for value in graph.output}:
        raise ModelRecipeError("producer-incompatible", "ONNX encoder output disagrees with recipe")
    occupied = {
        name
        for node in graph.node
        for name in (*node.input, *node.output)
        if name
    } | {item.name for item in graph.initializer}
    prefix = "omnitensor_embedding_"
    if any(name.startswith(prefix) for name in occupied):
        raise ModelRecipeError("producer-incompatible", "ONNX graph collides with wrapper names")
    nodes, initializers, pooled = _pooling_nodes(
        source_name,
        producer["postprocessing"],
        tensor_proto,
        helper,
        prefix,
    )
    epsilon_name = f"{prefix}epsilon"
    norm_name = f"{prefix}norm"
    safe_norm_name = f"{prefix}safe_norm"
    output_name = "embedding"
    initializers.append(helper.make_tensor(epsilon_name, tensor_proto.FLOAT, [1], [_EPSILON]))
    nodes.extend(
        (
            helper.make_node("ReduceL2", [pooled], [norm_name], axes=[1], keepdims=1),
            helper.make_node("Clip", [norm_name, epsilon_name, ""], [safe_norm_name]),
            helper.make_node("Div", [pooled, safe_norm_name], [output_name]),
        )
    )
    graph.node.extend(nodes)
    graph.initializer.extend(initializers)
    del graph.output[:]
    graph.output.append(
        helper.make_tensor_value_info(output_name, tensor_proto.FLOAT, producer["outputShape"])
    )


def _pooling_nodes(
    hidden_name: str,
    method: str,
    tensor_proto: object,
    helper: object,
    prefix: str,
) -> tuple[list, list, str]:
    if method == "cls-token-l2":
        index_name = f"{prefix}cls_index"
        pooled = f"{prefix}pooled"
        return (
            [helper.make_node("Gather", [hidden_name, index_name], [pooled], axis=1)],
            [helper.make_tensor(index_name, tensor_proto.INT64, [], [0])],
            pooled,
        )
    if method != "attention-mask-mean-pool-l2":
        raise ModelRecipeError("producer-incompatible", "unsupported embedding pooling")
    mask_float = f"{prefix}mask_float"
    axes_two = f"{prefix}axes_two"
    mask_3d = f"{prefix}mask_3d"
    masked = f"{prefix}masked"
    axes_one = f"{prefix}axes_one"
    summed = f"{prefix}summed"
    count = f"{prefix}count"
    one = f"{prefix}one"
    safe_count = f"{prefix}safe_count"
    count_2d = f"{prefix}count_2d"
    pooled = f"{prefix}pooled"
    nodes = [
        helper.make_node("Cast", ["attention_mask"], [mask_float], to=tensor_proto.FLOAT),
        helper.make_node("Unsqueeze", [mask_float, axes_two], [mask_3d]),
        helper.make_node("Mul", [hidden_name, mask_3d], [masked]),
        helper.make_node("ReduceSum", [masked, axes_one], [summed], keepdims=0),
        helper.make_node("ReduceSum", [mask_float, axes_one], [count], keepdims=0),
        helper.make_node("Clip", [count, one, ""], [safe_count]),
        helper.make_node("Unsqueeze", [safe_count, axes_one], [count_2d]),
        helper.make_node("Div", [summed, count_2d], [pooled]),
    ]
    initializers = [
        helper.make_tensor(axes_two, tensor_proto.INT64, [1], [2]),
        helper.make_tensor(axes_one, tensor_proto.INT64, [1], [1]),
        helper.make_tensor(one, tensor_proto.FLOAT, [1], [1.0]),
    ]
    return nodes, initializers, pooled


def _save_onnx_atomic(model: object, destination: Path, onnx: object) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".embedding-model-", suffix=".onnx", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        onnx.save_model(model, temporary, save_as_external_data=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _embedding_report(
    source: FetchedModelSource,
    model_path: Path,
    holdout: EmbeddingHoldout,
    evidence: EmbeddingGateEvidence,
) -> dict:
    source_digests = {item.role: item.sha256 for item in source.recipe.sources}
    return production_report(
        source,
        model_path,
        kind="sentence-embedding-source-production",
        source_identity={"sourceDigests": source_digests},
        holdout={
            "licenseId": holdout.license_id,
            "corpusSha256": holdout.corpus_sha256,
            "queryCount": evidence.query_count,
            "documentCount": evidence.document_count,
        },
        portable_source_gate={
            "minimumCosineSimilarity": evidence.minimum_cosine_similarity,
            "minimumTop10Overlap": evidence.minimum_top10_overlap,
            "nonfiniteOutputRate": evidence.nonfinite_output_rate,
            "vectorCount": evidence.vector_count,
            "accepted": evidence.accepted,
        },
    )
