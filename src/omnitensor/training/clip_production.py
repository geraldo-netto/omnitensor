"""Portable CLIP image-encoder production with source-only quality evidence."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..atomicio import write_json_atomic
from .embedding_production import l2_normalize
from .recipes import FetchedModelSource, ModelRecipeError, open_fetched_model_source

CLIP_RECIPE_ID = "clip-vit-b-32-image"
CLIP_IMAGE_SIZE = 224
CLIP_EMBEDDING_WIDTH = 512
MAX_IMAGE_EDGE = 16_384
MAX_HOLDOUT_IMAGES = 256
MIN_HOLDOUT_IMAGES = 10
MIN_CLIP_COSINE_PARITY = 0.999
MIN_ZERO_SHOT_AGREEMENT = 0.99


class ClipImageRunner(Protocol):
    """Run one exact image encoder from an operator-owned image reference."""

    def embed_image(self, image_ref: str) -> Sequence[float]: ...


class ClipImageExporter(Protocol):
    """Export the verified TorchScript source into fixed image-only ONNX."""

    def export(self, source: FetchedModelSource, destination: Path) -> None: ...


@dataclass(frozen=True, slots=True)
class ClipHoldout:
    """Reviewed local image references and reviewed prompt embeddings."""

    image_refs: tuple[str, ...]
    label_embeddings: tuple[tuple[float, ...], ...]
    license_id: str
    corpus_sha256: str

    def __post_init__(self) -> None:
        if not MIN_HOLDOUT_IMAGES <= len(self.image_refs) <= MAX_HOLDOUT_IMAGES:
            raise ValueError(
                f"CLIP holdout requires {MIN_HOLDOUT_IMAGES}..{MAX_HOLDOUT_IMAGES} images"
            )
        if any(not isinstance(ref, str) or not ref.strip() for ref in self.image_refs):
            raise ValueError("CLIP holdout image references must be non-empty strings")
        if len(set(self.image_refs)) != len(self.image_refs):
            raise ValueError("CLIP holdout image references must be unique")
        if len(self.label_embeddings) < 2:
            raise ValueError("CLIP holdout requires at least two label embeddings")
        _validate_label_embeddings(self.label_embeddings)
        if not self.license_id.strip():
            raise ValueError("CLIP holdout license id is required")
        if len(self.corpus_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.corpus_sha256
        ):
            raise ValueError("CLIP holdout corpus digest must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class ClipGateEvidence:
    minimum_cosine_similarity: float
    zero_shot_top1_agreement: float
    nonfinite_output_rate: float
    image_count: int
    accepted: bool


@dataclass(frozen=True, slots=True)
class ProducedClipSource:
    model_path: Path
    report_path: Path
    evidence: ClipGateEvidence


def clip_resize_geometry(width: int, height: int) -> tuple[int, int, int, int]:
    """Return cover-resize dimensions and centered 224-square crop origin."""
    if isinstance(width, bool) or isinstance(height, bool):
        raise ValueError("CLIP image dimensions must be integers")
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError("CLIP image dimensions must be integers")
    if width < 1 or height < 1 or width > MAX_IMAGE_EDGE or height > MAX_IMAGE_EDGE:
        raise ValueError(f"CLIP image dimensions must be within 1..{MAX_IMAGE_EDGE}")
    if width <= height:
        resized_width = CLIP_IMAGE_SIZE
        resized_height = math.ceil(height * CLIP_IMAGE_SIZE / width)
    else:
        resized_height = CLIP_IMAGE_SIZE
        resized_width = math.ceil(width * CLIP_IMAGE_SIZE / height)
    crop_left = (resized_width - CLIP_IMAGE_SIZE) // 2
    crop_top = (resized_height - CLIP_IMAGE_SIZE) // 2
    return resized_width, resized_height, crop_left, crop_top


def normalize_clip_rgb(rgb: bytes) -> tuple[float, ...]:
    """Normalize one already bicubic-resized/center-cropped RGB image to NCHW."""
    expected = CLIP_IMAGE_SIZE * CLIP_IMAGE_SIZE * 3
    if not isinstance(rgb, bytes) or len(rgb) != expected:
        raise ValueError(f"CLIP RGB crop must contain exactly {expected} bytes")
    means = (122.7709383, 116.7460125, 104.09373615)
    scales = (0.01459842661924292, 0.015007768493717056, 0.014220065717024088)
    return tuple(
        (rgb[pixel * 3 + channel] - means[channel]) * scales[channel]
        for channel in range(3)
        for pixel in range(CLIP_IMAGE_SIZE * CLIP_IMAGE_SIZE)
    )


def evaluate_clip_gate(
    holdout: ClipHoldout,
    source_runner: ClipImageRunner,
    portable_runner: ClipImageRunner,
) -> ClipGateEvidence:
    """Gate portable embeddings and zero-shot decisions against the source."""
    labels = tuple(l2_normalize(label) for label in holdout.label_embeddings)
    cosines = []
    agreements = 0
    for image_ref in holdout.image_refs:
        source = _clip_vector(source_runner.embed_image(image_ref))
        portable = _clip_vector(portable_runner.embed_image(image_ref))
        cosines.append(_dot(source, portable))
        source_label = _best_label(source, labels)
        portable_label = _best_label(portable, labels)
        agreements += source_label == portable_label
    minimum_cosine = min(cosines)
    agreement = agreements / len(holdout.image_refs)
    return ClipGateEvidence(
        minimum_cosine,
        agreement,
        0.0,
        len(holdout.image_refs),
        minimum_cosine >= MIN_CLIP_COSINE_PARITY and agreement >= MIN_ZERO_SHOT_AGREEMENT,
    )


class TorchScriptClipOnnxExporter:
    """Load the pinned CLIP checkpoint and export only normalized image embeddings."""

    def export(self, source: FetchedModelSource, destination: Path) -> None:
        try:
            import onnx
            import torch
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise ModelRecipeError(
                "producer-dependency-missing",
                "install the model-producers extra for Torch and ONNX",
            ) from error
        recipe = source.recipe
        if recipe.id != CLIP_RECIPE_ID or not recipe.producer:
            raise ModelRecipeError(
                "producer-incompatible", "recipe is not the pinned CLIP image model"
            )
        encoder = torch.jit.load(
            str(source.root / recipe.model_source.filename), map_location="cpu"
        ).eval()

        class ImageOnlyWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, image):
                embedding = self.model.encode_image(image)
                return torch.nn.functional.normalize(embedding, p=2, dim=1, eps=1e-12)

        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, staged_name = tempfile.mkstemp(
            prefix=".clip-image-", suffix=".onnx", dir=destination.parent
        )
        os.close(descriptor)
        staged = Path(staged_name)
        try:
            dummy = torch.zeros((1, 3, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE), dtype=torch.float32)
            torch.onnx.export(
                ImageOnlyWrapper(encoder).eval(),
                dummy,
                staged,
                input_names=["image"],
                output_names=["embedding"],
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
            model = onnx.load(staged, load_external_data=False)
            onnx.checker.check_model(model)
            _validate_clip_graph(model)
            os.replace(staged, destination)
        finally:
            staged.unlink(missing_ok=True)


def produce_clip_source(
    recipe_path: Path | str,
    source_root: Path | str,
    output_directory: Path | str,
    holdout: ClipHoldout,
    source_runner_factory: Callable[[FetchedModelSource], ClipImageRunner],
    portable_runner_factory: Callable[[Path], ClipImageRunner],
    *,
    exporter: ClipImageExporter | None = None,
) -> ProducedClipSource:
    """Export and gate CLIP without claiming compiler or device evidence."""
    fetched = open_fetched_model_source(recipe_path, source_root)
    if fetched.recipe.id != CLIP_RECIPE_ID:
        raise ModelRecipeError("producer-incompatible", "recipe is not the pinned CLIP image model")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "clip-vit-b-32-image.onnx"
    report_path = output / "clip-vit-b-32-image-production-report.json"
    if model_path.exists() or report_path.exists():
        raise ModelRecipeError("producer-conflict", "CLIP production output already exists")
    adapter = exporter or TorchScriptClipOnnxExporter()
    try:
        adapter.export(fetched, model_path)
        evidence = evaluate_clip_gate(
            holdout,
            source_runner_factory(fetched),
            portable_runner_factory(model_path),
        )
        if not evidence.accepted:
            raise ModelRecipeError("portable-parity-failed", "CLIP portable-source gate failed")
        write_json_atomic(
            report_path,
            _clip_report(fetched, model_path, holdout, evidence),
            prefix=".clip-report-",
        )
    except BaseException:
        model_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        raise
    return ProducedClipSource(model_path, report_path, evidence)


def _clip_vector(vector: Sequence[float]) -> tuple[float, ...]:
    normalized = l2_normalize(vector)
    if len(normalized) != CLIP_EMBEDDING_WIDTH:
        raise ValueError("CLIP runner output width must be 512")
    return normalized


def _validate_label_embeddings(labels: tuple[tuple[float, ...], ...]) -> None:
    for embedding in labels:
        if len(embedding) != CLIP_EMBEDDING_WIDTH:
            raise ValueError("CLIP label embedding width must be 512")
        l2_normalize(embedding)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))  # noqa: B905 - fixed widths validated


def _best_label(vector: Sequence[float], labels: Sequence[Sequence[float]]) -> int:
    return max(range(len(labels)), key=lambda index: (_dot(vector, labels[index]), -index))


def _validate_clip_graph(model: object) -> None:
    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    if len(inputs) != 1 or inputs[0].name != "image":
        raise ModelRecipeError("producer-invalid", "CLIP ONNX must have one image input")
    if len(outputs) != 1 or outputs[0].name != "embedding":
        raise ModelRecipeError("producer-invalid", "CLIP ONNX must have one embedding output")
    input_shape = [dimension.dim_value for dimension in inputs[0].type.tensor_type.shape.dim]
    output_shape = [dimension.dim_value for dimension in outputs[0].type.tensor_type.shape.dim]
    if input_shape != [1, 3, 224, 224] or output_shape != [1, 512]:
        raise ModelRecipeError("producer-invalid", "CLIP ONNX shapes disagree with recipe")


def _clip_report(
    source: FetchedModelSource,
    model_path: Path,
    holdout: ClipHoldout,
    evidence: ClipGateEvidence,
) -> dict:
    return {
        "reportVersion": 1,
        "kind": "clip-image-source-production",
        "recipeId": source.recipe.id,
        "recipeVersion": source.recipe.version,
        "recipeSha256": source.recipe.document_sha256,
        "sourceReceiptSha256": _digest(source.receipt_path),
        "sourceModelSha256": source.recipe.model_source.sha256,
        "portableModel": {
            "format": "onnx",
            "sha256": _digest(model_path),
            "tensorContract": source.recipe.tensor_contract,
            "outputContract": source.recipe.output_contract,
            "producer": source.recipe.producer,
        },
        "holdout": {
            "licenseId": holdout.license_id,
            "corpusSha256": holdout.corpus_sha256,
            "imageCount": evidence.image_count,
            "labelCount": len(holdout.label_embeddings),
        },
        "portableSourceGate": {
            "minimumCosineSimilarity": evidence.minimum_cosine_similarity,
            "zeroShotTop1Agreement": evidence.zero_shot_top1_agreement,
            "nonfiniteOutputRate": evidence.nonfinite_output_rate,
            "accepted": evidence.accepted,
        },
        "nativeTargets": {
            target: {
                "status": "unqualified",
                "reason": "no target compiler, native parity, or named-device evidence was run",
            }
            for target in ("gpu", "npu", "tpu")
        },
        "cpuFallback": "forbidden",
    }


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
