"""Pinned Retinexformer reconstruction, portable export, and fidelity gates."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..atomicio import write_json_atomic
from .recipes import FetchedModelSource, ModelRecipeError, open_fetched_model_source

RETINEXFORMER_RECIPE_ID = "retinexformer-lol-v1"
RETINEXFORMER_IMAGE_SIZE = 256
RETINEXFORMER_IMAGE_COMPONENTS = 3 * RETINEXFORMER_IMAGE_SIZE * RETINEXFORMER_IMAGE_SIZE
MIN_RETINEXFORMER_PAIRS = 4
MAX_RETINEXFORMER_PAIRS = 128
MIN_PORTABLE_SOURCE_SSIM = 0.999
MIN_PAIRED_HOLDOUT_SSIM = 0.8


class RetinexformerRunner(Protocol):
    def enhance(self, image_ref: str) -> Sequence[float]: ...


class PairedTargetLoader(Protocol):
    def load(self, image_ref: str) -> Sequence[float]: ...


class RetinexformerExporter(Protocol):
    def export(self, source: FetchedModelSource, destination: Path) -> None: ...


class RetinexformerModuleLoader(Protocol):
    def __call__(self, source: FetchedModelSource) -> object: ...


@dataclass(frozen=True, slots=True)
class RetinexformerHoldout:
    input_refs: tuple[str, ...]
    target_refs: tuple[str, ...]
    license_id: str
    corpus_sha256: str

    def __post_init__(self) -> None:
        count = len(self.input_refs)
        if not MIN_RETINEXFORMER_PAIRS <= count <= MAX_RETINEXFORMER_PAIRS:
            raise ValueError(
                f"Retinexformer holdout requires {MIN_RETINEXFORMER_PAIRS}.."
                f"{MAX_RETINEXFORMER_PAIRS} pairs"
            )
        if len(self.target_refs) != count:
            raise ValueError("Retinexformer holdout arrays must have equal lengths")
        references = self.input_refs + self.target_refs
        if any(not isinstance(reference, str) or not reference.strip() for reference in references):
            raise ValueError("Retinexformer holdout references must be non-empty strings")
        if len(set(self.input_refs)) != count or len(set(self.target_refs)) != count:
            raise ValueError("Retinexformer holdout references must be unique within each side")
        if not isinstance(self.license_id, str) or not self.license_id.strip():
            raise ValueError("Retinexformer holdout license id is required")
        if len(self.corpus_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.corpus_sha256
        ):
            raise ValueError("Retinexformer corpus digest must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class RetinexformerGateEvidence:
    minimum_portable_source_ssim: float
    minimum_paired_holdout_ssim: float
    output_range_violation_rate: float
    pair_count: int
    accepted: bool


@dataclass(frozen=True, slots=True)
class ProducedRetinexformerSource:
    model_path: Path
    report_path: Path
    evidence: RetinexformerGateEvidence


class PinnedRetinexformerLoader:
    """Execute the verified pinned architecture and strictly load official parameters."""

    def __call__(self, source: FetchedModelSource) -> object:
        if source.recipe.id != RETINEXFORMER_RECIPE_ID:
            raise ModelRecipeError("producer-incompatible", "recipe is not pinned Retinexformer")
        try:
            import torch
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise ModelRecipeError(
                "producer-dependency-missing", "install the retinexformer-producers extra"
            ) from error
        architecture = _source_path(source, "architecture")
        checkpoint_path = _source_path(source, "model")
        module_name = f"_omnitensor_retinexformer_{source.recipe.document_sha256[:16]}"
        spec = importlib.util.spec_from_file_location(module_name, architecture)
        if spec is None or spec.loader is None:
            raise ModelRecipeError(
                "source-invalid", "cannot load pinned Retinexformer architecture"
            )
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            model_type = module.RetinexFormer
            model = model_type(
                in_channels=3,
                out_channels=3,
                n_feat=40,
                stage=1,
                num_blocks=[1, 2, 2],
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("params"), dict):
                raise ModelRecipeError(
                    "source-invalid", "Retinexformer checkpoint must contain a params mapping"
                )
            model.load_state_dict(checkpoint["params"], strict=True)
        except ModelRecipeError:
            raise
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise ModelRecipeError(
                "source-invalid", f"cannot reconstruct pinned Retinexformer: {error}"
            ) from error
        return model.eval()


class TorchRetinexformerOnnxExporter:
    """Export the strictly reconstructed model with in-graph unit-range clamp."""

    def __init__(self, loader: RetinexformerModuleLoader | None = None):
        self._loader = loader or PinnedRetinexformerLoader()

    def export(self, source: FetchedModelSource, destination: Path) -> None:
        try:
            import onnx
            import torch
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise ModelRecipeError(
                "producer-dependency-missing", "install the retinexformer-producers extra"
            ) from error
        if source.recipe.id != RETINEXFORMER_RECIPE_ID or not source.recipe.producer:
            raise ModelRecipeError("producer-incompatible", "recipe is not pinned Retinexformer")
        model = self._loader(source)
        if not hasattr(model, "eval"):
            raise ModelRecipeError(
                "producer-incompatible", "Retinexformer loader returned no module"
            )
        model = model.eval()

        class ClampedRetinexformer(torch.nn.Module):
            def __init__(self, candidate):
                super().__init__()
                self.candidate = candidate

            def forward(self, image):
                return torch.clamp(self.candidate(image), min=0.0, max=1.0)

        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, staged_name = tempfile.mkstemp(
            prefix=".retinexformer-", suffix=".onnx", dir=destination.parent
        )
        os.close(descriptor)
        staged = Path(staged_name)
        try:
            torch.onnx.export(
                ClampedRetinexformer(model).eval(),
                torch.zeros((1, 3, 256, 256), dtype=torch.float32),
                staged,
                input_names=["image"],
                output_names=["enhanced"],
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
            graph = onnx.load(staged, load_external_data=False)
            onnx.checker.check_model(graph)
            _validate_retinexformer_graph(graph)
            os.replace(staged, destination)
        finally:
            staged.unlink(missing_ok=True)


def structural_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return deterministic whole-image SSIM for two unit-range NCHW vectors."""
    if len(left) != len(right) or not left:
        raise ValueError("SSIM vectors must be non-empty and equal length")
    count = len(left)
    left_mean = sum(left) / count
    right_mean = sum(right) / count
    left_variance = sum((value - left_mean) ** 2 for value in left) / count
    right_variance = sum((value - right_mean) ** 2 for value in right) / count
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    ) / count
    luminance = 2 * left_mean * right_mean + 0.0001
    contrast = 2 * covariance + 0.0009
    denominator = (left_mean**2 + right_mean**2 + 0.0001) * (
        left_variance + right_variance + 0.0009
    )
    return max(-1.0, min(1.0, luminance * contrast / denominator))


def evaluate_retinexformer_gate(
    holdout: RetinexformerHoldout,
    source_runner: RetinexformerRunner,
    portable_runner: RetinexformerRunner,
    target_loader: PairedTargetLoader,
) -> RetinexformerGateEvidence:
    """Gate source parity, paired fidelity, and every portable output component."""
    parity = []
    quality = []
    violations = 0
    for input_ref, target_ref in zip(holdout.input_refs, holdout.target_refs, strict=True):
        source = _bounded_image(source_runner.enhance(input_ref), "source")
        portable, portable_violations = _portable_image(portable_runner.enhance(input_ref))
        target = _bounded_image(target_loader.load(target_ref), "target")
        violations += portable_violations
        parity.append(structural_similarity(source, portable))
        quality.append(structural_similarity(portable, target))
    minimum_parity = min(parity)
    minimum_quality = min(quality)
    violation_rate = violations / (len(holdout.input_refs) * RETINEXFORMER_IMAGE_COMPONENTS)
    accepted = (
        minimum_parity >= MIN_PORTABLE_SOURCE_SSIM
        and minimum_quality >= MIN_PAIRED_HOLDOUT_SSIM
        and violations == 0
    )
    return RetinexformerGateEvidence(
        minimum_parity,
        minimum_quality,
        violation_rate,
        len(holdout.input_refs),
        accepted,
    )


def produce_retinexformer_source(
    recipe_path: Path | str,
    source_root: Path | str,
    output_directory: Path | str,
    holdout: RetinexformerHoldout,
    source_runner_factory: Callable[[FetchedModelSource], RetinexformerRunner],
    portable_runner_factory: Callable[[Path], RetinexformerRunner],
    target_loader: PairedTargetLoader,
    exporter: RetinexformerExporter | None = None,
) -> ProducedRetinexformerSource:
    """Publish a portable graph only; never infer native accelerator evidence."""
    fetched = open_fetched_model_source(recipe_path, source_root)
    if fetched.recipe.id != RETINEXFORMER_RECIPE_ID:
        raise ModelRecipeError("producer-incompatible", "recipe is not pinned Retinexformer")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "retinexformer-lol-v1.onnx"
    report_path = output / "retinexformer-lol-v1-production-report.json"
    if model_path.exists() or report_path.exists():
        raise ModelRecipeError("producer-conflict", "Retinexformer production output exists")
    try:
        (exporter or TorchRetinexformerOnnxExporter()).export(fetched, model_path)
        evidence = evaluate_retinexformer_gate(
            holdout,
            source_runner_factory(fetched),
            portable_runner_factory(model_path),
            target_loader,
        )
        if not evidence.accepted:
            raise ModelRecipeError("quality-gate-failed", "Retinexformer did not pass gates")
        write_json_atomic(
            report_path,
            _retinexformer_report(fetched, model_path, holdout, evidence),
            prefix=".retinexformer-report-",
        )
    except BaseException:
        model_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        raise
    return ProducedRetinexformerSource(model_path, report_path, evidence)


def _source_path(source: FetchedModelSource, role: str) -> Path:
    item = next((item for item in source.recipe.sources if item.role == role), None)
    if item is None:
        raise ModelRecipeError("source-invalid", f"Retinexformer source lacks {role}")
    return source.root / item.filename


def _bounded_image(values: Sequence[float], label: str) -> tuple[float, ...]:
    result, violations = _image_values(values)
    if violations:
        raise ValueError(f"Retinexformer {label} image must be finite and within [0,1]")
    return result


def _portable_image(values: Sequence[float]) -> tuple[tuple[float, ...], int]:
    return _image_values(values)


def _image_values(values: Sequence[float]) -> tuple[tuple[float, ...], int]:
    if len(values) != RETINEXFORMER_IMAGE_COMPONENTS:
        raise ValueError(
            f"Retinexformer image must contain {RETINEXFORMER_IMAGE_COMPONENTS} components"
        )
    result = []
    violations = 0
    for value in values:
        try:
            scalar = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("Retinexformer image components must be numeric") from error
        if not math.isfinite(scalar) or not 0 <= scalar <= 1:
            violations += 1
            scalar = 0.0 if not math.isfinite(scalar) else max(0.0, min(1.0, scalar))
        result.append(scalar)
    return tuple(result), violations


def _validate_retinexformer_graph(model: object) -> None:
    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    if len(inputs) != 1 or inputs[0].name != "image":
        raise ModelRecipeError("producer-invalid", "Retinexformer ONNX needs one image input")
    if len(outputs) != 1 or outputs[0].name != "enhanced":
        raise ModelRecipeError("producer-invalid", "Retinexformer ONNX needs one enhanced output")
    expected = [1, 3, 256, 256]
    input_shape = [dimension.dim_value for dimension in inputs[0].type.tensor_type.shape.dim]
    output_shape = [dimension.dim_value for dimension in outputs[0].type.tensor_type.shape.dim]
    if input_shape != expected or output_shape != expected:
        raise ModelRecipeError("producer-invalid", "Retinexformer ONNX shapes disagree with recipe")


def _retinexformer_report(
    source: FetchedModelSource,
    model_path: Path,
    holdout: RetinexformerHoldout,
    evidence: RetinexformerGateEvidence,
) -> dict:
    generic_reason = "no target compiler, native parity, or named-device evidence was run"
    return {
        "reportVersion": 1,
        "kind": "retinexformer-source-production",
        "recipeId": source.recipe.id,
        "recipeVersion": source.recipe.version,
        "recipeSha256": source.recipe.document_sha256,
        "sourceReceiptSha256": _digest(source.receipt_path),
        "sourceDigests": {item.role: item.sha256 for item in source.recipe.sources},
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
            "pairCount": evidence.pair_count,
        },
        "portableSourceGate": {
            "minimumPortableSourceSsim": evidence.minimum_portable_source_ssim,
            "minimumPairedHoldoutSsim": evidence.minimum_paired_holdout_ssim,
            "outputRangeViolationRate": evidence.output_range_violation_rate,
            "accepted": evidence.accepted,
        },
        "nativeTargets": {
            "gpu": {"status": "unqualified", "reason": generic_reason},
            "npu": {"status": "unqualified", "reason": generic_reason},
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


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
