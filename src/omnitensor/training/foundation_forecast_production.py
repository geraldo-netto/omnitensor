"""Portable production gates for pinned foundation time-series candidates."""

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
from .recipes import FetchedModelSource, ModelRecipeError, open_fetched_model_source

FORECAST_RECIPE_IDS = frozenset({"amazon-chronos-bolt-tiny", "ibm-granite-ttm-r2"})
FORECAST_CONTEXT_WIDTH = 512
MIN_FORECAST_SAMPLES = 32
MAX_FORECAST_SAMPLES = 512
MIN_REPEAT_LAST_SKILL = 0.35
MAX_PORTABLE_EXPORT_ERROR = 0.0001


class ForecastRunner(Protocol):
    def predict(self, context: Sequence[float]) -> float: ...


class FoundationForecastExporter(Protocol):
    def export(self, source: FetchedModelSource, destination: Path) -> None: ...


class FoundationModuleLoader(Protocol):
    """Recipe-family adapter supplied by an explicitly installed producer package."""

    def __call__(self, source: FetchedModelSource) -> object: ...


@dataclass(frozen=True, slots=True)
class ForecastHoldout:
    contexts: tuple[tuple[float, ...], ...]
    targets: tuple[float, ...]
    observed_at_ms: tuple[int, ...]
    corpus_sha256: str

    def __post_init__(self) -> None:
        count = len(self.contexts)
        if not MIN_FORECAST_SAMPLES <= count <= MAX_FORECAST_SAMPLES:
            raise ValueError(
                f"forecast holdout requires {MIN_FORECAST_SAMPLES}..{MAX_FORECAST_SAMPLES} samples"
            )
        if len(self.targets) != count or len(self.observed_at_ms) != count:
            raise ValueError("forecast holdout arrays must have equal lengths")
        adjacent_times = zip(self.observed_at_ms, self.observed_at_ms[1:], strict=False)
        if any(later <= earlier for earlier, later in adjacent_times):
            raise ValueError("forecast holdout timestamps must be strictly increasing")
        for context in self.contexts:
            _validate_context(context)
        if not all(math.isfinite(target) for target in self.targets):
            raise ValueError("forecast targets must be finite")
        if len(self.corpus_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.corpus_sha256
        ):
            raise ValueError("forecast corpus digest must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class FoundationForecastEvidence:
    candidate_mae: float
    source_mae: float
    repeat_last_mae: float
    local_linear_mae: float
    repeat_last_skill: float
    portable_export_max_error: float
    sample_count: int
    accepted: bool


@dataclass(frozen=True, slots=True)
class ProducedFoundationForecast:
    model_path: Path
    report_path: Path
    evidence: FoundationForecastEvidence


def select_point_forecast(output: Sequence[Sequence[Sequence[float]]]) -> float:
    """Select TTM's first horizon from exact `[1,96,1]` source output."""
    _validate_output_shape(output, 96, 1)
    return _finite_scalar(output[0][0][0], "TTM first-horizon output")


def select_median_quantile(output: Sequence[Sequence[Sequence[float]]]) -> float:
    """Select Chronos' index-4 median and first horizon from exact `[1,9,64]`."""
    _validate_output_shape(output, 9, 64)
    return _finite_scalar(output[0][4][0], "Chronos median first-horizon output")


def evaluate_foundation_forecast(
    holdout: ForecastHoldout,
    source_runner: ForecastRunner,
    portable_runner: ForecastRunner,
    local_linear_runner: ForecastRunner,
) -> FoundationForecastEvidence:
    """Require parity and beat both repeat-last and the local fitted baseline."""
    source = tuple(_prediction(source_runner, context) for context in holdout.contexts)
    portable = tuple(_prediction(portable_runner, context) for context in holdout.contexts)
    local = tuple(_prediction(local_linear_runner, context) for context in holdout.contexts)
    repeat = tuple(context[-1] for context in holdout.contexts)
    source_mae = _mae(source, holdout.targets)
    candidate_mae = _mae(portable, holdout.targets)
    local_mae = _mae(local, holdout.targets)
    repeat_mae = _mae(repeat, holdout.targets)
    if repeat_mae == 0:
        raise ValueError("repeat-last MAE is zero; skill is undefined")
    skill = 1 - candidate_mae / repeat_mae
    export_error = max(abs(left - right) for left, right in zip(source, portable, strict=True))
    accepted = (
        skill >= MIN_REPEAT_LAST_SKILL
        and candidate_mae < repeat_mae
        and candidate_mae < local_mae
        and export_error <= MAX_PORTABLE_EXPORT_ERROR
    )
    return FoundationForecastEvidence(
        candidate_mae,
        source_mae,
        repeat_mae,
        local_mae,
        skill,
        export_error,
        len(holdout.contexts),
        accepted,
    )


class TorchFoundationForecastOnnxExporter:
    """Export a loader-supplied pinned model with the recipe's scalar selector."""

    def __init__(self, loader: FoundationModuleLoader):
        self._loader = loader

    def export(self, source: FetchedModelSource, destination: Path) -> None:
        try:
            import onnx
            import torch
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise ModelRecipeError(
                "producer-dependency-missing",
                "install the foundation-producers extra and its reviewed loader",
            ) from error
        recipe = source.recipe
        if recipe.id not in FORECAST_RECIPE_IDS or not recipe.producer:
            raise ModelRecipeError(
                "producer-incompatible", "recipe is not a pinned forecast candidate"
            )
        model = self._loader(source)
        if not hasattr(model, "eval"):
            raise ModelRecipeError(
                "producer-incompatible", "forecast loader did not return a module"
            )
        model = model.eval()
        producer_kind = recipe.producer["kind"]
        output_name = recipe.producer["sourceOutputName"]

        class ScalarForecastWrapper(torch.nn.Module):
            def __init__(self, candidate):
                super().__init__()
                self.candidate = candidate

            def forward(self, context):
                result = self.candidate(context)
                values = getattr(result, output_name)
                if producer_kind == "timeseries-point-forecast":
                    return values[:, 0, 0].reshape(1, 1)
                return values[:, 4, 0].reshape(1, 1)

        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, staged_name = tempfile.mkstemp(
            prefix=".foundation-forecast-", suffix=".onnx", dir=destination.parent
        )
        os.close(descriptor)
        staged = Path(staged_name)
        try:
            torch.onnx.export(
                ScalarForecastWrapper(model).eval(),
                torch.zeros((1, FORECAST_CONTEXT_WIDTH), dtype=torch.float32),
                staged,
                input_names=["context"],
                output_names=["forecast"],
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
            graph = onnx.load(staged, load_external_data=False)
            onnx.checker.check_model(graph)
            _validate_forecast_graph(graph)
            os.replace(staged, destination)
        finally:
            staged.unlink(missing_ok=True)


def produce_foundation_forecast(
    recipe_path: Path | str,
    source_root: Path | str,
    output_directory: Path | str,
    holdout: ForecastHoldout,
    source_runner_factory: Callable[[FetchedModelSource], ForecastRunner],
    portable_runner_factory: Callable[[Path], ForecastRunner],
    local_linear_runner: ForecastRunner,
    exporter: FoundationForecastExporter,
) -> ProducedFoundationForecast:
    """Publish a portable candidate only after parity and both baseline gates."""
    fetched = open_fetched_model_source(recipe_path, source_root)
    if fetched.recipe.id not in FORECAST_RECIPE_IDS:
        raise ModelRecipeError("producer-incompatible", "recipe is not a pinned forecast candidate")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / f"{fetched.recipe.id}.onnx"
    report_path = output / f"{fetched.recipe.id}-production-report.json"
    if model_path.exists() or report_path.exists():
        raise ModelRecipeError("producer-conflict", "forecast production output already exists")
    try:
        exporter.export(fetched, model_path)
        evidence = evaluate_foundation_forecast(
            holdout,
            source_runner_factory(fetched),
            portable_runner_factory(model_path),
            local_linear_runner,
        )
        if not evidence.accepted:
            raise ModelRecipeError("quality-gate-failed", "foundation forecast did not pass gates")
        write_json_atomic(
            report_path,
            _forecast_report(fetched, model_path, holdout, evidence),
            prefix=".foundation-report-",
        )
    except BaseException:
        model_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        raise
    return ProducedFoundationForecast(model_path, report_path, evidence)


def _validate_context(context: Sequence[float]) -> None:
    if len(context) != FORECAST_CONTEXT_WIDTH:
        raise ValueError("forecast context width must be 512")
    if not all(math.isfinite(value) for value in context):
        raise ValueError("forecast context values must be finite")


def _validate_output_shape(output: Sequence, middle: int, last: int) -> None:
    try:
        valid = (
            len(output) == 1
            and len(output[0]) == middle
            and all(len(row) == last for row in output[0])
        )
    except (IndexError, TypeError):
        valid = False
    if not valid:
        raise ValueError(f"forecast source output shape must be [1,{middle},{last}]")


def _finite_scalar(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _prediction(runner: ForecastRunner, context: tuple[float, ...]) -> float:
    return _finite_scalar(runner.predict(context), "forecast prediction")


def _mae(predictions: Sequence[float], targets: Sequence[float]) -> float:
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("MAE arrays must be non-empty and equal length")
    pairs = zip(predictions, targets, strict=True)
    return sum(abs(prediction - target) for prediction, target in pairs) / len(targets)


def _validate_forecast_graph(model: object) -> None:
    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    if len(inputs) != 1 or inputs[0].name != "context":
        raise ModelRecipeError("producer-invalid", "forecast ONNX must have one context input")
    if len(outputs) != 1 or outputs[0].name != "forecast":
        raise ModelRecipeError("producer-invalid", "forecast ONNX must have one scalar output")
    input_shape = [dimension.dim_value for dimension in inputs[0].type.tensor_type.shape.dim]
    output_shape = [dimension.dim_value for dimension in outputs[0].type.tensor_type.shape.dim]
    if input_shape != [1, 512] or output_shape != [1, 1]:
        raise ModelRecipeError("producer-invalid", "forecast ONNX shapes disagree with recipe")


def _forecast_report(
    source: FetchedModelSource,
    model_path: Path,
    holdout: ForecastHoldout,
    evidence: FoundationForecastEvidence,
) -> dict:
    return {
        "reportVersion": 1,
        "kind": "foundation-forecast-source-production",
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
            "corpusSha256": holdout.corpus_sha256,
            "sampleCount": evidence.sample_count,
            "timeOrdered": True,
        },
        "portableSourceGate": {
            "candidateMae": evidence.candidate_mae,
            "sourceMae": evidence.source_mae,
            "repeatLastMae": evidence.repeat_last_mae,
            "localLinearMae": evidence.local_linear_mae,
            "repeatLastSkill": evidence.repeat_last_skill,
            "portableExportMaxError": evidence.portable_export_max_error,
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
