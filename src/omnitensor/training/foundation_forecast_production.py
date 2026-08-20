"""Portable production gates for pinned foundation time-series candidates."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..atomicio import write_json_atomic
from .production_pipeline import (
    export_torch_onnx_atomic,
    production_report,
    run_production_pipeline,
    validate_onnx_io,
)
from .recipe_fetch import open_fetched_model_source
from .recipe_model import FetchedModelSource, ModelRecipeError

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
            import onnx  # noqa: PLC0415 - deferred: an optional or heavy dependency
            import torch  # noqa: PLC0415 - deferred: an optional or heavy dependency
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

        def build_arguments() -> tuple[object, object]:
            return (
                ScalarForecastWrapper(model).eval(),
                torch.zeros((1, FORECAST_CONTEXT_WIDTH), dtype=torch.float32),
            )

        export_torch_onnx_atomic(
            destination,
            prefix=".foundation-forecast-",
            build_arguments=build_arguments,
            input_names=("context",),
            output_names=("forecast",),
            validate=_validate_forecast_graph,
            torch=torch,
            onnx=onnx,
        )


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
    return run_production_pipeline(
        model_path,
        report_path,
        conflict_detail="forecast production output already exists",
        export=lambda: exporter.export(fetched, model_path),
        evaluate=lambda: evaluate_foundation_forecast(
            holdout,
            source_runner_factory(fetched),
            portable_runner_factory(model_path),
            local_linear_runner,
        ),
        accepted=lambda evidence: evidence.accepted,
        rejection_code="quality-gate-failed",
        rejection_detail="foundation forecast did not pass gates",
        report=lambda evidence: _forecast_report(fetched, model_path, holdout, evidence),
        report_prefix=".foundation-report-",
        result=ProducedFoundationForecast,
        writer=write_json_atomic,
    )


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
    validate_onnx_io(
        model,
        input_name="context",
        input_shape=(1, 512),
        input_detail="forecast ONNX must have one context input",
        output_name="forecast",
        output_shape=(1, 1),
        output_detail="forecast ONNX must have one scalar output",
        shape_detail="forecast ONNX shapes disagree with recipe",
    )


def _forecast_report(
    source: FetchedModelSource,
    model_path: Path,
    holdout: ForecastHoldout,
    evidence: FoundationForecastEvidence,
) -> dict:
    return production_report(
        source,
        model_path,
        kind="foundation-forecast-source-production",
        source_identity={
            "sourceDigests": {item.role: item.sha256 for item in source.recipe.sources}
        },
        holdout={
            "corpusSha256": holdout.corpus_sha256,
            "sampleCount": evidence.sample_count,
            "timeOrdered": True,
        },
        portable_source_gate={
            "candidateMae": evidence.candidate_mae,
            "sourceMae": evidence.source_mae,
            "repeatLastMae": evidence.repeat_last_mae,
            "localLinearMae": evidence.local_linear_mae,
            "repeatLastSkill": evidence.repeat_last_skill,
            "portableExportMaxError": evidence.portable_export_max_error,
            "accepted": evidence.accepted,
        },
    )
