"""Self-supervised forecasting: the only profile family that needs no labels.

`resource-scheduler` predicts the next window's utilisation from the previous
few.  That makes the label the future value, which arrives on its own — no
corpus has to be annotated, and no public dataset has to match this host's
hardware.  A week of recording is a training set.

The model is ridge-regularised least squares, fitted with a Gauss-Jordan solve
over the normal equations.  That is a deliberate choice, not a placeholder.  A
linear model on lagged features is a genuinely strong baseline for short-horizon
utilisation, it fits in milliseconds on a laptop, its coefficients can be read
and argued with, and — most importantly — it establishes the accuracy any
heavier model has to beat.  Shipping a neural network first would leave nobody
able to say whether it helped.

Two rules keep an honest model from becoming a dishonest one.

*A fit reports its own error.*  Every fit is scored against data it did not
see, and a model that cannot beat predicting "the same as last time" is
reported as not worth using rather than served anyway.

*A forecast is refused when the input does not match what was fitted.*  Feature
order and window size are part of the model's identity; serving a vector that
disagrees produces a confident number with no relationship to the input.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from omnitensor.telemetry_types import ForecastError, ForecastQuality, LinearForecaster

DEFAULT_RIDGE = 1e-6
MAX_FEATURES_PER_FIT = 512
MIN_TRAINING_ROWS = 8

def fit_forecaster(
    inputs: Sequence[Sequence[float]],
    targets: Sequence[float],
    feature_names: Sequence[str],
    window: int,
    *,
    ridge: float = DEFAULT_RIDGE,
    holdout: float = 0.25,
) -> LinearForecaster:
    """Fit and honestly score a forecaster over recorded windows."""
    _validate_shape(inputs, targets, feature_names, window)
    if isinstance(ridge, bool) or not isinstance(ridge, (int, float)) or ridge < 0:
        raise ForecastError("bounds-invalid", "ridge must be a non-negative number")
    if not isinstance(holdout, float) or not 0 < holdout < 1:
        raise ForecastError("bounds-invalid", "holdout must be a fraction in (0, 1)")

    # At least one row is always held out: holdout is in (0, 1), so the
    # truncated split is strictly smaller than the number of windows.
    split = max(1, int(len(inputs) * (1 - holdout)))
    # Split by time, never at random: shuffling lets the model see the future
    # of the very window it is scored on, which flatters it enormously.
    train_inputs, test_inputs = list(inputs[:split]), list(inputs[split:])
    train_targets, test_targets = list(targets[:split]), list(targets[split:])

    weights, intercept = _solve(train_inputs, train_targets, ridge)
    quality = _score(test_inputs, test_targets, weights, intercept, len(feature_names))
    return LinearForecaster(tuple(feature_names), window, tuple(weights), intercept, quality)


def _validate_shape(
    inputs: Sequence[Sequence[float]],
    targets: Sequence[float],
    feature_names: Sequence[str],
    window: int,
) -> None:
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ForecastError("bounds-invalid", "window must be a positive integer")
    if not feature_names:
        raise ForecastError("features-invalid", "an explicit feature order is required")
    if len(inputs) != len(targets):
        raise ForecastError("training-invalid", "inputs and targets must be the same length")
    if len(inputs) < MIN_TRAINING_ROWS:
        # Fitting on a handful of rows produces a model whose error estimate
        # is noise, which is worse than saying there is not enough history.
        raise ForecastError(
            "insufficient-history",
            f"at least {MIN_TRAINING_ROWS} windows are required, got {len(inputs)}",
        )
    width = len(feature_names) * window
    if width > MAX_FEATURES_PER_FIT:
        raise ForecastError(
            "features-invalid", f"at most {MAX_FEATURES_PER_FIT} inputs are allowed"
        )
    for row in inputs:
        if len(row) != width:
            raise ForecastError("training-invalid", f"every input row must have {width} values")
        _require_numbers(row, "training inputs")
    _require_numbers(targets, "targets")


def _require_numbers(values: Sequence[object], label: str) -> None:
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ForecastError("training-invalid", f"{label} must be numbers")
        if not math.isfinite(value):
            raise ForecastError("training-invalid", f"{label} must be finite")


def _solve(
    inputs: Sequence[Sequence[float]], targets: Sequence[float], ridge: float
) -> tuple[list[float], float]:
    """Ridge least squares through the normal equations, with an intercept."""
    width = len(inputs[0]) + 1
    # Augmented with a constant column so the intercept is solved for rather
    # than assumed to be zero.
    rows = [[1.0, *(float(value) for value in row)] for row in inputs]

    normal = [[0.0] * width for _ in range(width)]
    right = [0.0] * width
    for row, target in zip(rows, targets, strict=True):
        for i in range(width):
            right[i] += row[i] * float(target)
            for j in range(width):
                normal[i][j] += row[i] * row[j]
    for i in range(1, width):
        # The intercept is deliberately not penalised: shrinking it would bias
        # every prediction toward zero rather than toward the mean.
        normal[i][i] += ridge

    solution = _gauss_jordan(normal, right)
    return solution[1:], solution[0]


def _gauss_jordan(matrix: list[list[float]], right: list[float]) -> list[float]:
    size = len(right)
    augmented = [[*row, value] for row, value in zip(matrix, right, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            # Collinear features leave the system singular; refusing beats
            # returning coefficients that are an artefact of rounding.
            raise ForecastError(
                "fit-degenerate",
                "the recorded features are collinear; the fit has no unique solution",
            )
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(augmented[row], augmented[column], strict=True)
                ]
    return [row[-1] for row in augmented]


def _score(
    inputs: Sequence[Sequence[float]],
    targets: Sequence[float],
    weights: Sequence[float],
    intercept: float,
    feature_count: int,
) -> ForecastQuality:
    """Mean absolute error against held-out data, beside the naive baseline."""
    predictions = []
    baselines = []
    for row in inputs:
        predictions.append(
            intercept
            + sum(weight * float(value) for weight, value in zip(weights, row, strict=True))
        )
        # "The same as last time": the most recent value of the first feature,
        # which is the forecast anyone gets for free.
        baselines.append(float(row[-feature_count]))
    return evaluate_forecast_predictions(targets, predictions, baselines)


def evaluate_forecast_predictions(
    targets: Sequence[float],
    predictions: Sequence[float],
    repeat_last: Sequence[float],
) -> ForecastQuality:
    """Apply the same finite, held-out repeat-last gate to every model family."""
    if not targets or len(targets) != len(predictions) or len(targets) != len(repeat_last):
        raise ForecastError(
            "evaluation-invalid",
            "targets, predictions and baselines must have equal nonzero length",
        )
    _require_numbers(targets, "evaluation targets")
    _require_numbers(predictions, "evaluation predictions")
    _require_numbers(repeat_last, "evaluation baselines")
    error = sum(
        abs(float(predicted) - float(target))
        for predicted, target in zip(predictions, targets, strict=True)
    )
    baseline = sum(
        abs(float(previous) - float(target))
        for previous, target in zip(repeat_last, targets, strict=True)
    )
    return ForecastQuality(len(targets), error / len(targets), baseline / len(targets))


for _legacy_value in (fit_forecaster, evaluate_forecast_predictions):
    _legacy_value.__module__ = "omnitensor.plugins.forecasting"
