"""Deterministic fitting and metric primitives for numeric trainers."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

from ..contracts import TrainingError

Example = TypeVar("Example")


def stable_sigmoid(value: float) -> float:
    """Return a numerically stable logistic probability."""
    if value >= 0:
        inverse = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(max(value, -700.0))
    return exponential / (1.0 + exponential)


def normalized_features(
    features: Sequence[float], means: Sequence[float], scales: Sequence[float]
) -> tuple[float, ...]:
    """Normalize one build-family feature row with stable refusal text."""
    if len(features) != len(means) or len(scales) != len(means):
        raise TrainingError("features-invalid", "build feature width disagrees")
    values = []
    for value, mean, scale in zip(features, means, scales, strict=True):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TrainingError("features-invalid", "build features must be numbers")
        if not math.isfinite(value):
            raise TrainingError("features-invalid", "build features must be finite")
        values.append((float(value) - mean) / scale)
    return tuple(values)


def population_normalization(
    examples: Sequence[Example],
    feature_of: Callable[[Example], Sequence[float]],
    *,
    scale_floor: float,
    width: int | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return population means and standard deviations with an explicit floor."""
    feature_width = len(feature_of(examples[0])) if width is None else width
    means = tuple(
        sum(feature_of(item)[index] for item in examples) / len(examples)
        for index in range(feature_width)
    )
    scales = tuple(
        max(
            scale_floor,
            math.sqrt(
                sum((feature_of(item)[index] - means[index]) ** 2 for item in examples)
                / len(examples)
            ),
        )
        for index in range(feature_width)
    )
    return means, scales


def unchecked_normalized_features(
    features: Sequence[float], means: Sequence[float], scales: Sequence[float]
) -> tuple[float, ...]:
    """Normalize an already validated family row without changing legacy failures."""
    return tuple(
        (value - mean) / scale for value, mean, scale in zip(features, means, scales, strict=True)
    )


def normalization(
    examples: Sequence[Example],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Build-family population normalization with the historical 1e-9 floor."""
    return population_normalization(
        examples,
        lambda item: item.features,
        scale_floor=1e-9,
    )


def fit_balanced_logistic(
    examples: Sequence[Example],
    label_of: Callable[[Example], int],
    feature_of: Callable[[Example], Sequence[float]],
    means: tuple[float, ...],
    scales: tuple[float, ...],
    *,
    iterations: int,
    normalize: Callable[
        [Sequence[float], Sequence[float], Sequence[float]], tuple[float, ...]
    ] = normalized_features,
    sigmoid: Callable[[float], float] = stable_sigmoid,
) -> tuple[tuple[float, ...], float]:
    """Fit one balanced logistic output with deterministic gradient descent."""
    labels = tuple(label_of(item) for item in examples)
    positives = sum(labels)
    positive_weight = (len(labels) - positives) / positives
    total_weight = 2 * (len(labels) - positives)
    weights = [0.0] * len(means)
    intercept = 0.0
    for step in range(iterations):
        gradients = [0.0] * len(weights)
        intercept_gradient = 0.0
        for item, label in zip(examples, labels, strict=True):
            features = normalize(feature_of(item), means, scales)
            probability = sigmoid(
                intercept
                + sum(weight * value for weight, value in zip(weights, features, strict=True))
            )
            sample_weight = positive_weight if label else 1.0
            error = sample_weight * (probability - label)
            intercept_gradient += error
            for index, value in enumerate(features):
                gradients[index] += error * value
        rate = 0.2 / math.sqrt(step + 1)
        intercept -= rate * intercept_gradient / total_weight
        for index in range(len(weights)):
            weights[index] -= rate * (gradients[index] / total_weight + 1e-4 * weights[index])
    return tuple(weights), intercept


def fit_output(
    examples: Sequence[Example],
    label_of: Callable[[Example], int],
    means: tuple[float, ...],
    scales: tuple[float, ...],
) -> tuple[tuple[float, ...], float]:
    """Fit one build-family output using the historical 240 iterations."""
    return fit_balanced_logistic(
        examples,
        label_of,
        lambda item: item.features,
        means,
        scales,
        iterations=240,
    )


def ranked_auc(
    scored: Iterable[tuple[float, int]],
    *,
    empty_class_detail: str | None,
) -> float:
    """Return tie-aware binary AUC, optionally refusing a missing class."""
    ordered = sorted(scored)
    positives = sum(label for _score, label in ordered)
    negatives = len(ordered) - positives
    if empty_class_detail is not None and (positives == 0 or negatives == 0):
        raise TrainingError("class-imbalance", empty_class_detail)
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        rank_sum += average_rank * sum(label for _score, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def binary_auc(scored: Iterable[tuple[float, int]]) -> float:
    """Build-family AUC with the historical missing-class refusal."""
    return ranked_auc(scored, empty_class_detail="AUC requires both build outcomes")


def binary_class_counts(
    examples: Sequence[Example], label_of: Callable[[Example], int]
) -> tuple[int, int]:
    """Return positive and negative example counts."""
    positives = sum(label_of(item) for item in examples)
    return positives, len(examples) - positives


def require_binary_class_floor(
    examples: Sequence[Example],
    label_of: Callable[[Example], int],
    minimum: int,
    detail: str,
) -> None:
    """Refuse a binary split unless both classes meet one floor."""
    positives, negatives = binary_class_counts(examples, label_of)
    if positives < minimum or negatives < minimum:
        raise TrainingError("class-imbalance", detail)


def unit_interval(value: object, name: str) -> float:
    """Return one finite inclusive unit-interval metric."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be in [0, 1]")
    return float(value)
