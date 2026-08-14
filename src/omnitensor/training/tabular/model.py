"""Neutral value types for normalized multi-output logistic models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .fit import normalized_features, stable_sigmoid


@dataclass(frozen=True, slots=True)
class BuildExample:
    started_at_ms: int
    features: tuple[float, ...]
    build_failed: int
    executed_optional: tuple[str, ...]
    failed_optional: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BuildAdvisorModel:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]
    intercepts: tuple[float, ...]
    _normalize = staticmethod(normalized_features)
    _probability = staticmethod(stable_sigmoid)

    def predict(self, features: Sequence[float]) -> tuple[float, ...]:
        normalized = self._normalize(features, self.means, self.scales)
        return tuple(
            self._probability(
                intercept
                + sum(normalized[row] * self.weights[row][column] for row in range(len(normalized)))
            )
            for column, intercept in enumerate(self.intercepts)
        )


# The neutral names are the package's vocabulary.  The defining names/module
# remain the one-release public build contract so repr and pickle globals do
# not drift while the family facades migrate.
BuildExample.__module__ = "omnitensor.training.build"
BuildAdvisorModel.__module__ = "omnitensor.training.build"
TabularExample = BuildExample
NormalizedLogisticModel = BuildAdvisorModel
