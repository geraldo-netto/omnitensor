"""Bounded public meaning for a locally trained forecast output."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

FORECAST_KIND = "forecast"
MAX_FORECAST_NESTING = 3
MAX_TARGET_CHARS = 64
MAX_HORIZON = 128
_READING_FIELDS = frozenset({"kind", "targetFeature", "horizon", "value"})


class ForecastResultError(ValueError):
    """A forecast model produced an output that cannot mean one scalar."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def forecast_reading(model: Mapping | None, tensors: object) -> dict | None:
    """Describe one forecast scalar, or ``None`` for a non-forecast model.

    The executor-specific raw tensors remain beside this reading.  This only
    adds target and observation-count horizon metadata that shape alone cannot
    recover.
    """
    contract = model.get("featureContract") if isinstance(model, Mapping) else None
    if not isinstance(contract, Mapping) or contract.get("recipe") != "forecast-v1":
        return None
    return {
        "kind": FORECAST_KIND,
        "targetFeature": contract["targetFeature"],
        "horizon": contract["horizon"],
        "value": _single_finite_value(tensors),
    }


def parse_forecast_reading(document: object) -> dict | None:
    """Defensively accept only the exact public forecast-reading shape."""
    if not isinstance(document, Mapping) or document.get("kind") != FORECAST_KIND:
        return None
    if set(document) != _READING_FIELDS:
        return None
    target = document.get("targetFeature")
    horizon = document.get("horizon")
    value = document.get("value")
    if not isinstance(target, str) or not 1 <= len(target) <= MAX_TARGET_CHARS:
        return None
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= MAX_HORIZON:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return {
        "kind": FORECAST_KIND,
        "targetFeature": target,
        "horizon": horizon,
        "value": float(value),
    }


def _single_finite_value(tensors: object) -> float:
    if not isinstance(tensors, Sequence) or isinstance(tensors, (str, bytes)):
        raise ForecastResultError(
            "forecast-output-invalid", "forecast must contain exactly one output tensor"
        )
    current = tensors
    for _depth in range(MAX_FORECAST_NESTING):
        if not isinstance(current, Sequence) or isinstance(current, (str, bytes)):
            break
        if len(current) != 1:
            raise ForecastResultError(
                "forecast-output-invalid", "forecast must produce exactly one numeric value"
            )
        current = current[0]
    if isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
        raise ForecastResultError(
            "forecast-output-invalid", "forecast output nesting exceeds supported lanes"
        )
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise ForecastResultError("forecast-output-invalid", "forecast value must be numeric")
    if not math.isfinite(current):
        raise ForecastResultError("forecast-output-invalid", "forecast value must be finite")
    return float(current)
