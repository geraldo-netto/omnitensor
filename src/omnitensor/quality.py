"""Shared fail-closed quality thresholds."""

from __future__ import annotations

import math

DEFAULT_THRESHOLD = 80.0


def validate_threshold(threshold: object) -> float:
    """Return a finite percentage or reject a gate that could fail open."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("quality threshold must be a finite percentage")
    if not math.isfinite(threshold) or not 0 <= threshold <= 100:
        raise ValueError("quality threshold must be between 0 and 100")
    return float(threshold)
