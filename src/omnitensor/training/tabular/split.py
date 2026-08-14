"""Chronological split policies shared by numeric trainers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from ..contracts import TrainingError

Example = TypeVar("Example")


def chronological_split(
    examples: Sequence[Example], *, insufficient_detail: str
) -> tuple[tuple[Example, ...], tuple[Example, ...]]:
    """Return the historical bounded 80/20 chronological split."""
    if len(examples) < 2:
        raise TrainingError("insufficient-history", insufficient_detail)
    split = min(len(examples) - 1, max(1, int(len(examples) * 0.8)))
    return tuple(examples[:split]), tuple(examples[split:])


def floor_chronological_split(
    rows: Sequence[Example],
    minimum_training: int,
    minimum_holdout: int,
    *,
    insufficient_detail: str,
) -> tuple[tuple[Example, ...], tuple[Example, ...]]:
    """Return an 80/20 split constrained by explicit side floors."""
    if len(rows) < minimum_training + minimum_holdout:
        raise TrainingError("insufficient-history", insufficient_detail)
    split = max(minimum_training, int(len(rows) * 0.8))
    split = min(split, len(rows) - minimum_holdout)
    return tuple(rows[:split]), tuple(rows[split:])
