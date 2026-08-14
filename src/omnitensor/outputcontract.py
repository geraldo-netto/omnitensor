"""What a model's output means, as its publisher states it.

A classifier answered with a thousand bare floats.  Correct, and unusable:
the only thing a consumer could do with it was pick the largest and report
"class 421", and to do better it had to know the reduction and hold a label
list of its own.  Both are properties of the model, so a consumer holding them
is a consumer that silently relabels every result the day the artifact
changes — which the digest exists to prevent everywhere else.

So the manifest states them.  ``kind`` says whether the output reduces at all,
``topK`` how far, and ``labels`` names a file installed beside the weights and
digest-verified with them.

Two absences are deliberate.  No contract at all returns the tensors exactly
as before, so every profile written before this field is unchanged.  And a
classification with no labels reports indices rather than inventing names: a
guessed label list is worse than an honest number, because it is wrong in a
form that reads as right.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

CLASSIFICATION = "classification"
DEFAULT_TOP_K = 5
MAX_TOP_K = 100
MAX_LABELS = 100_000
MAX_LABEL_CHARS = 160
_TOP_K_ERROR = f"output contract topK must be in [1, {MAX_TOP_K}]"


@dataclass(frozen=True, slots=True)
class OutputSpec:
    """The declared meaning of one model's output."""

    kind: str
    top_k: int = DEFAULT_TOP_K
    labels: str | None = None

    @property
    def reduces(self) -> bool:
        return self.kind == CLASSIFICATION


def declared_output(model: Mapping | None) -> OutputSpec | None:
    """The contract a manifest's model declares, or ``None``."""
    if not isinstance(model, Mapping):
        return None
    contract = model.get("outputContract")
    if not isinstance(contract, Mapping) or "kind" not in contract:
        return None
    top_k = contract.get("topK", DEFAULT_TOP_K)
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        top_k = DEFAULT_TOP_K
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(_TOP_K_ERROR)
    return OutputSpec(
        contract["kind"],
        top_k,
        contract.get("labels"),
    )


def parse_labels(text: str) -> tuple[str, ...]:
    """One label per line, blank lines dropped, each bounded.

    Bounded because the file is read into a snapshot a desktop renders, and a
    label list is an artifact like any other: its size is not a promise
    anybody made.
    """
    labels = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            labels.append(stripped[:MAX_LABEL_CHARS])
        if len(labels) >= MAX_LABELS:
            break
    return tuple(labels)


def _scores(tensor: object) -> list[float] | None:
    """The innermost run of numbers, since a 1000-class output arrives nested."""
    current = tensor
    while isinstance(current, list) and current and isinstance(current[0], list):
        if len(current) != 1:
            # More than one row is not a single classification, and picking one
            # would report a confident answer about an arbitrary slice.
            return None
        current = current[0]
    if not isinstance(current, list) or not current:
        return None
    numeric = (int, float)
    if not all(
        isinstance(value, numeric) and not isinstance(value, bool) for value in current
    ):
        return None
    return [float(value) for value in current]


def reduce_output(
    spec: OutputSpec | None,
    tensors: Sequence,
    labels: Sequence[str] = (),
) -> dict | None:
    """Reduce a model's raw output as its contract says, or ``None``.

    ``None`` means "nothing to add" — no contract, a kind that does not reduce,
    or an output whose shape the reduction cannot honestly describe. The caller
    keeps the raw tensors either way; this only ever adds a reading of them.
    """
    if spec is None:
        return None
    if (
        isinstance(spec.top_k, bool)
        or not isinstance(spec.top_k, int)
        or not 1 <= spec.top_k <= MAX_TOP_K
    ):
        raise ValueError(_TOP_K_ERROR)
    if not spec.reduces or not tensors:
        return None
    scores = _scores(tensors[0])
    if scores is None:
        return None
    ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    top = [index for index in ranked[:spec.top_k] if math.isfinite(scores[index])]
    return {
        "kind": spec.kind,
        "top": [_entry(index, scores[index], labels) for index in top],
    }


def _entry(index: int, score: float, labels: Sequence[str]) -> dict:
    entry = {"index": index, "score": score}
    if index < len(labels):
        entry["label"] = labels[index]
    return entry
