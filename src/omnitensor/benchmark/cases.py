"""The labelled work a model is judged on, and how a case states its answer.

A benchmark is only as honest as what it asks for, so a case never says "good"
— it states checkable properties of a correct answer: which words must survive
a translation, how many tasks are in a paragraph, that a date is 2026-09-03,
that every citation names a source that was actually supplied.

Cases live as JSON beside the code rather than inside it so that adding one is
not a code change, and so the set can be read by somebody deciding whether the
score means anything.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parents[3] / "benchmarks" / "cases"

# A case file that is larger than this is not a case file.
MAX_CASE_BYTES = 4 * 1024 * 1024


class CaseError(ValueError):
    """A case file that cannot be read as cases."""


@dataclass(frozen=True, slots=True)
class Case:
    """One piece of labelled work.

    ``sources`` is what the model is given, in order, as fragments. ``expect``
    is what a correct answer must show, and every key in it is checked by a
    named rule in :mod:`omnitensor.benchmark.scoring` — an expectation nobody
    checks is a comment pretending to be data.
    """

    id: str
    workload: str
    sources: tuple[str, ...]
    expect: dict = field(default_factory=dict)
    note: str = ""

    @property
    def answerable(self) -> bool:
        """Whether a refusal is wrong for this case.

        Some cases exist to check that a workload *does* refuse — a document
        with no date for event extraction — and scoring them the same way would
        reward the opposite behaviour.
        """
        return self.expect.get("refuses") is not True


def load(workload: str, root: Path = CASE_ROOT) -> tuple[Case, ...]:
    """Every case for one workload, in file order."""
    path = Path(root) / f"{workload}.json"
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise CaseError(f"no case file for {workload}: {path}") from error
    if len(raw) > MAX_CASE_BYTES:
        raise CaseError(f"case file for {workload} is oversized")
    try:
        document = json.loads(raw)
    except (UnicodeError, ValueError) as error:
        raise CaseError(f"case file for {workload} is not JSON") from error
    if not isinstance(document, dict) or not isinstance(document.get("cases"), list):
        raise CaseError(f"case file for {workload} holds no cases")
    return tuple(_case(workload, entry, index) for index, entry in enumerate(document["cases"]))


def _case(workload: str, entry: object, index: int) -> Case:
    if not isinstance(entry, dict):
        raise CaseError(f"{workload} case {index} is not an object")
    sources = entry.get("sources")
    if not isinstance(sources, list) or not sources or not all(
        isinstance(source, str) and source.strip() for source in sources
    ):
        raise CaseError(f"{workload} case {index} supplies no sources")
    expect = entry.get("expect", {})
    if not isinstance(expect, dict):
        raise CaseError(f"{workload} case {index} states no checkable expectation")
    return Case(
        id=str(entry.get("id") or f"{workload}-{index}"),
        workload=workload,
        sources=tuple(sources),
        expect=expect,
        note=str(entry.get("note", "")),
    )


def available(root: Path = CASE_ROOT) -> tuple[str, ...]:
    """Every workload that has a case file."""
    return tuple(sorted(path.stem for path in Path(root).glob("*.json")))


def counted(workloads: Iterable[str], root: Path = CASE_ROOT) -> dict[str, int]:
    return {workload: len(load(workload, root)) for workload in workloads}


__all__ = ["CASE_ROOT", "Case", "CaseError", "available", "counted", "load"]
