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
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

# Where the labelled cases are, in the order they are looked for.
#
# The corpus used to be source-tree only: `omnitensor.benchmark` shipped in
# the wheel and `benchmarks/` did not, so the one command that can say whether
# a workload still answers correctly refused on every installed host — "no
# case file for selected-text-tools" naming a path under `site-packages` that
# was never going to exist. Twenty-four kilobytes of JSON is not what kept it
# out; nothing did. They ship beside this module now.
#
# The source tree still wins when it is there, because that is the copy
# somebody edits, and `OMNITENSOR_BENCHMARK_ROOT` still outranks both for a
# corpus kept somewhere else entirely.
BENCHMARK_ROOT_VARIABLE = "OMNITENSOR_BENCHMARK_ROOT"
SOURCE_BENCHMARK_ROOT = Path(__file__).resolve().parents[3] / "benchmarks"
# Where `pyproject.toml` force-includes `benchmarks/cases` in the wheel.
PACKAGED_CASE_ROOT = Path(__file__).resolve().parent / "cases"


def benchmark_root(environ: Mapping[str, str] | None = None) -> Path:
    """Where the benchmark corpus lives on this machine."""
    configured = (environ if environ is not None else os.environ).get(BENCHMARK_ROOT_VARIABLE)
    return Path(configured).expanduser() if configured else SOURCE_BENCHMARK_ROOT


def case_root(environ: Mapping[str, str] | None = None) -> Path:
    """The cases directory: configured, then the source tree, then packaged."""
    configured = (environ if environ is not None else os.environ).get(BENCHMARK_ROOT_VARIABLE)
    if configured:
        return Path(configured).expanduser() / "cases"
    source = SOURCE_BENCHMARK_ROOT / "cases"
    return source if source.is_dir() else PACKAGED_CASE_ROOT


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


def load(workload: str, root: Path | None = None) -> tuple[Case, ...]:
    """Every case for one workload, in file order."""
    path = (Path(root) if root is not None else case_root()) / f"{workload}.json"
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise CaseError(
            f"no case file for {workload}: {path}"
            f" (set {BENCHMARK_ROOT_VARIABLE} to a corpus that carries one)"
        ) from error
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
    if (
        not isinstance(sources, list)
        or not sources
        or not all(isinstance(source, str) and source.strip() for source in sources)
    ):
        raise CaseError(f"{workload} case {index} supplies no sources")
    expect = entry.get("expect", {})
    if not isinstance(expect, dict):
        raise CaseError(f"{workload} case {index} states no checkable expectation")
    # A rule this code does not have is silently *not applied* by the judge,
    # which scores an answer as if the case had never asked. That is exactly
    # what happened on 2026-08-17: a long-running process held an older module,
    # a Hebrew translation containing Cyrillic passed a rule forbidding Cyrillic,
    # and the run reported 10/10. Refusing here turns a silent wrong score into
    # a loud failure to start.
    from .scoring import RULES  # noqa: PLC0415 - avoids a circular import

    unknown = set(expect) - set(RULES) - {"refuses"}
    if unknown:
        raise CaseError(
            f"{workload} case {index} names rules this build does not have: "
            f"{', '.join(sorted(unknown))}"
        )
    return Case(
        id=str(entry.get("id") or f"{workload}-{index}"),
        workload=workload,
        sources=tuple(sources),
        expect=expect,
        note=str(entry.get("note", "")),
    )


def available(root: Path | None = None) -> tuple[str, ...]:
    """Every workload that has a case file.

    Refuses when the corpus is not where it was looked for: this is an
    argparse default, so answering ``()`` made an installed package report
    "nothing was measured" instead of naming the directory it could not find.
    """
    resolved = Path(root) if root is not None else case_root()
    if not resolved.is_dir():
        raise CaseError(
            f"benchmark cases are not installed at {resolved}; "
            f"set {BENCHMARK_ROOT_VARIABLE} to the checkout's benchmarks/ directory"
        )
    return tuple(sorted(path.stem for path in resolved.glob("*.json")))


def counted(workloads: Iterable[str], root: Path | None = None) -> dict[str, int]:
    return {workload: len(load(workload, root)) for workload in workloads}


__all__ = [
    "BENCHMARK_ROOT_VARIABLE",
    "Case",
    "CaseError",
    "available",
    "benchmark_root",
    "case_root",
    "counted",
    "load",
]
