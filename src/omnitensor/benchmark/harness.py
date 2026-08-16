"""Run labelled cases against one model, through the real generation path.

Fidelity is the whole point, so this does not paraphrase the production path:
it loads the model with the same flags the provider uses, builds the same
`GenerationTask` the workload runs, publishes the same kind of source
fragments, and asks the same runtime to generate. A benchmark that measures
something adjacent to what ships is a benchmark that lies.

That is also why this is the harness OMNI-0361 qualifies with: a pair that
scores here has been exercised exactly as it will run, and the numbers it
produces — layers offloaded, context, cache — are the numbers a receipt
records.

Nothing here bounds an answer. Cases are scored on whether they are right, and
the token budget is the task's own.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..plugins.event_workload import MemoryFragmentStore
from ..plugins.events import SourceFragment
from ..plugins.generation import GenerationRequest
from .cases import Case
from .scoring import Judgement, Tally, judge


@dataclass(frozen=True, slots=True)
class Timing:
    """What one generation cost, in the terms a person compares."""

    seconds: float
    output_characters: int

    @property
    def characters_per_second(self) -> float:
        return self.output_characters / self.seconds if self.seconds > 0 else 0.0


@dataclass(frozen=True, slots=True)
class Outcome:
    """One case, one model: what came back, whether it was right, what it cost."""

    case_id: str
    judgement: Judgement
    timing: Timing
    document: dict | None
    error: str = ""


@dataclass(frozen=True, slots=True)
class Run:
    """Every case for one (workload, model, device)."""

    workload: str
    model_id: str
    device: str
    outcomes: tuple[Outcome, ...]
    load_seconds: float
    layers_offloaded: int
    context_tokens: int
    cache: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def tally(self) -> Tally:
        return Tally(tuple(outcome.judgement for outcome in self.outcomes))

    @property
    def seconds_per_case(self) -> float:
        timings = [outcome.timing.seconds for outcome in self.outcomes]
        return sum(timings) / len(timings) if timings else 0.0

    @property
    def characters_per_second(self) -> float:
        seconds = sum(outcome.timing.seconds for outcome in self.outcomes)
        characters = sum(outcome.timing.output_characters for outcome in self.outcomes)
        return characters / seconds if seconds > 0 else 0.0

    @property
    def errors(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.error)


def fragments(case: Case, request_id: str) -> tuple[SourceFragment, ...]:
    """A case's sources as the fragments a workload would have published."""
    import hashlib

    built = []
    for index, text in enumerate(case.sources):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        built.append(
            SourceFragment(
                f"case:{request_id}:{index}",
                digest,
                1,
                text,
                digest,
            )
        )
    return tuple(built)


async def run_case(
    runtime,
    task,
    case: Case,
    store: MemoryFragmentStore,
    *,
    progress,
    cancellation,
) -> Outcome:
    """One case, timed, judged, and never allowed to take the run down."""
    request_id = f"bench-{case.id}"
    published = fragments(case, request_id)
    await store.publish(request_id, published)
    request = GenerationRequest(
        request_id=request_id,
        task_id=task.task_id,
        content_references=tuple(fragment.reference for fragment in published),
    )
    started = time.monotonic()
    document: dict | None = None
    error = ""
    raw = ""
    try:
        raw = await runtime.generate(task, request, cancellation, progress)
        document = json.loads(raw)
    except json.JSONDecodeError:
        error = "output was not JSON"
    except Exception as failure:  # a failed case is data, not a stopped run
        error = f"{type(failure).__name__}: {failure}"[:200]
    finally:
        elapsed = time.monotonic() - started
        await store.discard(request_id)
    return Outcome(
        case_id=case.id,
        judgement=judge(case, document, [fragment.reference for fragment in published]),
        timing=Timing(elapsed, len(raw)),
        document=document,
        error=error,
    )


async def run_cases(
    runtime,
    task,
    cases: Sequence[Case],
    store: MemoryFragmentStore,
    *,
    progress,
    cancellation,
    on_case: Callable[[Outcome], None] | None = None,
) -> tuple[Outcome, ...]:
    outcomes = []
    for case in cases:
        outcome = await run_case(
            runtime, task, case, store, progress=progress, cancellation=cancellation
        )
        outcomes.append(outcome)
        if on_case is not None:
            on_case(outcome)
    return tuple(outcomes)


class NoCancellation:
    """Nothing cancels a benchmark but the person who started it."""

    def raise_if_cancelled(self) -> None:
        return None

    @property
    def cancelled(self) -> bool:
        return False


class SilentProgress:
    """The runtime reports progress; a benchmark has its own log."""

    async def report(self, _event) -> None:
        return None


def lease_file(directory: Path) -> Path:
    """The cross-worker GPU lease the runtime insists on holding.

    It exists to stop two workers loading a model onto one card at once, which
    is exactly as true for a benchmark as for a job — so this takes a real one
    rather than working around it.
    """
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / "accelerator.lease"
    if not path.is_file():
        path.write_bytes(b"")
    return path


def asyncio_run(coroutine):
    """One place to change if this ever needs its own loop policy."""
    return asyncio.run(coroutine)


__all__ = [
    "NoCancellation",
    "Outcome",
    "Run",
    "SilentProgress",
    "Timing",
    "asyncio_run",
    "fragments",
    "lease_file",
    "run_case",
    "run_cases",
]
