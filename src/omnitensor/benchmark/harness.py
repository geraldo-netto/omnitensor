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
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from ..plugins.event_workload import MemoryFragmentStore
from ..plugins.fragments import SourceFragment
from ..plugins.generation import GenerationRequest
from .cases import Case
from .scoring import Judgement, Tally, judge
from .vulkan_devices import confirm

# The KV cache the provider itself uses; recorded in every Run so a receipt
# says what was measured rather than what was assumed.
BENCHMARK_CACHE_TYPE = "q8_0"


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
        error = f"{type(failure).__name__}: {failure}"
    finally:
        elapsed = time.monotonic() - started
        await store.discard(request_id)
    return Outcome(
        case_id=case.id,
        judgement=judge(
            case,
            document,
            [fragment.reference for fragment in published],
            [fragment.text for fragment in published],
        ),
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


def lease_file(directory: Path, device_index: int = 0) -> Path:
    """The cross-worker GPU lease the runtime insists on holding.

    It exists to stop two workers loading a model onto one card at once, which
    is exactly as true for a benchmark as for a job — so this takes a real one
    rather than working around it.

    One lease per card, because "one card" is what it protects. A single shared
    lease also serialised the discrete and integrated lanes against each other,
    which was never the point: they are different memory and different queues,
    and measuring both takes twice as long for no reason.
    """
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / f"accelerator-{device_index}.lease"
    if not path.is_file():
        path.write_bytes(b"")
    return path


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """One model on one card, with what its load reported."""

    runtime: object
    store: MemoryFragmentStore
    load_seconds: float
    layers_offloaded: int
    total_layers: int
    device_name: str
    context_tokens: int

    def run_of(self, workload: str, model_id: str, outcomes) -> Run:
        """The Run these outcomes make, with the load's own figures attached."""
        return Run(
            workload=workload,
            model_id=model_id,
            device=self.device_name,
            outcomes=tuple(outcomes),
            load_seconds=self.load_seconds,
            layers_offloaded=self.layers_offloaded,
            context_tokens=self.context_tokens,
            cache=BENCHMARK_CACHE_TYPE,
        )


@contextmanager
def loaded_model(
    model_path: Path,
    device,
    *,
    lease_root: Path | None = None,
    say: Callable[[str], None] = lambda _line: None,
) -> Iterator[LoadedModel]:
    """Load one model on one card, and give the card back however this ends.

    Both benchmark entry points need exactly this - the visible-device
    variable set before the runtime initialises Vulkan, the lease taken per
    card, the load timed, the physical device confirmed against the one asked
    for, and `terminate` in a `finally` so a raising caller does not leave a
    model holding the VRAM. It was written twice, and the copies had already
    begun to differ.
    """
    import os  # noqa: PLC0415 - set before the runtime initialises Vulkan

    from omnitensor_vulkan_runtime.runtime import (  # noqa: PLC0415 - provider is optional
        MAX_RUNTIME_CONTEXT_TOKENS,
        LlamaVulkanRuntime,
    )

    # Vulkan device selection, done the way llama.cpp offers it: the provider
    # always takes device 0, so the choice is which device *is* 0.
    os.environ["GGML_VK_VISIBLE_DEVICES"] = str(device.index)
    store = MemoryFragmentStore()
    runtime = LlamaVulkanRuntime(
        store, lease_file(lease_root or Path.home() / ".cache/omnitensor-bench", device.index)
    )
    try:
        say(f"loading {model_path.name} on {device.name}")
        started = time.monotonic()
        report = asyncio_run(runtime.load((model_path,), "gpu"))
        load_seconds = time.monotonic() - started
        say(
            f"loaded in {load_seconds:.1f}s — {report.accelerator_layers}/"
            f"{report.total_model_layers} layers on {report.device}"
        )
        # The measurement is worthless if it describes a card nobody asked for,
        # and that has happened: refuse rather than write it down.
        confirm(device, runtime.physical_device)
        yield LoadedModel(
            runtime,
            store,
            load_seconds,
            report.accelerator_layers,
            report.total_model_layers,
            device.name,
            MAX_RUNTIME_CONTEXT_TOKENS,
        )
    finally:
        # The lease and the VRAM belong to the runtime, not to this frame.
        asyncio_run(runtime.terminate("__startup__"))


def case_line(outcome, width: int = 34) -> str:
    """The one-line verdict both benchmarks print per case."""
    verdict = "ok  " if outcome.judgement.correct else "FAIL"
    detail = f"  {outcome.error}" if outcome.error else ""
    failures = (
        "" if outcome.judgement.correct else "  failed: " + ", ".join(outcome.judgement.failed)
    )
    return f"  {outcome.case_id:{width}} {verdict} {outcome.timing.seconds:6.1f}s{detail}{failures}"


def asyncio_run(coroutine):
    """One place to change if this ever needs its own loop policy."""
    return asyncio.run(coroutine)


__all__ = [
    "BENCHMARK_CACHE_TYPE",
    "LoadedModel",
    "NoCancellation",
    "Outcome",
    "Run",
    "SilentProgress",
    "Timing",
    "asyncio_run",
    "case_line",
    "fragments",
    "loaded_model",
    "lease_file",
    "run_case",
    "run_cases",
]
