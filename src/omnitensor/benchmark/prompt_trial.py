"""Try a different wording of a task's prompt against the labelled cases.

A workload that answers badly is usually blamed on its model, and the blame is
usually wrong: `event-extraction` refuses every answerable document on two
model generations and both cards, which is a property of what it asks for
rather than of who is asked. Fixing that means changing the prompt, and
changing a prompt on a hunch is how a workload arrives at the wording it has.

So this runs one variant against the same labelled cases the benchmark uses
and reports what moved. It writes nothing, installs nothing, and cannot change
the shipped task: the variant lives in a JSON file passed on the command line,
and the qualified task is read only to be copied.

    python -m omnitensor.benchmark.prompt_trial \\
        --workload event-extraction --variant trials/event-positive.json

The variant file may override `system`, `instructionTemplate`, and
`outputTokens`. Anything it omits keeps what the shipped task says.

One thing to know before believing a good result: the task digest binds a
qualification receipt, so a prompt that wins here is not deployable on its own.
It needs a new receipt, or the worker refuses to start — which is exactly the
failure OMNI-0352 was.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from ..plugins.generation import GenerationTask
from . import cases as case_files
from .bench_cli import DEFAULT_ARTIFACT_ROOT, DEFAULT_LEASE_ROOT, model_path, tasks
from .harness import (
    NoCancellation,
    Run,
    SilentProgress,
    asyncio_run,
    case_line,
    loaded_model,
    run_cases,
)
from .vulkan_devices import DeviceError
from .vulkan_devices import devices as vulkan_devices
from .vulkan_devices import select as select_device

# A variant file is a wording, not a payload.
MAX_VARIANT_BYTES = 256 * 1024


class VariantError(ValueError):
    """A variant file that cannot be read as a variant."""


def load_variant(path: Path) -> dict:
    """Read a variant, refusing anything that is not one."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise VariantError(f"no variant file: {path}") from error
    if len(raw) > MAX_VARIANT_BYTES:
        raise VariantError("variant file is oversized")
    try:
        document = json.loads(raw)
    except (UnicodeError, ValueError) as error:
        raise VariantError("variant file is not JSON") from error
    if not isinstance(document, dict):
        raise VariantError("variant file is not an object")
    unknown = set(document) - {"name", "note", "system", "instructionTemplate", "outputTokens"}
    if unknown:
        raise VariantError(f"variant names fields this tool does not apply: {sorted(unknown)}")
    template = document.get("instructionTemplate")
    if template is not None and "{{UNTRUSTED_CONTENT}}" not in template:
        # Without the placeholder the model is asked to work on nothing, and
        # every case would fail for a reason that has nothing to do with the
        # wording being tried.
        raise VariantError("instructionTemplate must keep the {{UNTRUSTED_CONTENT}} placeholder")
    return document


def varied(task: GenerationTask, variant: dict) -> GenerationTask:
    """The shipped task with only the wording the variant states replaced."""
    limits = task.limits
    tokens = variant.get("outputTokens")
    if tokens is not None:
        limits = replace(limits, output_tokens=int(tokens))
    return replace(
        task,
        system_prompt=variant.get("system", task.system_prompt),
        instruction_template=variant.get("instructionTemplate", task.instruction_template),
        limits=limits,
    )


def differences(before: GenerationTask, after: GenerationTask) -> tuple[str, ...]:
    """What the variant actually changed, so a null result is recognisable."""
    changed = []
    if before.system_prompt != after.system_prompt:
        changed.append("system")
    if before.instruction_template != after.instruction_template:
        changed.append("instructionTemplate")
    if before.limits.output_tokens != after.limits.output_tokens:
        changed.append(
            f"outputTokens {before.limits.output_tokens} to {after.limits.output_tokens}"
        )
    return tuple(changed)


def trial(
    workload: str,
    task: GenerationTask,
    *,
    model_id: str,
    device,
    artifact_root: Path,
    case_root: Path,
    lease_root: Path,
    answerable_only: bool,
    say,
) -> Run:
    """Load the model once and run every selected case on one wording."""
    with loaded_model(
        model_path(artifact_root, model_id), device, lease_root=lease_root, say=say
    ) as loaded:
        cases = case_files.load(workload, case_root)
        if answerable_only:
            # A wording that refuses everything scores well on cases that expect
            # a refusal, which is how a broken task can look half right.
            cases = tuple(case for case in cases if case.answerable)
        say(f"{len(cases)} cases")
        outcomes = asyncio_run(
            run_cases(
                loaded.runtime,
                task,
                cases,
                loaded.store,
                progress=SilentProgress(),
                cancellation=NoCancellation(),
                on_case=lambda outcome: say(case_line(outcome, width=32)),
            )
        )
        return loaded.run_of(workload, model_id, outcomes)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omnitensor-prompt-trial",
        description="Run one prompt wording against the labelled cases. Changes nothing.",
    )
    parser.add_argument("--workload", required=True)
    parser.add_argument(
        "--variant", type=Path, help="JSON wording to try; omit to run the shipped task"
    )
    parser.add_argument("--model", default="qwen3-4b-q4-k-m")
    parser.add_argument("--device", default="0")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument(
        "--lease-root",
        type=Path,
        default=Path(DEFAULT_LEASE_ROOT).expanduser(),
        help="where the per-card GPU lease is taken",
    )
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="include cases that expect a refusal; by default only answerable ones run",
    )
    arguments = parser.parse_args(argv)

    def say(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    factory = tasks().get(arguments.workload)
    if factory is None:
        say(f"no provider task for {arguments.workload}")
        return 1
    shipped = factory()
    task = shipped
    if arguments.variant is not None:
        try:
            variant = load_variant(arguments.variant)
        except VariantError as refusal:
            say(str(refusal))
            return 1
        task = varied(shipped, variant)
        changed = differences(shipped, task)
        if not changed:
            say("this variant changes nothing about the shipped task")
            return 1
        say(f"variant {variant.get('name', arguments.variant.stem)}: changed {', '.join(changed)}")
    else:
        say("no variant: measuring the shipped task as it is")

    try:
        device = select_device(arguments.device, vulkan_devices())
    except DeviceError as refusal:
        say(str(refusal))
        return 1

    run = trial(
        arguments.workload,
        task,
        model_id=arguments.model,
        device=device,
        artifact_root=arguments.artifacts,
        case_root=arguments.cases or case_files.case_root(),
        lease_root=arguments.lease_root,
        answerable_only=not arguments.all_cases,
        say=say,
    )
    tally = run.tally
    say(
        f"{tally.correct}/{tally.total} correct ({tally.accuracy:.0%}), "
        f"rules {tally.rule_score:.0%}, {run.seconds_per_case:.1f}s per case"
    )
    if tally.failures():
        say(f"what failed: {json.dumps(tally.failures())}")
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())
