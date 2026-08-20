"""Run the labelled cases against models, and write down what happened.

    python -m omnitensor.benchmark.bench_cli --models qwen3-8b-q4-k-m,qwen3-4b-q4-k-m

Grouped by model rather than by workload, because loading a model is the
expensive part: one load, every case, then the next model. Results are written
after each model finishes, so a run that is interrupted after two hours still
leaves two hours of findings behind.

It changes nothing. No policy is written, no receipt is touched, no default
moves — this exists to learn the tradeoff, and deciding on it is a separate
act with its own commit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from ..fit import device_memory, estimate, gibibytes, verdict
from ..gguf import GgufError, read_shape
from . import cases as case_files
from .harness import (
    NoCancellation,
    Run,
    SilentProgress,
    asyncio_run,
    lease_file,
    run_cases,
)
from .report import write_results
from .vulkan_devices import DeviceError, VulkanDevice, confirm
from .vulkan_devices import devices as vulkan_devices
from .vulkan_devices import select as select_device

DEFAULT_ARTIFACT_ROOT = Path.home() / ".local/share/omnitensor/artifacts"

DEFAULT_MODELS = ("qwen3-8b-q4-k-m", "qwen3-4b-q4-k-m")


def model_path(artifact_root: Path, model_id: str) -> Path:
    """The one GGUF inside an installed artifact."""
    found = sorted((artifact_root / model_id).glob("**/*.gguf"))
    readable = []
    for path in found:
        try:
            read_shape(path)
        except (GgufError, OSError):
            continue  # a projector or a sidecar, not the model
        readable.append(path)
    if not readable:
        raise FileNotFoundError(f"no readable model inside {artifact_root / model_id}")
    return readable[0]


def tasks() -> dict:
    """The task factories the provider itself runs, by workload id."""
    from omnitensor_vulkan_runtime import workload_tasks  # noqa: PLC0415 - provider is optional

    return workload_tasks()


def run_model(
    model_id: str,
    workloads: Sequence[str],
    *,
    artifact_root: Path,
    case_root: Path,
    device: VulkanDevice,
    say,
) -> tuple[Run, ...]:
    """Load one model once, then every case of every workload on it."""
    from omnitensor_vulkan_runtime.runtime import (  # noqa: PLC0415
        MAX_RUNTIME_CONTEXT_TOKENS,
        LlamaVulkanRuntime,
    )

    from omnitensor.plugins.event_workload import MemoryFragmentStore  # noqa: PLC0415

    path = model_path(artifact_root, model_id)
    # Vulkan device selection, done the way llama.cpp offers it: the provider
    # always takes device 0, so the choice is which device *is* 0. Set before
    # the runtime loads anything, and confirmed against the card that answers.
    os.environ["GGML_VK_VISIBLE_DEVICES"] = str(device.index)

    store = MemoryFragmentStore()
    runtime = LlamaVulkanRuntime(
        store, lease_file(Path.home() / ".cache/omnitensor-bench", device.index)
    )

    try:
        say(f"loading {model_id} from {path.name} on {device.name}")
        started = time.monotonic()
        report = asyncio_run(runtime.load((path,), "gpu"))
        load_seconds = time.monotonic() - started
        say(
            f"loaded in {load_seconds:.1f}s — {report.accelerator_layers}/"
            f"{report.total_model_layers} layers on {report.device}"
        )
        # The measurement is worthless if it describes a card nobody asked for,
        # and that has happened: refuse rather than write it down.
        confirm(device, runtime.physical_device)

        every_task = tasks()
        runs = []
        for workload in workloads:
            task_factory = every_task.get(workload)
            if task_factory is None:
                say(f"skipping {workload}: this provider does not run it")
                continue
            loaded = case_files.load(workload, case_root)
            say(f"{model_id} · {workload}: {len(loaded)} cases")
            outcomes = asyncio_run(
                run_cases(
                    runtime,
                    task_factory(),
                    loaded,
                    store,
                    progress=SilentProgress(),
                    cancellation=NoCancellation(),
                    on_case=lambda outcome: say(
                        f"  {outcome.case_id:34} "
                        f"{'ok  ' if outcome.judgement.correct else 'FAIL'} "
                        f"{outcome.timing.seconds:6.1f}s"
                        + (f"  {outcome.error}" if outcome.error else "")
                        + (
                            ""
                            if outcome.judgement.correct
                            else "  failed: " + ", ".join(outcome.judgement.failed)
                        )
                    ),
                )
            )
            run = Run(
                workload=workload,
                model_id=model_id,
                device=device.name,
                outcomes=outcomes,
                load_seconds=load_seconds,
                layers_offloaded=report.accelerator_layers,
                context_tokens=MAX_RUNTIME_CONTEXT_TOKENS,
                cache="q8_0",
            )
            tally = run.tally
            say(
                f"{model_id} · {workload}: {tally.correct}/{tally.total} correct "
                f"({tally.accuracy:.0%}), rules {tally.rule_score:.0%}, "
                f"{run.seconds_per_case:.1f}s per case"
            )
            if tally.failures():
                say(f"  what failed: {json.dumps(tally.failures())}")
            runs.append(run)

        return tuple(runs)
    finally:
        # The lease and the VRAM belong to the runtime, not to this call
        # frame. `measure()` treats one model failing as a result and goes
        # on to the next, so a `confirm()`, `case_files.load()` or
        # `run_cases()` that raises must still give the card back — or the
        # next `load()` runs beside a model that never let go.
        asyncio_run(runtime.terminate("__startup__"))


def fit_note(model_id: str, artifact_root: Path, device: VulkanDevice, say) -> bool:
    """Say whether this is even expected to fit, before spending the minutes.

    Reuses the prober rather than repeating it, and does not refuse on its own
    estimate: the estimate is an estimate, and a load that succeeds where this
    predicted otherwise is a finding worth having.
    """
    try:
        shape = read_shape(model_path(artifact_root, model_id))
    except (FileNotFoundError, GgufError, OSError) as error:
        say(f"cannot read {model_id}: {error}")
        return False
    # sysfs orders render nodes its own way, so the card is found by what it
    # is — integrated or not — rather than by reusing the Vulkan index.
    matching = [
        card
        for card in device_memory()
        if card.capacity_known and card.integrated == device.integrated
    ]
    if len(matching) != 1:
        say(f"cannot tell which render node is {device.name}; proceeding anyway")
        return True
    memory = matching[0]
    answer = verdict(estimate(shape, 32_768), memory)
    say(
        f"{model_id}: {gibibytes(answer.estimate.weight_bytes)} weights + "
        f"{gibibytes(answer.estimate.cache_bytes)} cache at 32,768 tokens against "
        f"{gibibytes(memory.usable_free_bytes)} free on {memory.device_id} — "
        + ("expected to fit" if answer.fits else f"short {gibibytes(answer.short_by_bytes)}")
    )
    return True


def measure(
    models: Sequence[str],
    workloads: Sequence[str],
    *,
    device: VulkanDevice,
    artifact_root: Path,
    case_root: Path,
    result_root: Path,
    say,
) -> list[Run]:
    """Every model in turn, writing results after each so nothing is lost."""
    every_run: list[Run] = []
    for model_id in models:
        if not fit_note(model_id, artifact_root, device, say):
            continue
        try:
            every_run.extend(
                run_model(
                    model_id,
                    workloads,
                    artifact_root=artifact_root,
                    case_root=case_root,
                    device=device,
                    say=say,
                )
            )
        except Exception as failure:  # one model failing is a result, not the end
            say(f"{model_id} could not be measured: {type(failure).__name__}: {failure}")
        written = write_results(every_run, result_root)
        say(f"results written to {written}")
    return every_run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omnitensor-bench",
        description="Accuracy and speed per (workload, model). Changes nothing.",
    )
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--workloads", default="")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument(
        "--device",
        default="",
        help=(
            "which card to measure, by name — 'RX 6600' or '610M'. An index still "
            "works and is checked against the list. Default: the first device."
        ),
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="print the Vulkan devices this machine offers, and their indices",
    )
    arguments = parser.parse_args(argv)

    def say(message: str) -> None:
        # Unbuffered on purpose: this runs for hours and somebody reads the
        # log while it is still going.
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

    available = vulkan_devices()
    if arguments.list_devices:
        for card in available:
            say(str(card))
        return 0 if available else 1
    try:
        device = select_device(arguments.device or "0", available)
    except DeviceError as refusal:
        say(str(refusal))
        return 1
    say(f"measuring on {device}")

    models = [name for name in arguments.models.split(",") if name.strip()]
    case_root = arguments.cases or case_files.case_root()
    # Resolved here rather than as an argparse default, so a machine without
    # the corpus is told which directory is missing instead of measuring
    # nothing and reporting success.
    requested = arguments.workloads or ",".join(case_files.available(case_root))
    workloads = [name for name in requested.split(",") if name.strip()]
    say(f"models: {', '.join(models)}")
    say(f"workloads: {', '.join(workloads)}")

    every_run = measure(
        models,
        workloads,
        device=device,
        artifact_root=arguments.artifacts,
        case_root=case_root,
        result_root=arguments.results or case_files.benchmark_root() / "results",
        say=say,
    )

    if not every_run:
        say("nothing was measured")
        return 1
    say("done")
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())
