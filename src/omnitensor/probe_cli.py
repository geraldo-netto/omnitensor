"""What this machine can run, printed as a table.

Read-only by construction: it opens model headers, reads two files per card in
sysfs, and prints. It never loads a model, never touches policy, and never
changes which model a workload runs — the point is to learn the tradeoff before
anybody decides anything, not to decide it.

    python -m omnitensor.probe_cli
    python -m omnitensor.probe_cli --context 32768 --cache q8_0

Each row is one (model, card, context, cache) and says whether the model fits
there, how it splits between weights and cache, and what would be left over.
The cache column is why the answers surprise people: at 32,768 tokens it is
larger than several of these models.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

from . import paths
from .fit import (
    CACHE_BYTES,
    DEFAULT_OVERHEAD_BYTES,
    DeviceMemory,
    Verdict,
    device_memory,
    estimate,
    gibibytes,
    verdict,
)
from .gguf import GgufError, ModelShape, read_shape

DEFAULT_ARTIFACT_ROOT = Path(paths.ARTIFACT_ROOT).expanduser()
# The context the four text workloads declare in their task limits. Their own
# number, not a preference: it is what lets sixteen documents be read at once.
DEFAULT_CONTEXT = 32_768


def artifacts(root: Path) -> tuple[tuple[str, ModelShape], ...]:
    """Every readable GGUF under ``root``, by artifact directory name.

    A file that is not a language model — a vision projector, a stray export —
    is skipped rather than reported as a failure: it was never a candidate.
    """
    found: list[tuple[str, ModelShape]] = []
    for path in sorted(root.glob("*/**/*.gguf")):
        try:
            shape = read_shape(path)
        except (GgufError, OSError):
            continue
        found.append((_artifact_id(path, root), shape))
    return tuple(found)


def _artifact_id(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).parts[0]
    except ValueError:
        return path.stem


def rows(
    models: Iterable[tuple[str, ModelShape]],
    devices: Sequence[DeviceMemory],
    *,
    context_tokens: int,
    cache: str,
    overhead_bytes: int = DEFAULT_OVERHEAD_BYTES,
) -> tuple[tuple[str, ...], ...]:
    """One row per model and card, widest card first."""
    built = []
    for artifact_id, shape in models:
        estimated = estimate(shape, context_tokens, cache=cache, overhead_bytes=overhead_bytes)
        for device in devices:
            answer = verdict(estimated, device)
            built.append(
                (
                    artifact_id,
                    device.device_id,
                    gibibytes(estimated.weight_bytes),
                    gibibytes(estimated.cache_bytes),
                    gibibytes(estimated.total_bytes),
                    _free_cell(device),
                    _fits_cell(answer),
                    _headroom_cell(answer),
                )
            )
    return tuple(built)


def _free_cell(device: DeviceMemory) -> str:
    if not device.capacity_known:
        return "unknown"
    return gibibytes(device.usable_free_bytes) + (" (mapped)" if device.integrated else "")


def _fits_cell(answer: Verdict) -> str:
    if answer.fits is None:
        return "unknown"
    return "yes" if answer.fits else "no"


def _headroom_cell(answer: Verdict) -> str:
    """What is left once the safety margin stays unclaimed, or how short it is."""
    if answer.headroom_bytes is None:
        return "card does not report its memory"
    if answer.fits:
        return gibibytes(answer.headroom_bytes)
    return f"short {gibibytes(answer.short_by_bytes)}"


HEADINGS = ("artifact", "card", "weights", "cache", "total", "free", "fits", "headroom")


def table(built: Sequence[Sequence[str]], headings: Sequence[str] = HEADINGS) -> str:
    widths = [
        max(len(str(row[column])) for row in (headings, *built)) for column in range(len(headings))
    ]
    lines = [_line(headings, widths), _line(["-" * width for width in widths], widths)]
    lines.extend(_line(row, widths) for row in built)
    return "\n".join(lines)


def _line(row: Sequence[str], widths: Sequence[int]) -> str:
    return "  ".join(str(cell).ljust(width) for cell, width in zip(row, widths, strict=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omnitensor-probe",
        description="What fits on the cards this machine has. Reads only.",
    )
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--cache", default="q8_0", choices=sorted(CACHE_BYTES))
    parser.add_argument(
        "--overhead",
        type=int,
        default=DEFAULT_OVERHEAD_BYTES,
        help="bytes the runtime allocates beyond weights and cache (estimated)",
    )
    arguments = parser.parse_args(argv)

    devices = device_memory()
    if not devices:
        print("No render node reports its memory; nothing to measure against.")
        return 1
    models = artifacts(arguments.artifacts)
    if not models:
        print(f"No readable model under {arguments.artifacts}")
        return 1

    print(
        f"context {arguments.context} tokens · cache {arguments.cache} · "
        f"overhead {gibibytes(arguments.overhead)} (estimated)"
    )
    print()
    print(
        table(
            rows(
                models,
                devices,
                context_tokens=arguments.context,
                cache=arguments.cache,
                overhead_bytes=arguments.overhead,
            )
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
