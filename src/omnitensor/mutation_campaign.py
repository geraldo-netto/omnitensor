"""Run and gate one exact mutation-selector shard."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from .mutation_manifest import MutationShard, load_mutation_manifest
from .mutation_quality import mutation_failures
from .quality import DEFAULT_THRESHOLD

DEFAULT_MUTMUT_EXECUTABLE = str(Path(sys.executable).with_name("mutmut"))


def mutation_patterns(selectors: Sequence[str]) -> tuple[str, ...]:
    """Return exact mutmut globs without involving a shell."""
    if not selectors:
        raise ValueError("mutation shard has no selectors")
    return tuple(f"{selector}__mutmut_*" for selector in selectors)


def mutation_commands(
    shard: MutationShard,
    *,
    executable: str = DEFAULT_MUTMUT_EXECUTABLE,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Build the deterministic run and result commands for ``shard``."""
    return (
        (executable, "run", *mutation_patterns(shard.selectors)),
        (executable, "results", "--all", "true"),
    )


def execute_mutation_shard(
    shard: MutationShard,
    report_path: Path | str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    executable: str = DEFAULT_MUTMUT_EXECUTABLE,
    runner: Callable = subprocess.run,
) -> tuple[str, ...]:
    """Run one shard, persist its report, and return per-callable failures."""
    report = Path(report_path)
    report.unlink(missing_ok=True)
    run_command, results_command = mutation_commands(shard, executable=executable)
    completed = runner(run_command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"mutmut run exited {completed.returncode}")
    results = runner(
        results_command,
        check=False,
        capture_output=True,
        text=True,
    )
    if results.returncode != 0:
        raise RuntimeError(f"mutmut results exited {results.returncode}")
    report.write_text(results.stdout, encoding="utf-8")
    return mutation_failures(results.stdout, shard.selectors, threshold)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--shard")
    mode.add_argument("--list-shards", action="store_true")
    parser.add_argument("--source-root", type=Path, default=Path("src"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--mutmut-executable", default=DEFAULT_MUTMUT_EXECUTABLE)
    arguments = parser.parse_args(argv)
    try:
        manifest = load_mutation_manifest(
            arguments.manifest,
            source_root=arguments.source_root,
        )
        if arguments.list_shards:
            if arguments.report is not None:
                raise ValueError("--report is invalid with --list-shards")
            print(json.dumps(manifest.shard_names(), separators=(",", ":")))
            return 0
        if arguments.report is None:
            raise ValueError("--report is required with --shard")
        shard = manifest.shard(arguments.shard)
        failures = execute_mutation_shard(
            shard,
            arguments.report,
            threshold=arguments.threshold,
            executable=arguments.mutmut_executable,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"mutation campaign failed: {error}")
        return 2
    if failures:
        print(f"Per-callable mutation score below {arguments.threshold:g}%:")
        print("\n".join(failures))
        return 1
    print(
        f"mutation shard {shard.name}: {len(shard.selectors)} callables "
        f"at or above {arguments.threshold:g}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
