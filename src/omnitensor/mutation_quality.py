"""Fail-closed per-callable mutation-score gate for an exact mutmut scope."""

from __future__ import annotations

import argparse
import re
from collections.abc import Collection
from pathlib import Path

from .quality import DEFAULT_THRESHOLD, validate_threshold

_RESULT = re.compile(r"^\s*(\S+): (.+?)\s*$")
_DETECTED = frozenset({"killed", "timeout"})
_TESTED = frozenset({"killed", "survived", "timeout"})


def parse_results(text: str) -> dict[str, str]:
    """Parse mutmut's stable line-oriented result listing."""
    results = {}
    for line in text.splitlines():
        match = _RESULT.fullmatch(line)
        if match:
            name, status = match.groups()
            if name in results:
                raise ValueError(f"mutation report repeats result: {name}")
            results[name] = status
    if not results:
        raise ValueError("mutation report contains no results")
    return results


def mutation_failures(
    text: str,
    selectors: Collection[str],
    threshold: float,
) -> tuple[str, ...]:
    """Name exact callable selectors below ``threshold`` and reject stale results."""
    threshold = validate_threshold(threshold)
    if isinstance(selectors, (str, bytes)):
        raise ValueError("mutation selectors must be a collection")
    selected = tuple(selectors)
    if not selected or any(
        not selector or any(wildcard in selector for wildcard in "*?[")
        for selector in selected
    ):
        raise ValueError("mutation selectors must name exact callables without wildcards")
    if len(set(selected)) != len(selected):
        raise ValueError("mutation selectors must be unique")
    results = parse_results(text)
    failures = []
    for selector in selected:
        prefix = f"{selector}__mutmut_"
        statuses = [status for name, status in results.items() if name.startswith(prefix)]
        if not statuses:
            raise ValueError(f"mutation report omits callable: {selector}")
        untested = sorted(set(statuses) - _TESTED)
        if untested:
            raise ValueError(f"mutation report has untested {selector} result: {untested[0]}")
        detected = sum(status in _DETECTED for status in statuses)
        score = detected * 100 / len(statuses)
        if score < threshold:
            failures.append(f"{selector} {score:.1f}% ({detected}/{len(statuses)} detected)")
    return tuple(failures)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="output from mutmut results --all true")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--selector", action="append")
    selection.add_argument("--selector-file", type=Path)
    parser.add_argument("--shard")
    parser.add_argument("--source-root", type=Path, default=Path("src"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    arguments = parser.parse_args(argv)
    try:
        selectors = arguments.selector
        if arguments.selector_file is not None:
            if arguments.shard is None:
                raise ValueError("--shard is required with --selector-file")
            from .mutation_manifest import load_mutation_manifest  # noqa: PLC0415

            selectors = load_mutation_manifest(
                arguments.selector_file,
                source_root=arguments.source_root,
            ).shard(arguments.shard).selectors
        elif arguments.shard is not None:
            raise ValueError("--shard requires --selector-file")
        failures = mutation_failures(
            arguments.report.read_text(encoding="utf-8"),
            selectors,
            arguments.threshold,
        )
    except (OSError, ValueError) as error:
        print(f"mutation gate failed: {error}")
        return 2
    if failures:
        print(f"Per-callable mutation score below {arguments.threshold:g}%:")
        print("\n".join(failures))
        return 1
    print(
        f"per-callable mutation score: {len(selectors)} callables "
        f"at or above {arguments.threshold:g}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
