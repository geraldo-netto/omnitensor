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
            results[match.group(1)] = match.group(2)
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
    if not selected or any(not selector or "*" in selector for selector in selected):
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
    parser.add_argument("--selector", action="append", required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    arguments = parser.parse_args(argv)
    try:
        failures = mutation_failures(
            arguments.report.read_text(encoding="utf-8"),
            arguments.selector,
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
        f"per-callable mutation score: {len(arguments.selector)} callables "
        f"at or above {arguments.threshold:g}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
