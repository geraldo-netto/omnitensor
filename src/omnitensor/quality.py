"""Repository quality gates shared by local development and CI."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Collection
from pathlib import Path

DEFAULT_THRESHOLD = 80.0
DEFAULT_SOURCE_ROOT = Path("src/omnitensor")


def validate_threshold(threshold: object) -> float:
    """Return a finite percentage or reject a gate that could fail open."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("quality threshold must be a finite percentage")
    if not math.isfinite(threshold) or not 0 <= threshold <= 100:
        raise ValueError("quality threshold must be between 0 and 100")
    return float(threshold)


def source_files(root: Path | str) -> tuple[Path, ...]:
    """Return every Python source file the report must contain."""
    resolved = Path(root)
    if not resolved.is_dir():
        raise ValueError(f"coverage source root is not a directory: {resolved}")
    files = tuple(sorted(path.resolve() for path in resolved.rglob("*.py")))
    if not files:
        raise ValueError(f"coverage source root contains no Python files: {resolved}")
    return files


def _file_reports(document: object) -> dict:
    if not isinstance(document, dict) or not isinstance(document.get("files"), dict):
        raise ValueError("coverage report has no files object")
    reports = document["files"]
    if not reports:
        raise ValueError("coverage report contains no measured files")
    return reports


def _function_reports(filename: str, report: object) -> dict:
    functions = report.get("functions") if isinstance(report, dict) else None
    if not isinstance(functions, dict):
        raise ValueError(f"coverage report has no functions object for {filename}")
    return functions


def _function_percentage(filename: str, name: str, report: object) -> float:
    summary = report.get("summary") if isinstance(report, dict) else None
    percent = summary.get("percent_statements_covered") if isinstance(summary, dict) else None
    try:
        return validate_threshold(percent)
    except ValueError:
        raise ValueError(
            f"coverage report has no valid percentage for {filename}:{name}"
        ) from None


def coverage_failures(
    document: object,
    threshold: float,
    expected_files: Collection[Path | str],
) -> tuple[str, ...]:
    """Name every measured callable below ``threshold`` or reject an incomplete report."""
    threshold = validate_threshold(threshold)
    file_reports = _file_reports(document)
    expected = {Path(filename).resolve() for filename in expected_files}
    if not expected:
        raise ValueError("coverage source inventory is empty")
    reported = {Path(filename).resolve() for filename in file_reports}
    missing = sorted(expected - reported)
    if missing:
        raise ValueError(f"coverage report omits source file: {missing[0]}")

    failures = []
    measured = False
    for filename, file_report in sorted(file_reports.items()):
        functions = _function_reports(filename, file_report)
        for name, function_report in sorted(functions.items()):
            if not name:
                continue
            measured = True
            percent = _function_percentage(filename, name, function_report)
            if percent < threshold:
                line = function_report.get("start_line", "?")
                failures.append(f"{filename}:{line} {name} {percent:.1f}%")
    if not measured:
        raise ValueError("coverage report contains no measured functions")
    return tuple(failures)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="coverage.py JSON report")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    arguments = parser.parse_args(argv)
    try:
        document = json.loads(arguments.report.read_text(encoding="utf-8"))
        expected = source_files(arguments.source_root)
        failures = coverage_failures(document, arguments.threshold, expected)
    except (OSError, ValueError) as error:
        print(f"coverage gate failed: {error}")
        return 2
    if failures:
        print(f"Per-function coverage below {arguments.threshold:g}%:")
        print("\n".join(failures))
        return 1
    checked = sum(
        bool(name)
        for file_report in document["files"].values()
        for name in file_report["functions"]
    )
    print(f"per-function coverage: {checked} functions at or above {arguments.threshold:g}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
