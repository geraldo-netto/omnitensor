"""Turn one contract-valid runtime snapshot into bounded numeric history."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..atomicio import read_json_bounded
from ..registry import validate_document
from ..telemetry_recorder import TelemetryRecorder
from ..telemetry_types import FeatureRow, RecorderError

MAX_RUNTIME_SNAPSHOT_BYTES = 1024 * 1024
RUNTIME_SNAPSHOT_SELECTORS = (
    "queueDepth",
    "runningProfiles",
)


def load_runtime_snapshot(path: Path | str) -> dict:
    """Read one bounded snapshot and enforce the public runtime contract."""
    document = read_json_bounded(Path(path), MAX_RUNTIME_SNAPSHOT_BYTES)
    violations = validate_document("runtime-snapshot.schema.json", document)
    if violations:
        raise RecorderError("snapshot-invalid", "; ".join(violations))
    return document


def record_runtime_snapshot(
    document: object,
    *,
    profile_id: str,
    selectors: Sequence[str],
    recorder: TelemetryRecorder,
) -> FeatureRow | None:
    """Record selected aggregate numbers once per snapshot timestamp.

    ``None`` means this exact timestamp was already recorded. Older snapshots
    are refused because appending them would make the oldest-first training
    contract false.
    """
    violations = validate_document("runtime-snapshot.schema.json", document)
    if violations:
        raise RecorderError("snapshot-invalid", "; ".join(violations))
    if isinstance(selectors, (str, bytes)) or not selectors:
        raise RecorderError("selectors-invalid", "at least one selector is required")
    selected = tuple(selectors)
    if len(set(selected)) != len(selected):
        raise RecorderError("selectors-invalid", "selectors must be unique")
    unknown = [name for name in selected if name not in RUNTIME_SNAPSHOT_SELECTORS]
    if unknown:
        raise RecorderError("selectors-invalid", f"unsupported selector: {unknown[0]}")

    return recorder.record_unique(
        profile_id,
        _selector_values(document, selected),
        document["generatedAt"],
    )


def _selector_values(document: dict, selectors: Sequence[str]) -> dict[str, float]:
    values = {
        "queueDepth": document["metrics"]["queueDepth"],
        "runningProfiles": document["metrics"]["runningProfiles"],
    }
    return {name: float(values[name]) for name in selectors}
