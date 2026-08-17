"""Durably record collector observations, so there is history to learn from.

Every collector in this service emits and nothing writes, which is why the
host has no history to train on: waiting does not accumulate anything.  This is
the recorder, and it is deliberately the dullest component here — rows of
numbers, appended, rotated, bounded.

Three decisions carry the weight.

*Only numbers are recorded.*  A feature row is a mapping of name to float.  It
cannot hold a path, a window title, or a document's text, because a training
corpus is the longest-lived copy of anything that enters it — it outlives the
files it describes and is read again every time a model is fitted.  A row
carrying a string is refused, not coerced.

*Growth is bounded by deletion, not by hope.*  Old segments are removed once
the budget is reached.  Telemetry that grows forever fills the disk of the
machine it was meant to look after, and a recorder that stops recording when
full silently ends the history instead.

*A partial line is skipped, not repaired.*  Appends are not atomic against a
power failure, so the last line of a segment can be a fragment.  Reading skips
it, because guessing at the missing half would feed invented numbers to a
model that cannot tell them from measurements.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

from omnitensor.telemetry_types import RECORD_VERSION, FeatureRow, RecorderError

from .storelock import store_lock

DEFAULT_MAX_SEGMENT_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_SEGMENTS = 8
MAX_FEATURES = 128
MAX_NAME_LENGTH = 64
MAX_PROFILE_LENGTH = 64


def validated_features(features: object) -> dict[str, float]:
    """Bound a feature mapping to finite numbers, refusing anything else."""
    if not isinstance(features, Mapping):
        raise RecorderError("features-invalid", "features must be a mapping")
    if not features:
        raise RecorderError("features-invalid", "a row must carry at least one feature")
    if len(features) > MAX_FEATURES:
        raise RecorderError("features-invalid", f"at most {MAX_FEATURES} features are allowed")
    cleaned: dict[str, float] = {}
    for name, value in features.items():
        if not isinstance(name, str) or not 1 <= len(name) <= MAX_NAME_LENGTH:
            raise RecorderError("features-invalid", "feature names must be bounded strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            # A string here would make the corpus the longest-lived copy of
            # whatever it described.
            raise RecorderError("features-invalid", f"feature {name} is not a number")
        if not math.isfinite(value):
            raise RecorderError("features-invalid", f"feature {name} is not finite")
        cleaned[name] = float(value)
    return cleaned


class TelemetryRecorder:
    """Append-only, rotating, bounded storage for feature rows."""

    def __init__(
        self,
        root: Path | str,
        *,
        max_segment_bytes: int = DEFAULT_MAX_SEGMENT_BYTES,
        max_segments: int = DEFAULT_MAX_SEGMENTS,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        for name, value in (
            ("max_segment_bytes", max_segment_bytes),
            ("max_segments", max_segments),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise RecorderError("bounds-invalid", f"{name} must be a positive integer")
        self._root = Path(root)
        self._max_segment_bytes = max_segment_bytes
        self._max_segments = max_segments
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))

    @property
    def root(self) -> Path:
        return self._root

    def record(
        self, profile_id: str, features: Mapping[str, float], observed_at_ms: int | None = None
    ) -> FeatureRow:
        """Append one observation, rotating and pruning as needed."""
        row = self._validated_row(profile_id, features, observed_at_ms)
        self._append_row(row)
        return row

    def record_unique(
        self, profile_id: str, features: Mapping[str, float], observed_at_ms: int
    ) -> FeatureRow | None:
        """Append once per retained profile timestamp across cooperating processes."""
        row = self._validated_row(profile_id, features, observed_at_ms)
        with store_lock(self._root, ".telemetry.lock"):
            existing = self.rows(profile_id)
            if any(item.observed_at_ms == row.observed_at_ms for item in existing):
                return None
            if existing and row.observed_at_ms < max(item.observed_at_ms for item in existing):
                raise RecorderError(
                    "timestamp-unordered", "timestamp is older than recorded history"
                )
            self._append_row(row)
        return row

    def _validated_row(
        self,
        profile_id: str,
        features: Mapping[str, float],
        observed_at_ms: int | None,
    ) -> FeatureRow:
        if not isinstance(profile_id, str) or not 1 <= len(profile_id) <= MAX_PROFILE_LENGTH:
            raise RecorderError("profile-invalid", "profile id must be a bounded string")
        return FeatureRow(
            profile_id,
            self._timestamp(observed_at_ms),
            validated_features(features),
        )

    def _append_row(self, row: FeatureRow) -> None:
        line = json.dumps(row.document(), separators=(",", ":")) + "\n"
        self._append(row.profile_id, line.encode("utf-8"))

    def rows(self, profile_id: str) -> tuple[FeatureRow, ...]:
        """Every recorded row for one profile, oldest first, fragments skipped."""
        collected: list[FeatureRow] = []
        for segment in self._segments(profile_id):
            collected.extend(self._read_segment(segment))
        return tuple(collected)

    def windows(
        self, profile_id: str, size: int, *, horizon: int = 1
    ) -> tuple[tuple[tuple[FeatureRow, ...], FeatureRow], ...]:
        """Sliding (history, future) pairs — the labels of a self-supervised fit.

        The label is a later observation, so nothing has to be annotated by
        hand: the future is the ground truth and it arrives on its own.
        """
        for name, value in (("size", size), ("horizon", horizon)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise RecorderError("bounds-invalid", f"{name} must be a positive integer")
        rows = self.rows(profile_id)
        pairs = []
        for start in range(len(rows) - size - horizon + 1):
            history = rows[start : start + size]
            target = rows[start + size + horizon - 1]
            pairs.append((history, target))
        return tuple(pairs)

    def segment_count(self, profile_id: str) -> int:
        return len(self._segments(profile_id))

    def _timestamp(self, observed_at_ms: int | None) -> int:
        if observed_at_ms is None:
            return self._clock_ms()
        if isinstance(observed_at_ms, bool) or not isinstance(observed_at_ms, int):
            raise RecorderError("timestamp-invalid", "observed_at_ms must be an integer")
        if observed_at_ms < 0:
            raise RecorderError("timestamp-invalid", "observed_at_ms must not be negative")
        return observed_at_ms

    def _directory(self, profile_id: str) -> Path:
        directory = self._root / _segment_name(profile_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _segments(self, profile_id: str) -> list[Path]:
        directory = self._root / _segment_name(profile_id)
        if not directory.is_dir():
            return []
        return sorted(directory.glob("segment-*.jsonl"))

    def _append(self, profile_id: str, payload: bytes) -> None:
        directory = self._directory(profile_id)
        segments = sorted(directory.glob("segment-*.jsonl"))
        current = segments[-1] if segments else directory / "segment-00000000.jsonl"
        if current.exists() and current.stat().st_size + len(payload) > self._max_segment_bytes:
            index = int(current.stem.split("-")[1]) + 1
            current = directory / f"segment-{index:08d}.jsonl"
            segments.append(current)
        with current.open("ab") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        self._prune(directory)

    def _prune(self, directory: Path) -> None:
        segments = sorted(directory.glob("segment-*.jsonl"))
        # Deleting the oldest keeps the history current rather than freezing it
        # at whatever was recorded first.
        for stale in segments[: max(0, len(segments) - self._max_segments)]:
            stale.unlink(missing_ok=True)

    def _read_segment(self, segment: Path) -> Iterator[FeatureRow]:
        try:
            text = segment.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeDecodeError):
            return
        for line in text.splitlines():
            row = _parse_row(line)
            if row is not None:
                yield row


def _segment_name(profile_id: str) -> str:
    """A directory name derived from a profile id, never the id itself.

    The id reaches this from a manifest, so it is not trusted to be a safe
    path component.
    """
    safe = "".join(
        character for character in profile_id if character.isalnum() or character in "-_"
    )
    if not safe:
        raise RecorderError("profile-invalid", "profile id has no usable characters")
    return safe


def _parse_row(line: str) -> FeatureRow | None:
    """One recorded row, or ``None`` for a fragment or a row we cannot trust."""
    if not line.strip():
        return None
    try:
        document = json.loads(line)
    except ValueError:
        # A torn final line from an interrupted append. Guessing at the missing
        # half would feed invented numbers to a model.
        return None
    if not isinstance(document, dict) or document.get("v") != RECORD_VERSION:
        return None
    profile = document.get("p")
    observed = document.get("t")
    if not isinstance(profile, str) or not profile:
        return None
    if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
        return None
    try:
        features = validated_features(document.get("f"))
    except RecorderError:
        return None
    return FeatureRow(profile, observed, features)


def feature_matrix(
    windows: Sequence[tuple[Sequence[FeatureRow], FeatureRow]], names: Sequence[str]
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    """Flatten windows into (inputs, targets) over an explicit feature order.

    The order is passed in rather than inferred from the data: a model trained
    on one column order and served with another produces confident nonsense,
    and dictionary order is not a contract.
    """
    if not names:
        raise RecorderError("features-invalid", "an explicit feature order is required")
    inputs: list[tuple[float, ...]] = []
    targets: list[float] = []
    for history, target in windows:
        row: list[float] = []
        for observation in history:
            row.extend(float(observation.features.get(name, 0.0)) for name in names)
        inputs.append(tuple(row))
        targets.append(float(target.features.get(names[0], 0.0)))
    return tuple(inputs), tuple(targets)


for _legacy_value in (TelemetryRecorder, validated_features, feature_matrix):
    _legacy_value.__module__ = "omnitensor.plugins.recorder"
