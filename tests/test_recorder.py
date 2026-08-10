from __future__ import annotations

import json

import pytest

from omnitensor.plugins.recorder import (
    MAX_FEATURES,
    FeatureRow,
    RecorderError,
    TelemetryRecorder,
    feature_matrix,
    validated_features,
)


def recorder(tmp_path, **bounds):
    return TelemetryRecorder(tmp_path / "telemetry", **bounds)


def test_an_observation_is_recorded_and_read_back(tmp_path):
    subject = recorder(tmp_path)

    subject.record("resource-scheduler", {"cpu": 0.4, "queue": 2}, 1_000)

    [row] = subject.rows("resource-scheduler")
    assert row.features == {"cpu": 0.4, "queue": 2.0}
    assert row.observed_at_ms == 1_000
    assert row.profile_id == "resource-scheduler"


def test_history_accumulates_in_order(tmp_path):
    subject = recorder(tmp_path)
    for index in range(5):
        subject.record("resource-scheduler", {"cpu": index / 10}, 1_000 + index)

    rows = subject.rows("resource-scheduler")

    assert [row.observed_at_ms for row in rows] == [1_000, 1_001, 1_002, 1_003, 1_004]


def test_profiles_do_not_share_a_history(tmp_path):
    subject = recorder(tmp_path)
    subject.record("resource-scheduler", {"cpu": 0.1})
    subject.record("hardware-health", {"temp": 42.0})

    assert len(subject.rows("resource-scheduler")) == 1
    assert len(subject.rows("hardware-health")) == 1
    assert subject.rows("absent-profile") == ()


@pytest.mark.parametrize(
    "features",
    [
        {"path": "/home/u/secret.txt"},
        {"title": "Quarterly results"},
        {"cpu": "high"},
        {"cpu": None},
        {"cpu": True},
        {"cpu": float("nan")},
        {"cpu": float("inf")},
        {},
        [],
        {"": 1.0},
        {"x" * 100: 1.0},
        {f"f{index}": 1.0 for index in range(MAX_FEATURES + 1)},
    ],
)
def test_a_row_that_is_not_numbers_is_refused(features):
    """A corpus is the longest-lived copy of anything that enters it."""
    with pytest.raises(RecorderError, match="features-invalid"):
        validated_features(features)


def test_recording_a_non_numeric_row_is_refused(tmp_path):
    with pytest.raises(RecorderError, match="features-invalid"):
        recorder(tmp_path).record("resource-scheduler", {"path": "/home/u/a.txt"})


def test_growth_is_bounded_by_deleting_the_oldest(tmp_path):
    """Telemetry that grows forever fills the disk it was meant to look after."""
    subject = recorder(tmp_path, max_segment_bytes=200, max_segments=2)

    for index in range(200):
        subject.record("resource-scheduler", {"cpu": index / 1000}, 1_000 + index)

    assert subject.segment_count("resource-scheduler") == 2
    rows = subject.rows("resource-scheduler")
    assert rows
    # The history stayed current rather than freezing at what was recorded first.
    assert rows[-1].observed_at_ms == 1_199


def test_a_torn_final_line_is_skipped_not_repaired(tmp_path):
    """Appends are not atomic against power loss; the tail can be a fragment."""
    subject = recorder(tmp_path)
    subject.record("resource-scheduler", {"cpu": 0.5}, 1_000)
    segment = next((tmp_path / "telemetry" / "resource-scheduler").glob("segment-*.jsonl"))
    with segment.open("a") as stream:
        stream.write('{"v":1,"p":"resource-sched')

    rows = subject.rows("resource-scheduler")

    assert len(rows) == 1
    assert rows[0].features == {"cpu": 0.5}


@pytest.mark.parametrize(
    "line",
    [
        "",
        "   ",
        "not json",
        json.dumps({"v": 2, "p": "x", "t": 1, "f": {"a": 1}}),
        json.dumps({"v": 1, "p": "", "t": 1, "f": {"a": 1}}),
        json.dumps({"v": 1, "p": "x", "t": -1, "f": {"a": 1}}),
        json.dumps({"v": 1, "p": "x", "t": True, "f": {"a": 1}}),
        json.dumps({"v": 1, "p": "x", "t": 1, "f": {"a": "text"}}),
        json.dumps(["not", "an", "object"]),
    ],
)
def test_an_unusable_recorded_line_is_skipped(tmp_path, line):
    subject = recorder(tmp_path)
    subject.record("resource-scheduler", {"cpu": 0.5}, 1_000)
    segment = next((tmp_path / "telemetry" / "resource-scheduler").glob("segment-*.jsonl"))
    with segment.open("a") as stream:
        stream.write(line + "\n")

    assert len(subject.rows("resource-scheduler")) == 1


def test_windows_pair_history_with_a_later_observation(tmp_path):
    """The label is the future value, so nothing has to be annotated by hand."""
    subject = recorder(tmp_path)
    for index in range(6):
        subject.record("resource-scheduler", {"cpu": index / 10}, 1_000 + index)

    windows = subject.windows("resource-scheduler", 3)

    assert len(windows) == 3
    history, target = windows[0]
    assert [row.features["cpu"] for row in history] == [0.0, 0.1, 0.2]
    assert target.features["cpu"] == pytest.approx(0.3)


def test_a_longer_horizon_predicts_further_ahead(tmp_path):
    subject = recorder(tmp_path)
    for index in range(8):
        subject.record("resource-scheduler", {"cpu": index / 10}, 1_000 + index)

    windows = subject.windows("resource-scheduler", 2, horizon=3)

    _history, target = windows[0]
    assert target.features["cpu"] == pytest.approx(0.4)


def test_too_little_history_yields_no_windows(tmp_path):
    subject = recorder(tmp_path)
    subject.record("resource-scheduler", {"cpu": 0.1})
    assert subject.windows("resource-scheduler", 5) == ()


@pytest.mark.parametrize("bounds", [{"size": 0}, {"size": True}, {"horizon": 0}])
def test_window_bounds_are_validated(tmp_path, bounds):
    subject = recorder(tmp_path)
    with pytest.raises(RecorderError, match="bounds-invalid"):
        subject.windows("resource-scheduler", **{"size": 2, **bounds})


@pytest.mark.parametrize("bounds", [{"max_segment_bytes": 0}, {"max_segments": 0}])
def test_recorder_bounds_are_validated(tmp_path, bounds):
    with pytest.raises(RecorderError, match="bounds-invalid"):
        recorder(tmp_path, **bounds)


@pytest.mark.parametrize("profile", ["", "x" * 100, 7, "///"])
def test_a_profile_id_that_cannot_name_a_directory_is_refused(tmp_path, profile):
    with pytest.raises(RecorderError, match="profile-invalid"):
        recorder(tmp_path).record(profile, {"cpu": 0.1})


@pytest.mark.parametrize("timestamp", [-1, True, "now"])
def test_an_impossible_timestamp_is_refused(tmp_path, timestamp):
    with pytest.raises(RecorderError, match="timestamp-invalid"):
        recorder(tmp_path).record("resource-scheduler", {"cpu": 0.1}, timestamp)


def test_an_absent_timestamp_is_stamped_from_the_clock(tmp_path):
    subject = TelemetryRecorder(tmp_path / "t", clock_ms=lambda: 4242)
    assert subject.record("resource-scheduler", {"cpu": 0.1}).observed_at_ms == 4242


def test_a_matrix_uses_the_declared_feature_order(tmp_path):
    """Dictionary order is not a contract; a reordered column is nonsense."""
    subject = recorder(tmp_path)
    for index in range(4):
        subject.record("resource-scheduler", {"cpu": index / 10, "io": index}, 1_000 + index)

    inputs, targets = feature_matrix(subject.windows("resource-scheduler", 2), ("cpu", "io"))

    assert inputs[0] == (0.0, 0.0, 0.1, 1.0)
    assert targets[0] == pytest.approx(0.2)


def test_a_missing_feature_reads_as_zero_not_as_an_error(tmp_path):
    rows = (
        FeatureRow("p", 1, {"cpu": 0.5}),
        FeatureRow("p", 2, {"cpu": 0.6, "io": 3.0}),
    )
    inputs, _targets = feature_matrix([(rows, rows[1])], ("cpu", "io"))
    assert inputs[0] == (0.5, 0.0, 0.6, 3.0)


def test_a_matrix_requires_an_explicit_feature_order():
    with pytest.raises(RecorderError, match="features-invalid"):
        feature_matrix([], ())


def test_the_recorder_reports_where_it_writes(tmp_path):
    assert recorder(tmp_path).root == tmp_path / "telemetry"


def test_an_unreadable_segment_is_skipped_rather_than_raising(tmp_path):
    subject = recorder(tmp_path)
    subject.record("resource-scheduler", {"cpu": 0.5})
    segment = next((tmp_path / "telemetry" / "resource-scheduler").glob("segment-*.jsonl"))
    segment.write_bytes(b"\xff\xfe not utf-8")

    assert subject.rows("resource-scheduler") == ()
