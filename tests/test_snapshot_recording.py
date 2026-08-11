from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins.recorder import RecorderError, TelemetryRecorder
from omnitensor.training.cli import snapshot_record_main
from omnitensor.training.snapshot_recording import (
    MAX_RUNTIME_SNAPSHOT_BYTES,
    _selector_values,
    load_runtime_snapshot,
    record_runtime_snapshot,
)


def runtime_snapshot(*, timestamp=1_000, queue_depth=3, running_profiles=2):
    return {
        "version": 1,
        "generatedAt": timestamp,
        "devices": [
            {
                "id": "gpu0",
                "backend": "gpu",
                "available": True,
                "name": "GPU",
                "kind": "dri",
                "load": 42.5,
            }
        ],
        "metrics": {
            "queueDepth": queue_depth,
            "runningProfiles": running_profiles,
        },
        "profiles": {},
        "alerts": [],
    }


def test_snapshot_recorder_extracts_only_selected_aggregate_numbers(tmp_path):
    recorder = TelemetryRecorder(tmp_path / "records")

    row = record_runtime_snapshot(
        runtime_snapshot(),
        profile_id="resource-scheduler",
        selectors=("runningProfiles", "queueDepth"),
        recorder=recorder,
    )

    assert row is not None
    assert row.observed_at_ms == 1_000
    assert tuple(row.features) == ("runningProfiles", "queueDepth")
    assert row.features == {"runningProfiles": 2.0, "queueDepth": 3.0}
    stored = recorder.rows("resource-scheduler")
    assert stored == (row,)


def test_snapshot_recorder_deduplicates_and_refuses_time_travel(tmp_path):
    recorder = TelemetryRecorder(tmp_path / "records")
    latest = runtime_snapshot(timestamp=2_000)
    selectors = ("queueDepth",)

    assert record_runtime_snapshot(
        latest,
        profile_id="resource-scheduler",
        selectors=selectors,
        recorder=recorder,
    ) is not None
    changed_duplicate = runtime_snapshot(timestamp=2_000, queue_depth=99)
    assert (
        record_runtime_snapshot(
            changed_duplicate,
            profile_id="resource-scheduler",
            selectors=selectors,
            recorder=recorder,
        )
        is None
    )

    with pytest.raises(RecorderError) as captured:
        record_runtime_snapshot(
            runtime_snapshot(timestamp=1_999),
            profile_id="resource-scheduler",
            selectors=selectors,
            recorder=recorder,
        )

    assert captured.value.code == "timestamp-unordered"
    assert captured.value.detail == "timestamp is older than recorded history"
    assert len(recorder.rows("resource-scheduler")) == 1


@pytest.mark.parametrize(
    ("selectors", "detail"),
    [
        ((), "at least one selector is required"),
        ("queueDepth", "at least one selector is required"),
        (("queueDepth", "queueDepth"), "selectors must be unique"),
        (("deviceId",), "unsupported selector: deviceId"),
        (("acceleratorLoad",), "unsupported selector: acceleratorLoad"),
    ],
)
def test_snapshot_recorder_refuses_unbounded_or_ambiguous_selectors(
    tmp_path, selectors, detail
):
    with pytest.raises(RecorderError) as captured:
        record_runtime_snapshot(
            runtime_snapshot(),
            profile_id="resource-scheduler",
            selectors=selectors,
            recorder=TelemetryRecorder(tmp_path / "records"),
        )

    assert captured.value.code == "selectors-invalid"
    assert captured.value.detail == detail
    assert not (tmp_path / "records/resource-scheduler").exists()


def test_snapshot_recorder_rejects_invalid_contract_before_writing(tmp_path):
    document = runtime_snapshot()
    document["metrics"]["queueDepth"] = True

    with pytest.raises(RecorderError) as captured:
        record_runtime_snapshot(
            document,
            profile_id="resource-scheduler",
            selectors=("queueDepth",),
            recorder=TelemetryRecorder(tmp_path / "records"),
        )

    assert captured.value.code == "snapshot-invalid"
    assert "metrics/queueDepth" in captured.value.detail
    assert not (tmp_path / "records/resource-scheduler").exists()


def test_snapshot_loader_is_bounded_and_contract_validating(tmp_path):
    path = tmp_path / "snapshot.json"
    document = runtime_snapshot()
    path.write_text(json.dumps(document))
    assert load_runtime_snapshot(path) == document

    document["metrics"]["queueDepth"] = "private text"
    path.write_text(json.dumps(document))
    with pytest.raises(RecorderError) as captured:
        load_runtime_snapshot(path)
    assert captured.value.code == "snapshot-invalid"
    assert "metrics/queueDepth" in captured.value.detail

    path.write_bytes(b" " * (MAX_RUNTIME_SNAPSHOT_BYTES + 1))
    with pytest.raises(ValueError, match="document exceeds 1048576 bytes"):
        load_runtime_snapshot(path)


def test_duplicate_in_an_older_retained_segment_is_still_skipped(tmp_path):
    recorder = TelemetryRecorder(
        tmp_path / "records", max_segment_bytes=100, max_segments=8
    )
    for timestamp in range(1_000, 1_006):
        assert recorder.record_unique(
            "resource-scheduler", {"queueDepth": timestamp}, timestamp
        ) is not None
    assert recorder.segment_count("resource-scheduler") > 1
    before = recorder.rows("resource-scheduler")

    assert recorder.record_unique(
        "resource-scheduler", {"queueDepth": 999_999}, before[0].observed_at_ms
    ) is None
    assert recorder.rows("resource-scheduler") == before


def test_concurrent_recorders_serialize_duplicate_check_and_append(tmp_path):
    root = tmp_path / "records"
    recorders = (TelemetryRecorder(root), TelemetryRecorder(root))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                lambda subject: subject.record_unique(
                    "resource-scheduler", {"queueDepth": 3}, 1_000
                ),
                recorders,
            )
        )

    assert sum(result is not None for result in results) == 1
    assert len(TelemetryRecorder(root).rows("resource-scheduler")) == 1


def test_snapshot_record_cli_reports_recorded_and_duplicate_status(tmp_path, capsys):
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(runtime_snapshot(timestamp=4_242)))
    arguments = [
        "--profile",
        "resource-scheduler",
        "--selector",
        "queueDepth",
        "--snapshot",
        str(snapshot_path),
        "--records-root",
        str(tmp_path / "records"),
    ]

    assert snapshot_record_main(arguments) == 0
    recorded = {
        "v": 1,
        "p": "resource-scheduler",
        "t": 4_242,
        "f": {"queueDepth": 3.0},
        "status": "recorded",
    }
    assert capsys.readouterr().out == json.dumps(recorded, indent=2) + "\n"
    assert snapshot_record_main(arguments) == 0
    duplicate = {
        "status": "duplicate",
        "profileId": "resource-scheduler",
        "observedAtMs": 4_242,
    }
    assert capsys.readouterr().out == json.dumps(duplicate, indent=2) + "\n"


def test_snapshot_record_cli_help_and_required_arguments_are_operator_contracts(capsys):
    with pytest.raises(SystemExit, match="0"):
        snapshot_record_main(["--help"])
    help_text = capsys.readouterr().out
    assert "usage: omnitensor-record-runtime-snapshot" in help_text
    assert "Append allowlisted aggregate runtime measurements" in help_text
    assert "queueDepth or runningProfiles; repeat" in help_text

    with pytest.raises(SystemExit, match="2"):
        snapshot_record_main([])
    assert "--profile" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="2"):
        snapshot_record_main(["--profile", "resource-scheduler"])
    assert "--selector" in capsys.readouterr().err


def test_snapshot_record_cli_uses_environment_and_default_records_root(
    tmp_path, capsys, monkeypatch
):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(runtime_snapshot(timestamp=5_000)))
    monkeypatch.setenv("OMNITENSOR_STATE_PATH", str(path))
    monkeypatch.setattr(
        "omnitensor.training.cli.DEFAULT_RECORDS_ROOT", str(tmp_path / "default-records")
    )

    assert snapshot_record_main(
        ["--profile", "resource-scheduler", "--selector", "runningProfiles"]
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "recorded"
    [row] = TelemetryRecorder(tmp_path / "default-records").rows("resource-scheduler")
    assert row.features == {"runningProfiles": 2.0}


def test_snapshot_record_cli_honours_state_environment_and_has_stable_failure(
    tmp_path, capsys, monkeypatch
):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(runtime_snapshot()))
    monkeypatch.setenv("OMNITENSOR_STATE_PATH", str(path))

    assert snapshot_record_main(
        [
            "--profile",
            "resource-scheduler",
            "--selector",
            "identity",
            "--records-root",
            str(tmp_path / "records"),
        ]
    ) == 1
    assert (
        capsys.readouterr().err
        == "snapshot recording failed: selectors-invalid: unsupported selector: identity\n"
    )


@given(
    queue_depth=st.integers(min_value=0, max_value=1_000_000),
    running_profiles=st.integers(min_value=0, max_value=128),
    selectors=st.sampled_from(
        (("queueDepth",), ("runningProfiles",), ("queueDepth", "runningProfiles"))
    ),
)
def test_snapshot_selectors_preserve_exact_bounded_metrics(
    queue_depth, running_profiles, selectors
):
    features = _selector_values(
        runtime_snapshot(queue_depth=queue_depth, running_profiles=running_profiles),
        selectors,
    )

    expected = {
        "queueDepth": float(queue_depth),
        "runningProfiles": float(running_profiles),
    }
    assert features == {name: expected[name] for name in selectors}
