from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins import (
    ArtifactReference,
    CollectedOutput,
    DeliveredOutput,
    InferenceOutput,
    PipelineSnapshot,
    PipelineStage,
    PipelineStateMachine,
    PipelineTransitionError,
    PluginResult,
    PluginResultStatus,
    PostprocessedOutput,
    PreprocessedOutput,
    ResolvedOutput,
)


def artifact() -> ArtifactReference:
    return ArtifactReference(
        "echo-model",
        "1.0.0",
        "onnx",
        hashlib.sha256(b"model").hexdigest(),
    )


def outputs():
    return [
        CollectedOutput({"samples": [1, 2]}),
        PreprocessedOutput({"input": [0.1, 0.2]}, {"shape": [1, 2]}),
        ResolvedOutput(artifact(), Path("/models/echo/model.onnx")),
        InferenceOutput({"scores": [0.9]}),
        PostprocessedOutput({"label": "healthy"}),
        DeliveredOutput({"alertId": "alert-1"}, "published"),
    ]


def test_new_pipeline_exposes_one_small_queued_snapshot():
    machine = PipelineStateMachine("job-1")

    assert machine.snapshot == PipelineSnapshot(
        "job-1",
        PipelineStage.QUEUED,
        0,
        None,
        None,
    )
    assert machine.terminal_result is None


def test_pipeline_accepts_each_typed_stage_in_order_and_succeeds_once():
    machine = PipelineStateMachine("job-1")
    assert machine.start().stage is PipelineStage.COLLECT

    expected_stages = [
        PipelineStage.PREPROCESS,
        PipelineStage.RESOLVE,
        PipelineStage.INFER,
        PipelineStage.POSTPROCESS,
        PipelineStage.DELIVER,
        PipelineStage.TERMINAL,
    ]
    for timestamp, (output, expected_stage) in enumerate(
        zip(outputs(), expected_stages, strict=True),
        start=10,
    ):
        snapshot = machine.advance(output, completed_at_ms=timestamp)
        assert snapshot.stage is expected_stage
        assert snapshot.revision == timestamp - 8
        assert snapshot.last_output == output

    expected = PluginResult(
        "job-1",
        PluginResultStatus.SUCCEEDED,
        {"alertId": "alert-1"},
        "published",
        15,
    )
    assert machine.terminal_result == expected
    assert machine.snapshot.terminal_result == expected
    assert machine.snapshot.revision == 7


@pytest.mark.parametrize(
    ("accepted", "wrong", "stage", "required"),
    [
        ([], PreprocessedOutput({}, {}), "collect", "CollectedOutput"),
        ([CollectedOutput({})], InferenceOutput({}), "preprocess", "PreprocessedOutput"),
        (
            [CollectedOutput({}), PreprocessedOutput({}, {})],
            CollectedOutput({}),
            "resolve",
            "ResolvedOutput",
        ),
        (
            [
                CollectedOutput({}),
                PreprocessedOutput({}, {}),
                ResolvedOutput(artifact(), Path("model")),
            ],
            PostprocessedOutput({}),
            "infer",
            "InferenceOutput",
        ),
        (outputs()[:4], DeliveredOutput({}), "postprocess", "PostprocessedOutput"),
        (outputs()[:5], InferenceOutput({}), "deliver", "DeliveredOutput"),
    ],
)
def test_pipeline_rejects_a_wrong_output_without_changing_state(
    accepted,
    wrong,
    stage,
    required,
):
    machine = PipelineStateMachine("job")
    machine.start()
    for timestamp, output in enumerate(accepted):
        machine.advance(output, completed_at_ms=timestamp)
    before = machine.snapshot

    with pytest.raises(PipelineTransitionError) as excinfo:
        machine.advance(wrong, completed_at_ms=99)

    assert excinfo.value.code == "stage-output-mismatch"
    assert str(excinfo.value) == (
        f"stage-output-mismatch: {stage} requires {required}; "
        f"received {type(wrong).__name__}"
    )
    assert machine.snapshot == before


def test_pipeline_cannot_start_twice_or_advance_outside_runnable_stages():
    queued = PipelineStateMachine("queued")
    with pytest.raises(PipelineTransitionError) as not_started:
        queued.advance(CollectedOutput({}), completed_at_ms=0)
    assert str(not_started.value) == "stage-not-runnable: cannot advance from queued"

    started = PipelineStateMachine("started")
    started.start()
    with pytest.raises(PipelineTransitionError) as duplicate:
        started.start()
    assert str(duplicate.value) == "already-started: job is already at collect"

    for timestamp, output in enumerate(outputs()):
        started.advance(output, completed_at_ms=timestamp)
    with pytest.raises(PipelineTransitionError) as terminal:
        started.advance(DeliveredOutput({}), completed_at_ms=10)
    assert str(terminal.value) == "stage-not-runnable: cannot advance from terminal"


@pytest.mark.parametrize(
    ("operation", "status", "detail"),
    [
        ("fail", PluginResultStatus.FAILED, "collector unavailable"),
        ("cancel", PluginResultStatus.CANCELLED, "shutdown"),
    ],
)
def test_failure_or_cancellation_can_terminalize_any_nonterminal_stage(
    operation,
    status,
    detail,
):
    machine = PipelineStateMachine("job")
    machine.start()
    machine.advance(CollectedOutput({}), completed_at_ms=1)

    snapshot = getattr(machine, operation)(detail, completed_at_ms=2)

    assert snapshot.stage is PipelineStage.TERMINAL
    assert snapshot.revision == 3
    assert snapshot.terminal_result == PluginResult("job", status, {}, detail, 2)


def test_first_terminal_result_wins_without_revision_or_payload_replacement():
    machine = PipelineStateMachine("job")
    first = machine.fail("first", completed_at_ms=1)

    second = machine.cancel("second", completed_at_ms=-1)
    third = machine.fail(123, completed_at_ms=3)

    assert second == third == first
    assert machine.snapshot.revision == 1
    assert machine.terminal_result == PluginResult(
        "job",
        PluginResultStatus.FAILED,
        {},
        "first",
        1,
    )


@pytest.mark.parametrize("job_id", ["", "x" * 129, None, 1])
def test_job_identity_is_nonempty_and_bounded(job_id):
    with pytest.raises(ValueError, match="job_id must contain 1-128 characters"):
        PipelineStateMachine(job_id)
    assert PipelineStateMachine("x" * 128).snapshot.job_id == "x" * 128


@pytest.mark.parametrize("timestamp", [-1, True, 1.5, "1", None])
def test_terminal_timestamps_are_non_negative_integers(timestamp):
    machine = PipelineStateMachine("job")
    with pytest.raises(ValueError, match="completed_at_ms must be a non-negative integer"):
        machine.fail("failed", completed_at_ms=timestamp)

    machine.start()
    with pytest.raises(ValueError, match="completed_at_ms must be a non-negative integer"):
        machine.advance(CollectedOutput({}), completed_at_ms=timestamp)


def test_terminal_detail_must_be_a_string():
    with pytest.raises(TypeError, match="terminal detail must be a string"):
        PipelineStateMachine("job").fail(None, completed_at_ms=0)


def test_snapshots_and_stage_outputs_are_immutable():
    snapshot = PipelineStateMachine("job").snapshot
    output = CollectedOutput({})
    with pytest.raises(FrozenInstanceError):
        snapshot.revision = 2
    with pytest.raises(FrozenInstanceError):
        output.payload = {"changed": True}


@given(
    first=st.sampled_from(["fail", "cancel"]),
    repeats=st.lists(st.sampled_from(["fail", "cancel"]), min_size=1, max_size=20),
)
def test_arbitrary_repeated_terminal_signals_never_publish_a_second_result(first, repeats):
    machine = PipelineStateMachine("job")
    expected = getattr(machine, first)(first, completed_at_ms=0)

    for timestamp, operation in enumerate(repeats, start=1):
        assert getattr(machine, operation)(operation, completed_at_ms=timestamp) == expected
    assert machine.snapshot.revision == 1
    assert machine.terminal_result is expected.terminal_result
