from __future__ import annotations

import asyncio

import pytest

from omnitensor.plugins.artifacts import ArtifactReference
from omnitensor.plugins.cancellation import CancellationReason, JobCancellationRegistry
from omnitensor.plugins.flow import PluginFlowController
from omnitensor.plugins.pipeline import (
    CollectedOutput,
    DeliveredOutput,
    InferenceOutput,
    PipelineStage,
    PostprocessedOutput,
    PreprocessedOutput,
    ResolvedOutput,
)
from omnitensor.plugins.policy import PipelinePolicyGate
from omnitensor.plugins.protocol import PluginResultStatus
from omnitensor.plugins.runner import STAGE_ORDER, PipelineRunner

PLUGIN = "hardware-health"
ARTIFACT = ArtifactReference("sample-model", "1.2.3", "ncnn", "a" * 64)


def run(coroutine, timeout=10):
    return asyncio.run(asyncio.wait_for(coroutine, timeout=timeout))


def stages(record=None, delays=None):
    """One well-behaved implementation of every stage."""
    record = record if record is not None else []
    delays = delays or {}

    async def make(stage, output):
        async def stage_callable(carried):
            record.append(stage)
            if stage in delays:
                await asyncio.sleep(delays[stage])
            return output

        return stage_callable

    from pathlib import Path

    outputs = {
        PipelineStage.COLLECT: CollectedOutput({"samples": [1]}),
        PipelineStage.PREPROCESS: PreprocessedOutput({"tensor": [1]}, {}),
        PipelineStage.RESOLVE: ResolvedOutput(ARTIFACT, Path("/models/x")),
        PipelineStage.INFER: InferenceOutput({"scores": [0.5]}),
        PipelineStage.POSTPROCESS: PostprocessedOutput({"verdict": "ok"}),
        PipelineStage.DELIVER: DeliveredOutput({"verdict": "ok"}, "delivered"),
    }

    def build(stage):
        async def stage_callable(carried):
            record.append(stage)
            if stage in delays:
                await asyncio.sleep(delays[stage])
            return outputs[stage]

        return stage_callable

    return {stage: build(stage) for stage in STAGE_ORDER}, record


def runner(*, policy=None, flow=None, cancellations=None, stage_map=None, record=None):
    built, record = stage_map if stage_map else stages(record)
    return (
        PipelineRunner(
            PLUGIN,
            built,
            policy=policy
            or PipelinePolicyGate(PLUGIN, is_paused=lambda: False, is_enabled=lambda _p: True),
            flow=flow if flow is not None else PluginFlowController(PLUGIN, max_in_flight=4),
            cancellations=(
                cancellations if cancellations is not None else JobCancellationRegistry()
            ),
            clock_ms=lambda: 1_700_000_000_000,
        ),
        record,
    )


def test_a_job_runs_every_stage_in_order_and_succeeds():
    subject, record = runner()

    result = run(subject.run("job-1", {"trigger": "manual"}))

    assert record == list(STAGE_ORDER)
    assert result.status is PluginResultStatus.SUCCEEDED
    assert result.output == {"verdict": "ok"}
    assert result.job_id == "job-1"


def test_every_stage_must_be_supplied():
    partial = {stage: (lambda carried: None) for stage in STAGE_ORDER[:3]}
    with pytest.raises(ValueError, match="missing"):
        PipelineRunner(
            PLUGIN,
            partial,
            policy=PipelinePolicyGate(PLUGIN, is_paused=lambda: False, is_enabled=lambda _p: True),
            flow=PluginFlowController(PLUGIN),
            cancellations=JobCancellationRegistry(),
        )


def test_a_paused_runtime_stops_the_job_before_any_stage_runs():
    paused = PipelinePolicyGate(PLUGIN, is_paused=lambda: True, is_enabled=lambda _p: True)
    subject, record = runner(policy=paused)

    result = run(subject.run("job-1", {}))

    assert record == [], "a stage ran while the runtime was paused"
    assert result.status is PluginResultStatus.CANCELLED
    assert "paused" in result.detail


def test_policy_withdrawn_mid_flight_stops_the_next_stage():
    """The answer legitimately changes while a job is running."""
    paused = [False]
    policy = PipelinePolicyGate(PLUGIN, is_paused=lambda: paused[0], is_enabled=lambda _p: True)
    record = []
    built, record = stages(record)
    original = built[PipelineStage.PREPROCESS]

    async def pause_then_run(carried):
        paused[0] = True
        return await original(carried)

    built[PipelineStage.PREPROCESS] = pause_then_run
    subject, _ = runner(policy=policy, stage_map=(built, record))

    result = run(subject.run("job-1", {}))

    assert record == [PipelineStage.COLLECT, PipelineStage.PREPROCESS]
    assert result.status is PluginResultStatus.CANCELLED


def test_a_caller_cancellation_stops_the_running_stage():
    registry = JobCancellationRegistry()
    built, record = stages(delays={PipelineStage.INFER: 5})
    subject, _ = runner(cancellations=registry, stage_map=(built, record))

    async def scenario():
        job = asyncio.create_task(subject.run("job-1", {}))
        for _ in range(200):
            if PipelineStage.INFER in record:
                break
            await asyncio.sleep(0.005)
        subject.cancel("job-1", "user withdrew it")
        return await job

    result = run(scenario())

    assert result.status is PluginResultStatus.CANCELLED
    assert "user withdrew it" in result.detail
    assert PipelineStage.POSTPROCESS not in record


def test_shutdown_cancels_every_running_job():
    registry = JobCancellationRegistry()
    built, record = stages(delays={PipelineStage.COLLECT: 5})
    subject, _ = runner(cancellations=registry, stage_map=(built, record))

    async def scenario():
        job = asyncio.create_task(subject.run("job-1", {}))
        for _ in range(200):
            if record:
                break
            await asyncio.sleep(0.005)
        registry.cancel_all(CancellationReason.SHUTDOWN, "service stopping")
        return await job

    result = run(scenario())

    assert result.status is PluginResultStatus.CANCELLED
    assert "service stopping" in result.detail


def test_a_stage_fault_ends_only_that_job():
    built, record = stages()

    async def broken(carried):
        raise ValueError("plugin defect")

    built[PipelineStage.INFER] = broken
    subject, _ = runner(stage_map=(built, record))

    result = run(subject.run("job-1", {}))

    assert result.status is PluginResultStatus.FAILED
    assert "plugin defect" in result.detail


def test_backpressure_is_reported_as_a_terminal_result_not_an_exception():
    flow = PluginFlowController(PLUGIN, max_in_flight=1)
    built, record = stages(delays={PipelineStage.COLLECT: 1})
    subject, _ = runner(flow=flow, stage_map=(built, record))

    async def scenario():
        first = asyncio.create_task(subject.run("job-1", {}))
        await asyncio.sleep(0.01)
        second = await subject.run("job-2", {})
        await first
        return second

    result = run(scenario())

    assert result.status is PluginResultStatus.FAILED
    assert "plugin-backpressure" in result.detail


def test_a_repeated_idempotency_key_replays_without_running_again():
    """A retrying caller must not cause the pipeline to deliver twice."""
    subject, record = runner()

    async def scenario():
        first = await subject.run("job-1", {}, idempotency_key="request-1")
        second = await subject.run("job-2", {}, idempotency_key="request-1")
        return first, second

    first, second = run(scenario())

    assert record == list(STAGE_ORDER), "the pipeline ran twice for one request"
    assert first is second
    assert first.status is PluginResultStatus.SUCCEEDED


def test_a_finished_job_is_released_from_the_cancellation_registry():
    registry = JobCancellationRegistry()
    subject, _ = runner(cancellations=registry)

    run(subject.run("job-1", {}))

    assert registry.get("job-1") is None
    assert len(registry) == 0


def test_a_failed_job_is_also_released():
    registry = JobCancellationRegistry()
    built, record = stages()

    async def broken(carried):
        raise ValueError("defect")

    built[PipelineStage.COLLECT] = broken
    subject, _ = runner(cancellations=registry, stage_map=(built, record))

    run(subject.run("job-1", {}))

    assert len(registry) == 0


def test_cancelling_an_unknown_job_is_not_an_error():
    subject, _ = runner()
    assert subject.cancel("absent") is False
