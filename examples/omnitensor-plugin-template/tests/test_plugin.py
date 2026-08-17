from __future__ import annotations

import asyncio
import math

import pytest
from omnitensor_template.plugin import (
    MAX_ABSOLUTE_VALUE,
    MAX_VALUES,
    TemplateCollector,
    TemplatePipeline,
    TemplatePlugin,
    TemplateResultConsumer,
)

from omnitensor.sdk import (
    CancellationController,
    CollectedOutput,
    InferenceOutput,
    PluginContext,
    PluginProgress,
    PluginRequest,
    PluginResultStatus,
    ProgressProbe,
    SDKContractError,
    Trigger,
    TriggerKind,
    run_cancellation_contract,
    run_plugin_contract,
)


def _request(payload=None) -> PluginRequest:
    return PluginRequest(
        "job-1",
        "template-workload",
        "manual",
        {"values": [-1, 0, 1]} if payload is None else payload,
        10,
        100,
    )


def test_plugin_contract_and_cancellation() -> None:
    async def scenario() -> None:
        context = PluginContext("template-workload", 1, {"scale": 2}, frozenset())
        report = await run_plugin_contract(TemplatePlugin(), context, _request())
        cancelled = await run_cancellation_contract(TemplatePlugin(), context, _request())

        assert report.result.status is PluginResultStatus.SUCCEEDED
        assert report.result.output == {"score": 0.5}
        assert report.result.detail == "template result accepted"
        assert [item.stage for item in report.progress] == [
            "collect",
            "preprocess",
            "deliver",
            "terminal",
        ]
        assert cancelled.result.status is PluginResultStatus.CANCELLED
        assert cancelled.result.detail == "cancelled before execution"

    asyncio.run(scenario())


def test_collector_accepts_a_bounded_numeric_vector() -> None:
    async def scenario() -> None:
        collector = TemplateCollector()
        readiness = await collector.readiness()
        assert readiness.detail == "manual input ready"
        assert readiness.checked_at_ms == 0
        output = await collector.collect(
            Trigger(
                "template-workload",
                "manual-1",
                TriggerKind.MANUAL,
                {"values": [1, 2.5]},
                0,
            )
        )
        assert output == CollectedOutput({"values": [1.0, 2.5]})

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "values",
    [
        None,
        [],
        list(range(MAX_VALUES + 1)),
        [True],
        [math.nan],
        [math.inf],
        [MAX_ABSOLUTE_VALUE + 1],
        ["1"],
    ],
)
def test_collector_rejects_invalid_vectors(values) -> None:
    async def scenario() -> None:
        trigger = Trigger(
            "template-workload",
            "invalid",
            TriggerKind.MANUAL,
            {"values": values},
            0,
        )
        with pytest.raises(SDKContractError) as error:
            await TemplateCollector().collect(trigger)
        assert error.value.code == "invalid-values"

    asyncio.run(scenario())


@pytest.mark.parametrize("scale", [None, True, "1", 0, -1, 1001, math.nan, math.inf])
def test_pipeline_rejects_invalid_scale(scale) -> None:
    with pytest.raises(SDKContractError) as error:
        TemplatePipeline(scale)
    assert error.value.code == "invalid-scale"


def test_pipeline_and_consumer_enforce_bounds_and_cancellation() -> None:
    async def scenario() -> None:
        cancellation = CancellationController()
        pipeline = TemplatePipeline(2)
        preprocessed = await pipeline.preprocess(CollectedOutput({"values": [-2, 2]}), cancellation)
        assert preprocessed.tensors == {"mean": 0.0}
        assert preprocessed.metadata == {"count": 2}
        postprocessed = await pipeline.postprocess(InferenceOutput({"score": 0.75}), cancellation)
        delivered = await TemplateResultConsumer().deliver(postprocessed, cancellation)
        assert delivered.output == {"score": 0.75}
        assert delivered.detail == "template result accepted"

        for score in (None, True, -0.1, 1.1, math.nan, math.inf):
            with pytest.raises(SDKContractError) as error:
                await pipeline.postprocess(InferenceOutput({"score": score}), cancellation)
            assert error.value.code == "invalid-score"

        cancellation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pipeline.preprocess(CollectedOutput({"values": [1]}), cancellation)
        with pytest.raises(asyncio.CancelledError):
            await pipeline.postprocess(InferenceOutput({"score": 0.5}), cancellation)
        with pytest.raises(asyncio.CancelledError):
            await TemplateResultConsumer().deliver(postprocessed, cancellation)

    asyncio.run(scenario())


def test_explicit_execution_reports_bounded_progress() -> None:
    async def scenario() -> None:
        plugin = TemplatePlugin()
        assert plugin.plugin_id == "template-workload"
        await plugin.start(PluginContext("template-workload", 1, {}, frozenset()))
        health = await plugin.health()
        assert health.detail == "template ready"
        progress = ProgressProbe("job-1")
        result = await plugin.execute(_request({"values": [1]}), CancellationController(), progress)
        await plugin.stop()

        assert result.output == {"score": 1.0}
        assert result.completed_at_ms == 10
        assert progress.items == [
            PluginProgress("job-1", "collect", 0.0, "", 10),
            PluginProgress("job-1", "preprocess", 0.4, "", 10),
            PluginProgress("job-1", "deliver", 0.8, "", 10),
            PluginProgress("job-1", "terminal", 1.0, "", 10),
        ]

    asyncio.run(scenario())
