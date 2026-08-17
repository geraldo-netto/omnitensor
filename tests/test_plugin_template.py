from __future__ import annotations

import asyncio
import json
import sys
import tomllib
from pathlib import Path

import pytest

from omnitensor.plugins import (
    PluginMetadata,
    PluginSource,
    resolve_plugin_identities,
)
from omnitensor.registry import validate_document
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

TEMPLATE_ROOT = Path(__file__).parents[1] / "examples/omnitensor-plugin-template"
TEMPLATE_SOURCE = TEMPLATE_ROOT / "src"


@pytest.fixture(scope="module")
def template_module():
    sys.path.insert(0, str(TEMPLATE_SOURCE))
    try:
        from omnitensor_template import plugin

        yield plugin
    finally:
        sys.path.remove(str(TEMPLATE_SOURCE))
        for name in tuple(sys.modules):
            if name == "omnitensor_template" or name.startswith("omnitensor_template."):
                sys.modules.pop(name)


def template_request(payload=None):
    return PluginRequest(
        "job-1",
        "template-workload",
        "manual",
        {"values": [-1, 0, 1]} if payload is None else payload,
        10,
        100,
    )


def test_template_build_metadata_manifest_and_identity_agree():
    project = tomllib.loads((TEMPLATE_ROOT / "pyproject.toml").read_text())
    manifest_path = TEMPLATE_ROOT / "src/omnitensor_template/omnitensor-plugin.json"
    manifest = json.loads(manifest_path.read_text())
    entry_points = project["project"]["entry-points"]["omnitensor.workloads"]

    assert project["project"]["name"] == "omnitensor-plugin-template"
    assert project["project"]["version"] == manifest["version"]
    assert entry_points == {"template-workload": "omnitensor_template.plugin:TemplatePlugin"}
    assert manifest["manifestVersion"] == 2
    assert manifest["id"] == manifest["plugin"]["entryPoint"] == "template-workload"
    assert validate_document("workload-manifest.schema.json", manifest) == []
    # Regression (OMNI-0143): installing the manifest at the wheel root made
    # any two plugins built from this template overwrite each other's manifest.
    assert "force-include" not in project["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert manifest_path.parent.name == "omnitensor_template"

    catalog = resolve_plugin_identities(
        (
            PluginMetadata(
                PluginSource.EXTERNAL,
                "template-workload",
                entry_points["template-workload"],
                project["project"]["name"],
                project["project"]["version"],
                (manifest_path,),
                None,
                "",
            ),
        )
    )
    assert [plugin.plugin_id for plugin in catalog.plugins] == ["template-workload"]
    assert catalog.rejections == ()


def test_template_imports_only_public_sdk_at_the_omnitensor_boundary():
    source = (TEMPLATE_SOURCE / "omnitensor_template/plugin.py").read_text()
    assert "from omnitensor.sdk import" in source
    assert "omnitensor.plugins" not in source
    for internal in ("service", "scheduler", "control", "registry", "executors", "state"):
        assert f"omnitensor.{internal}" not in source


def test_template_plugin_passes_normal_and_cancellation_conformance(template_module):
    async def scenario():
        context = PluginContext(
            "template-workload",
            1,
            {"scale": 2},
            frozenset(),
        )
        report = await run_plugin_contract(
            template_module.TemplatePlugin(),
            context,
            template_request(),
        )
        cancelled = await run_cancellation_contract(
            template_module.TemplatePlugin(),
            context,
            template_request(),
        )

        assert report.result.status is PluginResultStatus.SUCCEEDED
        assert report.result.output == {"score": 0.5}
        assert report.result.detail == "template result accepted"
        assert [progress.stage for progress in report.progress] == [
            "collect",
            "preprocess",
            "deliver",
            "terminal",
        ]
        assert cancelled.result.status is PluginResultStatus.CANCELLED
        assert cancelled.result.detail == "cancelled before execution"

    asyncio.run(scenario())


def test_template_collector_bounds_numeric_inputs(template_module):
    async def scenario():
        collector = template_module.TemplateCollector()
        assert (await collector.readiness()).detail == "manual input ready"
        trigger = Trigger(
            "template-workload",
            "manual-1",
            TriggerKind.MANUAL,
            {"values": [1, 2.5]},
            0,
        )
        assert await collector.collect(trigger) == CollectedOutput({"values": [1.0, 2.5]})

        for values in (
            None,
            [],
            list(range(template_module.MAX_VALUES + 1)),
            [True],
            [float("nan")],
            [template_module.MAX_ABSOLUTE_VALUE + 1],
            ["1"],
        ):
            invalid = Trigger(
                "template-workload",
                "invalid",
                TriggerKind.MANUAL,
                {"values": values},
                0,
            )
            with pytest.raises(SDKContractError, match="invalid-values"):
                await collector.collect(invalid)

    asyncio.run(scenario())


def test_template_pipeline_and_consumer_enforce_typed_bounds(template_module):
    async def scenario():
        cancellation = CancellationController()
        pipeline = template_module.TemplatePipeline(2)
        preprocessed = await pipeline.preprocess(CollectedOutput({"values": [-2, 2]}), cancellation)
        assert preprocessed.tensors == {"mean": 0.0}
        assert preprocessed.metadata == {"count": 2}
        postprocessed = await pipeline.postprocess(InferenceOutput({"score": 0.75}), cancellation)
        delivered = await template_module.TemplateResultConsumer().deliver(
            postprocessed, cancellation
        )
        assert delivered.output == {"score": 0.75}
        assert delivered.detail == "template result accepted"

        for score in (None, True, -0.1, 1.1, float("inf")):
            with pytest.raises(SDKContractError, match="invalid-score"):
                await pipeline.postprocess(InferenceOutput({"score": score}), cancellation)

        cancellation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pipeline.preprocess(CollectedOutput({"values": [1]}), cancellation)
        with pytest.raises(asyncio.CancelledError):
            await pipeline.postprocess(InferenceOutput({"score": 0.5}), cancellation)
        with pytest.raises(asyncio.CancelledError):
            await template_module.TemplateResultConsumer().deliver(postprocessed, cancellation)

    asyncio.run(scenario())


@pytest.mark.parametrize("scale", [None, True, "1", 0, -1, 1001, float("nan")])
def test_template_scale_is_finite_positive_and_bounded(template_module, scale):
    with pytest.raises(SDKContractError, match="invalid-scale"):
        template_module.TemplatePipeline(scale)


def test_template_progress_is_bounded_by_the_host_probe(template_module):
    async def scenario():
        plugin = template_module.TemplatePlugin()
        await plugin.start(PluginContext("template-workload", 1, {}, frozenset()))
        progress = ProgressProbe("job-1")
        result = await plugin.execute(
            template_request({"values": [1]}),
            CancellationController(),
            progress,
        )
        await plugin.stop()

        assert result.output == {"score": 1.0}
        assert progress.items[-1] == PluginProgress("job-1", "terminal", 1.0, "", 10)

    asyncio.run(scenario())
