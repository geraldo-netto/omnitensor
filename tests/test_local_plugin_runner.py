from __future__ import annotations

import asyncio
import hashlib

import pytest

import omnitensor.sdk as sdk


def artifact_reference(content=b"fixture-model"):
    return sdk.ArtifactReference(
        "fixture-model",
        "1.0.0",
        "tflite",
        hashlib.sha256(content).hexdigest(),
    )


class FakePipeline:
    def __init__(self, *, failure=None):
        self.calls = []
        self.failure = failure

    async def preprocess(self, collected, cancellation):
        cancellation.raise_if_cancelled()
        self.calls.append(("preprocess", collected))
        return sdk.PreprocessedOutput(
            {"input": collected.payload["value"]},
            {"source": "fixture"},
        )

    async def postprocess(self, inference, cancellation):
        cancellation.raise_if_cancelled()
        self.calls.append(("postprocess", inference))
        if self.failure is not None:
            raise self.failure
        return sdk.PostprocessedOutput({"score": inference.tensors["score"]})

    async def deliver(self, output, cancellation):
        cancellation.raise_if_cancelled()
        self.calls.append(("deliver", output))
        return sdk.DeliveredOutput(output.output, "delivered locally")


def runner_fixture(tmp_path, *, pipeline=None, executor_outputs=None, grants=None):
    clock = sdk.FakeClock(100)
    triggers = sdk.FakeTriggerFactory("fixture-plugin", clock)
    trigger = triggers.manual({"input": "sample"})
    collector = sdk.FakeCollector({trigger.trigger_id: sdk.CollectedOutput({"value": [1, 2, 3]})})
    artifact = artifact_reference()
    artifact_path = tmp_path / "model.tflite"
    artifact_path.write_bytes(b"fixture-model")
    artifacts = sdk.FakeArtifactProvider(
        [(artifact, sdk.ArtifactResolution(True, artifact_path, "", 13))]
    )
    executor = sdk.FakeExecutor(executor_outputs or [sdk.InferenceOutput({"score": 0.9})])
    permissions = sdk.FakePermissions(
        frozenset({"read:/fixture"}),
        frozenset({"read:/fixture"}) if grants is None else grants,
    )
    results = sdk.RecordingResultSink()
    progress = sdk.RecordingProgressSink()
    runner = sdk.LocalPipelineRunner(
        collector=collector,
        pipeline=pipeline or FakePipeline(),
        artifacts=artifacts,
        executor=executor,
        permissions=permissions.view,
        clock=clock,
        results=results,
        progress=progress,
    )
    return {
        "clock": clock,
        "triggers": triggers,
        "trigger": trigger,
        "collector": collector,
        "artifact": artifact,
        "artifact_path": artifact_path,
        "artifacts": artifacts,
        "executor": executor,
        "permissions": permissions,
        "results": results,
        "progress": progress,
        "runner": runner,
    }


def test_local_runner_executes_complete_pipeline_without_accelerator(tmp_path):
    async def scenario():
        fixture = runner_fixture(tmp_path)

        run = await fixture["runner"].run(
            fixture["trigger"],
            fixture["artifact"],
            required_permissions=frozenset({"read:/fixture"}),
        )

        assert run.request == sdk.PluginRequest(
            "manual-1",
            "fixture-plugin",
            "manual",
            {"input": "sample"},
            100,
            30_100,
        )
        assert run.result == sdk.PluginResult(
            "manual-1",
            sdk.PluginResultStatus.SUCCEEDED,
            {"score": 0.9},
            "delivered locally",
            100,
        )
        assert [item.stage for item in run.progress] == [
            "collect",
            "preprocess",
            "resolve",
            "infer",
            "postprocess",
            "deliver",
            "terminal",
        ]
        assert [item.fraction for item in run.progress] == [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
        assert fixture["progress"].items == list(run.progress)
        assert fixture["results"].items == [run.result]
        assert fixture["collector"].calls == [fixture["trigger"]]
        assert fixture["artifacts"].calls == [fixture["artifact"]]
        assert fixture["executor"].calls == [(fixture["artifact_path"], {"input": [1, 2, 3]})]

    asyncio.run(scenario())


def test_runner_fails_closed_before_collection_when_permission_is_missing(tmp_path):
    async def scenario():
        fixture = runner_fixture(tmp_path, grants=frozenset())

        run = await fixture["runner"].run(
            fixture["trigger"],
            fixture["artifact"],
            required_permissions=frozenset({"read:/fixture"}),
        )

        assert run.result.status is sdk.PluginResultStatus.FAILED
        assert run.result.detail == "local pipeline failed: SDKContractError"
        assert run.progress == ()
        assert fixture["collector"].calls == []
        assert fixture["results"].items == [run.result]

    asyncio.run(scenario())


def test_runner_maps_artifact_and_pipeline_failures_without_private_details(tmp_path):
    async def scenario():
        fixture = runner_fixture(
            tmp_path,
            pipeline=FakePipeline(failure=RuntimeError("private result")),
        )
        artifact = fixture["artifact"]
        fixture["artifacts"] = sdk.FakeArtifactProvider(
            [(artifact, sdk.ArtifactResolution(False, None, "private root", 0))]
        )
        unavailable_runner = sdk.LocalPipelineRunner(
            collector=fixture["collector"],
            pipeline=FakePipeline(),
            artifacts=fixture["artifacts"],
            executor=fixture["executor"],
            permissions=fixture["permissions"].view,
            clock=fixture["clock"],
            results=fixture["results"],
        )

        unavailable = await unavailable_runner.run(fixture["trigger"], artifact)
        failed = await fixture["runner"].run(fixture["trigger"], artifact)

        assert unavailable.result.detail == "local pipeline failed: SDKContractError"
        assert [item.stage for item in unavailable.progress] == [
            "collect",
            "preprocess",
            "resolve",
        ]
        assert failed.result.detail == "local pipeline failed: RuntimeError"
        assert "private" not in failed.result.detail

    asyncio.run(scenario())


def test_runner_maps_precancelled_and_expired_runs_to_cancellation(tmp_path):
    async def scenario():
        fixture = runner_fixture(tmp_path)
        cancellation = sdk.CancellationController()
        cancellation.cancel("user cancelled")

        cancelled = await fixture["runner"].run(
            fixture["trigger"],
            fixture["artifact"],
            cancellation=cancellation,
        )
        fixture["clock"].advance(11)
        expired = await fixture["runner"].run(
            fixture["trigger"],
            fixture["artifact"],
            deadline_ms=10,
        )

        assert cancelled.result.status is sdk.PluginResultStatus.CANCELLED
        assert cancelled.result.detail == "user cancelled"
        assert cancelled.progress == ()
        assert expired.result.status is sdk.PluginResultStatus.CANCELLED
        assert expired.result.detail == "deadline exceeded"
        assert expired.progress == ()

    asyncio.run(scenario())


def test_runner_validates_deadline_and_boundary_contracts(tmp_path):
    fixture = runner_fixture(tmp_path)

    async def invalid_deadline():
        with pytest.raises(ValueError, match="deadline_ms must be a positive integer"):
            await fixture["runner"].run(fixture["trigger"], fixture["artifact"], deadline_ms=0)

    asyncio.run(invalid_deadline())

    valid = {
        "collector": fixture["collector"],
        "pipeline": FakePipeline(),
        "artifacts": fixture["artifacts"],
        "executor": fixture["executor"],
        "permissions": fixture["permissions"].view,
        "clock": fixture["clock"],
        "results": fixture["results"],
    }
    for name in ("collector", "pipeline", "artifacts", "executor", "results"):
        arguments = dict(valid)
        arguments[name] = object()
        with pytest.raises(TypeError, match=f"{name} does not implement"):
            sdk.LocalPipelineRunner(**arguments)
    with pytest.raises(TypeError, match="progress does not implement"):
        sdk.LocalPipelineRunner(**valid, progress=object())


def test_fake_clock_and_trigger_factory_are_deterministic():
    clock = sdk.FakeClock(7)
    triggers = sdk.FakeTriggerFactory("plugin", clock)

    assert clock.advance(3) == 10
    assert triggers.manual({}).trigger_id == "manual-1"
    assert triggers.event({}).trigger_id == "event-2"
    assert triggers.periodic({}).trigger_id == "periodic-3"
    assert triggers.manual({}).created_at_ms == 10
    with pytest.raises(ValueError, match="now_ms must be a non-negative integer"):
        sdk.FakeClock(-1)
    with pytest.raises(ValueError, match="milliseconds must be a non-negative integer"):
        clock.advance(True)
    with pytest.raises(ValueError, match="plugin_id must be a non-empty string"):
        sdk.FakeTriggerFactory("", clock)
    with pytest.raises(TypeError, match="kind must be a TriggerKind"):
        triggers.create("manual", {})
    with pytest.raises(TypeError, match="payload must be a mapping"):
        triggers.manual([])


def test_fake_collector_reports_readiness_replays_errors_and_missing_ids():
    async def scenario():
        trigger = sdk.Trigger("plugin", "event-1", sdk.TriggerKind.EVENT, {}, 0)
        error_trigger = sdk.Trigger("plugin", "event-2", sdk.TriggerKind.EVENT, {}, 0)
        missing = sdk.Trigger("plugin", "event-3", sdk.TriggerKind.EVENT, {}, 0)
        readiness = sdk.CollectorReadiness(sdk.SourceStatus.DEGRADED, "fixture", 1)
        collector = sdk.FakeCollector(
            {
                "event-1": sdk.CollectedOutput({"ok": True}),
                "event-2": RuntimeError("fixture failure"),
            },
            readiness=readiness,
        )

        assert await collector.readiness() == readiness
        assert await collector.collect(trigger) == sdk.CollectedOutput({"ok": True})
        with pytest.raises(RuntimeError, match="fixture failure"):
            await collector.collect(error_trigger)
        with pytest.raises(sdk.SDKContractError, match="fixture-missing"):
            await collector.collect(missing)
        default = sdk.FakeCollector({})
        assert (await default.readiness()).status is sdk.SourceStatus.READY

    asyncio.run(scenario())


def test_fake_artifacts_and_executor_replay_in_order(tmp_path):
    async def scenario():
        artifact = artifact_reference()
        path = tmp_path / "model"
        provider = sdk.FakeArtifactProvider([(artifact, sdk.ArtifactResolution(True, path, "", 1))])
        assert provider.resolve(artifact).path == path
        missing = sdk.ArtifactReference("missing", "1.0.0", "tflite", "0" * 64)
        assert not provider.resolve(missing).ready
        with pytest.raises(ValueError, match="unique identities"):
            sdk.FakeArtifactProvider(
                [
                    (artifact, sdk.ArtifactResolution(True, path, "", 1)),
                    (artifact, sdk.ArtifactResolution(False, None, "duplicate", 0)),
                ]
            )

        executor = sdk.FakeExecutor([sdk.InferenceOutput({"a": 1}), RuntimeError("fixture")])
        cancellation = sdk.CancellationController()
        assert await executor.infer(path, {"input": 1}, cancellation) == sdk.InferenceOutput(
            {"a": 1}
        )
        with pytest.raises(RuntimeError, match="fixture"):
            await executor.infer(path, {}, cancellation)
        with pytest.raises(sdk.SDKContractError, match="no inference fixture remains"):
            await executor.infer(path, {}, cancellation)
        cancellation.cancel()
        with pytest.raises(sdk.PluginCancelledError):
            await executor.infer(path, {}, cancellation)

    asyncio.run(scenario())


def test_fake_permissions_grant_and_revoke_fail_closed():
    permissions = sdk.FakePermissions(frozenset({"read:/fixture"}))
    assert not permissions.view.allows("read:/fixture")
    assert permissions.grant("read:/fixture").allows("read:/fixture")
    assert not permissions.revoke("read:/fixture").allows("read:/fixture")
    with pytest.raises(sdk.SDKContractError, match="permission-undeclared"):
        permissions.grant("read:/private")


def test_recording_sinks_enforce_type_and_cardinality():
    async def scenario():
        results = sdk.RecordingResultSink(limit=1)
        progress = sdk.RecordingProgressSink(limit=1)
        result = sdk.PluginResult("job", sdk.PluginResultStatus.SUCCEEDED, {}, "", 0)
        update = sdk.PluginProgress("job", "collect", 0.0, "", 0)
        await results.publish(result)
        await progress.publish(update)
        with pytest.raises(sdk.SDKContractError, match="result-limit"):
            await results.publish(result)
        with pytest.raises(sdk.SDKContractError, match="progress-limit"):
            await progress.publish(update)
        with pytest.raises(TypeError, match="result must be"):
            await sdk.RecordingResultSink().publish(object())
        with pytest.raises(TypeError, match="progress must be"):
            await sdk.RecordingProgressSink().publish(object())

    asyncio.run(scenario())
    for sink in (sdk.RecordingResultSink, sdk.RecordingProgressSink):
        with pytest.raises(ValueError, match="limit must be a positive integer"):
            sink(limit=0)
