from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

import omnitensor.sdk as sdk


def request() -> sdk.PluginRequest:
    return sdk.PluginRequest("job-1", "sample-plugin", "manual", {}, 10, 100)


def reference(content: bytes = b"model") -> sdk.ArtifactReference:
    return sdk.ArtifactReference(
        "sample-model",
        "1.0.0",
        "tflite",
        hashlib.sha256(content).hexdigest(),
    )


def test_cancellation_controller_wakes_waiters_and_preserves_first_reason():
    async def scenario():
        cancellation = sdk.CancellationController()
        waiter = asyncio.create_task(cancellation.wait())
        assert not cancellation.cancelled
        assert cancellation.reason == ""
        cancellation.raise_if_cancelled()

        assert cancellation.cancel("policy revoked")
        await waiter
        assert cancellation.cancelled
        assert cancellation.reason == "policy revoked"
        assert not cancellation.cancel("later reason")
        assert cancellation.reason == "policy revoked"
        with pytest.raises(sdk.PluginCancelledError, match="policy revoked"):
            cancellation.raise_if_cancelled()

    asyncio.run(scenario())


def test_cancellation_default_reason_is_stable():
    cancellation = sdk.CancellationController()
    assert cancellation.cancel()
    assert cancellation.reason == "cancelled"


@pytest.mark.parametrize("reason", ["", "x" * (sdk.MAX_RESULT_DETAIL_CHARS + 1), None])
def test_cancellation_reason_is_bounded(reason):
    with pytest.raises(sdk.SDKContractError) as excinfo:
        sdk.CancellationController().cancel(reason)
    assert excinfo.value.code == "invalid-text"
    assert excinfo.value.detail == (
        "cancellation reason must contain 1-2048 characters"
    )


class RecordingProgressSink:
    def __init__(self):
        self.items = []

    async def publish(self, progress):
        self.items.append(progress)


def test_progress_emitter_publishes_bounded_monotonic_progress():
    async def scenario():
        sink = RecordingProgressSink()
        emitter = sdk.ProgressEmitter("job-1", sink)

        first = await emitter.report("collect", 0.25, "reading", observed_at_ms=10)
        second = await emitter.report("infer", 1, "", observed_at_ms=11)

        assert sink.items == [first, second]
        assert first == sdk.PluginProgress("job-1", "collect", 0.25, "reading", 10)
        assert second.fraction == 1.0

    asyncio.run(scenario())


@pytest.mark.parametrize("job_id", ["", "x" * 129, None])
def test_progress_job_id_is_bounded(job_id):
    with pytest.raises(sdk.SDKContractError) as excinfo:
        sdk.ProgressEmitter(job_id, RecordingProgressSink())
    assert excinfo.value.code == "invalid-text"
    assert excinfo.value.detail == "job_id must contain 1-128 characters"


def test_progress_sink_must_implement_contract():
    with pytest.raises(TypeError) as excinfo:
        sdk.ProgressEmitter("job", object())
    assert str(excinfo.value) == "sink must implement publish"


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan"), float("inf"), True, "1"])
def test_progress_fraction_is_finite_and_bounded(fraction):
    async def scenario():
        emitter = sdk.ProgressEmitter("job", RecordingProgressSink())
        with pytest.raises(sdk.SDKContractError, match="invalid-progress"):
            await emitter.report("infer", fraction, "", observed_at_ms=1)

    asyncio.run(scenario())


def test_progress_cannot_move_backward_and_fields_are_validated():
    async def scenario():
        emitter = sdk.ProgressEmitter("job", RecordingProgressSink())
        await emitter.report("collect", 0.5, "", observed_at_ms=1)
        with pytest.raises(sdk.SDKContractError, match="invalid-progress"):
            await emitter.report("infer", 0.4, "", observed_at_ms=2)
        with pytest.raises(sdk.SDKContractError) as stage_error:
            await emitter.report("", 0.5, "", observed_at_ms=2)
        assert stage_error.value.detail == (
            "progress stage must contain 1-80 characters"
        )
        with pytest.raises(sdk.SDKContractError) as detail_error:
            await emitter.report(
                "infer",
                0.5,
                "x" * (sdk.MAX_PROGRESS_DETAIL_CHARS + 1),
                observed_at_ms=2,
            )
        assert detail_error.value.detail == (
            "progress detail must contain 0-1024 characters"
        )
        with pytest.raises(sdk.SDKContractError, match="invalid-timestamp"):
            await emitter.report("infer", 0.5, "", observed_at_ms=-1)

    asyncio.run(scenario())


def test_configuration_view_reads_defensive_typed_values():
    source = {"count": 3, "enabled": True, "nested": {"items": [1]}}
    view = sdk.ConfigurationView(source)
    source["nested"]["items"].append(2)

    assert view.required("count", int) == 3
    assert view.required("enabled", bool) is True
    assert view.optional("name", str, "default") == "default"
    nested = view.required("nested", dict)
    assert nested == {"items": [1]}
    nested["items"].append(9)
    assert view.as_dict()["nested"] == {"items": [1]}


def test_configuration_view_fails_closed_for_missing_or_wrong_types():
    view = sdk.ConfigurationView({"flag": True, "count": "three"})
    with pytest.raises(sdk.SDKContractError, match="configuration-missing"):
        view.required("missing", str)
    with pytest.raises(sdk.SDKContractError, match="configuration-type"):
        view.required("count", int)
    with pytest.raises(sdk.SDKContractError, match="configuration-type"):
        view.required("flag", int)
    with pytest.raises(sdk.SDKContractError, match="invalid-text"):
        view.required("", str)
    with pytest.raises(TypeError, match="expected_type"):
        view.required("flag", "bool")
    with pytest.raises(TypeError, match="configuration must be a mapping"):
        sdk.ConfigurationView([])


def test_permission_view_requires_declared_grants():
    view = sdk.PermissionView(
        frozenset({"read:/sensors", "device:/dev/apex_0"}),
        frozenset({"read:/sensors"}),
    )
    assert view.granted == frozenset({"read:/sensors"})
    assert view.allows("read:/sensors")
    assert not view.allows("device:/dev/apex_0")
    with pytest.raises(sdk.SDKContractError, match="permission-denied"):
        view.require("device:/dev/apex_0")
    view.require("read:/sensors")


def test_permission_view_rejects_undeclared_or_invalid_grants():
    with pytest.raises(sdk.SDKContractError, match="undeclared-grant"):
        sdk.PermissionView(frozenset(), frozenset({"read:/private"}))
    with pytest.raises(sdk.SDKContractError, match="invalid-permissions"):
        sdk.PermissionView(frozenset({""}), frozenset())


def test_artifact_view_returns_only_declared_ready_identity(tmp_path):
    artifact = reference()
    path = tmp_path / "model.tflite"
    path.write_bytes(b"model")
    ready = sdk.ArtifactResolution(True, path, "", 5)
    view = sdk.ArtifactView([(artifact, ready)])

    assert view.resolution(artifact) == ready
    assert view.require_ready(artifact) == path

    other_digest = sdk.ArtifactReference(
        artifact.id,
        artifact.version,
        artifact.format,
        "0" * 64,
    )
    with pytest.raises(sdk.SDKContractError, match="artifact-identity-mismatch"):
        view.resolution(other_digest)
    undeclared = sdk.ArtifactReference("other", "1.0.0", "tflite", "0" * 64)
    with pytest.raises(sdk.SDKContractError, match="artifact-undeclared"):
        view.resolution(undeclared)


def test_artifact_view_rejects_invalid_duplicates_and_unready_paths(tmp_path):
    artifact = reference()
    unavailable = sdk.ArtifactResolution(False, None, "digest mismatch", 5)
    with pytest.raises(sdk.SDKContractError, match="duplicate-artifact"):
        sdk.ArtifactView([(artifact, unavailable), (artifact, unavailable)])
    invalid = sdk.ArtifactReference("Bad", "1.0.0", "tflite", "0" * 64)
    with pytest.raises(sdk.SDKContractError, match="invalid-artifact"):
        sdk.ArtifactView([(invalid, unavailable)])
    view = sdk.ArtifactView([(artifact, unavailable)])
    with pytest.raises(sdk.SDKContractError, match="digest mismatch"):
        view.require_ready(artifact)
    no_reason = sdk.ArtifactView(
        [(artifact, sdk.ArtifactResolution(False, None, "", 0))]
    )
    with pytest.raises(sdk.SDKContractError, match="artifact is not ready"):
        no_reason.require_ready(artifact)


def test_result_factories_copy_outputs_and_assign_terminal_statuses():
    output = {"labels": ["safe"]}
    succeeded = sdk.succeeded_result(request(), output, completed_at_ms=20, detail="done")
    output["labels"].append("mutated")
    failed = sdk.failed_result(request(), "failed safely", completed_at_ms=21)
    cancelled = sdk.cancelled_result(request(), "cancelled", completed_at_ms=22)

    assert succeeded == sdk.PluginResult(
        "job-1",
        sdk.PluginResultStatus.SUCCEEDED,
        {"labels": ["safe"]},
        "done",
        20,
    )
    assert failed.status is sdk.PluginResultStatus.FAILED
    assert failed.output == {}
    assert cancelled.status is sdk.PluginResultStatus.CANCELLED
    assert cancelled.output == {}


@pytest.mark.parametrize("timestamp", [-1, True, 1.5])
def test_result_factories_validate_timestamp(timestamp):
    with pytest.raises(sdk.SDKContractError, match="invalid-timestamp"):
        sdk.failed_result(request(), "failed", completed_at_ms=timestamp)


def test_result_factories_validate_request_output_and_detail():
    with pytest.raises(TypeError, match="request must be"):
        sdk.succeeded_result(object(), {}, completed_at_ms=1)
    with pytest.raises(TypeError, match="output must be"):
        sdk.succeeded_result(request(), [], completed_at_ms=1)
    with pytest.raises(sdk.SDKContractError, match="invalid-text"):
        sdk.failed_result(
            request(),
            "x" * (sdk.MAX_RESULT_DETAIL_CHARS + 1),
            completed_at_ms=1,
        )


class SamplePlugin(sdk.ManagedPlugin):
    plugin_id = "sample-plugin"

    def __init__(self, events, *, fail_start=False):
        super().__init__()
        self.events = events
        self.fail_start = fail_start

    async def on_start(self):
        self.events.append(("start", self.configuration.required("mode", str)))
        if self.fail_start:
            raise RuntimeError("start failed")

    async def on_health(self):
        return sdk.PluginHealth(sdk.PluginHealthStatus.DEGRADED, "fixture", 7)

    async def on_stop(self):
        self.events.append(("stop", self.permissions.granted))

    async def execute(self, request, cancellation, progress):
        cancellation.raise_if_cancelled()
        return sdk.succeeded_result(request, {"ok": True}, completed_at_ms=8)


def test_managed_plugin_provides_identity_checked_idempotent_lifecycle():
    async def scenario():
        events = []
        plugin = SamplePlugin(events)
        assert await plugin.health() == sdk.PluginHealth(
            sdk.PluginHealthStatus.STOPPED,
            "plugin is stopped",
            0,
        )
        with pytest.raises(sdk.SDKContractError, match="plugin-not-started"):
            _ = plugin.context
        context = sdk.PluginContext(
            "sample-plugin",
            1,
            {"mode": "safe"},
            frozenset({"read:/sensors"}),
        )
        await plugin.start(context)
        assert plugin.context == context
        assert await plugin.health() == sdk.PluginHealth(
            sdk.PluginHealthStatus.DEGRADED,
            "fixture",
            7,
        )
        with pytest.raises(sdk.SDKContractError) as already_started:
            await plugin.start(context)
        assert already_started.value.code == "plugin-already-started"
        assert already_started.value.detail == "plugin is already started"
        result = await plugin.execute(
            request(), sdk.CancellationController(), object()
        )
        assert result.status is sdk.PluginResultStatus.SUCCEEDED
        await plugin.stop()
        await plugin.stop()
        assert events == [
            ("start", "safe"),
            ("stop", frozenset({"read:/sensors"})),
        ]

    asyncio.run(scenario())


def test_managed_plugin_rejects_identity_and_rolls_back_failed_start():
    async def scenario():
        plugin = SamplePlugin([], fail_start=True)
        wrong = sdk.PluginContext("other-plugin", 1, {}, frozenset())
        with pytest.raises(sdk.SDKContractError) as mismatch:
            await plugin.start(wrong)
        assert mismatch.value.code == "plugin-identity-mismatch"
        assert mismatch.value.detail == (
            "expected sample-plugin; received other-plugin"
        )
        context = sdk.PluginContext(
            "sample-plugin", 1, {"mode": "safe"}, frozenset()
        )
        with pytest.raises(RuntimeError, match="start failed"):
            await plugin.start(context)
        assert plugin._context is None

    asyncio.run(scenario())


def test_default_managed_plugin_hooks_report_ready():
    class DefaultPlugin(sdk.ManagedPlugin):
        plugin_id = "default-plugin"

        async def execute(self, request, cancellation, progress):
            return sdk.cancelled_result(request, "unused", completed_at_ms=1)

    async def scenario():
        plugin = DefaultPlugin()
        await plugin.start(sdk.PluginContext("default-plugin", 1, {}, frozenset()))
        assert await plugin.health() == sdk.PluginHealth(
            sdk.PluginHealthStatus.READY,
            "plugin is ready",
            0,
        )
        await plugin.stop()

    asyncio.run(scenario())


def test_sdk_exports_contracts_without_service_internals():
    expected = {
        "ManagedPlugin",
        "Collector",
        "PipelineHooks",
        "ArtifactProvider",
        "ArtifactView",
        "ResultSink",
        "PluginConfigurationSpec",
        "PermissionView",
        "CancellationController",
    }
    assert expected <= set(sdk.__all__)
    sdk_root = Path(sdk.__file__).parent
    source = "\n".join(path.read_text() for path in sdk_root.glob("*.py"))
    for internal in ("service", "scheduler", "control", "registry", "executors", "state"):
        assert f"omnitensor.{internal}" not in source
        assert f"..{internal}" not in source


def test_optional_returns_none_for_an_absent_key():
    """The implicit None default was type-checked against the caller's type."""
    view = sdk.ConfigurationView({})
    assert view.optional("label", str) is None
    assert view.optional("count", int) is None


def test_optional_returns_a_typed_default_of_any_type():
    view = sdk.ConfigurationView({})
    assert view.optional("label", str, "fallback") == "fallback"
    assert view.optional("retries", int, 0) == 0
    assert view.optional("window", list, ()) == ()


def test_optional_still_rejects_a_present_value_of_the_wrong_type():
    view = sdk.ConfigurationView({"label": 7})
    with pytest.raises(sdk.SDKContractError) as excinfo:
        view.optional("label", str)
    assert excinfo.value.code == "configuration-type"


def test_required_is_unchanged_by_the_default_exemption():
    with pytest.raises(sdk.SDKContractError) as excinfo:
        sdk.ConfigurationView({}).required("label", str)
    assert excinfo.value.code == "configuration-missing"
