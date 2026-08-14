from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor import sdk
from omnitensor.plugins import (
    CancellationToken,
    PluginContext,
    PluginHealth,
    PluginHealthStatus,
    PluginLifecycle,
    PluginProgress,
    PluginRequest,
    PluginResult,
    PluginResultStatus,
    ProgressReporter,
    ScaledProgressReporter,
    WorkloadPlugin,
)
from omnitensor.plugins.document_qa import DocumentQuestionError
from omnitensor.plugins.event_workload import EventWorkloadError
from omnitensor.plugins.file_organizer import FileOrganizerError
from omnitensor.plugins.selected_text import SelectedTextError


class NeverCancelled:
    @property
    def cancelled(self) -> bool:
        return False

    async def wait(self) -> None:
        await asyncio.Event().wait()

    def raise_if_cancelled(self) -> None:
        return None


class RecordingProgress:
    def __init__(self):
        self.updates = []

    async def report(self, progress: PluginProgress) -> None:
        self.updates.append(progress)


class EchoPlugin:
    def __init__(self):
        self.context = None
        self.stopped = False

    async def start(self, context: PluginContext) -> None:
        self.context = context

    async def health(self) -> PluginHealth:
        return PluginHealth(PluginHealthStatus.READY, "ready", 2)

    async def execute(
        self,
        request: PluginRequest,
        cancellation: CancellationToken,
        progress: ProgressReporter,
    ) -> PluginResult:
        cancellation.raise_if_cancelled()
        await progress.report(PluginProgress(request.job_id, "echo", 1.0, "done", 3))
        return PluginResult(
            request.job_id,
            PluginResultStatus.SUCCEEDED,
            request.payload,
            "echoed",
            4,
        )

    async def stop(self) -> None:
        self.stopped = True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "offset", "scale", "error_type", "expected"),
    [
        (
            "suggest",
            0.55,
            0.4,
            FileOrganizerError,
            [0.55, 0.75, 0.9500000000000001],
        ),
        ("answer", 0.7, 0.25, DocumentQuestionError, [0.7, 0.825, 0.95]),
        ("generate", 0.6, 0.25, EventWorkloadError, [0.6, 0.725, 0.85]),
        ("generate", 0.1, 0.85, SelectedTextError, [0.1, 0.525, 0.95]),
    ],
)
async def test_scaled_progress_reporter_preserves_workflow_contract(
    stage, offset, scale, error_type, expected
):
    request = PluginRequest("outer-job", "workflow", "manual", {}, 1, None)
    target = RecordingProgress()
    reporter = ScaledProgressReporter(
        request,
        target,
        lambda: 17,
        stage=stage,
        offset=offset,
        scale=scale,
        error_type=error_type,
    )

    assert isinstance(reporter, ProgressReporter)
    for fraction in (0, 0.5, 1):
        await reporter.report(
            PluginProgress("provider-job", "provider-stage", fraction, "private", 2)
        )

    assert target.updates == [
        PluginProgress("outer-job", stage, fraction, "", 17) for fraction in expected
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type",
    [FileOrganizerError, DocumentQuestionError, EventWorkloadError, SelectedTextError],
)
@pytest.mark.parametrize("fraction", [None, True, "0.5", -0.1, 1.1, float("nan"), float("inf")])
async def test_scaled_progress_reporter_rejects_invalid_provider_fraction(error_type, fraction):
    target = RecordingProgress()
    reporter = ScaledProgressReporter(
        PluginRequest("job", "workflow", "manual", {}, 1, None),
        target,
        lambda: 9,
        stage="stage",
        offset=0.1,
        scale=0.8,
        error_type=error_type,
    )

    with pytest.raises(error_type) as caught:
        await reporter.report(PluginProgress("provider", "model", fraction, "private", 1))

    assert (caught.value.code, caught.value.detail) == (
        "progress-invalid",
        "provider progress is invalid",
    )
    assert target.updates == []


def test_scaled_progress_reporter_has_stable_sdk_export():
    assert sdk.ScaledProgressReporter is ScaledProgressReporter


@given(fraction=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False))
def test_scaled_progress_reporter_accepts_every_bounded_finite_fraction(fraction):
    async def scenario():
        target = RecordingProgress()
        reporter = ScaledProgressReporter(
            PluginRequest("job", "workflow", "manual", {}, 1, None),
            target,
            lambda: 11,
            stage="generate",
            offset=0.1,
            scale=0.85,
            error_type=SelectedTextError,
        )

        await reporter.report(PluginProgress("provider", "model", fraction, "private", 1))

        assert target.updates == [
            PluginProgress("job", "generate", 0.1 + (float(fraction) * 0.85), "", 11)
        ]

    asyncio.run(scenario())


def test_structural_protocols_accept_an_external_plugin_without_inheritance():
    plugin = EchoPlugin()
    assert isinstance(plugin, PluginLifecycle)
    assert isinstance(plugin, WorkloadPlugin)
    assert isinstance(NeverCancelled(), CancellationToken)
    assert isinstance(RecordingProgress(), ProgressReporter)
    assert not isinstance(object(), WorkloadPlugin)


def test_plugin_contract_drives_a_complete_lifecycle():
    async def scenario():
        plugin: WorkloadPlugin = EchoPlugin()
        context = PluginContext("echo", 1, {"prefix": "x"}, frozenset({"read:fixture"}))
        request = PluginRequest("job-1", "echo", "manual", {"value": [1, 2]}, 1, 100)
        progress = RecordingProgress()

        await plugin.start(context)
        assert await plugin.health() == PluginHealth(PluginHealthStatus.READY, "ready", 2)
        result = await plugin.execute(request, NeverCancelled(), progress)
        await plugin.stop()

        assert result == PluginResult(
            "job-1",
            PluginResultStatus.SUCCEEDED,
            {"value": [1, 2]},
            "echoed",
            4,
        )
        assert progress.updates == [PluginProgress("job-1", "echo", 1.0, "done", 3)]
        assert plugin.context == context
        assert plugin.stopped is True

    asyncio.run(scenario())


def test_contract_messages_are_frozen_and_statuses_are_wire_strings():
    context = PluginContext("echo", 1, {}, frozenset())
    with pytest.raises(FrozenInstanceError):
        context.plugin_id = "other"
    assert str(PluginResultStatus.CANCELLED) == "cancelled"
    assert str(PluginHealthStatus.UNAVAILABLE) == "unavailable"


@given(
    job_id=st.text(min_size=1, max_size=40),
    trigger=st.text(max_size=40),
    submitted_at=st.integers(min_value=0, max_value=2**53),
    deadline=st.none() | st.integers(min_value=0, max_value=2**53),
)
def test_request_contract_preserves_transport_metadata(job_id, trigger, submitted_at, deadline):
    request = PluginRequest(job_id, "plugin", trigger, {}, submitted_at, deadline)
    assert request.job_id == job_id
    assert request.trigger == trigger
    assert request.submitted_at_ms == submitted_at
    assert request.deadline_at_ms == deadline
