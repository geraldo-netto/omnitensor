from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError

import pytest
from hypothesis import given
from hypothesis import strategies as st

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
    WorkloadPlugin,
)


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
