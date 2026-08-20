"""A call that is still reporting is not a call to give up on.

Two clocks bound a worker call — the flow controller that admitted it and the
worker's own budget — and both used to measure total work. A three-hour
recording, or a document of a few hundred pages, was therefore killed at the
deadline with its answer half made, while the worker was streaming progress
the whole time. What a deadline is worth having for is a worker that has
stopped, and that is what these assert: silence ends a call, and work does not.
"""

from __future__ import annotations

import asyncio

import pytest

from omnitensor.plugins.budgets import (
    WorkerBudgetCode,
    WorkerBudgetEnforcer,
    WorkerBudgetExceededError,
    WorkerBudgetLimits,
)
from omnitensor.plugins.flow import FlowRefusal, FlowRefusedError, PluginFlowController
from omnitensor.plugins.liveness import CallHeartbeat, beating, report

TINY = 0.01


def enforcer(seconds=TINY):
    return WorkerBudgetEnforcer(WorkerBudgetLimits(call_timeout_seconds=seconds))


class TestTheWorkerBudget:
    def test_a_call_that_keeps_reporting_runs_past_its_deadline(self):
        heartbeat = CallHeartbeat()

        async def working():
            # Reporting more often than the deadline, which is what a worker
            # streaming progress through the transport actually does. The call
            # outlives the deadline several times over and is never cut off.
            for _step in range(12):
                await asyncio.sleep(TINY / 4)
                heartbeat.beat()
            return "transcribed"

        assert asyncio.run(enforcer().run(working, heartbeat)) == "transcribed"

    def test_a_call_that_goes_quiet_is_still_cut_off(self):
        heartbeat = CallHeartbeat()

        async def working_then_wedged():
            heartbeat.beat()
            await asyncio.sleep(60)

        with pytest.raises(WorkerBudgetExceededError) as refusal:
            asyncio.run(enforcer().run(working_then_wedged, heartbeat))

        assert refusal.value.code is WorkerBudgetCode.DEADLINE
        assert "reported nothing" in refusal.value.detail

    def test_a_call_that_cannot_report_at_all_is_bounded_as_before(self):
        """No heartbeat means no evidence of life, so the clock is the old one."""

        async def wedged():
            await asyncio.sleep(60)

        with pytest.raises(WorkerBudgetExceededError):
            asyncio.run(enforcer().run(wedged))


class TestTheFlowController:
    def controller(self):
        return PluginFlowController(
            "media-transcription", max_in_flight=2, deadline_seconds=TINY, max_retries=0
        )

    def test_a_job_that_keeps_reporting_is_not_retried_or_refused(self):
        heartbeat = CallHeartbeat()

        async def working():
            for _step in range(12):
                await asyncio.sleep(TINY / 4)
                heartbeat.beat()
            return "answer"

        assert asyncio.run(self.controller().submit("job-1", working, heartbeat)) == "answer"

    def test_a_job_that_says_nothing_is_given_up_on(self):
        async def wedged():
            await asyncio.sleep(60)

        with pytest.raises(FlowRefusedError) as refusal:
            asyncio.run(self.controller().submit("job-1", wedged, CallHeartbeat()))

        assert refusal.value.code == FlowRefusal.RETRIES_EXHAUSTED
        assert "reported nothing" in refusal.value.detail


class TestTheSeamBetweenThem:
    def test_whatever_reports_beats_the_heartbeat_the_clocks_are_watching(self):
        """The transport calls `report()`; nothing in between carries a token."""
        heartbeat = CallHeartbeat()

        with beating(heartbeat):
            report()

        assert heartbeat.consume() is True
        assert heartbeat.consume() is False

    def test_reporting_outside_a_job_is_not_an_error(self):
        report()
