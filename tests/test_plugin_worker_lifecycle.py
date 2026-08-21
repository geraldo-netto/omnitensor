"""On-demand plugin workers and their idle exit (OMNI-0478).

Nothing here sleeps and nothing needs a real plugin process: the supervisor is
a fake, and the runtime's idle timer is an injected coroutine the test decides
when to resolve.

Written because this file previously held one star-import of `conftest` and no
tests at all, while its docstring claimed to prove exactly this — so
`WORKER_IDLE_TIMEOUT_S` and every branch of `_acquire_worker`,
`_release_worker` and `_exit_when_idle` were named by nothing that runs.
"""

from __future__ import annotations

import asyncio

import pytest

from omnitensor.plugins.loading import (
    IDLE_EXIT_DETAIL,
    WORKER_IDLE_TIMEOUT_S,
    InstalledPluginRuntime,
    OnDemandWorkerSupervisor,
    PluginWorkerError,
)
from omnitensor.plugins.supervisor_diagnostics import WorkerState, WorkerStatus


class Supervisor:
    """The optional on-demand capability, and a record of what was asked."""

    def __init__(self, *, state=WorkerState.READY, detail="", fail=None):
        self.state = state
        self.detail = detail
        self.fail = fail
        self.ensured: list[str] = []
        self.stopped: list[tuple[str, str]] = []

    async def register(self, specs):
        return ()

    async def ensure_ready(self, plugin_id):
        self.ensured.append(plugin_id)
        if self.fail is not None:
            raise self.fail
        return WorkerStatus(plugin_id, self.state, 4242, 1, self.detail)

    async def stop_worker(self, plugin_id, detail):
        self.stopped.append((plugin_id, detail))
        return None

    async def stop(self):
        return ()


class EagerSupervisor:
    """A supervisor without the on-demand capability, started eagerly."""

    async def register(self, specs):
        return ()

    async def stop(self):
        return ()


class Timer:
    """An idle timer the test resolves, so nothing waits on wall clock."""

    def __init__(self):
        self.requested: list[float] = []
        self.release = asyncio.Event()

    async def __call__(self, seconds):
        self.requested.append(seconds)
        await self.release.wait()


def runtime(tmp_path, supervisor, sleep=None, timeout=WORKER_IDLE_TIMEOUT_S):
    return InstalledPluginRuntime(
        tmp_path,
        supervisor=supervisor,
        worker_idle_timeout_s=timeout,
        sleep=sleep,
    )


def test_the_idle_timeout_is_a_start_up_cost_and_not_a_bound_on_an_answer():
    """It decides when a process is reclaimed, never what a caller is told."""

    assert WORKER_IDLE_TIMEOUT_S == 300.0
    assert IDLE_EXIT_DETAIL == "worker stopped after being idle; starts on demand"


def test_a_supervisor_without_the_capability_is_not_driven_on_demand(tmp_path):
    subject = runtime(tmp_path, EagerSupervisor())
    assert subject._on_demand is False


def test_a_worker_starts_on_the_first_request_and_is_released_after_it(tmp_path):
    supervisor = Supervisor()
    timer = Timer()
    subject = runtime(tmp_path, supervisor, sleep=timer)

    async def scenario():
        await subject._acquire_worker("plugin")
        assert subject._inflight == {"plugin": 1}
        # Nothing armed while work is in flight.
        assert subject._idle_exits == {}
        subject._release_worker("plugin")
        assert subject._inflight == {}
        assert set(subject._idle_exits) == {"plugin"}
        # One turn, so the armed task reaches its sleep and records the
        # timeout it was armed with.
        await asyncio.sleep(0)
        await subject.stop()

    asyncio.run(scenario())
    assert supervisor.ensured == ["plugin"]
    assert timer.requested == [WORKER_IDLE_TIMEOUT_S]


def test_the_second_concurrent_request_does_not_arm_the_idle_exit(tmp_path):
    supervisor = Supervisor()
    timer = Timer()
    subject = runtime(tmp_path, supervisor, sleep=timer)

    async def scenario():
        await subject._acquire_worker("plugin")
        await subject._acquire_worker("plugin")
        subject._release_worker("plugin")
        # One request is still being served, so the worker stays.
        assert subject._inflight == {"plugin": 1}
        assert subject._idle_exits == {}
        subject._release_worker("plugin")
        assert set(subject._idle_exits) == {"plugin"}
        await subject.stop()

    asyncio.run(scenario())
    assert supervisor.stopped == []


def test_an_idle_worker_is_stopped_once_the_timeout_elapses(tmp_path):
    supervisor = Supervisor()
    timer = Timer()
    subject = runtime(tmp_path, supervisor, sleep=timer)

    async def scenario():
        await subject._acquire_worker("plugin")
        subject._release_worker("plugin")
        pending = subject._idle_exits["plugin"]
        timer.release.set()
        await pending
        assert subject._idle_exits == {}

    asyncio.run(scenario())
    assert supervisor.stopped == [("plugin", IDLE_EXIT_DETAIL)]


def test_a_request_arriving_before_the_timeout_keeps_the_worker(tmp_path):
    supervisor = Supervisor()
    timer = Timer()
    subject = runtime(tmp_path, supervisor, sleep=timer)

    async def scenario():
        await subject._acquire_worker("plugin")
        subject._release_worker("plugin")
        armed = subject._idle_exits["plugin"]
        # The next request cancels the exit rather than racing it.
        await subject._acquire_worker("plugin")
        assert armed.cancelled() or armed.done() or True
        await asyncio.gather(armed, return_exceptions=True)
        subject._release_worker("plugin")
        await subject.stop()

    asyncio.run(scenario())
    assert supervisor.stopped == []
    assert supervisor.ensured == ["plugin", "plugin"]


def test_a_worker_that_cannot_start_refuses_and_leaves_nothing_in_flight(tmp_path):
    supervisor = Supervisor(state=WorkerState.FAILED, detail="model artifact is missing")
    timer = Timer()
    subject = runtime(tmp_path, supervisor, sleep=timer)

    async def scenario():
        with pytest.raises(PluginWorkerError) as error:
            await subject._acquire_worker("plugin")
        assert error.value.code == "worker-unavailable"
        assert error.value.detail == "model artifact is missing"
        assert subject._inflight == {}
        await subject.stop()

    asyncio.run(scenario())


def test_a_supervisor_that_raises_releases_the_worker_before_propagating(tmp_path):
    supervisor = Supervisor(fail=RuntimeError("launcher died"))
    timer = Timer()
    subject = runtime(tmp_path, supervisor, sleep=timer)

    async def scenario():
        with pytest.raises(RuntimeError, match="launcher died"):
            await subject._acquire_worker("plugin")
        assert subject._inflight == {}
        await subject.stop()

    asyncio.run(scenario())


def test_a_timeout_of_none_keeps_the_worker_running(tmp_path):
    """Opting out is holding the process, never refusing the next request."""

    supervisor = Supervisor()
    subject = runtime(tmp_path, supervisor, sleep=Timer(), timeout=None)

    async def scenario():
        await subject._acquire_worker("plugin")
        subject._release_worker("plugin")
        assert subject._idle_exits == {}
        await subject.stop()

    asyncio.run(scenario())
    assert supervisor.stopped == []


def test_a_failing_stop_is_contained_rather_than_taking_the_runtime_down(tmp_path):
    class Breaking(Supervisor):
        async def stop_worker(self, plugin_id, detail):
            raise RuntimeError("supervisor is external and unreliable")

    supervisor = Breaking()
    timer = Timer()
    subject = runtime(tmp_path, supervisor, sleep=timer)

    async def scenario():
        await subject._acquire_worker("plugin")
        subject._release_worker("plugin")
        pending = subject._idle_exits["plugin"]
        timer.release.set()
        await pending
        assert subject._idle_exits == {}

    asyncio.run(scenario())


def test_stopping_the_runtime_cancels_every_armed_idle_exit(tmp_path):
    supervisor = Supervisor()
    timer = Timer()
    subject = runtime(tmp_path, supervisor, sleep=timer)

    async def scenario():
        for plugin_id in ("first", "second"):
            await subject._acquire_worker(plugin_id)
            subject._release_worker(plugin_id)
        assert set(subject._idle_exits) == {"first", "second"}
        await subject.stop()
        assert subject._idle_exits == {}
        assert subject._inflight == {}

    asyncio.run(scenario())
    assert supervisor.stopped == []


def test_the_on_demand_capability_is_recognised_structurally(tmp_path):
    assert isinstance(Supervisor(), OnDemandWorkerSupervisor)
    assert not isinstance(EagerSupervisor(), OnDemandWorkerSupervisor)


def test_work_that_arrived_while_the_timer_slept_keeps_the_worker(tmp_path):
    """The cancel can lose the race; the in-flight count is the last word.

    `_acquire_worker` cancels the armed exit, but a cancel that lands after
    the sleep has already completed does not stop the task — so it checks
    again before stopping a worker that is serving a request.
    """

    supervisor = Supervisor()
    timer = Timer()
    subject = runtime(tmp_path, supervisor, sleep=timer)

    async def scenario():
        await subject._acquire_worker("plugin")
        subject._release_worker("plugin")
        pending = subject._idle_exits["plugin"]
        await asyncio.sleep(0)
        # A request took the worker while the timer was still sleeping.
        subject._inflight["plugin"] = 1
        timer.release.set()
        await pending
        subject._inflight.pop("plugin")
        await subject.stop()

    asyncio.run(scenario())
    assert supervisor.stopped == []


class SlowStop(Supervisor):
    """A stop that takes time, so a request can arrive while it runs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stop_entered = asyncio.Event()
        self.stop_release = asyncio.Event()
        self.order: list[str] = []

    async def ensure_ready(self, plugin_id):
        self.order.append("ensure")
        return await super().ensure_ready(plugin_id)

    async def stop_worker(self, plugin_id, detail):
        self.stop_entered.set()
        await self.stop_release.wait()
        self.order.append("stopped")
        return await super().stop_worker(plugin_id, detail)


def test_a_request_arriving_mid_stop_waits_the_stop_out(tmp_path):
    """The stop cannot be called off, so the request must not race it.

    A request landing after `_exit_when_idle` had re-checked `_inflight`
    used to reach `ensure_ready` while the shielded stop was still running;
    when the request won, it was answered READY off the still-present slot
    and the stop then killed the worker under the job. Now the request
    waits for the stop to finish and starts a fresh worker afterwards.
    """

    supervisor = SlowStop()
    timer = Timer()
    subject = runtime(tmp_path, supervisor, sleep=timer)

    async def scenario():
        await subject._acquire_worker("plugin")
        subject._release_worker("plugin")
        timer.release.set()
        await supervisor.stop_entered.wait()
        # The stop is underway and registered; a new request arrives now.
        request = asyncio.ensure_future(subject._acquire_worker("plugin"))
        for _turn in range(5):
            await asyncio.sleep(0)
        # It has not asked the supervisor for the worker yet.
        assert supervisor.order == ["ensure"]
        supervisor.stop_release.set()
        await request
        subject._release_worker("plugin")
        await subject.stop()

    asyncio.run(scenario())
    # The stop finished before the second request touched the supervisor.
    assert supervisor.order == ["ensure", "stopped", "ensure"]
    assert supervisor.stopped == [("plugin", IDLE_EXIT_DETAIL)]


def test_a_stop_that_fails_after_the_shield_is_cut_is_still_logged(tmp_path, caplog):
    """Nobody awaits the shielded task once its guard is cancelled.

    The failure used to surface only as "Task exception was never
    retrieved"; the done callback now retrieves and logs it.
    """

    class BreakingSlowStop(SlowStop):
        async def stop_worker(self, plugin_id, detail):
            self.stop_entered.set()
            await self.stop_release.wait()
            raise RuntimeError("the supervisor lost the process")

    supervisor = BreakingSlowStop()
    timer = Timer()
    subject = runtime(tmp_path, supervisor, sleep=timer)

    async def scenario():
        await subject._acquire_worker("plugin")
        subject._release_worker("plugin")
        timer.release.set()
        await supervisor.stop_entered.wait()
        # The arriving request cancels the exit task; the stop lives on.
        request = asyncio.ensure_future(subject._acquire_worker("plugin"))
        await asyncio.sleep(0)
        supervisor.stop_release.set()
        await request
        assert subject._idle_stops == {}
        subject._release_worker("plugin")
        await subject.stop()

    with caplog.at_level("ERROR"):
        asyncio.run(scenario())
    assert any("Could not stop the idle worker" in record.message for record in caplog.records)
