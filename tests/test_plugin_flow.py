from __future__ import annotations

import asyncio

import pytest

from omnitensor.plugins.flow import (
    MAX_IDEMPOTENCY_KEY_CHARS,
    MAX_IN_FLIGHT_LIMIT,
    FlowRefusal,
    FlowRefusedError,
    PluginFlowController,
)

PLUGIN = "hardware-health"


def controller(**changes):
    options = {"max_in_flight": 2, "deadline_seconds": 5.0, "max_retries": 0}
    options.update(changes)
    return PluginFlowController(PLUGIN, **options)


def run(coroutine, timeout=10):
    return asyncio.run(asyncio.wait_for(coroutine, timeout=timeout))


async def value(result="collected"):
    return result


def test_an_accepted_operation_returns_its_result():
    async def scenario():
        flow = controller()
        result = await flow.submit("key-1", lambda: value())
        return result, flow.snapshot()

    result, snapshot = run(scenario())

    assert result == "collected"
    assert (snapshot.accepted, snapshot.in_flight) == (1, 0)


def test_backpressure_drops_work_beyond_the_in_flight_bound():
    """Queuing past the bound is unbounded memory, so it is refused instead."""

    async def scenario():
        flow = controller(max_in_flight=1)
        release = asyncio.Event()

        async def blocked():
            await release.wait()
            return "done"

        first = asyncio.create_task(flow.submit("key-1", lambda: blocked()))
        await asyncio.sleep(0)
        with pytest.raises(FlowRefusedError) as excinfo:
            await flow.submit("key-2", lambda: value())
        release.set()
        await first
        return excinfo.value, flow.snapshot()

    error, snapshot = run(scenario())

    assert error.refusal is FlowRefusal.BACKPRESSURE
    assert "already has 1 operations in flight" in error.detail
    assert snapshot.dropped == 1


def test_the_same_key_in_flight_is_coalesced_rather_than_run_twice():
    """Two triggers for one logical request must not both do the work."""
    runs = []

    async def scenario():
        flow = controller()
        release = asyncio.Event()

        async def once():
            runs.append(1)
            await release.wait()
            return "shared"

        first = asyncio.create_task(flow.submit("key-1", lambda: once()))
        await asyncio.sleep(0)
        second = asyncio.create_task(flow.submit("key-1", lambda: once()))
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(first, second)
        return results, flow.snapshot()

    results, snapshot = run(scenario())

    assert results == ["shared", "shared"]
    assert runs == [1], "the operation ran twice for one key"
    assert snapshot.coalesced == 1
    assert snapshot.accepted == 1


def test_a_repeated_key_replays_the_retained_result_without_delivering_again():
    """A retrying caller must not cause a second delivery."""
    deliveries = []

    async def scenario():
        flow = controller()

        async def deliver():
            deliveries.append(1)
            return "delivered"

        first = await flow.submit("key-1", lambda: deliver())
        second = await flow.submit("key-1", lambda: deliver())
        return first, second, flow.snapshot()

    first, second, snapshot = run(scenario())

    assert (first, second) == ("delivered", "delivered")
    assert deliveries == [1], "the operation was delivered twice"
    assert snapshot.replayed == 1


def test_a_retained_result_expires_so_retention_stays_bounded():
    now = [1000.0]
    runs = []

    async def scenario():
        flow = controller(result_ttl_seconds=10.0, clock=lambda: now[0])

        async def work():
            runs.append(1)
            return len(runs)

        first = await flow.submit("key-1", lambda: work())
        now[0] += 11.0
        second = await flow.submit("key-1", lambda: work())
        return first, second

    first, second = run(scenario())

    assert (first, second) == (1, 2)
    assert runs == [1, 1]


def test_retention_evicts_the_least_recently_used_key():
    async def scenario():
        flow = controller(result_retention=2)
        for index in range(3):
            await flow.submit(f"key-{index}", lambda index=index: value(index))
        return flow._completed

    completed = run(scenario())

    assert sorted(completed) == ["key-1", "key-2"]


def test_an_operation_past_its_deadline_is_retried_then_refused():
    attempts = []

    async def scenario():
        flow = controller(deadline_seconds=0.01, max_retries=2)

        async def never():
            attempts.append(1)
            await asyncio.Event().wait()

        with pytest.raises(FlowRefusedError) as excinfo:
            await flow.submit("key-1", lambda: never())
        return excinfo.value, flow.snapshot()

    error, snapshot = run(scenario())

    assert error.refusal is FlowRefusal.RETRIES_EXHAUSTED
    assert len(attempts) == 3, "one initial attempt plus two retries"
    assert (snapshot.retried, snapshot.failed) == (2, 1)


def test_a_retry_that_succeeds_delivers_exactly_once():
    """Only the attempt that succeeds produces a result at all."""
    attempts = []

    async def scenario():
        flow = controller(deadline_seconds=0.02, max_retries=2)

        async def slow_then_fast():
            attempts.append(1)
            if len(attempts) == 1:
                await asyncio.Event().wait()
            return "delivered"

        return await flow.submit("key-1", lambda: slow_then_fast())

    assert run(scenario()) == "delivered"
    assert len(attempts) == 2


def test_a_failing_operation_reports_its_own_error_and_is_not_retained():
    async def scenario():
        flow = controller()

        async def broken():
            raise ValueError("plugin defect")

        with pytest.raises(ValueError, match="plugin defect"):
            await flow.submit("key-1", lambda: broken())
        assert flow.snapshot().in_flight == 0
        return await flow.submit("key-1", lambda: value("recovered"))

    assert run(scenario()) == "recovered"


def test_a_dropped_submission_frees_its_slot_for_the_next_one():
    async def scenario():
        flow = controller(max_in_flight=1)
        await flow.submit("key-1", lambda: value("first"))
        return await flow.submit("key-2", lambda: value("second"))

    assert run(scenario()) == "second"


@pytest.mark.parametrize("key", ["", None, 7, "x" * (MAX_IDEMPOTENCY_KEY_CHARS + 1)])
def test_an_invalid_idempotency_key_is_refused(key):
    async def scenario():
        with pytest.raises(FlowRefusedError) as excinfo:
            await controller().submit(key, lambda: value())
        return excinfo.value

    assert run(scenario()).refusal is FlowRefusal.INVALID_KEY


def test_the_operation_must_be_callable():
    async def scenario():
        with pytest.raises(TypeError, match="callable"):
            await controller().submit("key-1", "not-callable")

    run(scenario())


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"max_in_flight": 0}, "max_in_flight"),
        ({"max_in_flight": MAX_IN_FLIGHT_LIMIT + 1}, "max_in_flight"),
        ({"max_in_flight": True}, "max_in_flight"),
        ({"deadline_seconds": 0}, "deadline_seconds"),
        ({"deadline_seconds": "1"}, "deadline_seconds"),
        ({"max_retries": -1}, "max_retries"),
        ({"max_retries": True}, "max_retries"),
        ({"result_retention": 0}, "result_retention"),
        ({"result_ttl_seconds": 0}, "result_ttl_seconds"),
    ],
)
def test_flow_limits_are_bounded(changes, message):
    with pytest.raises(ValueError, match=message):
        PluginFlowController(PLUGIN, **changes)


def test_cancelling_a_coalesced_caller_does_not_cancel_the_shared_work():
    """One impatient caller must not abort work another is still waiting on."""
    finished = []

    async def scenario():
        flow = controller()
        release = asyncio.Event()

        async def shared():
            await release.wait()
            finished.append(1)
            return "shared"

        first = asyncio.create_task(flow.submit("key-1", lambda: shared()))
        await asyncio.sleep(0)
        second = asyncio.create_task(flow.submit("key-1", lambda: shared()))
        await asyncio.sleep(0)
        second.cancel()
        await asyncio.gather(second, return_exceptions=True)
        release.set()
        return await first

    assert run(scenario()) == "shared"
    assert finished == [1]


def test_a_cancelled_primary_refuses_its_coalesced_callers_instead_of_cancelling_them():
    """OMNI-0420: an unrelated job used to abort silently, with no code."""

    async def scenario():
        subject = controller(deadline_seconds=30.0)
        started = asyncio.Event()
        release = asyncio.Event()

        async def operation():
            started.set()
            await release.wait()
            return "never"

        primary = asyncio.ensure_future(subject.submit("key-1", operation))
        await started.wait()
        coalesced = asyncio.ensure_future(subject.submit("key-1", operation))
        await asyncio.sleep(0)

        primary.cancel()
        with pytest.raises(asyncio.CancelledError):
            await primary

        # The second caller is refused with a stable code, in its own context,
        # and is emphatically not itself cancelled.
        with pytest.raises(FlowRefusedError) as refused:
            await coalesced
        assert refused.value.code == "primary-cancelled"
        assert not coalesced.cancelled()
        release.set()

    asyncio.run(scenario())
