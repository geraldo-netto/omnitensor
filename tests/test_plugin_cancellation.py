from __future__ import annotations

import asyncio
import json

import pytest

from omnitensor.plugins.cancellation import (
    JOURNAL_VERSION,
    Cancellation,
    CancellationReason,
    JobCancellationRegistry,
    JobCancellationToken,
    JobCancelledError,
)

EVERY_REASON = [reason for reason in CancellationReason]


def run(coroutine, timeout=10):
    return asyncio.run(asyncio.wait_for(coroutine, timeout=timeout))


def test_a_fresh_token_is_not_cancelled():
    token = JobCancellationToken("job-1")
    assert token.cancelled is False
    assert token.cancellation is None
    assert token.raise_if_cancelled() is None


@pytest.mark.parametrize("reason", EVERY_REASON)
def test_every_source_cancels_the_same_way(reason):
    """Caller, policy, deadline, worker, and shutdown look identical to a stage."""
    token = JobCancellationToken("job-1")

    assert token.cancel(reason, "because") is True

    assert token.cancelled is True
    assert token.cancellation == Cancellation("job-1", reason, "because")
    with pytest.raises(JobCancelledError) as excinfo:
        token.raise_if_cancelled()
    assert excinfo.value.code == str(reason)


def test_the_first_reason_wins():
    """A job its caller withdrew did not stop because a deadline later passed."""
    token = JobCancellationToken("job-1")

    assert token.cancel(CancellationReason.CALLER, "withdrawn") is True
    assert token.cancel(CancellationReason.DEADLINE, "too slow") is False

    assert token.cancellation.reason is CancellationReason.CALLER
    assert token.cancellation.detail == "withdrawn"


def test_a_cancellation_without_a_detail_still_explains_itself():
    token = JobCancellationToken("job-1")
    token.cancel(CancellationReason.SHUTDOWN)
    assert token.cancellation.detail == "service-shutdown"


def test_waiting_returns_the_cancellation():
    async def scenario():
        token = JobCancellationToken("job-1")
        waiter = asyncio.create_task(token.wait())
        await asyncio.sleep(0)
        token.cancel(CancellationReason.WORKER, "worker exited")
        return await waiter

    assert run(scenario()).reason is CancellationReason.WORKER


def test_guard_returns_the_result_when_nothing_cancels():
    async def scenario():
        token = JobCancellationToken("job-1")
        return await token.guard(lambda: _value("collected"))

    assert run(scenario()) == "collected"


def test_guard_refuses_to_start_an_already_cancelled_job():
    started = []

    async def scenario():
        token = JobCancellationToken("job-1")
        token.cancel(CancellationReason.POLICY, "profile disabled")

        async def work():
            started.append(1)

        with pytest.raises(JobCancelledError):
            await token.guard(lambda: work())

    run(scenario())
    assert started == [], "work started for an already cancelled job"


def test_guard_stops_a_running_stage_as_soon_as_it_is_cancelled():
    finished = []
    stopped = []

    async def scenario():
        token = JobCancellationToken("job-1")

        async def work():
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                stopped.append(1)
                raise
            finished.append(1)

        guarded = asyncio.create_task(token.guard(lambda: work()))
        await asyncio.sleep(0.01)
        token.cancel(CancellationReason.SHUTDOWN, "stopping")
        with pytest.raises(JobCancelledError) as excinfo:
            await guarded
        return excinfo.value

    error = run(scenario())

    assert error.cancellation.reason is CancellationReason.SHUTDOWN
    assert stopped == [1]
    assert finished == [], "the stage ran to completion after cancellation"


def test_guard_propagates_a_stage_failure_unchanged():
    async def scenario():
        token = JobCancellationToken("job-1")

        async def broken():
            raise ValueError("stage defect")

        with pytest.raises(ValueError, match="stage defect"):
            await token.guard(lambda: broken())

    run(scenario())


def test_the_registry_tracks_cancels_and_releases_jobs():
    registry = JobCancellationRegistry()
    token = registry.track("job-1", "hardware-health")

    assert registry.get("job-1") is token
    assert len(registry) == 1
    assert registry.cancel("job-1", CancellationReason.CALLER, "withdrawn") is True
    assert token.cancellation.reason is CancellationReason.CALLER

    registry.release("job-1")
    assert registry.get("job-1") is None
    assert list(registry) == []


def test_cancelling_an_unknown_job_is_not_an_error():
    assert JobCancellationRegistry().cancel("absent", CancellationReason.CALLER) is False


def test_cancel_all_stops_every_live_job():
    """Shutdown and a lost worker both need to stop everything at once."""
    registry = JobCancellationRegistry()
    tokens = [registry.track(f"job-{index}") for index in range(3)]

    assert registry.cancel_all(CancellationReason.SHUTDOWN, "stopping") == 3
    assert registry.cancel_all(CancellationReason.SHUTDOWN, "stopping") == 0
    assert all(token.cancellation.reason is CancellationReason.SHUTDOWN for token in tokens)


def test_tracking_the_same_job_twice_is_refused():
    registry = JobCancellationRegistry()
    registry.track("job-1")
    with pytest.raises(ValueError, match="already tracked"):
        registry.track("job-1")


@pytest.mark.parametrize("job_id", ["", None, 7])
def test_an_invalid_job_id_is_refused(job_id):
    with pytest.raises(ValueError, match="job_id"):
        JobCancellationRegistry().track(job_id)


def test_the_tracked_job_count_is_bounded():
    registry = JobCancellationRegistry(max_tracked_jobs=2)
    registry.track("job-1")
    registry.track("job-2")
    with pytest.raises(ValueError, match="at most 2 jobs"):
        registry.track("job-3")


@pytest.mark.parametrize("bound", [0, -1, True, "2"])
def test_the_tracked_job_bound_is_validated(bound):
    with pytest.raises(ValueError, match="max_tracked_jobs"):
        JobCancellationRegistry(max_tracked_jobs=bound)


def test_an_interrupted_job_is_reported_rather_than_left_claiming_to_run(tmp_path):
    """Nothing will ever finish it, so claiming it is running is a lie."""
    journal = tmp_path / "in-flight.json"
    first = JobCancellationRegistry(journal)
    first.track("job-1", "hardware-health")
    first.track("job-2", "visual-library")
    first.release("job-2")

    recovered = JobCancellationRegistry(journal).recover()

    assert [item.job_id for item in recovered] == ["job-1"]
    assert recovered[0].reason is CancellationReason.INTERRUPTED
    assert "hardware-health" in recovered[0].detail


def test_recovery_clears_the_journal_so_one_crash_does_not_haunt_every_start(tmp_path):
    journal = tmp_path / "in-flight.json"
    JobCancellationRegistry(journal).track("job-1")

    registry = JobCancellationRegistry(journal)
    assert len(registry.recover()) == 1
    assert registry.recover() == ()
    assert not journal.exists()


def test_releasing_the_last_job_removes_the_journal(tmp_path):
    journal = tmp_path / "in-flight.json"
    registry = JobCancellationRegistry(journal)
    registry.track("job-1")
    assert journal.exists()
    registry.release("job-1")
    assert not journal.exists()


def test_nothing_to_recover_without_a_journal(tmp_path):
    assert JobCancellationRegistry(tmp_path / "absent.json").recover() == ()
    assert JobCancellationRegistry().recover() == ()


@pytest.mark.parametrize(
    "content",
    [
        "{nope",
        json.dumps({"version": 99, "jobs": {"job-1": ""}}),
        json.dumps({"version": JOURNAL_VERSION, "jobs": "not-an-object"}),
        json.dumps(["not", "an", "object"]),
    ],
)
def test_an_unusable_journal_starts_clean_rather_than_refusing_to_start(
    tmp_path, content
):
    journal = tmp_path / "in-flight.json"
    journal.write_text(content, encoding="utf-8")
    assert JobCancellationRegistry(journal).recover() == ()


def test_a_registry_without_a_journal_still_tracks_in_memory():
    registry = JobCancellationRegistry()
    registry.track("job-1")
    assert registry.cancel("job-1", CancellationReason.DEADLINE) is True


async def _value(result):
    return result
