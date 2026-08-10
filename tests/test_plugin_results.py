from __future__ import annotations

import pytest

from omnitensor.plugins.protocol import PluginProgress, PluginResult, PluginResultStatus
from omnitensor.plugins.results import (
    MAX_RETAINED_JOBS_LIMIT,
    JobResultError,
    JobResultStore,
)

OWNER = ":1.42"
OTHER = ":1.99"


def progress(job_id="job-1", fraction=0.5, stage="infer"):
    return PluginProgress(job_id, stage, fraction, "working", 1_700_000_000_000)


def result(job_id="job-1", status=PluginResultStatus.SUCCEEDED):
    return PluginResult(job_id, status, {"verdict": "ok"}, "done", 1_700_000_000_000)


def test_progress_is_retrievable_by_its_owner():
    store = JobResultStore()
    store.record_progress("job-1", OWNER, progress())

    record = store.get("job-1", OWNER)

    assert record.progress.fraction == 0.5
    assert record.result is None
    assert record.terminal is False


def test_a_terminal_result_is_retrievable_and_keeps_the_last_progress():
    store = JobResultStore()
    store.record_progress("job-1", OWNER, progress(fraction=0.9))
    store.record_result("job-1", OWNER, result())

    record = store.get("job-1", OWNER)

    assert record.terminal is True
    assert record.result.status is PluginResultStatus.SUCCEEDED
    assert record.progress.fraction == 0.9


def test_progress_is_replaced_rather_than_accumulated():
    """Keeping every update turns one slow job into unbounded growth."""
    store = JobResultStore()
    for fraction in (0.1, 0.4, 0.8):
        store.record_progress("job-1", OWNER, progress(fraction=fraction))

    assert store.get("job-1", OWNER).progress.fraction == 0.8
    assert len(store) == 1


def test_a_late_progress_update_cannot_reopen_a_finished_job():
    store = JobResultStore()
    store.record_result("job-1", OWNER, result())
    store.record_progress("job-1", OWNER, progress(fraction=0.2))

    record = store.get("job-1", OWNER)

    assert record.terminal is True
    assert record.progress is None


def test_another_caller_gets_the_same_answer_as_for_a_job_that_never_existed():
    """Distinguishing them would confirm which job ids are real."""
    store = JobResultStore()
    store.record_result("job-1", OWNER, result())

    with pytest.raises(JobResultError) as foreign:
        store.get("job-1", OTHER)
    with pytest.raises(JobResultError) as absent:
        store.get("job-absent", OTHER)

    assert foreign.value.code == absent.value.code == "job-not-found"
    assert foreign.value.detail == "no such job: job-1"
    assert absent.value.detail == "no such job: job-absent"


def test_an_expired_result_is_no_longer_retrievable():
    now = [1000.0]
    store = JobResultStore(result_ttl_seconds=60.0, clock=lambda: now[0])
    store.record_result("job-1", OWNER, result())

    now[0] += 61.0

    with pytest.raises(JobResultError, match="job-not-found"):
        store.get("job-1", OWNER)
    assert len(store) == 0


def test_a_result_within_its_lifetime_survives():
    now = [1000.0]
    store = JobResultStore(result_ttl_seconds=60.0, clock=lambda: now[0])
    store.record_result("job-1", OWNER, result())
    now[0] += 59.0
    assert store.get("job-1", OWNER).terminal is True


def test_retention_is_bounded_and_evicts_the_oldest_first():
    """A burst must not evict results a caller is still waiting on."""
    now = [1000.0]
    store = JobResultStore(max_retained_jobs=2, clock=lambda: now[0])
    for index in range(3):
        now[0] += 1.0
        store.record_result(f"job-{index}", OWNER, result(f"job-{index}"))

    with pytest.raises(JobResultError):
        store.get("job-0", OWNER)
    assert store.get("job-1", OWNER).terminal
    assert store.get("job-2", OWNER).terminal


def test_forgetting_a_job_removes_it():
    store = JobResultStore()
    store.record_result("job-1", OWNER, result())

    assert store.forget("job-1", OWNER) is True
    assert store.forget("job-1", OWNER) is False
    with pytest.raises(JobResultError):
        store.get("job-1", OWNER)


def test_another_caller_cannot_forget_a_job():
    store = JobResultStore()
    store.record_result("job-1", OWNER, result())

    assert store.forget("job-1", OTHER) is False
    assert store.get("job-1", OWNER).terminal is True


@pytest.mark.parametrize("job_id", ["", None, 7, "x" * 121])
def test_an_invalid_job_id_is_refused(job_id):
    with pytest.raises(JobResultError) as excinfo:
        JobResultStore().get(job_id, OWNER)
    assert excinfo.value.code == "job-id-invalid"


@pytest.mark.parametrize("owner", ["", None, 7, "x" * 201])
def test_an_invalid_owner_is_refused(owner):
    with pytest.raises(JobResultError) as excinfo:
        JobResultStore().get("job-1", owner)
    assert excinfo.value.code == "owner-invalid"


def test_a_non_progress_value_is_refused():
    with pytest.raises(JobResultError, match="progress-invalid"):
        JobResultStore().record_progress("job-1", OWNER, {"fraction": 0.5})


def test_a_non_result_value_is_refused():
    with pytest.raises(JobResultError, match="result-invalid"):
        JobResultStore().record_result("job-1", OWNER, {"status": "succeeded"})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"max_retained_jobs": 0}, "max_retained_jobs"),
        ({"max_retained_jobs": MAX_RETAINED_JOBS_LIMIT + 1}, "max_retained_jobs"),
        ({"max_retained_jobs": True}, "max_retained_jobs"),
        ({"result_ttl_seconds": 0}, "result_ttl_seconds"),
        ({"result_ttl_seconds": "60"}, "result_ttl_seconds"),
    ],
)
def test_store_limits_are_bounded(changes, message):
    with pytest.raises(ValueError, match=message):
        JobResultStore(**changes)


def test_an_empty_store_is_still_truthy():
    store = JobResultStore()
    assert len(store) == 0
    assert (store or JobResultStore()) is store
