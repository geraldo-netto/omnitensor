from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable
from typing import TypeVar

import pytest

from omnitensor.jobs import (
    DEFAULT_MAX_JOB_REQUEST_BYTES,
    JobDispatchError,
    JobSubmissionService,
    PredicateJobAuthorizer,
)
from omnitensor.registry import validate_document

T = TypeVar("T")


def run_scenario(coroutine: Awaitable[T]) -> T:
    async def bounded() -> T:
        return await asyncio.wait_for(coroutine, timeout=0.5)

    return asyncio.run(bounded())


def submit_document(request_id="request-1", workload_id="visual-library", payload=None):
    return {
        "version": 1,
        "requestId": request_id,
        "workloadId": workload_id,
        "payload": payload or {},
    }


def cancel_document(job_id, request_id="cancel-1"):
    return {"version": 1, "requestId": request_id, "jobId": job_id}


def decode(text):
    document = json.loads(text)
    assert validate_document("runtime-job-acknowledgement.schema.json", document) == []
    return document


def allows_all():
    return PredicateJobAuthorizer(lambda action, workload_id: True)


class BlockingDispatcher:
    def __init__(self):
        self.calls = []
        self.cancelled = []
        self.release = asyncio.Event()

    def dispatch(self, job_id, workload_id, payload):
        self.calls.append((job_id, workload_id, payload))

        async def operation():
            try:
                await self.release.wait()
                return payload
            except asyncio.CancelledError:
                self.cancelled.append(job_id)
                raise

        return operation()


class ImmediateDispatcher:
    def dispatch(self, job_id, workload_id, payload):
        async def operation():
            return (job_id, workload_id, payload)

        return operation()


def test_submit_accepts_authorized_bounded_job_and_removes_completed_task():
    async def scenario():
        dispatcher = BlockingDispatcher()
        service = JobSubmissionService(
            dispatcher,
            allows_all(),
            id_factory=lambda: "job-1",
            clock_ms=lambda: 123,
        )
        request = submit_document(payload={"tensor": [1, 2]})

        reply = decode(await service.submit_job_text(json.dumps(request)))

        assert reply == {
            "version": 1,
            "requestId": "request-1",
            "jobId": "job-1",
            "status": "accepted",
            "code": "job-accepted",
            "message": "Job accepted",
            "timestamp": 123,
        }
        assert service.active_job_ids() == ("job-1",)
        assert dispatcher.calls == [
            ("job-1", "visual-library", {"tensor": [1, 2]})
        ]
        request["payload"]["tensor"].append(3)
        assert dispatcher.calls[0][2] == {"tensor": [1, 2]}
        dispatcher.release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert service.active_job_ids() == ()

    run_scenario(scenario())


def test_cancel_authorized_active_job_and_not_found_are_stable():
    async def scenario():
        dispatcher = BlockingDispatcher()
        service = JobSubmissionService(
            dispatcher,
            allows_all(),
            id_factory=lambda: "job-cancel",
        )
        await service.submit_job_text(json.dumps(submit_document()))
        await asyncio.sleep(0)

        cancelled = decode(
            await service.cancel_job_text(json.dumps(cancel_document("job-cancel")))
        )
        missing = decode(
            await service.cancel_job_text(json.dumps(cancel_document("job-cancel", "again")))
        )

        assert cancelled["status"] == "cancelled"
        assert cancelled["code"] == "job-cancelled"
        assert cancelled["jobId"] == "job-cancel"
        assert cancelled["message"] == "Job cancelled"
        assert dispatcher.cancelled == ["job-cancel"]
        assert service.active_job_ids() == ()
        assert missing["status"] == "not-found"
        assert missing["code"] == "job-not-found"
        assert missing["jobId"] == "job-cancel"
        assert missing["message"] == "Active job was not found"

    run_scenario(scenario())


def test_default_policy_denies_submission_before_dispatch():
    async def scenario():
        dispatcher = BlockingDispatcher()
        service = JobSubmissionService(dispatcher, clock_ms=lambda: 0)

        reply = decode(
            await service.submit_job_text(json.dumps(submit_document()))
        )

        assert reply["status"] == "rejected"
        assert reply["code"] == "not-authorized"
        assert reply["message"] == (
            "Job submission is not authorized for this workload"
        )
        assert reply["timestamp"] == 1
        assert dispatcher.calls == []

    run_scenario(scenario())


def test_cancel_authorization_is_evaluated_against_active_workload():
    async def scenario():
        dispatcher = BlockingDispatcher()
        actions = []

        def policy(action, workload_id):
            actions.append((action, workload_id))
            return action == "submit"

        service = JobSubmissionService(
            dispatcher,
            PredicateJobAuthorizer(policy),
            id_factory=lambda: "owned-job",
        )
        await service.submit_job_text(json.dumps(submit_document()))

        reply = decode(
            await service.cancel_job_text(json.dumps(cancel_document("owned-job")))
        )

        assert reply["code"] == "not-authorized"
        assert reply["jobId"] == "owned-job"
        assert reply["message"] == (
            "Job cancellation is not authorized for this workload"
        )
        assert actions == [
            ("submit", "visual-library"),
            ("cancel", "visual-library"),
        ]
        assert service.active_job_ids() == ("owned-job",)
        dispatcher.release.set()
        await asyncio.sleep(0)
        await service.stop()

    run_scenario(scenario())


def test_capacity_is_fail_fast_and_recovers_after_completion():
    async def scenario():
        dispatcher = BlockingDispatcher()
        ids = iter(("first-job", "second-job"))
        service = JobSubmissionService(
            dispatcher,
            allows_all(),
            max_active_jobs=1,
            id_factory=lambda: next(ids),
        )
        first = decode(await service.submit_job_text(json.dumps(submit_document())))
        full = decode(
            await service.submit_job_text(
                json.dumps(submit_document(request_id="request-2"))
            )
        )
        assert first["status"] == "accepted"
        assert full["code"] == "capacity-exceeded"
        assert full["message"] == "Active job capacity is exhausted"
        assert len(dispatcher.calls) == 1

        dispatcher.release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        second = decode(
            await service.submit_job_text(
                json.dumps(submit_document(request_id="request-3"))
            )
        )
        assert second["jobId"] == "second-job"
        await asyncio.sleep(0)

    run_scenario(scenario())


class RejectingDispatcher:
    def dispatch(self, job_id, workload_id, payload):
        raise JobDispatchError("artifact-not-ready", "Verified artifact is unavailable")


def test_dispatch_admission_error_is_versioned_and_preserves_request_id():
    async def scenario():
        service = JobSubmissionService(RejectingDispatcher(), allows_all())
        reply = decode(
            await service.submit_job_text(
                json.dumps(submit_document(request_id="admission-1"))
            )
        )

        assert reply["requestId"] == "admission-1"
        assert reply["status"] == "rejected"
        assert reply["code"] == "artifact-not-ready"
        assert reply["message"] == "Verified artifact is unavailable"
        assert service.active_job_ids() == ()

    run_scenario(scenario())


@pytest.mark.parametrize(
    "text, code, request_id",
    [
        ("{", "invalid-json", "invalid"),
        ("NaN", "invalid-json", "invalid"),
        ("[]", "invalid-request", "invalid"),
        (json.dumps({"requestId": "kept"}), "invalid-request", "kept"),
        (json.dumps({"version": 2, "requestId": "v2"}), "invalid-request", "v2"),
        ("\ud800", "invalid-request", "invalid"),
    ],
)
def test_invalid_submit_requests_return_stable_contract(text, code, request_id):
    async def scenario():
        service = JobSubmissionService(ImmediateDispatcher(), allows_all())
        reply = decode(await service.submit_job_text(text))
        assert reply["code"] == code
        assert reply["requestId"] == request_id
        assert reply["jobId"] is None

    run_scenario(scenario())


def test_request_byte_bound_precedes_json_parsing():
    async def scenario():
        service = JobSubmissionService(
            ImmediateDispatcher(), allows_all(), max_request_bytes=16
        )
        reply = decode(await service.submit_job_text("{" + "x" * 16))
        assert reply["code"] == "request-too-large"
        assert reply["message"] == "Request exceeds 16 bytes"

    run_scenario(scenario())


def test_invalid_cancel_contract_never_echoes_unsafe_identifiers():
    async def scenario():
        service = JobSubmissionService(ImmediateDispatcher(), allows_all())
        reply = decode(
            await service.cancel_job_text(
                json.dumps({"version": 1, "requestId": "bad id", "jobId": "x"})
            )
        )
        assert reply["requestId"] == "invalid"
        assert reply["code"] == "invalid-request"
        assert reply["jobId"] is None

    run_scenario(scenario())


class UncancellableFuture(asyncio.Future):
    def cancel(self, msg=None):
        return False


class UncancellableDispatcher:
    def __init__(self):
        self.future = None

    def dispatch(self, job_id, workload_id, payload):
        self.future = UncancellableFuture()
        return self.future


def test_cancellation_deadline_is_bounded_and_job_remains_accounted():
    async def scenario():
        dispatcher = UncancellableDispatcher()
        service = JobSubmissionService(
            dispatcher,
            allows_all(),
            id_factory=lambda: "stubborn-job",
            cancel_timeout_seconds=0.001,
        )
        await service.submit_job_text(json.dumps(submit_document()))

        reply = decode(
            await service.cancel_job_text(json.dumps(cancel_document("stubborn-job")))
        )

        assert reply["code"] == "cancel-timeout"
        assert reply["jobId"] == "stubborn-job"
        assert reply["message"] == (
            "Job did not stop within the cancellation deadline"
        )
        assert service.active_job_ids() == ("stubborn-job",)
        dispatcher.future.set_result(None)
        await asyncio.sleep(0)
        assert service.active_job_ids() == ()

    run_scenario(scenario())


def test_stop_cancels_every_active_job_and_is_idempotent():
    async def scenario():
        dispatcher = BlockingDispatcher()
        ids = iter(("job-a", "job-b"))
        service = JobSubmissionService(
            dispatcher,
            allows_all(),
            id_factory=lambda: next(ids),
        )
        await service.submit_job_text(json.dumps(submit_document(request_id="a")))
        await service.submit_job_text(json.dumps(submit_document(request_id="b")))
        await asyncio.sleep(0)

        await service.stop()
        await service.stop()

        assert sorted(dispatcher.cancelled) == ["job-a", "job-b"]
        assert service.active_job_ids() == ()

    run_scenario(scenario())


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"max_active_jobs": True}, "max_active_jobs"),
        ({"max_active_jobs": 0}, "max_active_jobs"),
        ({"max_active_jobs": 1025}, "max_active_jobs"),
        ({"max_request_bytes": 0}, "max_request_bytes"),
        (
            {"max_request_bytes": DEFAULT_MAX_JOB_REQUEST_BYTES * 17},
            "max_request_bytes",
        ),
        ({"cancel_timeout_seconds": True}, "cancel_timeout_seconds"),
        ({"cancel_timeout_seconds": 0}, "cancel_timeout_seconds"),
        ({"cancel_timeout_seconds": math.inf}, "cancel_timeout_seconds"),
        ({"cancel_timeout_seconds": 31}, "cancel_timeout_seconds"),
    ],
)
def test_service_limits_are_validated(changes, match):
    with pytest.raises(ValueError, match=match):
        JobSubmissionService(**changes)


@pytest.mark.parametrize(
    "code, message",
    [
        ("Bad", "job dispatch code"),
        ("-bad", "job dispatch code"),
        ("bad--code", "job dispatch code"),
        ("bad", "job dispatch message"),
    ],
)
def test_dispatch_errors_validate_public_fields(code, message):
    value = "" if message == "job dispatch message" else "message"
    with pytest.raises(ValueError, match=message):
        JobDispatchError(code, value)


def test_generated_job_id_failure_and_collision_are_redacted():
    async def scenario():
        invalid = JobSubmissionService(
            ImmediateDispatcher(), allows_all(), id_factory=lambda: "bad id"
        )
        invalid_reply = decode(
            await invalid.submit_job_text(json.dumps(submit_document()))
        )
        assert invalid_reply["code"] == "internal-error"

        dispatcher = BlockingDispatcher()
        collision = JobSubmissionService(
            dispatcher, allows_all(), id_factory=lambda: "same-job"
        )
        await collision.submit_job_text(json.dumps(submit_document()))
        collision_reply = decode(
            await collision.submit_job_text(
                json.dumps(submit_document(request_id="second"))
            )
        )
        assert collision_reply["requestId"] == "second"
        assert collision_reply["code"] == "internal-error"
        await collision.stop()

    run_scenario(scenario())


def test_a_large_valid_request_keeps_its_request_id_in_the_error_reply():
    """The id lookup used the module default, not the configured bound, so a
    request the service accepts by size still echoed requestId "invalid"."""

    async def scenario():
        service = JobSubmissionService(
            None,
            allows_all(),
            max_request_bytes=DEFAULT_MAX_JOB_REQUEST_BYTES * 4,
            clock_ms=lambda: 0,
        )
        padding = "x" * DEFAULT_MAX_JOB_REQUEST_BYTES
        request = submit_document(payload={"tensor": padding})
        return decode(await service.submit_job_text(json.dumps(request)))

    reply = run_scenario(scenario())

    assert reply["requestId"] == "request-1"
    assert reply["status"] == "rejected"


def test_request_id_recovery_measures_encoded_bytes_not_characters():
    """A multi-byte body is larger on the wire than len() suggests."""

    async def scenario():
        service = JobSubmissionService(None, allows_all(), clock_ms=lambda: 0)
        oversized = submit_document(payload={"tensor": "é" * DEFAULT_MAX_JOB_REQUEST_BYTES})
        text = json.dumps(oversized, ensure_ascii=False)
        assert len(text) < DEFAULT_MAX_JOB_REQUEST_BYTES * 2
        assert len(text.encode()) > DEFAULT_MAX_JOB_REQUEST_BYTES
        return decode(await service.submit_job_text(text))

    reply = run_scenario(scenario())

    assert reply["requestId"] == "invalid"
    assert reply["status"] == "rejected"


class SwallowingDispatcher:
    """A plugin that catches cancellation and reaches a terminal state anyway."""

    def __init__(self, outcome):
        self._outcome = outcome

    def dispatch(self, job_id, workload_id, payload):
        async def operation():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if isinstance(self._outcome, Exception):
                    raise self._outcome from None
                return self._outcome

        return operation()


def test_cancel_reports_completion_when_the_job_finished_as_cancellation_landed():
    """Reporting "cancelled" credited the caller's request with an outcome it
    did not cause."""

    async def scenario():
        service = JobSubmissionService(
            SwallowingDispatcher("finished"),
            allows_all(),
            id_factory=lambda: "job-1",
            clock_ms=lambda: 1,
        )
        await service.submit_job_text(json.dumps(submit_document()))
        await asyncio.sleep(0)  # let the job reach its await point
        return decode(await service.cancel_job_text(json.dumps(cancel_document("job-1"))))

    reply = run_scenario(scenario())

    assert reply["status"] == "rejected"
    assert reply["code"] == "job-already-completed"
    assert validate_document("runtime-job-acknowledgement.schema.json", reply) == []


def test_cancel_reports_failure_when_the_job_failed_as_cancellation_landed():
    async def scenario():
        service = JobSubmissionService(
            SwallowingDispatcher(ValueError("private")),
            allows_all(),
            id_factory=lambda: "job-1",
            clock_ms=lambda: 1,
        )
        await service.submit_job_text(json.dumps(submit_document()))
        await asyncio.sleep(0)  # let the job reach its await point
        return decode(await service.cancel_job_text(json.dumps(cancel_document("job-1"))))

    reply = run_scenario(scenario())

    assert reply["status"] == "rejected"
    assert reply["code"] == "job-already-failed"
    assert validate_document("runtime-job-acknowledgement.schema.json", reply) == []


def test_cancel_still_reports_cancellation_when_the_job_really_stopped():
    async def scenario():
        service = JobSubmissionService(
            BlockingDispatcher(),
            allows_all(),
            id_factory=lambda: "job-1",
            clock_ms=lambda: 1,
        )
        await service.submit_job_text(json.dumps(submit_document()))
        await asyncio.sleep(0)  # let the job reach its await point
        return decode(await service.cancel_job_text(json.dumps(cancel_document("job-1"))))

    reply = run_scenario(scenario())

    assert reply["status"] == "cancelled"
    assert reply["code"] == "job-cancelled"


def test_another_caller_cannot_cancel_a_job_it_did_not_submit():
    """The answer is identical to a job that never existed, deliberately."""
    dispatcher = BlockingDispatcher()

    async def scenario():
        service = JobSubmissionService(dispatcher, allows_all())
        accepted = decode(
            await service.submit_job_text(json.dumps(submit_document()), owner="uid:1000")
        )
        job_id = accepted["jobId"]
        stranger = decode(
            await service.cancel_job_text(
                json.dumps(cancel_document(job_id)), owner="uid:1001"
            )
        )
        absent = decode(
            await service.cancel_job_text(
                json.dumps(cancel_document("job-that-never-existed")), owner="uid:1001"
            )
        )
        dispatcher.release.set()
        owner = decode(
            await service.cancel_job_text(
                json.dumps(cancel_document(job_id, "cancel-2")), owner="uid:1000"
            )
        )
        return stranger, absent, owner

    stranger, absent, owner = run_scenario(scenario())

    assert stranger["status"] == absent["status"] == "not-found"
    assert stranger["code"] == absent["code"] == "job-not-found"
    assert stranger["message"] == absent["message"]
    assert owner["status"] != "not-found"


def test_an_unowned_job_is_cancellable_by_an_unidentified_caller():
    """Without credentials every caller is anonymous, and anonymous is one owner."""
    dispatcher = BlockingDispatcher()

    async def scenario():
        service = JobSubmissionService(dispatcher, allows_all())
        accepted = decode(await service.submit_job_text(json.dumps(submit_document())))
        dispatcher.release.set()
        return decode(
            await service.cancel_job_text(json.dumps(cancel_document(accepted["jobId"])))
        )

    assert run_scenario(scenario())["status"] != "not-found"


def result_document(job_id, request_id="result-1"):
    return {"version": 1, "requestId": request_id, "jobId": job_id}


def decode_result(text):
    document = json.loads(text)
    assert validate_document("runtime-job-result.schema.json", document) == []
    return document


def test_a_submitted_job_can_be_asked_about_until_it_finishes():
    """Submission only answers 'accepted'; without this the outcome is lost."""
    from omnitensor.plugins.results import JobResultStore

    dispatcher = BlockingDispatcher()

    async def scenario():
        service = JobSubmissionService(dispatcher, allows_all(), results=JobResultStore())
        accepted = decode(
            await service.submit_job_text(json.dumps(submit_document()), owner="uid:1000")
        )
        job_id = accepted["jobId"]
        running = decode_result(
            await service.job_result_text(json.dumps(result_document(job_id)), owner="uid:1000")
        )
        dispatcher.release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        finished = decode_result(
            await service.job_result_text(
                json.dumps(result_document(job_id, "result-2")), owner="uid:1000"
            )
        )
        return running, finished

    running, finished = run_scenario(scenario())

    assert running["state"] == "running"
    assert finished["state"] == "succeeded"
    assert finished["jobId"] == running["jobId"]


def test_a_failed_job_reports_why_rather_than_vanishing():
    from omnitensor.plugins.results import JobResultStore

    class FailingDispatcher:
        def dispatch(self, job_id, workload_id, payload):
            async def fail():
                raise RuntimeError("the accelerator went away")

            return fail()

    async def scenario():
        service = JobSubmissionService(FailingDispatcher(), allows_all(), results=JobResultStore())
        accepted = decode(
            await service.submit_job_text(json.dumps(submit_document()), owner="uid:1000")
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return decode_result(
            await service.job_result_text(
                json.dumps(result_document(accepted["jobId"])), owner="uid:1000"
            )
        )

    finished = run_scenario(scenario())

    assert finished["state"] == "failed"
    assert "accelerator went away" in finished["message"]


def test_another_caller_cannot_read_a_job_it_does_not_own():
    """Identical to a job that never existed, so ids are never confirmed."""
    from omnitensor.plugins.results import JobResultStore

    dispatcher = BlockingDispatcher()

    async def scenario():
        service = JobSubmissionService(dispatcher, allows_all(), results=JobResultStore())
        accepted = decode(
            await service.submit_job_text(json.dumps(submit_document()), owner="uid:1000")
        )
        stranger = decode_result(
            await service.job_result_text(
                json.dumps(result_document(accepted["jobId"])), owner="uid:1001"
            )
        )
        absent = decode_result(
            await service.job_result_text(
                json.dumps(result_document("job-never-existed", "result-3")), owner="uid:1001"
            )
        )
        dispatcher.release.set()
        return stranger, absent

    stranger, absent = run_scenario(scenario())

    assert stranger["state"] == absent["state"] == "unknown"
    assert stranger["code"] == absent["code"] == "job-not-found"
    assert stranger["message"] == absent["message"]


def test_a_malformed_result_request_is_answered_not_raised():
    async def scenario():
        service = JobSubmissionService(BlockingDispatcher(), allows_all())
        return [
            decode_result(await service.job_result_text(text, owner="uid:1000"))
            for text in ("not json", json.dumps({"version": 2}), json.dumps({"jobId": "x"}))
        ]

    replies = run_scenario(scenario())

    assert [reply["state"] for reply in replies] == ["unknown"] * 3


def test_a_service_without_a_result_store_still_answers_about_a_live_job():
    """Degrades to liveness rather than failing the call outright."""
    dispatcher = BlockingDispatcher()

    async def scenario():
        service = JobSubmissionService(dispatcher, allows_all())
        accepted = decode(
            await service.submit_job_text(json.dumps(submit_document()), owner="uid:1000")
        )
        reply = decode_result(
            await service.job_result_text(
                json.dumps(result_document(accepted["jobId"])), owner="uid:1000"
            )
        )
        dispatcher.release.set()
        return reply

    assert run_scenario(scenario())["state"] == "running"
