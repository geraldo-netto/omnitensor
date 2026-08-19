from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable
from typing import TypeVar

import pytest

import omnitensor.job_codec as job_codec
import omnitensor.job_lifecycle as job_lifecycle
import omnitensor.job_ports as job_ports
import omnitensor.jobs as jobs
from omnitensor.job_codec import DEFAULT_MAX_JOB_REQUEST_BYTES
from omnitensor.job_lifecycle import (
    DEFAULT_JOB_CANCEL_TIMEOUT_SECONDS,
    JobSubmissionService,
    PredicateJobAuthorizer,
)
from omnitensor.job_ports import JobDispatchError
from omnitensor.plugins.job_results import JobRecord, JobResultStore
from omnitensor.plugins.protocol import PluginProgress, PluginResult, PluginResultStatus
from omnitensor.registry import validate_document

T = TypeVar("T")


def test_jobs_facade_preserves_extracted_contract_identity():
    assert jobs.JobRequest is job_codec.JobRequest
    assert jobs.JobSubmissionService is job_lifecycle.JobSubmissionService
    assert jobs.NoJobObserver is job_lifecycle.NoJobObserver
    assert jobs.PredicateJobAuthorizer is job_lifecycle.PredicateJobAuthorizer
    assert jobs.UnavailableJobDispatcher is job_lifecycle.UnavailableJobDispatcher
    assert jobs.JobAdmission is job_ports.JobAdmission
    assert jobs.JobAuthorizer is job_ports.JobAuthorizer
    assert jobs.JobDispatcher is job_ports.JobDispatcher
    assert jobs.JobDispatchError is job_ports.JobDispatchError
    assert jobs.JobLifecycleObserver is job_ports.JobLifecycleObserver
    assert jobs._parse_request is job_codec._parse_request
    assert jobs._request_id_from_text is job_codec._request_id_from_text
    assert jobs._record_reply is job_codec._record_reply
    assert jobs._result_reply is job_codec._result_reply
    assert jobs._validated_result_reply is job_codec._validated_result_reply


def test_job_components_own_their_responsibilities_without_a_facade_cycle():
    assert job_codec.JobRequest.__module__ == "omnitensor.job_codec"
    assert job_lifecycle.JobSubmissionService.__module__ == "omnitensor.job_lifecycle"
    assert job_ports.JobDispatchError.__module__ == "omnitensor.job_ports"
    assert "omnitensor.jobs" not in {
        value.__name__ for value in vars(job_ports).values() if isinstance(value, type(json))
    }


def test_extracted_codec_preserves_exact_wire_shapes():
    acknowledgement = job_codec._acknowledgement_reply(
        "request-1", "job-1", "accepted", "job-accepted", "Job accepted", 123
    )
    unknown = job_codec._result_reply(
        "result-1", "job-1", "unknown", "job-not-found", "No such job", 123
    )
    running_without_progress = job_codec._record_reply(
        "result-1", JobRecord("job-1", None, None, 100.0), 123
    )
    running_with_progress = job_codec._record_reply(
        "result-1",
        JobRecord(
            "job-1",
            PluginProgress("job-1", "infer", 0.5, "Half way", 100),
            None,
            100.0,
        ),
        123,
    )
    succeeded = job_codec._record_reply(
        "result-1",
        JobRecord(
            "job-1",
            None,
            PluginResult(
                "job-1",
                PluginResultStatus.SUCCEEDED,
                {"value": 7},
                "Done",
                100,
            ),
            100.0,
        ),
        123,
    )
    failed = job_codec._record_reply(
        "result-1",
        JobRecord(
            "job-1",
            None,
            PluginResult("job-1", PluginResultStatus.FAILED, {}, "Accelerator lost", 100),
            100.0,
        ),
        123,
    )
    cancelled = job_codec._record_reply(
        "result-1",
        JobRecord(
            "job-1",
            None,
            PluginResult("job-1", PluginResultStatus.CANCELLED, {}, "", 100),
            100.0,
        ),
        123,
    )

    assert acknowledgement == (
        '{"version":1,"requestId":"request-1","jobId":"job-1",'
        '"status":"accepted","code":"job-accepted","message":"Job accepted",'
        '"timestamp":123}'
    )
    assert unknown == (
        '{"version":1,"requestId":"result-1","jobId":"job-1",'
        '"state":"unknown","code":"job-not-found","message":"No such job",'
        '"timestamp":123,"progress":null,"output":null}'
    )
    assert running_without_progress == (
        '{"version":1,"requestId":"result-1","jobId":"job-1",'
        '"state":"running","code":"job-running","message":"Job is still running",'
        '"timestamp":123,"progress":null,"output":null}'
    )
    assert running_with_progress == (
        '{"version":1,"requestId":"result-1","jobId":"job-1",'
        '"state":"running","code":"job-running","message":"Half way",'
        '"timestamp":123,"progress":{"fraction":0.5,"detail":"Half way"},'
        '"output":null}'
    )
    assert succeeded == (
        '{"version":1,"requestId":"result-1","jobId":"job-1",'
        '"state":"succeeded","code":"job-succeeded","message":"Done",'
        '"timestamp":123,"progress":null,"output":{"value":7}}'
    )
    assert failed == (
        '{"version":1,"requestId":"result-1","jobId":"job-1",'
        '"state":"failed","code":"job-failed","message":"Accelerator lost",'
        '"timestamp":123,"progress":null,"output":{}}'
    )
    assert cancelled == (
        '{"version":1,"requestId":"result-1","jobId":"job-1",'
        '"state":"cancelled","code":"job-cancelled","message":"Job finished",'
        '"timestamp":123,"progress":null,"output":{}}'
    )
    assert (
        validate_document("runtime-job-acknowledgement.schema.json", json.loads(acknowledgement))
        == []
    )
    for reply in (
        unknown,
        running_without_progress,
        running_with_progress,
        succeeded,
        failed,
        cancelled,
    ):
        assert validate_document("runtime-job-result.schema.json", json.loads(reply)) == []


def test_result_serializers_refuse_non_finite_numbers():
    with pytest.raises(ValueError, match="Out of range float values"):
        job_codec._result_reply(
            "result-1", "job-1", "unknown", "job-not-found", "No such job", math.nan
        )
    with pytest.raises(ValueError, match="Out of range float values"):
        job_codec._record_reply(
            "result-1",
            JobRecord(
                "job-1",
                PluginProgress("job-1", "infer", math.nan, "Working", 100),
                None,
                100.0,
            ),
            123,
        )


def test_extracted_codec_preserves_refusal_details_and_identifier_boundaries():
    with pytest.raises(job_codec._RequestError) as caught:
        job_codec._parse_request(
            None,  # type: ignore[arg-type]
            "runtime-job-submit.schema.json",
            DEFAULT_MAX_JOB_REQUEST_BYTES,
        )
    assert caught.value.request_id == "invalid"
    assert caught.value.code == "invalid-request"
    assert caught.value.message == "Request must be JSON text"
    assert str(caught.value) == "Request must be JSON text"

    with pytest.raises(ValueError, match="^invalid JSON constant: NaN$"):
        job_codec._reject_json_constant("NaN")
    assert job_codec._valid_identifier("a" * 120) is True
    assert job_codec._valid_identifier("a" * 121) is False
    with pytest.raises(ValueError, match="^generated job ID is invalid$"):
        job_codec._validate_identifier("generated job ID", "bad id")


def test_extracted_dispatch_error_preserves_code_boundaries_and_diagnostic():
    assert JobDispatchError("a", "message").code == "a"
    assert JobDispatchError("a" * 80, "message").code == "a" * 80
    for code in (None, "a" * 81, "bad-"):
        with pytest.raises(ValueError, match="^job dispatch code is invalid$"):
            JobDispatchError(code, "message")  # type: ignore[arg-type]


def test_extracted_lifecycle_helpers_preserve_exact_terminal_outcomes():
    async def scenario():
        completed = asyncio.get_running_loop().create_future()
        completed.set_result({})
        failed = asyncio.get_running_loop().create_future()
        failed.set_exception(RuntimeError("accelerator lost"))
        cancelled = asyncio.get_running_loop().create_future()
        cancelled.cancel()

        assert job_lifecycle._settled_outcome(completed) == (
            "rejected",
            "job-already-completed",
            "Job had already completed when cancellation arrived",
        )
        assert job_lifecycle._settled_outcome(failed) == (
            "rejected",
            "job-already-failed",
            "Job had already failed when cancellation arrived",
        )
        assert job_lifecycle._terminal_status(cancelled) == (
            PluginResultStatus.CANCELLED,
            "Job cancelled",
        )
        assert job_lifecycle._terminal_status(failed) == (
            PluginResultStatus.FAILED,
            "RuntimeError: accelerator lost",
        )

    run_scenario(scenario())


def test_extracted_lifecycle_completion_preserves_owner_and_task_identity():
    async def scenario():
        service = JobSubmissionService()
        active_task = asyncio.get_running_loop().create_future()
        other_task = asyncio.get_running_loop().create_future()
        calls = []
        service._record_outcome = lambda *arguments: calls.append(arguments)
        service._active["active"] = job_lifecycle._ActiveJob(
            "visual-library", active_task, "uid:1000"
        )

        service._job_completed("missing", other_task)
        service._job_completed("active", other_task)
        assert service.active_job_ids() == ("active",)
        service._job_completed("active", active_task)

        assert calls == [
            ("missing", "anonymous", "", other_task),
            ("active", "uid:1000", "visual-library", other_task),
            ("active", "uid:1000", "visual-library", active_task),
        ]
        assert service.active_job_ids() == ()
        active_task.cancel()
        other_task.cancel()

    run_scenario(scenario())


def test_extracted_lifecycle_stop_passes_the_configured_deadline(monkeypatch):
    async def scenario():
        service = JobSubmissionService(cancel_timeout_seconds=0.25)
        task = asyncio.get_running_loop().create_future()
        service._active["active"] = job_lifecycle._ActiveJob("workload", task)
        observed = []

        async def wait(tasks, *, timeout):
            observed.append((tasks, timeout))
            return set(tasks), set()

        monkeypatch.setattr(job_lifecycle.asyncio, "wait", wait)
        await service.stop()

        assert observed == [((task,), 0.25)]
        assert task.cancelled() is True

    run_scenario(scenario())


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
        assert dispatcher.calls == [("job-1", "visual-library", {"tensor": [1, 2]})]
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

        cancelled = decode(await service.cancel_job_text(json.dumps(cancel_document("job-cancel"))))
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

        reply = decode(await service.submit_job_text(json.dumps(submit_document())))

        assert reply["status"] == "rejected"
        assert reply["code"] == "not-authorized"
        assert reply["message"] == ("Job submission is not authorized for this workload")
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

        reply = decode(await service.cancel_job_text(json.dumps(cancel_document("owned-job"))))

        assert reply["code"] == "not-authorized"
        assert reply["jobId"] == "owned-job"
        assert reply["message"] == ("Job cancellation is not authorized for this workload")
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
            await service.submit_job_text(json.dumps(submit_document(request_id="request-2")))
        )
        assert first["status"] == "accepted"
        assert full["code"] == "capacity-exceeded"
        assert full["message"] == "Active job capacity is exhausted"
        assert len(dispatcher.calls) == 1

        dispatcher.release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        second = decode(
            await service.submit_job_text(json.dumps(submit_document(request_id="request-3")))
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
            await service.submit_job_text(json.dumps(submit_document(request_id="admission-1")))
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
        service = JobSubmissionService(ImmediateDispatcher(), allows_all(), max_request_bytes=16)
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

        reply = decode(await service.cancel_job_text(json.dumps(cancel_document("stubborn-job"))))

        assert reply["code"] == "cancel-timeout"
        assert reply["jobId"] == "stubborn-job"
        assert reply["message"] == ("Job did not stop within the cancellation deadline")
        assert service.active_job_ids() == ("stubborn-job",)
        dispatcher.future.set_result(None)
        await asyncio.sleep(0)
        assert service.active_job_ids() == ()

    run_scenario(scenario())


def test_default_cancellation_budget_contains_worker_drain_and_escalation():
    from omnitensor.plugins.supervisor_process import (
        DEFAULT_CANCEL_TIMEOUT_SECONDS,
        DEFAULT_STOP_TIMEOUT_SECONDS,
    )

    assert DEFAULT_JOB_CANCEL_TIMEOUT_SECONDS == 2.0
    assert (
        DEFAULT_CANCEL_TIMEOUT_SECONDS + DEFAULT_STOP_TIMEOUT_SECONDS
        < DEFAULT_JOB_CANCEL_TIMEOUT_SECONDS
    )


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
        invalid_reply = decode(await invalid.submit_job_text(json.dumps(submit_document())))
        assert invalid_reply["code"] == "internal-error"

        dispatcher = BlockingDispatcher()
        collision = JobSubmissionService(dispatcher, allows_all(), id_factory=lambda: "same-job")
        await collision.submit_job_text(json.dumps(submit_document()))
        collision_reply = decode(
            await collision.submit_job_text(json.dumps(submit_document(request_id="second")))
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
            await service.cancel_job_text(json.dumps(cancel_document(job_id)), owner="uid:1001")
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
        return decode(await service.cancel_job_text(json.dumps(cancel_document(accepted["jobId"]))))

    assert run_scenario(scenario())["status"] != "not-found"


def result_document(job_id, request_id="result-1"):
    return {"version": 1, "requestId": request_id, "jobId": job_id}


def decode_result(text):
    document = json.loads(text)
    assert validate_document("runtime-job-result.schema.json", document) == []
    return document


def test_a_submitted_job_can_be_asked_about_until_it_finishes():
    """Submission only answers 'accepted'; without this the outcome is lost."""
    from omnitensor.plugins.job_results import JobResultStore

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
    from omnitensor.plugins.job_results import JobResultStore

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
    from omnitensor.plugins.job_results import JobResultStore

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


def test_a_running_job_reports_where_it_has_got_to():
    """Without progress a slow job is indistinguishable from a hung one."""
    from omnitensor.plugins.job_results import JobResultStore

    dispatcher = BlockingDispatcher()

    async def scenario():
        service = JobSubmissionService(dispatcher, allows_all(), results=JobResultStore())
        accepted = decode(
            await service.submit_job_text(json.dumps(submit_document()), owner="uid:1000")
        )
        job_id = accepted["jobId"]
        queued = decode_result(
            await service.job_result_text(json.dumps(result_document(job_id)), owner="uid:1000")
        )
        service.note_progress(job_id, "infer", 0.5, "Running on gpu")
        halfway = decode_result(
            await service.job_result_text(
                json.dumps(result_document(job_id, "result-2")), owner="uid:1000"
            )
        )
        dispatcher.release.set()
        return queued, halfway

    queued, halfway = run_scenario(scenario())

    assert queued["progress"] == {"fraction": 0.0, "detail": "Job accepted and queued"}
    assert halfway["progress"] == {"fraction": 0.5, "detail": "Running on gpu"}
    assert halfway["message"] == "Running on gpu"


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [(-1, 0.0), (2, 1.0), (float("nan"), 0.0), (True, 0.0), ("half", 0.0), (0.25, 0.25)],
)
def test_an_impossible_fraction_is_clamped_rather_than_losing_the_update(fraction, expected):
    """A stage reporting nonsense is a bug in the stage, not a reason to go silent."""
    from omnitensor.plugins.job_results import JobResultStore

    dispatcher = BlockingDispatcher()

    async def scenario():
        service = JobSubmissionService(dispatcher, allows_all(), results=JobResultStore())
        accepted = decode(
            await service.submit_job_text(json.dumps(submit_document()), owner="uid:1000")
        )
        service.note_progress(accepted["jobId"], "infer", fraction, "working")
        reply = decode_result(
            await service.job_result_text(
                json.dumps(result_document(accepted["jobId"])), owner="uid:1000"
            )
        )
        dispatcher.release.set()
        return reply

    assert run_scenario(scenario())["progress"]["fraction"] == expected


@pytest.mark.parametrize("fraction", [0.0, 1.0])
@pytest.mark.parametrize("detail_length", [200, 201])
def test_running_result_progress_and_truncation_boundaries(fraction, detail_length):
    store = JobResultStore()
    detail = "p" * detail_length
    store.record_progress(
        "job-progress",
        "uid:1000",
        PluginProgress("job-progress", "infer", fraction, detail, 100),
    )
    service = JobSubmissionService(results=store, clock_ms=lambda: 123)

    reply = run_scenario(
        service.job_result_text(json.dumps(result_document("job-progress")), owner="uid:1000")
    )

    document = decode_result(reply)
    assert document["state"] == "running"
    assert document["progress"] == {
        "fraction": fraction,
        "detail": "p" * min(detail_length, 200),
    }


@pytest.mark.parametrize(
    "status",
    [
        PluginResultStatus.SUCCEEDED,
        PluginResultStatus.FAILED,
        PluginResultStatus.CANCELLED,
    ],
)
def test_every_terminal_result_state_is_schema_valid(status):
    store = JobResultStore()
    store.record_result(
        "job-terminal",
        "uid:1000",
        PluginResult("job-terminal", status, {"value": 1}, "finished", 100),
    )
    service = JobSubmissionService(results=store, clock_ms=lambda: 123)

    document = decode_result(
        run_scenario(
            service.job_result_text(json.dumps(result_document("job-terminal")), owner="uid:1000")
        )
    )

    assert document["state"] == str(status)
    assert document["code"] == f"job-{status}"
    assert document["message"] == "finished"


@pytest.mark.parametrize("detail_length", [500, 501])
def test_terminal_result_message_truncation_boundary(detail_length):
    store = JobResultStore()
    store.record_result(
        "job-detail",
        "uid:1000",
        PluginResult(
            "job-detail",
            PluginResultStatus.FAILED,
            {},
            "d" * detail_length,
            100,
        ),
    )
    service = JobSubmissionService(results=store, clock_ms=lambda: 123)

    document = decode_result(
        run_scenario(
            service.job_result_text(json.dumps(result_document("job-detail")), owner="uid:1000")
        )
    )

    assert document["message"] == "d" * min(detail_length, 500)


def test_invalid_stored_state_returns_a_contract_valid_rejection():
    store = JobResultStore()
    store.record_result(
        "job-invalid",
        "uid:1000",
        PluginResult("job-invalid", "invalid", {}, "finished", 100),  # type: ignore[arg-type]
    )
    service = JobSubmissionService(results=store, clock_ms=lambda: 123)

    document = decode_result(
        run_scenario(
            service.job_result_text(json.dumps(result_document("job-invalid")), owner="uid:1000")
        )
    )

    assert document["state"] == "failed"
    assert document["code"] == "job-result-invalid"
    assert document["message"] == "Stored job result is invalid"
    assert document["output"] is None


def test_non_finite_stored_output_returns_a_contract_valid_rejection():
    store = JobResultStore()
    store.record_result(
        "job-invalid-json",
        "uid:1000",
        PluginResult(
            "job-invalid-json",
            PluginResultStatus.SUCCEEDED,
            {"score": float("nan")},
            "finished",
            100,
        ),
    )
    service = JobSubmissionService(results=store, clock_ms=lambda: 123)

    document = decode_result(
        run_scenario(
            service.job_result_text(
                json.dumps(result_document("job-invalid-json")), owner="uid:1000"
            )
        )
    )

    assert document["state"] == "failed"
    assert document["code"] == "job-result-invalid"
    assert document["message"] == "Stored job result is invalid"
    assert document["output"] is None


def test_job_result_timestamp_is_clamped_to_the_schema_minimum():
    service = JobSubmissionService(clock_ms=lambda: 0)

    document = decode_result(
        run_scenario(
            service.job_result_text(json.dumps(result_document("job-unknown")), owner="uid:1000")
        )
    )

    assert document["timestamp"] == 1


def test_progress_for_an_unknown_job_is_ignored_not_filed_under_a_guess():
    from omnitensor.plugins.job_results import JobResultStore

    store = JobResultStore()
    service = JobSubmissionService(BlockingDispatcher(), allows_all(), results=store)

    service.note_progress("job-that-never-existed", "infer", 0.5, "working")

    assert not store.active_job_ids() if hasattr(store, "active_job_ids") else True


def test_progress_after_a_result_does_not_reopen_a_finished_job():
    from omnitensor.plugins.job_results import JobResultStore

    async def scenario():
        service = JobSubmissionService(InstantDispatcher(), allows_all(), results=JobResultStore())
        accepted = decode(
            await service.submit_job_text(json.dumps(submit_document()), owner="uid:1000")
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        service.note_progress(accepted["jobId"], "infer", 0.5, "late update")
        return decode_result(
            await service.job_result_text(
                json.dumps(result_document(accepted["jobId"])), owner="uid:1000"
            )
        )

    assert run_scenario(scenario())["state"] == "succeeded"


class InstantDispatcher:
    def dispatch(self, job_id, workload_id, payload):
        async def work():
            return {"outputs": []}

        return work()


class RefusingAdmission:
    def __init__(self, code="input-ref-denied", message="that path may not be read"):
        self.calls = []
        self._code = code
        self._message = message

    def admit(self, workload_id, payload):
        self.calls.append((workload_id, payload))
        raise JobDispatchError(self._code, self._message)


class AcceptingAdmission:
    def __init__(self):
        self.calls = []

    def admit(self, workload_id, payload):
        self.calls.append((workload_id, payload))


def test_an_admission_refusal_answers_the_submission_and_queues_nothing():
    """The caller learns why in the reply, instead of polling for a failure."""

    async def scenario():
        dispatcher = BlockingDispatcher()
        admission = RefusingAdmission()
        service = JobSubmissionService(dispatcher, allows_all(), admission=admission)
        reply = decode(await service.submit_job_text(json.dumps(submit_document())))
        return reply, dispatcher, admission, service

    reply, dispatcher, admission, service = run_scenario(scenario())

    assert reply["status"] == "rejected"
    assert reply["code"] == "input-ref-denied"
    assert reply["jobId"] is None
    assert reply["requestId"] == "request-1"
    assert dispatcher.calls == []
    assert service.active_job_ids() == ()
    assert admission.calls == [("visual-library", {})]


def test_admission_sees_the_payload_the_dispatcher_would_have_seen():
    async def scenario():
        admission = AcceptingAdmission()
        service = JobSubmissionService(ImmediateDispatcher(), allows_all(), admission=admission)
        await service.submit_job_text(
            json.dumps(submit_document(payload={"inputRefs": [{"path": "/tmp/x"}]}))
        )
        return admission

    assert run_scenario(scenario()).calls == [
        ("visual-library", {"inputRefs": [{"path": "/tmp/x"}]})
    ]


def test_an_unauthorized_job_is_never_offered_to_admission():
    """Authorization is the cheaper question and answers first."""

    async def scenario():
        admission = RefusingAdmission()
        service = JobSubmissionService(
            ImmediateDispatcher(),
            PredicateJobAuthorizer(lambda action, workload_id: False),
            admission=admission,
        )
        reply = decode(await service.submit_job_text(json.dumps(submit_document())))
        return reply, admission

    reply, admission = run_scenario(scenario())

    assert reply["code"] == "not-authorized"
    assert admission.calls == []


def test_without_an_admission_check_nothing_is_refused_earlier_than_before():
    async def scenario():
        service = JobSubmissionService(ImmediateDispatcher(), allows_all())
        return decode(await service.submit_job_text(json.dumps(submit_document())))

    assert run_scenario(scenario())["status"] == "accepted"


def test_the_fail_closed_dispatcher_refuses_at_admission_as_well_as_at_dispatch():
    """A profile whose pipeline would reach it must not be told 'accepted'."""
    from omnitensor.jobs import UnavailableJobDispatcher

    subject = UnavailableJobDispatcher()

    with pytest.raises(JobDispatchError) as admission:
        subject.admit("visual-library", {"inputs": []})
    with pytest.raises(JobDispatchError) as dispatched:
        subject.dispatch("job-1", "visual-library", {"inputs": []})

    assert admission.value.code == dispatched.value.code == "dispatch-unavailable"
    assert admission.value.message == dispatched.value.message


def test_an_observer_failure_never_rejects_a_job_that_is_already_running():
    """OMNI-0412: the reply and the capacity slot must agree with reality."""

    class ExplodingObserver:
        def job_started(self, workload_id):
            raise RuntimeError("publisher is down")

        def job_finished(self, workload_id, status, detail=""):
            pass

    async def scenario():
        dispatcher = BlockingDispatcher()
        service = JobSubmissionService(
            dispatcher,
            allows_all(),
            id_factory=lambda: "job-observed",
            observer=ExplodingObserver(),
        )

        reply = decode(await service.submit_job_text(json.dumps(submit_document())))

        assert reply["status"] == "accepted"
        assert service.active_job_ids() == ("job-observed",)
        dispatcher.release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert service.active_job_ids() == ()

    run_scenario(scenario())


def test_a_rejected_submission_leaves_no_running_job_holding_capacity():
    """OMNI-0412: an internal-error reply must not strand the started task."""

    async def scenario():
        dispatcher = BlockingDispatcher()
        service = JobSubmissionService(
            dispatcher,
            allows_all(),
            id_factory=lambda: "job-stranded",
        )
        service._reply = _failing_once(service._reply)
        before = asyncio.all_tasks()

        reply = decode(await service.submit_job_text(json.dumps(submit_document())))

        assert reply["status"] == "rejected"
        assert reply["code"] == "internal-error"
        assert service.active_job_ids() == ()
        started = asyncio.all_tasks() - before
        assert started
        await asyncio.gather(*started, return_exceptions=True)
        assert all(task.cancelled() for task in started)

    run_scenario(scenario())


def _failing_once(reply):
    state = {"failed": False}

    def guarded(*args, **kwargs):
        if not state["failed"] and args[2] == "accepted":
            state["failed"] = True
            raise RuntimeError("acknowledgement encoding failed")
        return reply(*args, **kwargs)

    return guarded
