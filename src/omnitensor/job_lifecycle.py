"""Runtime job admission, execution lifecycle, cancellation, and results."""

from __future__ import annotations

import asyncio
import copy
import math
import secrets
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass

from .callers import ANONYMOUS_OWNER
from .job_codec import (
    DEFAULT_MAX_JOB_REQUEST_BYTES,
    MAX_JOB_REQUEST_BYTES_LIMIT,
    JobRequest,
    _acknowledgement_reply,
    _parse_request,
    _record_reply,
    _request_id_from_text,
    _RequestError,
    _result_reply,
    _validate_identifier,
    _validated_result_reply,
)
from .job_ports import (
    JobAdmission,
    JobAuthorizer,
    JobDispatcher,
    JobDispatchError,
    JobLifecycleObserver,
)
from .plugins.protocol import PluginProgress, PluginResult, PluginResultStatus
from .plugins.results import JobRecord, JobResultError, JobResultStore

DEFAULT_MAX_ACTIVE_JOBS = 128
MAX_ACTIVE_JOBS_LIMIT = 1024
DEFAULT_JOB_CANCEL_TIMEOUT_SECONDS = 2.0
MAX_JOB_CANCEL_TIMEOUT_SECONDS = 30.0
_UNAVAILABLE_DISPATCHER = "No verified inference dispatcher is ready for this workload"


@dataclass(slots=True)
class _ActiveJob:
    workload_id: str
    task: asyncio.Future
    owner: str = ANONYMOUS_OWNER


class _DenyAllAuthorizer:
    def allows(self, action: str, workload_id: str) -> bool:
        return False


class UnavailableJobDispatcher:
    """Fail-closed wiring used until a verified inference dispatcher is injected."""

    def admit(self, workload_id: str, payload: dict) -> None:
        # Also refused at admission: with no verified dispatcher there is
        # nothing that could succeed later, so accepting the job would promise
        # a result that a caller then has to poll for to be told this.
        raise JobDispatchError("dispatch-unavailable", _UNAVAILABLE_DISPATCHER)

    def dispatch(self, job_id: str, workload_id: str, payload: dict) -> Awaitable[object]:
        raise JobDispatchError("dispatch-unavailable", _UNAVAILABLE_DISPATCHER)


class NoJobObserver:
    """The default: nothing watches, and nothing is recorded."""

    def job_started(self, workload_id: str) -> None:
        return None

    def job_finished(self, workload_id: str, status: str, detail: str) -> None:
        return None


class PredicateJobAuthorizer:
    """Small adapter keeping service policy outside the job domain."""

    def __init__(self, predicate: Callable[[str, str], bool]) -> None:
        self._predicate = predicate

    def allows(self, action: str, workload_id: str) -> bool:
        return self._predicate(action, workload_id) is True


class JobSubmissionService:
    """Own active job tasks while exposing only bounded JSON contracts."""

    def __init__(
        self,
        dispatcher: JobDispatcher | None = None,
        authorizer: JobAuthorizer | None = None,
        *,
        max_active_jobs: int = DEFAULT_MAX_ACTIVE_JOBS,
        max_request_bytes: int = DEFAULT_MAX_JOB_REQUEST_BYTES,
        cancel_timeout_seconds: float = DEFAULT_JOB_CANCEL_TIMEOUT_SECONDS,
        id_factory: Callable[[], str] | None = None,
        clock_ms: Callable[[], int] | None = None,
        results: JobResultStore | None = None,
        admission: JobAdmission | None = None,
        observer: JobLifecycleObserver | None = None,
    ) -> None:
        _validate_integer_bound("max_active_jobs", max_active_jobs, 1, MAX_ACTIVE_JOBS_LIMIT)
        _validate_integer_bound(
            "max_request_bytes",
            max_request_bytes,
            1,
            MAX_JOB_REQUEST_BYTES_LIMIT,
        )
        if (
            isinstance(cancel_timeout_seconds, bool)
            or not isinstance(cancel_timeout_seconds, (int, float))
            or not math.isfinite(cancel_timeout_seconds)
            or not 0 < cancel_timeout_seconds <= MAX_JOB_CANCEL_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "cancel_timeout_seconds must be finite and between 0 and "
                f"{MAX_JOB_CANCEL_TIMEOUT_SECONDS}"
            )
        self._dispatcher = dispatcher or UnavailableJobDispatcher()
        self._authorizer = authorizer or _DenyAllAuthorizer()
        self._max_active_jobs = max_active_jobs
        self._max_request_bytes = max_request_bytes
        self._cancel_timeout_seconds = cancel_timeout_seconds
        self._id_factory = id_factory or (lambda: secrets.token_hex(16))
        self._clock_ms = clock_ms or (lambda: max(1, int(time.time() * 1000)))
        self._active: dict[str, _ActiveJob] = {}
        self._results = results
        # Absent by default: with no admission check a job is refused wherever
        # it was refused before, never earlier and never for a new reason.
        self._admission = admission
        self._observer = observer or NoJobObserver()

    def note_progress(self, job_id: str, stage: str, fraction: float, detail: str) -> None:
        """Record where a running job has got to, against its owner.

        The owner lives here rather than in the dispatcher, so progress is
        reported through this method instead of written to the store directly:
        a dispatcher that guessed an owner could file one caller's progress
        under another's job.
        """
        active = self._active.get(job_id)
        if active is None or self._results is None:
            return
        with suppress(JobResultError):
            self._results.record_progress(
                job_id,
                active.owner,
                PluginProgress(
                    job_id,
                    stage,
                    _clamped(fraction),
                    str(detail)[:200],
                    self._clock_ms(),
                ),
            )

    def active_job_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._active))

    async def submit_job_text(self, text: str, *, owner: str = ANONYMOUS_OWNER) -> str:
        try:
            document = _parse_request(
                text,
                "runtime-job-submit.schema.json",
                self._max_request_bytes,
            )
            request = JobRequest(
                document["requestId"],
                document["workloadId"],
                copy.deepcopy(document["payload"]),
            )
            if not self._authorizer.allows("submit", request.workload_id):
                return self._reply(
                    request.request_id,
                    None,
                    "rejected",
                    "not-authorized",
                    "Job submission is not authorized for this workload",
                )
            if len(self._active) >= self._max_active_jobs:
                return self._reply(
                    request.request_id,
                    None,
                    "rejected",
                    "capacity-exceeded",
                    "Active job capacity is exhausted",
                )
            if self._admission is not None:
                # Before an id is minted and before anything is queued, so a
                # refused job leaves nothing behind for a caller to poll and
                # occupies no scheduler slot.
                self._admission.admit(request.workload_id, request.payload)
            job_id = self._id_factory()
            _validate_identifier("generated job ID", job_id)
            if job_id in self._active:
                raise RuntimeError("job ID collision")
            operation = self._dispatcher.dispatch(job_id, request.workload_id, request.payload)
            task = asyncio.ensure_future(operation)
            self._active[job_id] = _ActiveJob(request.workload_id, task, owner)
            self._observer.job_started(request.workload_id)
            # Recorded before the reply so a caller that polls immediately sees
            # a queued job rather than a job it cannot distinguish from a hang.
            self.note_progress(job_id, "queued", 0.0, "Job accepted and queued")
            task.add_done_callback(
                lambda completed, current_id=job_id: self._job_completed(current_id, completed)
            )
            return self._reply(
                request.request_id,
                job_id,
                "accepted",
                "job-accepted",
                "Job accepted",
            )
        except _RequestError as error:
            return self._reply(
                error.request_id,
                None,
                "rejected",
                error.code,
                error.message,
            )
        except JobDispatchError as error:
            request_id = _request_id_from_text(text, self._max_request_bytes)
            return self._reply(
                request_id,
                None,
                "rejected",
                error.code,
                error.message,
            )
        except Exception:
            return self._reply(
                _request_id_from_text(text, self._max_request_bytes),
                None,
                "rejected",
                "internal-error",
                "Internal error while submitting the job",
            )

    async def cancel_job_text(self, text: str, *, owner: str = ANONYMOUS_OWNER) -> str:
        try:
            document = _parse_request(
                text,
                "runtime-job-cancel.schema.json",
                self._max_request_bytes,
            )
            request_id = document["requestId"]
            job_id = document["jobId"]
            active = self._active.get(job_id)
            if active is None or active.owner != owner:
                # Someone else's job answers exactly as a job that never
                # existed: any other answer tells an unauthorized caller which
                # job ids are real, which is worth nothing to a legitimate
                # caller and quite a lot to anyone else.
                return self._reply(
                    request_id,
                    job_id,
                    "not-found",
                    "job-not-found",
                    "Active job was not found",
                )
            if not self._authorizer.allows("cancel", active.workload_id):
                return self._reply(
                    request_id,
                    job_id,
                    "rejected",
                    "not-authorized",
                    "Job cancellation is not authorized for this workload",
                )
            active.task.cancel()
            _done, pending = await asyncio.wait(
                (active.task,), timeout=self._cancel_timeout_seconds
            )
            if pending:
                return self._reply(
                    request_id,
                    job_id,
                    "rejected",
                    "cancel-timeout",
                    "Job did not stop within the cancellation deadline",
                )
            status, code, message = _settled_outcome(active.task)
            return self._reply(request_id, job_id, status, code, message)
        except _RequestError as error:
            return self._reply(
                error.request_id,
                None,
                "rejected",
                error.code,
                error.message,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return self._reply(
                _request_id_from_text(text, self._max_request_bytes),
                None,
                "rejected",
                "internal-error",
                "Internal error while cancelling the job",
            )

    async def stop(self) -> None:
        tasks = tuple(active.task for active in self._active.values())
        for task in tasks:
            task.cancel()
        if not tasks:
            return
        await asyncio.wait(tasks, timeout=self._cancel_timeout_seconds)

    def _job_completed(self, job_id: str, task: asyncio.Future) -> None:
        active = self._active.get(job_id)
        owner = active.owner if active is not None else ANONYMOUS_OWNER
        workload_id = active.workload_id if active is not None else ""
        self._record_outcome(job_id, owner, workload_id, task)
        if active is not None and active.task is task:
            self._active.pop(job_id, None)

    def _record_outcome(
        self, job_id: str, owner: str, workload_id: str, task: asyncio.Future
    ) -> None:
        """Remember how a job ended, because the caller has already been replied to."""
        status, detail = _terminal_status(task)
        # Before the early return below: a profile's counters must not depend
        # on whether a result store happens to be wired.
        self._observer.job_finished(workload_id, str(status), detail[:200])
        if self._results is None:
            # Still consulted the task above: an exception nobody retrieves is
            # reported by asyncio as an unhandled error at collection time.
            return
        output = task.result() if status is PluginResultStatus.SUCCEEDED else {}
        self._results.record_result(
            job_id,
            owner,
            PluginResult(
                job_id,
                status,
                output if isinstance(output, dict) else {"value": output},
                detail[:200],
                self._clock_ms(),
            ),
        )

    async def job_result_text(self, text: str, *, owner: str = ANONYMOUS_OWNER) -> str:
        """Report what is known about one job, to the caller that owns it."""
        try:
            document = _parse_request(
                text, "runtime-job-result-request.schema.json", self._max_request_bytes
            )
            request_id = document["requestId"]
            job_id = document["jobId"]
        except _RequestError as error:
            reply = _result_reply(
                error.request_id,
                "unknown",
                "unknown",
                error.code,
                error.message,
                max(1, int(self._clock_ms())),
            )
        else:
            record = self._lookup(job_id, owner)
            if record is None:
                # A foreign job is indistinguishable from one that never existed.
                reply = _result_reply(
                    request_id,
                    job_id,
                    "unknown",
                    "job-not-found",
                    "No such job for this caller",
                    max(1, int(self._clock_ms())),
                )
            else:
                reply = _record_reply(request_id, record, max(1, int(self._clock_ms())))
        return _validated_result_reply(reply)

    def _lookup(self, job_id: str, owner: str):
        if not isinstance(job_id, str) or not job_id:
            return None
        if self._results is not None:
            try:
                record = self._results.get(job_id, owner)
            except JobResultError:
                record = None
            if record is not None:
                return record
        active = self._active.get(job_id)
        if active is None or active.owner != owner:
            return None
        return JobRecord(job_id, None, None, 0.0)

    def _reply(
        self,
        request_id: str,
        job_id: str | None,
        status: str,
        code: str,
        message: str,
    ) -> str:
        return _acknowledgement_reply(
            request_id,
            job_id,
            status,
            code,
            message,
            max(1, int(self._clock_ms())),
        )


def _settled_outcome(task: asyncio.Task) -> tuple[str, str, str]:
    """Describe how a settled job actually ended, not how it was asked to end."""
    if task.cancelled():
        return "cancelled", "job-cancelled", "Job cancelled"
    if task.exception() is not None:
        return (
            "rejected",
            "job-already-failed",
            "Job had already failed when cancellation arrived",
        )
    return (
        "rejected",
        "job-already-completed",
        "Job had already completed when cancellation arrived",
    )


def _validate_integer_bound(name: str, value: object, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _clamped(fraction: object) -> float:
    """Progress outside [0, 1] is a stage bug, not a lost update."""
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        return 0.0
    if not math.isfinite(fraction):
        return 0.0
    return float(min(max(fraction, 0.0), 1.0))


def _terminal_status(task: asyncio.Future) -> tuple[PluginResultStatus, str]:
    """How a job actually ended, from the task rather than from intent."""
    if task.cancelled():
        return PluginResultStatus.CANCELLED, "Job cancelled"
    error = task.exception()
    if error is not None:
        return PluginResultStatus.FAILED, f"{type(error).__name__}: {error}"
    return PluginResultStatus.SUCCEEDED, "Job completed"
