"""Bounded, transport-neutral runtime job submission and cancellation."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import secrets
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .callers import ANONYMOUS_OWNER
from .plugins.protocol import PluginProgress, PluginResult, PluginResultStatus
from .plugins.results import JobRecord, JobResultError, JobResultStore
from .registry import validate_document

JOB_API_VERSION = 1
DEFAULT_MAX_ACTIVE_JOBS = 128
MAX_ACTIVE_JOBS_LIMIT = 1024
DEFAULT_MAX_JOB_REQUEST_BYTES = 64 * 1024
MAX_JOB_REQUEST_BYTES_LIMIT = 1024 * 1024
DEFAULT_JOB_CANCEL_TIMEOUT_SECONDS = 2.0
MAX_JOB_CANCEL_TIMEOUT_SECONDS = 30.0
MAX_JOB_MESSAGE_CHARS = 240
_UNAVAILABLE_DISPATCHER = "No verified inference dispatcher is ready for this workload"


class JobAuthorizer(Protocol):
    def allows(self, action: str, workload_id: str) -> bool: ...


class JobDispatcher(Protocol):
    def dispatch(
        self, job_id: str, workload_id: str, payload: dict
    ) -> Awaitable[object]: ...


@runtime_checkable
class JobAdmission(Protocol):
    """Everything about a job that can be refused before it becomes active.

    Separate from :class:`JobDispatcher` because dispatching is asynchronous by
    the time a pipeline runs it: whatever the runner's stages decide is decided
    after this service has already answered ``accepted``.  A refusal a caller
    can act on has to be reachable synchronously, in the submitting call, which
    is what this port is for.  ``runtime_checkable`` so wiring can offer a
    dispatcher that also admits without a second registration.
    """

    def admit(self, workload_id: str, payload: dict) -> None:
        """Return normally, or raise :class:`JobDispatchError` with a reason."""


class JobDispatchError(RuntimeError):
    """Stable admission failure raised before a job becomes active."""

    def __init__(self, code: str, message: str) -> None:
        _validate_code(code)
        if not isinstance(message, str) or not message:
            raise ValueError("job dispatch message must be non-empty")
        super().__init__(message)
        self.code = code
        self.message = message[:MAX_JOB_MESSAGE_CHARS]


@dataclass(frozen=True, slots=True)
class JobRequest:
    request_id: str
    workload_id: str
    payload: dict


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


class JobLifecycleObserver(Protocol):
    """Told when a job starts and how it ended, for surfaces that only watch.

    A port rather than the telemetry registry itself: whether a finished job
    becomes a counter, a summary, or nothing is a question for the service
    wiring, and the job service should not learn the vocabulary of whichever
    answer is current.  Observation must never change an outcome, so an
    implementation that raises is a defect in the implementation.
    """

    def job_started(self, workload_id: str) -> None: ...

    def job_finished(self, workload_id: str, status: str, detail: str) -> None: ...


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


class _RequestError(ValueError):
    def __init__(self, request_id: str, code: str, message: str) -> None:
        super().__init__(message)
        self.request_id = request_id
        self.code = code
        self.message = message


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
        _validate_integer_bound(
            "max_active_jobs", max_active_jobs, 1, MAX_ACTIVE_JOBS_LIMIT
        )
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
                PluginProgress(job_id, stage, _clamped(fraction), str(detail)[:200],
                               self._clock_ms()),
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
            operation = self._dispatcher.dispatch(
                job_id, request.workload_id, request.payload
            )
            task = asyncio.ensure_future(operation)
            self._active[job_id] = _ActiveJob(request.workload_id, task, owner)
            self._observer.job_started(request.workload_id)
            # Recorded before the reply so a caller that polls immediately sees
            # a queued job rather than a job it cannot distinguish from a hang.
            self.note_progress(job_id, "queued", 0.0, "Job accepted and queued")
            task.add_done_callback(
                lambda completed, current_id=job_id: self._job_completed(
                    current_id, completed
                )
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
        """Remember how a job ended, because the caller has already been replied to.

        Submission answers with an acceptance, so this is the only place the
        outcome can be kept; dropping it here is what made a submitted job
        unanswerable for the rest of its life.
        """
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
        request_id = "unknown"
        try:
            document = _parse_request(
                text, "runtime-job-result-request.schema.json", self._max_request_bytes
            )
            request_id = document["requestId"]
            job_id = document["jobId"]
        except _RequestError as error:
            return _result_reply(
                error.request_id, "unknown", "unknown", error.code, error.message,
                self._clock_ms(),
            )
        record = self._lookup(job_id, owner)
        if record is None:
            # A job belonging to someone else answers exactly as one that never
            # existed, so a reply never confirms which job ids are real.
            return _result_reply(
                request_id, job_id, "unknown", "job-not-found",
                "No such job for this caller", self._clock_ms(),
            )
        return _record_reply(request_id, record, self._clock_ms())

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
        document = {
            "version": JOB_API_VERSION,
            "requestId": request_id,
            "jobId": job_id,
            "status": status,
            "code": code,
            "message": message[:MAX_JOB_MESSAGE_CHARS],
            "timestamp": max(1, int(self._clock_ms())),
        }
        violations = validate_document(
            "runtime-job-acknowledgement.schema.json", document
        )
        if violations:  # pragma: no cover - contract self-check
            raise RuntimeError(f"job acknowledgement violates contract: {violations}")
        return json.dumps(document, separators=(",", ":"), allow_nan=False)


def _parse_request(text: str, schema: str, max_bytes: int) -> dict:
    if not isinstance(text, str):
        raise _RequestError("invalid", "invalid-request", "Request must be JSON text")
    try:
        encoded_size = len(text.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise _RequestError(
            "invalid", "invalid-request", "Request is not valid UTF-8 text"
        ) from error
    if encoded_size > max_bytes:
        raise _RequestError(
            "invalid", "request-too-large", f"Request exceeds {max_bytes} bytes"
        )
    try:
        document = json.loads(text, parse_constant=_reject_json_constant)
    except (ValueError, RecursionError) as error:
        raise _RequestError(
            "invalid", "invalid-json", "Request is not valid JSON"
        ) from error
    request_id = _request_id(document)
    if not isinstance(document, dict):
        raise _RequestError(
            request_id, "invalid-request", "Request must be a JSON object"
        )
    if validate_document(schema, document):
        raise _RequestError(
            request_id,
            "invalid-request",
            "Request does not match the version 1 contract",
        )
    return document


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _settled_outcome(task: asyncio.Task) -> tuple[str, str, str]:
    """Describe how a settled job actually ended, not how it was asked to end.

    A job can reach a terminal state before the cancellation reaches it.
    Reporting "cancelled" then credits the caller's request with an outcome
    something else decided.
    """
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


def _request_id(document: object) -> str:
    if isinstance(document, dict):
        candidate = document.get("requestId")
        if _valid_identifier(candidate):
            return candidate
    return "invalid"


def _request_id_from_text(text: object, max_request_bytes: int) -> str:
    """Recover the request id from a request that failed elsewhere.

    The bound must be the caller's configured one, measured in encoded bytes:
    the module default measured in characters both rejects a valid request that
    a larger configured bound admits — echoing requestId "invalid" back for it —
    and admits a multi-byte request larger than the bound actually allows.
    """
    if not isinstance(text, str):
        return "invalid"
    if len(text.encode("utf-8", "surrogatepass")) > max_request_bytes:
        return "invalid"
    try:
        return _request_id(json.loads(text, parse_constant=_reject_json_constant))
    except (ValueError, RecursionError):
        return "invalid"


def _valid_identifier(value: object) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 120
        and all(
            character.isascii() and (character.isalnum() or character in "._-")
            for character in value
        )
    )


def _validate_identifier(name: str, value: object) -> None:
    if not _valid_identifier(value):
        raise ValueError(f"{name} is invalid")


def _validate_code(code: object) -> None:
    if (
        not isinstance(code, str)
        or not 1 <= len(code) <= 80
        or code.strip("abcdefghijklmnopqrstuvwxyz0123456789-")
        or code.startswith("-")
        or code.endswith("-")
        or "--" in code
    ):
        raise ValueError("job dispatch code is invalid")


def _validate_integer_bound(name: str, value: object, minimum: int, maximum: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _clamped(fraction: object) -> float:
    """Progress outside [0, 1] is a bug in a stage, not a reason to lose the update."""
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


def _result_reply(
    request_id: str, job_id: str, state: str, code: str, message: str, timestamp: int
) -> str:
    return json.dumps(
        {
            "version": 1,
            "requestId": request_id if isinstance(request_id, str) else "unknown",
            "jobId": job_id if isinstance(job_id, str) and job_id else "unknown",
            "state": state,
            "code": code,
            "message": message[:500],
            "timestamp": timestamp,
            "progress": None,
            "output": None,
        },
        separators=(",", ":"),
    )


def _record_reply(request_id: str, record: JobRecord, timestamp: int) -> str:
    """Project a stored record into the published result contract."""
    result = record.result
    progress = record.progress
    if result is None:
        return json.dumps(
            {
                "version": 1,
                "requestId": request_id,
                "jobId": record.job_id,
                "state": "running",
                "code": "job-running",
                "message": progress.detail[:500] if progress else "Job is still running",
                "timestamp": timestamp,
                "progress": (
                    None
                    if progress is None
                    else {"fraction": progress.fraction, "detail": progress.detail[:200]}
                ),
                "output": None,
            },
            separators=(",", ":"),
        )
    return json.dumps(
        {
            "version": 1,
            "requestId": request_id,
            "jobId": record.job_id,
            "state": str(result.status),
            "code": f"job-{result.status}",
            "message": result.detail[:500] or "Job finished",
            "timestamp": timestamp,
            "progress": None,
            "output": result.output if isinstance(result.output, dict) else None,
        },
        separators=(",", ":"),
    )
