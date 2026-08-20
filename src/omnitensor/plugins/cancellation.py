"""Cancellation that reaches every stage, and survives a restart.

Five different things can stop a job — the caller withdraws it, policy stops
admitting it, its deadline passes, its worker dies, or the service shuts down —
and to whatever is running the stage they must all look the same: one signal,
delivered once, carrying why.  Without that, each stage grows its own idea of
"should I stop", and the ones that forget keep running.

The second half is the part that is easy to skip.  A job in flight when the
service dies leaves a record claiming it is still running, and nothing will
ever finish it.  On the next start that record has to be reconciled — the job
is not resumable, because the process that owned its stage is gone — so it is
reported as cancelled with the reason, rather than left claiming to run.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..atomicio import JsonTooLargeError, read_json_bounded, remove_durable, write_json_atomic

_LOGGER = logging.getLogger(__name__)

JOURNAL_VERSION = 1
DEFAULT_MAX_TRACKED_JOBS = 256
MAX_JOURNAL_BYTES = 512 * 1024


class CancellationReason(StrEnum):
    """Why a job stopped, in the vocabulary a user or an operator needs."""

    CALLER = "caller-cancelled"
    POLICY = "policy-refused"
    DEADLINE = "deadline-exceeded"
    WORKER = "worker-lost"
    SHUTDOWN = "service-shutdown"
    INTERRUPTED = "service-interrupted"


@dataclass(frozen=True, slots=True)
class Cancellation:
    """One delivered cancellation, with the reason attached."""

    job_id: str
    reason: CancellationReason
    detail: str


class JobCancelledError(RuntimeError):
    """A stage stopped because its job was cancelled."""

    def __init__(self, cancellation: Cancellation):
        self.cancellation = cancellation
        self.code = str(cancellation.reason)
        self.detail = cancellation.detail
        super().__init__(f"{self.code}: {cancellation.detail}")


class JobCancellationToken:
    """The single signal every stage of one job watches."""

    def __init__(self, job_id: str) -> None:
        self._job_id = job_id
        self._event = asyncio.Event()
        self._cancellation: Cancellation | None = None

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def cancelled(self) -> bool:
        return self._cancellation is not None

    @property
    def cancellation(self) -> Cancellation | None:
        return self._cancellation

    def cancel(self, reason: CancellationReason, detail: str = "") -> bool:
        """Record the first reason and wake every waiter.

        The first reason wins: a job cancelled by its caller that then also
        trips its deadline stopped because the caller withdrew it, and saying
        anything else would misreport what happened.
        """
        if self._cancellation is not None:
            return False
        self._cancellation = Cancellation(self._job_id, reason, detail or str(reason))
        self._event.set()
        return True

    def raise_if_cancelled(self) -> None:
        """The check a synchronous stage boundary makes."""
        if self._cancellation is not None:
            raise JobCancelledError(self._cancellation)

    async def wait(self) -> Cancellation:
        await self._event.wait()
        assert self._cancellation is not None  # noqa: S101 - set before the event
        return self._cancellation

    async def guard(self, operation) -> object:
        """Run ``operation``, stopping it as soon as this token is cancelled."""
        if self._cancellation is not None:
            raise JobCancelledError(self._cancellation)
        task = asyncio.ensure_future(operation())
        watcher = asyncio.ensure_future(self._event.wait())
        try:
            done, _pending = await asyncio.wait(
                (task, watcher), return_when=asyncio.FIRST_COMPLETED
            )
            if task in done:
                return await task
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise JobCancelledError(self._cancellation)
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


@runtime_checkable
class CancellationJournal(Protocol):
    """Crash-recovery view of jobs left in flight by a stopped service."""

    def recover(self) -> tuple[Cancellation, ...]: ...


@runtime_checkable
class CancellationRegistry(CancellationJournal, Protocol):
    """Job-runner view of cancellation state and its recovery journal."""

    def track(self, job_id: str, profile_id: str = "") -> JobCancellationToken: ...

    def cancel(self, job_id: str, reason: CancellationReason, detail: str = "") -> bool: ...

    def release(self, job_id: str) -> None: ...


class JobCancellationRegistry:
    """Own every live job's token and the record that survives a restart.

    Where that record is kept is somebody else's problem: this holds a
    :class:`CancellationJournalStore`, never a path, so cancellation logic can
    be exercised — and reasoned about — without a filesystem underneath it.
    """

    def __init__(
        self,
        journal_path: Path | None = None,
        *,
        max_tracked_jobs: int = DEFAULT_MAX_TRACKED_JOBS,
        journal: CancellationJournalStore | None = None,
    ) -> None:
        if isinstance(max_tracked_jobs, bool) or not isinstance(max_tracked_jobs, int):
            raise ValueError("max_tracked_jobs must be a positive integer")
        if max_tracked_jobs < 1:
            raise ValueError("max_tracked_jobs must be a positive integer")
        if journal is None and journal_path is not None:
            journal = JsonCancellationJournal(Path(journal_path))
        self._journal = journal
        self._max_tracked_jobs = max_tracked_jobs
        self._tokens: dict[str, JobCancellationToken] = {}
        self._entries: dict[str, str] | None = None

    def __len__(self) -> int:
        return len(self._tokens)

    def __bool__(self) -> bool:
        """A registry always exists, however many jobs it happens to hold.

        Without this, ``__len__`` makes an empty registry falsy, so the common
        ``registry or default()`` wiring silently substitutes a second registry
        and cancellation reaches jobs nobody is tracking.
        """
        return True

    def __iter__(self) -> Iterator[str]:
        return iter(tuple(self._tokens))

    def track(self, job_id: str, profile_id: str = "") -> JobCancellationToken:
        """Register a job and record that it is in flight."""
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be a non-empty string")
        if job_id in self._tokens:
            raise ValueError(f"job is already tracked: {job_id}")
        if len(self._tokens) >= self._max_tracked_jobs:
            raise ValueError(f"at most {self._max_tracked_jobs} jobs may be tracked")
        token = JobCancellationToken(job_id)
        self._tokens[job_id] = token
        self._persist(profile_id=profile_id, job_id=job_id, adding=True)
        return token

    def get(self, job_id: str) -> JobCancellationToken | None:
        return self._tokens.get(job_id)

    def cancel(self, job_id: str, reason: CancellationReason, detail: str = "") -> bool:
        token = self._tokens.get(job_id)
        return bool(token and token.cancel(reason, detail))

    def cancel_all(self, reason: CancellationReason, detail: str = "") -> int:
        """Stop everything, which is what shutdown and a lost worker need."""
        return sum(token.cancel(reason, detail) for token in tuple(self._tokens.values()))

    def release(self, job_id: str) -> None:
        """Forget a finished job so neither memory nor the journal grows.

        The journal is updated first: dropping the token first and then failing
        to write left a finished job recorded as in flight, and the next start
        reported it as interrupted.
        """
        if job_id in self._tokens:
            self._persist(job_id=job_id, adding=False)
            del self._tokens[job_id]

    def recover(self) -> tuple[Cancellation, ...]:
        """Reconcile jobs the previous process left claiming to be in flight.

        They are not resumable: whatever owned their stage is gone.  Reporting
        them as interrupted is the only honest outcome, and clearing the
        journal afterwards stops one crash from haunting every later start.
        """
        entries = self._journal_entries()
        if not entries:
            return ()
        self._entries = {}
        if self._journal is not None:
            self._journal.clear()
        return tuple(
            Cancellation(
                job_id,
                CancellationReason.INTERRUPTED,
                "the service stopped while this job was running"
                + (f" for {profile}" if profile else ""),
            )
            for job_id, profile in sorted(entries.items())
        )

    def _persist(self, *, job_id: str, adding: bool, profile_id: str = "") -> None:
        """Record the change, or say it could not be — never refuse the job.

        A full or read-only disk says nothing about whether the job can run:
        the tokens live in memory. Failing here used to fail submission itself.
        """
        if self._journal is None:
            return
        entries = self._journal_entries()
        if adding:
            entries[job_id] = profile_id
        else:
            entries.pop(job_id, None)
        try:
            if entries:
                self._journal.save(entries)
            else:
                self._journal.clear()
        except OSError as error:
            _LOGGER.warning(
                "in-flight journal could not be updated for %s: %s; "
                "an interrupted job may not be reported after a restart",
                job_id,
                error,
            )

    def _journal_entries(self) -> dict[str, str]:
        """The journal's content, read once rather than on every job start."""
        if self._entries is None:
            self._entries = {} if self._journal is None else dict(self._journal.load())
        return self._entries


@runtime_checkable
class CancellationJournalStore(Protocol):
    """Where the record of in-flight jobs is kept between two processes."""

    def load(self) -> Mapping[str, str]:
        """Every job the previous process left in flight, by id to profile."""

    def save(self, entries: Mapping[str, str]) -> None:
        """Replace the record; raises :class:`OSError` when it cannot."""

    def clear(self) -> None:
        """Remove the record entirely, durably."""


class JsonCancellationJournal:
    """The in-flight journal as one atomically rewritten JSON file."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def load(self) -> dict[str, str]:
        try:
            document = read_json_bounded(self._path, MAX_JOURNAL_BYTES)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, JsonTooLargeError) as error:
            # A journal we cannot read tells us nothing about what was running,
            # and refusing to start over it would be worse than starting clean.
            # Saying nothing is the part that was wrong: "no job was
            # interrupted" and "the record of what was interrupted is
            # unreadable" are different facts, and the runtime reported the
            # first for both.
            return self._discarded(f"it could not be read: {error}")
        if not isinstance(document, dict) or document.get("version") != JOURNAL_VERSION:
            return self._discarded(
                f"it is not a version {JOURNAL_VERSION} in-flight journal document"
            )
        jobs = document.get("jobs")
        if not isinstance(jobs, dict):
            return self._discarded("its job map is missing or not a map")
        return {
            str(job_id): str(profile) for job_id, profile in jobs.items() if isinstance(job_id, str)
        }

    def _discarded(self, reason: str) -> dict[str, str]:
        """Start clean, having said which record was thrown away and why."""
        _LOGGER.warning(
            "in-flight journal %s was discarded because %s;"
            " a job the previous process left running is not reported as interrupted",
            self._path,
            reason,
        )
        return {}

    def save(self, entries: Mapping[str, str]) -> None:
        write_json_atomic(
            self._path,
            {"version": JOURNAL_VERSION, "jobs": dict(entries)},
            prefix=".in-flight-",
        )

    def clear(self) -> None:
        remove_durable(self._path)
