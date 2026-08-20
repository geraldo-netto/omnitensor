"""Retrievable job progress and terminal results, bounded and owner-scoped.

A caller that submits a job needs to ask about it later, which means the
service has to remember something — and remembering per job is exactly how a
long-running process runs out of memory.  So retention is bounded twice, by
age and by count, and eviction is by age so a burst cannot push out the results
of jobs still being watched.

Ownership is enforced with a deliberate flatness: asking for someone else's job
returns exactly what asking for a job that never existed returns.  Any other
answer tells an unauthorised caller which job ids are real, which is a probe
worth nothing to a legitimate caller and quite a lot to anyone else.

Progress is replaceable rather than accumulated.  A caller wants to know where
a job is now, and keeping every update it ever reported turns one slow job into
unbounded growth.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from ..stable_error import StableError
from .protocol import PluginProgress, PluginResult

DEFAULT_MAX_RETAINED_JOBS = 256
DEFAULT_RESULT_TTL_SECONDS = 900.0
MAX_RETAINED_JOBS_LIMIT = 4096


class JobResultError(StableError, RuntimeError):
    """Stable retrieval failure safe to return over the bus."""


@dataclass(frozen=True, slots=True)
class JobRecord:
    """What is currently known about one job."""

    job_id: str
    progress: PluginProgress | None
    result: PluginResult | None
    updated_at: float

    @property
    def terminal(self) -> bool:
        return self.result is not None


class JobResultStore:
    """Bounded, expiring, owner-scoped job progress and results."""

    def __init__(
        self,
        *,
        max_retained_jobs: int = DEFAULT_MAX_RETAINED_JOBS,
        result_ttl_seconds: float = DEFAULT_RESULT_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            isinstance(max_retained_jobs, bool)
            or not isinstance(max_retained_jobs, int)
            or not 1 <= max_retained_jobs <= MAX_RETAINED_JOBS_LIMIT
        ):
            raise ValueError(
                f"max_retained_jobs must be an integer in [1, {MAX_RETAINED_JOBS_LIMIT}]"
            )
        if (
            isinstance(result_ttl_seconds, bool)
            or not isinstance(result_ttl_seconds, (int, float))
            or result_ttl_seconds <= 0
        ):
            raise ValueError("result_ttl_seconds must be a positive number")
        self._max_retained_jobs = max_retained_jobs
        self._result_ttl_seconds = result_ttl_seconds
        self._clock = clock
        self._records: OrderedDict[str, tuple[str, JobRecord]] = OrderedDict()

    def __len__(self) -> int:
        self._expire()
        return len(self._records)

    def __bool__(self) -> bool:
        return True

    def record_progress(self, job_id: str, owner: str, progress: PluginProgress) -> None:
        """Replace what is known about a running job's progress."""
        self._validate(job_id, owner)
        if not isinstance(progress, PluginProgress):
            raise JobResultError("progress-invalid", "progress must be a PluginProgress")
        existing = self._records.get(job_id)
        if existing is not None and existing[1].terminal:
            # A terminal result is final; a late progress update from a stage
            # that has not noticed yet must not reopen a finished job.
            return
        self._store(job_id, owner, JobRecord(job_id, progress, None, self._clock()))

    def record_result(self, job_id: str, owner: str, result: PluginResult) -> None:
        """Record the single terminal result for a job."""
        self._validate(job_id, owner)
        if not isinstance(result, PluginResult):
            raise JobResultError("result-invalid", "result must be a PluginResult")
        existing = self._records.get(job_id)
        progress = existing[1].progress if existing is not None else None
        self._store(job_id, owner, JobRecord(job_id, progress, result, self._clock()))

    def get(self, job_id: str, owner: str) -> JobRecord:
        """Return what is known, or refuse identically for absent and foreign."""
        self._validate(job_id, owner)
        self._expire()
        entry = self._records.get(job_id)
        if entry is None or entry[0] != owner:
            # Deliberately the same answer for "never existed" and "not yours":
            # distinguishing them would confirm which job ids are real.
            raise JobResultError("job-not-found", f"no such job: {job_id}")
        return entry[1]

    def forget(self, job_id: str, owner: str) -> bool:
        """Drop a job the caller no longer needs."""
        self._validate(job_id, owner)
        entry = self._records.get(job_id)
        if entry is None or entry[0] != owner:
            return False
        del self._records[job_id]
        return True

    def _store(self, job_id: str, owner: str, record: JobRecord) -> None:
        self._expire()
        self._records[job_id] = (owner, record)
        self._records.move_to_end(job_id)
        while len(self._records) > self._max_retained_jobs:
            # Oldest first, so a burst of new jobs cannot evict the results of
            # jobs a caller is still waiting on more recently than the burst.
            self._records.popitem(last=False)

    def _expire(self) -> None:
        deadline = self._clock() - self._result_ttl_seconds
        for job_id in [
            job_id
            for job_id, (_owner, record) in self._records.items()
            if record.updated_at < deadline
        ]:
            del self._records[job_id]

    def _validate(self, job_id: str, owner: str) -> None:
        if not isinstance(job_id, str) or not job_id or len(job_id) > 120:
            raise JobResultError("job-id-invalid", "job id must contain 1-120 characters")
        if not isinstance(owner, str) or not owner or len(owner) > 200:
            raise JobResultError("owner-invalid", "owner must contain 1-200 characters")
