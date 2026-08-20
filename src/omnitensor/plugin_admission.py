"""Weighted admission for jobs that run in plugin workers.

Inference jobs queue in the :mod:`~omnitensor.scheduler` backend queues, where
a profile's ``weight`` decides how often it is served under contention.  Jobs
that run in an external plugin worker never entered those queues: they were
dispatched the moment they were accepted, so every plugin ran as many jobs at
once as callers asked for, and ``weight`` decided nothing for the six workloads
a person actually starts by hand.

This is the missing half.  A plugin job now takes a slot from a bounded pool
before its worker call begins, and when the pool is full the waiting profiles
are served by the same stride rule the scheduler uses — smallest pass value
first, pass advanced by ``1 / weight`` — so a weight-5 profile is served about
five times per weight-1 serve while both have work waiting.

Two properties are deliberate:

* **Held, not refused.**  A profile the policy predicate rejects (disabled, or
  the whole runtime paused) keeps its place and waits.  Its pass is clamped to
  the pool's virtual time so it neither banks credit while held nor is
  penalised for having been held — the same clamp ``_BackendQueue`` applies.
* **The slot spans the worker call, not the queue.**  Releasing on dispatch
  would make the pool a rate limiter over submission rather than a bound on
  concurrent work, and weight would again decide nothing.
* **A full host holds the pool, it does not empty it.**  When the machine is
  above its memory threshold no *additional* job starts, but one always may:
  the pressure is often somebody else's browser, and a runtime that stopped
  entirely until an unrelated process let go would be worse than one that
  slows down.

Tuning lives in ``docs/plugin-admission.md``.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable

from .job_ports import JobDispatchError

# How many plugin jobs may run at once, across every plugin.  Four: wide enough
# that ordinary use does not queue behind one long answer, narrow enough that
# four model workers still fit while the host-pressure gate decides whether a
# fifth would.  Weight still orders whatever waits.
DEFAULT_MAX_CONCURRENT = 4
# The ceiling the pool may be tuned to, so a bad configuration cannot ask for
# more concurrent model processes than the host can hold.
MAX_CONCURRENT_LIMIT = 32
# Waiting jobs, across every profile.  A job that cannot even queue is refused
# at admission, before an id is minted, rather than waiting forever.
DEFAULT_MAX_WAITING = 64
MAX_WAITING_LIMIT = 1024
# The stride ceiling, matching the scheduler: policy weights are 1..5.
MAX_STRIDE_WEIGHT = 5


def _bounded(value: object, name: str, limit: int) -> int:
    if type(value) is not int or not 1 <= value <= limit:
        raise ValueError(f"{name} must be an integer between 1 and {limit}")
    return value


class PluginAdmissionQueue:
    """A weighted, bounded pool of concurrent plugin-worker jobs."""

    def __init__(
        self,
        *,
        weight_of: Callable[[str], int],
        admits: Callable[[str], bool] | None = None,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        max_waiting: int = DEFAULT_MAX_WAITING,
        pressure: Callable[[], object] | None = None,
    ) -> None:
        self._weight_of = weight_of
        self._admits = admits
        # Read at the moment a slot would be handed out, not on a timer: a
        # reading from two seconds ago is a reading from before the model that
        # just loaded.
        self._pressure = pressure
        self._max_concurrent = _bounded(max_concurrent, "max_concurrent", MAX_CONCURRENT_LIMIT)
        self._max_waiting = _bounded(max_waiting, "max_waiting", MAX_WAITING_LIMIT)
        self._waiting: dict[str, deque[asyncio.Future]] = {}
        self._order: list[str] = []
        self._passes: dict[str, float] = {}
        self._running: dict[str, int] = {}

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    def waiting(self) -> int:
        return sum(len(queue) for queue in self._waiting.values())

    def running(self) -> int:
        return sum(self._running.values())

    def has_room(self) -> bool:
        """Whether one more job could at least wait."""
        return self.waiting() < self._max_waiting

    def profile_stats(self) -> dict[str, dict[str, int]]:
        """Queued and running counts per profile, shaped as the scheduler's are."""
        stats: dict[str, dict[str, int]] = {}
        for profile_id, queue in self._waiting.items():
            stats.setdefault(profile_id, {"queued": 0, "running": 0})["queued"] += len(queue)
        for profile_id, count in self._running.items():
            stats.setdefault(profile_id, {"queued": 0, "running": 0})["running"] += count
        return stats

    def kick(self) -> None:
        """Re-evaluate waiters against the current policy.

        Called when policy changes: a profile disabled while its jobs waited is
        now skipped, and one re-enabled is now eligible without anything else
        having to finish first.
        """
        self._pump()

    async def run(self, profile_id: str, start: Callable[[], object]):
        """Await a slot, then run ``start()`` inside it.

        ``start`` is called only once the slot is held, so a queued job has not
        yet touched its worker; cancelling while it waits leaves nothing to
        stop and frees no slot it never took.
        """
        await self._acquire(profile_id)
        try:
            return await start()
        finally:
            self._release(profile_id)

    async def _acquire(self, profile_id: str) -> None:
        # Every job joins the queue, including one arriving at an empty pool:
        # taking a slot directly would jump whatever is already waiting, and
        # testing "is anything waiting?" instead would let a profile held by
        # policy block an eligible one from a slot that is free.  The queue
        # decides; with a free slot and no rival, it decides immediately.
        waiter: asyncio.Future = asyncio.get_running_loop().create_future()
        self._push(profile_id, waiter)
        self._pump()
        try:
            await waiter
        except asyncio.CancelledError:
            self._discard(profile_id, waiter)
            # A slot handed over in the same tick the waiter was cancelled is
            # already counted as running and would otherwise leak: give it back.
            if waiter.done() and not waiter.cancelled():
                self._release(profile_id)
            raise

    def _push(self, profile_id: str, waiter: asyncio.Future) -> None:
        if self.waiting() >= self._max_waiting:
            # Enforced here, not only in has_room(): a dispatcher reached
            # without going through admit() - the documented admission=None
            # default, or a runner-driven re-dispatch - used to queue without
            # any bound at all, so the module's promise above was advisory.
            raise JobDispatchError(
                "plugin-queue-full", "Too many plugin jobs are waiting for a worker slot"
            )
        if profile_id not in self._waiting:
            self._waiting[profile_id] = deque()
            self._order.append(profile_id)
            # A newcomer joins at the pool's current virtual time rather than
            # at zero, so it cannot monopolise the pool catching up on service
            # it never waited for.
            self._passes[profile_id] = min(self._passes.values(), default=0.0)
        self._waiting[profile_id].append(waiter)

    def _discard(self, profile_id: str, waiter: asyncio.Future) -> None:
        queue = self._waiting.get(profile_id)
        if queue is None:
            return
        try:
            queue.remove(waiter)
        except ValueError:
            return
        if not queue:
            self._prune(profile_id)

    def _prune(self, profile_id: str) -> None:
        self._waiting.pop(profile_id, None)
        if profile_id in self._order:
            self._order.remove(profile_id)
        self._passes.pop(profile_id, None)

    def _release(self, profile_id: str) -> None:
        remaining = self._running.get(profile_id, 0) - 1
        if remaining > 0:
            self._running[profile_id] = remaining
        else:
            self._running.pop(profile_id, None)
        self._pump()

    def width(self) -> int:
        """How many jobs may run right now.

        The configured width while the host has room; one while it does not, so
        work continues at the slowest rate that still makes progress. Never
        zero: a queue that can never be served is a queue that never empties.
        """
        if self._pressure is None:
            return self._max_concurrent
        try:
            reading = self._pressure()
        except Exception:  # noqa: BLE001 - a failed probe must not stop work
            return self._max_concurrent
        return 1 if getattr(reading, "saturated", False) else self._max_concurrent

    def _pump(self) -> None:
        while self.running() < self.width():
            profile_id = self._next_profile()
            if profile_id is None:
                return
            waiter = self._waiting[profile_id].popleft()
            if not self._waiting[profile_id]:
                self._prune(profile_id)
            if waiter.cancelled():
                # Cancelled between being queued and being chosen: it never
                # took the slot, so the next waiter gets it instead.
                continue
            self._running[profile_id] = self._running.get(profile_id, 0) + 1
            waiter.set_result(None)

    def _next_profile(self) -> str | None:
        """Stride selection: smallest pass first, advanced by ``1 / weight``."""
        pending = [
            profile_id
            for profile_id in self._order
            if self._waiting[profile_id] and (self._admits is None or self._admits(profile_id))
        ]
        if not pending:
            return None
        chosen = min(
            pending,
            key=lambda profile_id: (self._passes[profile_id], self._order.index(profile_id)),
        )
        weight = self._weight_of(chosen)
        weight = weight if isinstance(weight, int) and not isinstance(weight, bool) else 1
        self._passes[chosen] += 1.0 / max(1, min(MAX_STRIDE_WEIGHT, weight))
        self._clamp_held_out()
        return chosen

    def _clamp_held_out(self) -> None:
        """Keep a policy-held profile level with the pool's virtual time."""
        if self._admits is None:
            return
        admitted = [
            profile_id
            for profile_id, queue in self._waiting.items()
            if queue and self._admits(profile_id)
        ]
        if not admitted:
            return
        virtual_time = min(self._passes[profile_id] for profile_id in admitted)
        for profile_id, queue in self._waiting.items():
            if queue and not self._admits(profile_id):
                self._passes[profile_id] = max(self._passes[profile_id], virtual_time)
