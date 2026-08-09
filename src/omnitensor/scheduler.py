"""Weighted workload scheduling with per-device serialized dispatch.

One worker per backend device serializes dispatch to that accelerator, as
one physical device cannot be partitioned.  Under contention profiles
share a device by deficit round robin: each round a profile earns credit
equal to its weight (1–5), so weights approximate proportional shares
without hard isolation.  Per-device load is an exponential moving average
of busy time between snapshot ticks.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter, deque
from dataclasses import dataclass, field

from .executors.base import Executor, InferenceResult, supports_model
from .registry import Workload

LOAD_SMOOTHING = 0.5


def pick_backend(workload: Workload, executors: dict[str, Executor]) -> tuple[str | None, str]:
    """First backend in the workload's preference that is available and
    format-compatible; otherwise ``(None, reason)``."""
    reasons: list[str] = []
    for backend in workload.preference:
        executor = executors.get(backend)
        if executor is None:
            reasons.append(f"{backend}: no executor")
            continue
        if not supports_model(executor, workload.model):
            reasons.append(f"{backend}: model format not supported")
            continue
        availability = executor.availability()
        if not availability.available:
            reasons.append(f"{backend}: {availability.reason}")
            continue
        return backend, ""
    return None, "; ".join(reasons) or "No backend in preference list"


@dataclass
class _Job:
    workload_id: str
    model_path: str
    inputs: list
    future: asyncio.Future


@dataclass
class _BackendQueue:
    profiles: dict[str, deque] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    passes: dict[str, float] = field(default_factory=dict)
    busy_ms: float = 0.0
    load: float | None = None

    def push(self, job: _Job) -> None:
        if job.workload_id not in self.profiles:
            self.profiles[job.workload_id] = deque()
            self.order.append(job.workload_id)
            # Anti-burst: a newcomer joins at the device's current virtual
            # time (the smallest tracked pass) rather than 0, so it cannot
            # monopolize the device "catching up" on service it never queued
            # for while others advanced their passes.
            self.passes[job.workload_id] = min(self.passes.values(), default=0.0)
        self.profiles[job.workload_id].append(job)

    def depth(self) -> int:
        return sum(len(queue) for queue in self.profiles.values())

    def pop_weighted(self, weight_of, admits=None) -> _Job | None:
        """Stride scheduling: the pending profile with the smallest pass value
        runs next, and its pass advances by 1/weight — so a weight-5 profile
        is served roughly five times per weight-1 serve under contention.
        Profiles the ``admits`` policy predicate rejects stay queued."""
        pending = [
            profile_id
            for profile_id in self.order
            if self.profiles[profile_id] and (admits is None or admits(profile_id))
        ]
        if not pending:
            return None
        chosen = min(
            pending,
            key=lambda profile_id: (self.passes[profile_id], self.order.index(profile_id)),
        )
        stride = 1.0 / max(1, min(5, weight_of(chosen)))
        self.passes[chosen] += stride
        job = self.profiles[chosen].popleft()
        if not self.profiles[chosen]:
            # Prune drained profiles so profiles/order/passes cannot grow
            # without bound; a returning profile re-joins at virtual time.
            del self.profiles[chosen]
            self.order.remove(chosen)
            del self.passes[chosen]
        return job


class Scheduler:
    def __init__(self, executors: dict[str, Executor], weight_of, admits=None):
        self._executors = dict(executors)
        self._weight_of = weight_of
        self._admits = admits
        self._queues: dict[str, _BackendQueue] = {backend: _BackendQueue() for backend in executors}
        # Per-profile count of in-flight jobs: the same profile can run on
        # several backends at once, so membership alone would undercount.
        self._running: Counter[str] = Counter()
        self._workers: dict[str, asyncio.Task] = {}
        self._retired: list[asyncio.Task] = []
        self._wakeups: dict[str, asyncio.Event] = {
            backend: asyncio.Event() for backend in executors
        }
        self._started = False
        self._stopped = False
        self._last_tick = time.monotonic()

    def start(self) -> None:
        self._started = True
        for backend in self._executors:
            self._spawn_worker(backend)

    def _spawn_worker(self, backend: str) -> None:
        self._workers[backend] = asyncio.get_running_loop().create_task(self._worker(backend))

    def update_executors(self, executors: dict[str, Executor]) -> None:
        """Swap in the executors from the latest discovery pass so dispatch
        never uses stale adapters.  Existing backends keep their queue and
        worker (the worker looks the executor up per job); new backends get a
        queue and, when running, a worker; queued jobs of removed backends are
        cancelled because no device can ever serve them again."""
        removed = [backend for backend in self._executors if backend not in executors]
        self._executors = dict(executors)
        # Reap retired workers that have finished cancelling so the list
        # cannot grow without bound under repeated backend churn; tasks
        # still unwinding stay tracked until stop() awaits them.
        self._retired = [task for task in self._retired if not task.done()]
        for backend in removed:
            worker = self._workers.pop(backend, None)
            if worker is not None:
                worker.cancel()
                self._retired.append(worker)
            self._wakeups.pop(backend, None)
            self._cancel_queued(self._queues.pop(backend))
        for backend in executors:
            if backend not in self._queues:
                self._queues[backend] = _BackendQueue()
                self._wakeups[backend] = asyncio.Event()
                if self._started and not self._stopped:
                    self._spawn_worker(backend)

    async def stop(self) -> None:
        """Stop dispatch deterministically: cancel the workers, then cancel the
        future of every job still queued so no awaiting caller can hang."""
        self._stopped = True
        for event in self._wakeups.values():
            event.set()
        workers = [*self._workers.values(), *self._retired]
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        self._workers.clear()
        self._retired.clear()
        for queue in self._queues.values():
            self._cancel_queued(queue)

    @staticmethod
    def _cancel_queued(queue: _BackendQueue) -> None:
        for jobs in queue.profiles.values():
            while jobs:
                job = jobs.popleft()
                if not job.future.done():
                    job.future.cancel()

    def submit(
        self, backend: str, workload_id: str, model_path: str, inputs: list,
    ) -> asyncio.Future:
        if self._stopped:
            raise RuntimeError("scheduler is stopped")
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._queues[backend].push(_Job(workload_id, model_path, inputs, future))
        self._wakeups[backend].set()
        return future

    def stats(self) -> dict:
        return {
            "queueDepth": sum(queue.depth() for queue in self._queues.values()),
            "runningProfiles": len(self._running),
            "loads": {backend: queue.load for backend, queue in self._queues.items()},
        }

    def profile_stats(self) -> dict[str, dict[str, int]]:
        """Per-profile queued and running job counts across all backends;
        profiles with no queued or running work are absent."""
        stats: dict[str, dict[str, int]] = {}
        for queue in self._queues.values():
            for profile_id, jobs in queue.profiles.items():
                entry = stats.setdefault(profile_id, {"queued": 0, "running": 0})
                entry["queued"] += len(jobs)
        for profile_id, count in self._running.items():
            entry = stats.setdefault(profile_id, {"queued": 0, "running": 0})
            entry["running"] += count
        return stats

    def tick(self) -> None:
        """Fold busy time since the last snapshot tick into each device's load EMA."""
        now = time.monotonic()
        elapsed_ms = max(1.0, (now - self._last_tick) * 1000)
        self._last_tick = now
        for queue in self._queues.values():
            ratio = min(100.0, 100.0 * queue.busy_ms / elapsed_ms)
            queue.busy_ms = 0.0
            queue.load = ratio if queue.load is None else (
                LOAD_SMOOTHING * ratio + (1 - LOAD_SMOOTHING) * queue.load
            )

    def kick(self) -> None:
        """Wake every worker so held jobs are re-evaluated against the current
        admission policy (call after a pause/enable policy change)."""
        for event in self._wakeups.values():
            event.set()

    async def _worker(self, backend: str) -> None:
        queue = self._queues[backend]
        while not self._stopped:
            job = queue.pop_weighted(self._weight_of, self._admits)
            if job is None:
                self._wakeups[backend].clear()
                await self._wakeups[backend].wait()
                continue
            await self._execute(backend, queue, job)

    async def _execute(self, backend: str, queue: _BackendQueue, job: _Job) -> None:
        executor = self._executors[backend]
        self._running[job.workload_id] += 1
        started = time.monotonic()
        try:
            result: InferenceResult = await asyncio.to_thread(
                executor.run, job.model_path, job.inputs,
            )
        except asyncio.CancelledError:
            # Worker cancellation (stop) is not an executor failure: cancel the
            # caller's future and let the cancellation propagate.
            if not job.future.done():
                job.future.cancel()
            raise
        except BaseException as error:  # noqa: BLE001 - failures propagate to the caller
            if not job.future.done():
                job.future.set_exception(
                    error if isinstance(error, Exception) else RuntimeError(repr(error)),
                )
            if not isinstance(error, Exception):
                raise
        else:
            if not job.future.done():
                job.future.set_result(result)
        finally:
            queue.busy_ms += (time.monotonic() - started) * 1000
            self._job_finished(job.workload_id)

    def _job_finished(self, workload_id: str) -> None:
        self._running[workload_id] -= 1
        if self._running[workload_id] <= 0:
            del self._running[workload_id]
