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
import logging
import time
from collections import Counter, deque
from dataclasses import dataclass

from .discovery import BACKENDS
from .executors.base import (
    FORMAT_UNSUPPORTED,
    Executor,
    InferenceResult,
    availability_for_model,
    most_actionable,
    run_executor,
    supports_model,
)
from .registry import Workload

LOGGER = logging.getLogger(__name__)

LOAD_SMOOTHING = 0.5
MAX_SNAPSHOT_QUEUE_DEPTH = 1_000_000
MAX_BACKEND_QUEUE_DEPTH = MAX_SNAPSHOT_QUEUE_DEPTH // len(BACKENDS)


class QueueFullError(RuntimeError):
    """A backend rejected work because its bounded queue is full."""


NO_EXECUTOR = "no-executor"
NO_PREFERENCE = "no-preference"


@dataclass(frozen=True, slots=True)
class BackendChoice:
    """Which backend will serve a workload, and why not when none will.

    ``reason`` names the device or package and is written for a person;
    ``code`` says what kind of problem it is and is written for a program.
    Both are carried because neither can be derived from the other: a caller
    choosing a remedy must not parse prose, and a person reading a code learns
    nothing about which of three accelerators is missing.
    """

    backend: str | None
    reason: str
    code: str


def select_backend(workload: Workload, executors: dict[str, Executor]) -> BackendChoice:
    """First backend in the workload's preference that is available and
    format-compatible; otherwise the collected reasons and the most actionable
    of their codes."""
    reasons: list[str] = []
    codes: list[str] = []
    for backend in workload.preference:
        executor = executors.get(backend)
        if executor is None:
            reasons.append(f"{backend}: no executor")
            codes.append(NO_EXECUTOR)
            continue
        # A profile may declare a model per lane, so the question is not
        # "does this backend run *the* model" but "does it run any model this
        # profile declares" — which is what makes acceleratorPreference mean
        # anything for a profile that declares one at all.
        model = _runnable_model(executor, workload)
        if model is _UNSUPPORTED:
            reasons.append(f"{backend}: model format not supported")
            codes.append(FORMAT_UNSUPPORTED)
            continue
        availability = availability_for_model(executor, model)
        if not availability.available:
            reasons.append(f"{backend}: {availability.reason}")
            codes.append(availability.code)
            continue
        return BackendChoice(backend, "", "")
    if not reasons:
        return BackendChoice(None, "No backend in preference list", NO_PREFERENCE)
    return BackendChoice(None, "; ".join(reasons), most_actionable(codes) or NO_EXECUTOR)


# Distinct from None, which means "declares no model and so runs anywhere".
_UNSUPPORTED = object()


def _runnable_model(executor: Executor, workload: Workload):
    """The model this executor could run for this profile, or ``_UNSUPPORTED``."""
    models = workload.models
    if not models:
        return None
    for model in models:
        if supports_model(executor, model):
            return model
    return _UNSUPPORTED


def runnable_model(workload: Workload, backend: str, executors: dict[str, Executor]) -> dict | None:
    """The model a chosen backend will actually run, or ``None``."""
    executor = executors.get(backend)
    if executor is None:
        return None
    model = _runnable_model(executor, workload)
    return None if model is _UNSUPPORTED else model


def pick_backend(workload: Workload, executors: dict[str, Executor]) -> tuple[str | None, str]:
    """The choice above without its code, for callers that only render prose."""
    choice = select_backend(workload, executors)
    return choice.backend, choice.reason


@dataclass
class _Job:
    workload_id: str
    model_path: str
    inputs: list
    future: asyncio.Future
    model_format: str | None = None


class _BackendQueue:
    """Mutable backend scheduling state.

    Kept as a regular class because mutmut 3.7 skips methods on decorated
    classes; explicit initialization keeps stride logic inside mutation scope.
    """

    def __init__(self) -> None:
        self.profiles: dict[str, deque] = {}
        self.order: list[str] = []
        self.passes: dict[str, float] = {}
        self.busy_ms = 0.0
        self.load: float | None = None

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
        self._clamp_held_out(admits)
        return job

    def _clamp_held_out(self, admits) -> None:
        """Keep a policy-held profile level with the device's virtual time.

        A profile held out by ``admits`` keeps its queue and its pass while the
        admitted ones advance theirs.  Left alone it banks credit for service
        it was never eligible for and monopolizes the device catching up the
        moment it is re-enabled — the same burst the newcomer clamp in
        :meth:`push` exists to prevent.  Raising it to virtual time (never
        past it) neither rewards nor penalizes being held.
        """
        if admits is None:
            return
        admitted = [
            profile_id
            for profile_id, queue in self.profiles.items()
            if queue and admits(profile_id)
        ]
        if not admitted:
            return
        virtual_time = min(self.passes[profile_id] for profile_id in admitted)
        for profile_id, queue in self.profiles.items():
            if queue and not admits(profile_id):
                self.passes[profile_id] = max(self.passes[profile_id], virtual_time)


class Scheduler:
    def __init__(
        self,
        executors: dict[str, Executor],
        weight_of,
        admits=None,
        *,
        max_backend_queue_depth: int = MAX_BACKEND_QUEUE_DEPTH,
    ):
        if max_backend_queue_depth < 1:
            raise ValueError("max_backend_queue_depth must be positive")
        self._executors = dict(executors)
        self._weight_of = weight_of
        self._admits = admits
        self._max_backend_queue_depth = max_backend_queue_depth
        self._queues: dict[str, _BackendQueue] = {backend: _BackendQueue() for backend in executors}
        # Per-profile count of in-flight jobs: the same profile can run on
        # several backends at once, so membership alone would undercount.
        self._running: Counter[str] = Counter()
        self._workers: dict[str, asyncio.Task] = {}
        self._retired: list[asyncio.Task] = []
        self._wakeups: dict[str, asyncio.Event] = {
            backend: asyncio.Event() for backend in executors
        }
        self._degraded: dict[str, str] = {}
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
            self._clear_degradation(backend)
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
        self,
        backend: str,
        workload_id: str,
        model_path: str,
        inputs: list,
        *,
        model_format: str | None = None,
    ) -> asyncio.Future:
        if self._stopped:
            raise RuntimeError("scheduler is stopped")
        queue = self._queues[backend]
        if queue.depth() >= self._max_backend_queue_depth:
            raise QueueFullError(
                f"{backend} queue is full ({self._max_backend_queue_depth} jobs)",
            )
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        queue.push(_Job(workload_id, model_path, inputs, future, model_format))
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
            try:
                job = queue.pop_weighted(self._weight_of, self._admits)
                if job is None:
                    self._wakeups[backend].clear()
                    await self._wakeups[backend].wait()
                    continue
                await self._execute(backend, queue, job)
            except asyncio.CancelledError:
                raise
            except BaseException as error:  # noqa: BLE001 - the loop must outlive one job
                # Letting this end the task killed the backend silently: every
                # future already queued for it would never be resolved, and
                # callers would wait forever on a scheduler that looked healthy.
                self._degrade(backend, queue, error)

    def _degrade(self, backend: str, queue: _BackendQueue, error: BaseException) -> None:
        """Fail everything queued for a backend that just failed unexpectedly."""
        LOGGER.exception(
            "Backend %s failed outside job execution; failing its queued jobs", backend
        )
        self._degraded[backend] = type(error).__name__
        for profile_id in list(queue.profiles):
            for job in queue.profiles.pop(profile_id):
                if not job.future.done():
                    job.future.set_exception(
                        RuntimeError(f"{backend} scheduling failed: {type(error).__name__}")
                    )
            queue.order.remove(profile_id)
            queue.passes.pop(profile_id, None)

    def degraded_backends(self) -> dict[str, str]:
        """Backends that failed outside job execution, by failure type."""
        return dict(self._degraded)

    def _clear_degradation(self, backend: str) -> None:
        """Forget stale failure evidence after recovery or device removal."""
        self._degraded.pop(backend, None)

    async def _execute(self, backend: str, queue: _BackendQueue, job: _Job) -> None:
        executor = self._executors[backend]
        self._running[job.workload_id] += 1
        started = time.monotonic()
        try:
            result: InferenceResult = await asyncio.to_thread(
                run_executor,
                executor,
                job.model_path,
                job.inputs,
                model_format=job.model_format,
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
            self._clear_degradation(backend)
            if not job.future.done():
                job.future.set_result(result)
        finally:
            queue.busy_ms += (time.monotonic() - started) * 1000
            self._job_finished(job.workload_id)

    def _job_finished(self, workload_id: str) -> None:
        self._running[workload_id] -= 1
        if self._running[workload_id] <= 0:
            del self._running[workload_id]
