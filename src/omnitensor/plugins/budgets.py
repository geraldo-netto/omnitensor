"""Fail-closed per-worker call and resource budgets."""

from __future__ import annotations

import asyncio
import errno
import math
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from ..stable_error import StableError

DEFAULT_CALL_TIMEOUT_SECONDS = 30.0
# Unbounded by default. These ceilings refused real work — a Qwen worker
# legitimately holding 831 MB was killed against a 512 MiB limit once its model
# had loaded — and the in-process check was already observe-only for that
# reason. What protects the machine now is host pressure, measured across the
# whole host rather than guessed per worker. A caller may still set any of
# these; ``None`` means no ceiling.
DEFAULT_MAX_PROCESSES = None
DEFAULT_MAX_MEMORY_BYTES = None
DEFAULT_MAX_DESCRIPTORS = None
DEFAULT_MAX_CONCURRENCY = 1
MAX_CALL_TIMEOUT_SECONDS = 3600.0
MAX_PROCESSES_LIMIT = 128
MAX_MEMORY_BYTES_LIMIT = 64 * 1024 * 1024 * 1024
MAX_DESCRIPTORS_LIMIT = 65_536
MAX_CONCURRENCY_LIMIT = 32
MAX_PROCFS_PROCESSES = 4096


class WorkerBudgetCode(StrEnum):
    """Why a call was refused. Every member here is raised by this module.

    The resource codes that once lived beside these — process, memory,
    descriptor, usage-unavailable — were never raised after the ceilings became
    observe-only, and a refusal code nothing can produce is a branch every
    client has to carry for nothing.
    """

    DEADLINE = "deadline-exceeded"
    CONCURRENCY = "concurrency-budget-exceeded"


class WorkerBudgetExceededError(StableError, RuntimeError):
    """Stable worker-call failure safe to cross the service boundary."""


@dataclass(frozen=True, slots=True)
class WorkerBudgetLimits:
    call_timeout_seconds: float = DEFAULT_CALL_TIMEOUT_SECONDS
    max_processes: int | None = DEFAULT_MAX_PROCESSES
    max_memory_bytes: int | None = DEFAULT_MAX_MEMORY_BYTES
    max_descriptors: int | None = DEFAULT_MAX_DESCRIPTORS
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY

    def __post_init__(self) -> None:
        _bounded_number(
            self.call_timeout_seconds,
            "call_timeout_seconds",
            MAX_CALL_TIMEOUT_SECONDS,
        )
        _optional_bounded_integer(self.max_processes, "max_processes", MAX_PROCESSES_LIMIT)
        _optional_bounded_integer(self.max_memory_bytes, "max_memory_bytes", MAX_MEMORY_BYTES_LIMIT)
        _optional_bounded_integer(self.max_descriptors, "max_descriptors", MAX_DESCRIPTORS_LIMIT)
        _bounded_integer(self.max_concurrency, "max_concurrency", MAX_CONCURRENCY_LIMIT)


@dataclass(frozen=True, slots=True)
class WorkerResourceUsage:
    processes: int
    memory_bytes: int
    descriptors: int

    def __post_init__(self) -> None:
        for name in ("processes", "memory_bytes", "descriptors"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class WorkerBudgetSnapshot:
    active_calls: int
    peak_calls: int
    completed_calls: int
    rejected_calls: int


class WorkerBudgetEnforcer:
    """Run one async operation under this worker's call deadline and concurrency.

    It takes no usage probe. Resource ceilings became observe-only when they
    started killing work that was about to succeed, and the observation that
    replaced them sampled the whole process tree every 50 ms for the life of
    every call — 12,000 walks of ``/proc`` for one 600-second generative call —
    and dropped every sample on the floor. What protects the machine now is
    host pressure, measured across the whole host rather than per worker.
    """

    def __init__(self, limits: WorkerBudgetLimits) -> None:
        self._limits = limits
        self._active = 0
        self._peak = 0
        self._completed = 0
        self._rejected = 0
        self._lock = asyncio.Lock()

    def snapshot(self) -> WorkerBudgetSnapshot:
        return WorkerBudgetSnapshot(
            self._active,
            self._peak,
            self._completed,
            self._rejected,
        )

    async def run(self, operation: Callable[[], Awaitable]) -> object:
        if not callable(operation):
            raise TypeError("operation must be callable")
        await self._enter()
        task = None
        try:
            task = asyncio.create_task(operation())
            result = await self._wait(task)
            # Nothing here inspects the size of the answer. A result too large
            # for one IPC frame is split across frames by the transport; it was
            # refused here once, which discarded a whole document-QA answer and
            # cost a model reload on the next request.
            self._completed += 1
            return result
        finally:
            # Release the slot before any await, and without the lock: taking
            # the lock is itself a suspension point, so cancellation delivered
            # there — or at the task cleanup below — would abandon the
            # decrement and leak a concurrency slot for the life of the
            # enforcer.  A bare decrement runs to completion between awaits,
            # which is all the mutual exclusion this counter needs.
            self._active -= 1
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def _enter(self) -> None:
        async with self._lock:
            if self._active >= self._limits.max_concurrency:
                self._rejected += 1
                raise WorkerBudgetExceededError(
                    WorkerBudgetCode.CONCURRENCY,
                    f"at most {self._limits.max_concurrency} calls may run concurrently",
                )
            self._active += 1
            self._peak = max(self._peak, self._active)

    async def _wait(self, task: asyncio.Task) -> object:
        """The call's answer, or its deadline — one sleep, no polling.

        The event loop wakes this exactly twice: once when the call answers and
        once if it does not answer in time. An idle worker costs no wakeups at
        all, which is what an idle runtime is required to cost.
        """
        done, _pending = await asyncio.wait((task,), timeout=self._limits.call_timeout_seconds)
        if not done:
            self._rejected += 1
            raise WorkerBudgetExceededError(
                WorkerBudgetCode.DEADLINE,
                f"call exceeded {self._limits.call_timeout_seconds:g} seconds",
            )
        return await task


class ProcfsWorkerUsageProbe:
    """Read one Linux worker process tree without importing plugin code."""

    def __init__(self, pid: int, *, proc_root: Path = Path("/proc")) -> None:
        if type(pid) is not int or pid < 1:
            raise ValueError("pid must be a positive integer")
        self._pid = pid
        self._root = Path(proc_root)

    def __call__(self) -> WorkerResourceUsage:
        processes = self._process_tree()
        memory = sum(self._resident_bytes(pid) for pid in processes)
        descriptors = sum(self._descriptor_count(pid) for pid in processes)
        return WorkerResourceUsage(len(processes), memory, descriptors)

    def _process_tree(self) -> tuple[int, ...]:
        pending = deque((self._pid,))
        observed = []
        seen = set()
        while pending:
            pid = pending.popleft()
            if pid in seen:
                continue
            seen.add(pid)
            observed.append(pid)
            if len(observed) > MAX_PROCFS_PROCESSES:
                raise OSError(f"process tree exceeds {MAX_PROCFS_PROCESSES} entries")
            pending.extend(self._children(pid))
        return tuple(observed)

    def _children(self, pid: int) -> list[int]:
        """Children of every thread of ``pid``, not only of its main thread.

        The kernel records a child under the task that forked it, so reading
        ``task/<pid>/children`` alone misses anything a plugin forked from a
        worker thread — and what it misses is exactly what escapes the
        process, memory, and descriptor limits.
        """
        children: list[int] = []
        try:
            threads = sorted((self._root / str(pid) / "task").iterdir(), key=lambda p: p.name)
        except OSError as error:
            if _vanished(error):
                return children
            raise
        for thread in threads:
            try:
                text = (thread / "children").read_text(encoding="ascii").strip()
            except OSError as error:
                if _vanished(error):
                    continue
                raise
            children.extend(int(child) for child in text.split())
        return children

    def _resident_bytes(self, pid: int) -> int:
        try:
            status = (self._root / str(pid) / "status").read_text(encoding="ascii")
        except OSError as error:
            if _vanished(error):
                return 0
            raise
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB":
                    return int(fields[1]) * 1024
        raise OSError("VmRSS is unavailable")

    def _descriptor_count(self, pid: int) -> int:
        return _descriptor_count(self._root, pid)


def _descriptor_count(self, pid: int) -> int:
    return _descriptor_count(self._root, pid)


def _descriptor_count(proc_root: Path, pid: int) -> int:
    try:
        return sum(1 for _entry in (proc_root / str(pid) / "fd").iterdir())
    except OSError as error:
        if _vanished(error):
            return 0
        raise


def _vanished(error: OSError) -> bool:
    """A pid that exited mid-probe contributes no usage; it is not a failure.

    Enumerating a tree and then reading each entry can never be atomic, so a
    process exiting between the two steps is routine.  Treating it as an error
    turned an ordinary race into a spurious usage-unavailable rejection.
    """
    return error.errno in (errno.ENOENT, errno.ESRCH)


def _optional_bounded_integer(value: object, name: str, maximum: int) -> None:
    """Validate a ceiling that may be absent; ``None`` is "no ceiling"."""
    if value is None:
        return
    _bounded_integer(value, name, maximum)


def _bounded_integer(value: int, name: str, maximum: int) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")


def _bounded_number(value: float, name: str, maximum: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value <= maximum
    ):
        raise ValueError(f"{name} must be finite and in (0, {maximum:g}]")
