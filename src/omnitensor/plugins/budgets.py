"""Fail-closed per-worker call and resource budgets."""

from __future__ import annotations

import asyncio
import json
import math
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

DEFAULT_CALL_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_PROCESSES = 1
DEFAULT_MAX_MEMORY_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_DESCRIPTORS = 128
DEFAULT_MAX_OUTPUT_BYTES = 1024 * 1024
DEFAULT_MAX_CONCURRENCY = 1
DEFAULT_RESOURCE_POLL_SECONDS = 0.05
MAX_CALL_TIMEOUT_SECONDS = 3600.0
MAX_PROCESSES_LIMIT = 128
MAX_MEMORY_BYTES_LIMIT = 64 * 1024 * 1024 * 1024
MAX_DESCRIPTORS_LIMIT = 65_536
MAX_CONCURRENCY_LIMIT = 32
MAX_PROCFS_PROCESSES = 4096


class WorkerBudgetCode(StrEnum):
    DEADLINE = "deadline-exceeded"
    PROCESS = "process-budget-exceeded"
    MEMORY = "memory-budget-exceeded"
    DESCRIPTOR = "descriptor-budget-exceeded"
    OUTPUT = "output-budget-exceeded"
    CONCURRENCY = "concurrency-budget-exceeded"
    USAGE_UNAVAILABLE = "usage-unavailable"


class WorkerBudgetExceededError(RuntimeError):
    """Stable worker-call failure safe to cross the service boundary."""

    def __init__(self, code: WorkerBudgetCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class WorkerBudgetLimits:
    call_timeout_seconds: float = DEFAULT_CALL_TIMEOUT_SECONDS
    max_processes: int = DEFAULT_MAX_PROCESSES
    max_memory_bytes: int = DEFAULT_MAX_MEMORY_BYTES
    max_descriptors: int = DEFAULT_MAX_DESCRIPTORS
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    resource_poll_seconds: float = DEFAULT_RESOURCE_POLL_SECONDS

    def __post_init__(self) -> None:
        _bounded_number(
            self.call_timeout_seconds,
            "call_timeout_seconds",
            MAX_CALL_TIMEOUT_SECONDS,
        )
        _bounded_integer(self.max_processes, "max_processes", MAX_PROCESSES_LIMIT)
        _bounded_integer(
            self.max_memory_bytes, "max_memory_bytes", MAX_MEMORY_BYTES_LIMIT
        )
        _bounded_integer(
            self.max_descriptors, "max_descriptors", MAX_DESCRIPTORS_LIMIT
        )
        _bounded_integer(
            self.max_output_bytes, "max_output_bytes", DEFAULT_MAX_OUTPUT_BYTES
        )
        _bounded_integer(
            self.max_concurrency, "max_concurrency", MAX_CONCURRENCY_LIMIT
        )
        _bounded_number(
            self.resource_poll_seconds,
            "resource_poll_seconds",
            min(1.0, self.call_timeout_seconds),
        )


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
    """Run one async operation while polling all declared worker limits."""

    def __init__(
        self,
        limits: WorkerBudgetLimits,
        usage_probe: Callable[[], WorkerResourceUsage],
    ) -> None:
        if not callable(usage_probe):
            raise TypeError("usage_probe must be callable")
        self._limits = limits
        self._usage_probe = usage_probe
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
            self._check_usage()
            task = asyncio.create_task(operation())
            result = await self._wait(task)
            self._check_usage()
            try:
                _check_output(result, self._limits.max_output_bytes)
            except WorkerBudgetExceededError:
                self._rejected += 1
                raise
            self._completed += 1
            return result
        finally:
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            async with self._lock:
                self._active -= 1

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
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._limits.call_timeout_seconds
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                self._rejected += 1
                raise WorkerBudgetExceededError(
                    WorkerBudgetCode.DEADLINE,
                    f"call exceeded {self._limits.call_timeout_seconds:g} seconds",
                )
            done, _pending = await asyncio.wait(
                (task,),
                timeout=min(remaining, self._limits.resource_poll_seconds),
            )
            if done:
                return await task
            self._check_usage()

    def _check_usage(self) -> None:
        try:
            usage = self._usage_probe()
        except Exception as error:
            self._rejected += 1
            raise WorkerBudgetExceededError(
                WorkerBudgetCode.USAGE_UNAVAILABLE,
                f"resource usage probe failed: {type(error).__name__}",
            ) from error
        if not isinstance(usage, WorkerResourceUsage):
            self._rejected += 1
            raise WorkerBudgetExceededError(
                WorkerBudgetCode.USAGE_UNAVAILABLE,
                "resource usage probe returned an invalid value",
            )
        checks = (
            (
                usage.processes,
                self._limits.max_processes,
                WorkerBudgetCode.PROCESS,
                "processes",
            ),
            (
                usage.memory_bytes,
                self._limits.max_memory_bytes,
                WorkerBudgetCode.MEMORY,
                "memory bytes",
            ),
            (
                usage.descriptors,
                self._limits.max_descriptors,
                WorkerBudgetCode.DESCRIPTOR,
                "descriptors",
            ),
        )
        for observed, limit, code, label in checks:
            if observed > limit:
                self._rejected += 1
                raise WorkerBudgetExceededError(
                    code,
                    f"worker reported {observed} {label}; limit is {limit}",
                )


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
            children = self._root / str(pid) / "task" / str(pid) / "children"
            text = children.read_text(encoding="ascii").strip()
            if text:
                pending.extend(int(child) for child in text.split())
        return tuple(observed)

    def _resident_bytes(self, pid: int) -> int:
        status = (self._root / str(pid) / "status").read_text(encoding="ascii")
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB":
                    return int(fields[1]) * 1024
        raise OSError("VmRSS is unavailable")

    def _descriptor_count(self, pid: int) -> int:
        return sum(1 for _entry in (self._root / str(pid) / "fd").iterdir())


def _check_output(value: object, limit: int) -> None:
    try:
        body = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise WorkerBudgetExceededError(
            WorkerBudgetCode.OUTPUT,
            "worker output is not finite JSON",
        ) from error
    if len(body) > limit:
        raise WorkerBudgetExceededError(
            WorkerBudgetCode.OUTPUT,
            f"worker output is {len(body)} bytes; limit is {limit}",
        )


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
