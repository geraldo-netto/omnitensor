"""Fail-closed per-worker call and resource budgets."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import math
import os
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, is_dataclass
from enum import StrEnum
from pathlib import Path

DEFAULT_CALL_TIMEOUT_SECONDS = 30.0
# Bubblewrap retains a supervisor and namespace init around the worker.  Four
# covers that fixed three-process sandbox topology plus no more than one
# short-lived helper; additional forks are rejected.
DEFAULT_MAX_PROCESSES = 4
DEFAULT_MAX_MEMORY_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_DESCRIPTORS = 128
DEFAULT_MAX_OUTPUT_BYTES = 1024 * 1024
DEFAULT_MAX_CONCURRENCY = 1
DEFAULT_RESOURCE_POLL_SECONDS = 0.05
MAX_CALL_TIMEOUT_SECONDS = 3600.0
MAX_PROCESSES_LIMIT = 128
MAX_MEMORY_BYTES_LIMIT = 64 * 1024 * 1024 * 1024
MAX_DESCRIPTORS_LIMIT = 65_536
MAX_OUTPUT_BYTES_LIMIT = 64 * 1024 * 1024
MAX_CONCURRENCY_LIMIT = 32
MAX_PROCFS_PROCESSES = 4096
CGROUP_PROCS_FILE = "cgroup.procs"
CGROUP_MEMORY_FILE = "memory.current"
CGROUP_MEMORY_LIMIT_FILE = "memory.max"
CGROUP_PROCESS_LIMIT_FILE = "pids.max"
CGROUP_KILL_FILE = "cgroup.kill"


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
            self.max_output_bytes, "max_output_bytes", MAX_OUTPUT_BYTES_LIMIT
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


class CgroupWorkerUsageProbe:
    """Account one worker by cgroup membership instead of process parentage.

    Walking ``children`` links cannot see a process that double-forked or was
    reparented: severing the parent link is precisely what daemonising does,
    so a plugin that daemonises escapes every process, memory, and descriptor
    limit.  Cgroup membership survives both, so a worker cannot leave its
    accounting by forking.

    The service must run with a delegated cgroup subtree (``Delegate=yes``)
    and place each worker in its own sub-cgroup for this to observe anything.
    """

    def __init__(self, cgroup: Path, *, proc_root: Path = Path("/proc")) -> None:
        self._cgroup = Path(cgroup)
        self._root = Path(proc_root)

    def __call__(self) -> WorkerResourceUsage:
        members = self._members()
        descriptors = sum(_descriptor_count(self._root, pid) for pid in members)
        return WorkerResourceUsage(len(members), self._memory_bytes(), descriptors)

    def _members(self) -> tuple[int, ...]:
        text = (self._cgroup / CGROUP_PROCS_FILE).read_text(encoding="ascii")
        members = tuple(int(entry) for entry in text.split())
        if len(members) > MAX_PROCFS_PROCESSES:
            raise OSError(f"worker cgroup exceeds {MAX_PROCFS_PROCESSES} entries")
        return members

    def _memory_bytes(self) -> int:
        text = (self._cgroup / CGROUP_MEMORY_FILE).read_text(encoding="ascii").strip()
        # The controller reports "max" when it is enabled but unbounded; that
        # is a configuration state, not a measurement.
        if not text.isdigit():
            raise OSError(f"cgroup memory accounting is unavailable: {text!r}")
        return int(text)


def create_worker_cgroup(parent: Path, name: str) -> Path:
    """Create the sub-cgroup one worker will be confined to.

    ``parent`` is the service's own delegated cgroup.  The directory is the
    entire creation protocol: the kernel populates its interface files.
    """
    if not name or "/" in name or name in (".", ".."):
        raise ValueError(f"invalid worker cgroup name: {name!r}")
    cgroup = Path(parent) / name
    cgroup.mkdir(parents=False, exist_ok=True)
    return cgroup


def configure_worker_cgroup(cgroup: Path, limits: WorkerBudgetLimits) -> None:
    """Install kernel-enforced process and aggregate-memory ceilings."""
    root = Path(cgroup)
    (root / CGROUP_PROCESS_LIMIT_FILE).write_text(
        str(limits.max_processes), encoding="ascii"
    )
    (root / CGROUP_MEMORY_LIMIT_FILE).write_text(
        str(limits.max_memory_bytes), encoding="ascii"
    )


def current_process_cgroup(
    *,
    membership_path: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> Path | None:
    """Resolve this process's unified cgroup-v2 directory, if available."""
    try:
        membership = Path(membership_path).read_text(encoding="ascii")
    except OSError:
        return None
    root = Path(cgroup_root).resolve()
    for line in membership.splitlines():
        try:
            hierarchy, controllers, member = line.split(":", 2)
        except ValueError:
            continue
        if hierarchy != "0" or controllers:
            continue
        candidate = (root / member.lstrip("/")).resolve()
        if (candidate == root or root in candidate.parents) and os.access(
            candidate, os.W_OK
        ):
            return candidate
        return None
    return None


def join_worker_cgroup(cgroup: Path) -> None:
    """Move the calling process into ``cgroup``.

    Written from the child between fork and exec, this closes the window in
    which a worker could fork before it has been confined.
    """
    with open(Path(cgroup) / CGROUP_PROCS_FILE, "w", encoding="ascii") as stream:
        stream.write("0")


def kill_worker_cgroup(cgroup: Path) -> None:
    """Kill every process retained in a worker's cgroup-v2 subtree."""
    (Path(cgroup) / CGROUP_KILL_FILE).write_text("1", encoding="ascii")


def remove_worker_cgroup(cgroup: Path) -> None:
    """Remove an empty worker cgroup, tolerating one that is already gone."""
    with contextlib.suppress(FileNotFoundError):
        Path(cgroup).rmdir()


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


def _check_output(value: object, limit: int) -> None:
    document = asdict(value) if is_dataclass(value) and not isinstance(value, type) else value
    try:
        body = json.dumps(
            document,
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
