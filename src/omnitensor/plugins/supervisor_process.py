"""Worker process contracts, launch, validation, and reaping."""

from __future__ import annotations

import asyncio
import math
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .budgets import (
    ProcfsWorkerUsageProbe,
    WorkerBudgetLimits,
    WorkerResourceUsage,
    kill_worker_cgroup,
    remove_worker_cgroup,
)
from .ipc import HandshakeOffer, handshake_frame
from .sandbox import FilesystemSandbox

MAX_SUPERVISED_WORKERS = 128
MAX_WORKER_ARGUMENTS = 128
MAX_WORKER_ARGUMENT_CHARS = 4096
DEFAULT_HANDSHAKE_TIMEOUT_SECONDS = 5.0
DEFAULT_STARTUP_TIMEOUT_SECONDS = 60.0
DEFAULT_STOP_TIMEOUT_SECONDS = 0.5
DEFAULT_CANCEL_TIMEOUT_SECONDS = 0.25


def _facade_value(name: str, fallback):
    facade = sys.modules.get("omnitensor.plugins.supervisor")
    return getattr(facade, name, fallback) if facade is not None else fallback


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    """Validated launch and protocol offer for one active plugin."""

    plugin_id: str
    argv: tuple[str, ...]
    minimum_protocol: int = 1
    maximum_protocol: int = 1
    capabilities: frozenset[str] = frozenset()
    sandbox: FilesystemSandbox | None = None
    budget_limits: WorkerBudgetLimits = field(default_factory=WorkerBudgetLimits)

    def offer(self) -> HandshakeOffer:
        offer_type = _facade_value("HandshakeOffer", HandshakeOffer)
        return offer_type(
            self.plugin_id,
            self.minimum_protocol,
            self.maximum_protocol,
            self.capabilities,
        )


class WorkerProcess(Protocol):
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    pid: int
    usage_probe: Callable[[], WorkerResourceUsage]

    @property
    def returncode(self) -> int | None: ...

    async def wait(self) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


class WorkerLauncher(Protocol):
    async def launch(self, spec: WorkerSpec) -> WorkerProcess: ...


class _SubprocessWorker:
    def __init__(
        self,
        process: asyncio.subprocess.Process,
        usage_probe: Callable[[], WorkerResourceUsage],
        cgroup: Path | None = None,
    ) -> None:
        if process.stdout is None or process.stdin is None:
            raise RuntimeError("worker pipes are unavailable")
        self._process = process
        self.reader = process.stdout
        self.writer = process.stdin
        self.pid = process.pid
        self.usage_probe = usage_probe
        self._cgroup = cgroup

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    async def wait(self) -> int:
        returncode = await self._process.wait()
        if self._cgroup is not None:
            with suppress(OSError):
                remove_worker_cgroup(self._cgroup)
        return returncode

    def terminate(self) -> None:
        self._process.terminate()

    def kill(self) -> None:
        if self._cgroup is not None:
            try:
                kill_worker_cgroup(self._cgroup)
                return
            except OSError:
                pass
        self._process.kill()


class AsyncioSubprocessLauncher:
    """Launch workers without a shell, inherited descriptors, or a shared session."""

    # Workers are launched without a cgroup and without CPU or memory bounds.
    #
    # The unit delegated a cgroup subtree and every worker was given a child of
    # it carrying `pids.max` and `memory.max`. That could never work here: the
    # service's own PID stays in the delegated root, the kernel's
    # no-internal-process rule then refuses to populate
    # `cgroup.subtree_control`, and a child of a controller-less parent has no
    # `pids.max` to write. Every external worker failed at launch with
    # `FileNotFoundError`, which — with stderr discarded — surfaced in the
    # applet as "Configure and qualify …", sending users to look for artifacts
    # that were already installed and ready.
    #
    # Confinement that matters is still in force: `spec.sandbox` wraps the argv
    # and seccomp still gates whether a worker may start at all.
    def __init__(self, *, proc_root: Path = Path("/proc")) -> None:
        self._proc_root = Path(proc_root)

    async def launch(self, spec: WorkerSpec) -> WorkerProcess:
        argv = spec.sandbox.wrap(spec.argv) if spec.sandbox is not None else spec.argv
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            # Kept rather than discarded: a worker that dies on import said why,
            # and throwing that away is what made this failure invisible.
            stderr=asyncio.subprocess.PIPE,
            close_fds=True,
            start_new_session=True,
        )
        wrapper = _facade_value("_SubprocessWorker", _SubprocessWorker)
        probe = ProcfsWorkerUsageProbe(process.pid, proc_root=self._proc_root)
        return wrapper(process, probe, None)


def validate_specs(
    specs: Sequence[WorkerSpec],
    *,
    max_workers: int = MAX_SUPERVISED_WORKERS,
    max_arguments: int = MAX_WORKER_ARGUMENTS,
    max_argument_chars: int = MAX_WORKER_ARGUMENT_CHARS,
    validate_offer: Callable[[HandshakeOffer], object] = handshake_frame,
) -> tuple[WorkerSpec, ...]:
    if len(specs) > max_workers:
        raise ValueError(f"at most {max_workers} workers may be supervised")
    ordered = tuple(sorted(specs, key=lambda spec: spec.plugin_id))
    if len({spec.plugin_id for spec in ordered}) != len(ordered):
        raise ValueError("worker plugin IDs must be unique")
    for spec in ordered:
        if not spec.argv or len(spec.argv) > max_arguments:
            raise ValueError(f"worker argv must contain 1-{max_arguments} arguments")
        if any(
            not isinstance(argument, str)
            or not argument
            or "\0" in argument
            or len(argument) > max_argument_chars
            for argument in spec.argv
        ):
            raise ValueError("worker argv contains an invalid argument")
        validate_offer(spec.offer())
    return ordered


def validate_timeout(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive")


async def cancel_monitor(task: asyncio.Task | None) -> None:
    if task is None:
        return
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with suppress(ConnectionError, RuntimeError):
        await writer.wait_closed()


async def force_stop(
    process: WorkerProcess,
    timeout: float,
    *,
    close_writer_of=close_writer,
    terminate_of=None,
) -> bool:
    await close_writer_of(process.writer)
    return await (terminate_of or terminate_after_timeout)(process, timeout)


async def terminate_after_timeout(process: WorkerProcess, timeout: float) -> bool:
    """Escalate to SIGTERM then SIGKILL and report whether the child was reaped."""
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
        return True
    except TimeoutError:
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
        return True
    except TimeoutError:
        process.kill()
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except TimeoutError:
        return False
    return True


for _legacy_type in (WorkerSpec, WorkerProcess, WorkerLauncher, AsyncioSubprocessLauncher):
    _legacy_type.__module__ = "omnitensor.plugins.supervisor"


__all__ = [
    "AsyncioSubprocessLauncher",
    "DEFAULT_CANCEL_TIMEOUT_SECONDS",
    "DEFAULT_HANDSHAKE_TIMEOUT_SECONDS",
    "DEFAULT_STARTUP_TIMEOUT_SECONDS",
    "DEFAULT_STOP_TIMEOUT_SECONDS",
    "MAX_SUPERVISED_WORKERS",
    "MAX_WORKER_ARGUMENT_CHARS",
    "MAX_WORKER_ARGUMENTS",
    "WorkerLauncher",
    "WorkerProcess",
    "WorkerSpec",
    "cancel_monitor",
    "close_writer",
    "force_stop",
    "terminate_after_timeout",
    "validate_specs",
    "validate_timeout",
]
