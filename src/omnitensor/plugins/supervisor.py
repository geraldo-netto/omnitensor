"""Ordered process supervision for identity-validated plugin workers."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from .ipc import (
    FRAME_FORMAT_VERSION,
    HandshakeAgreement,
    HandshakeOffer,
    IPCFrame,
    IPCProtocolError,
    WorkerMessageType,
    handshake_frame,
    perform_service_handshake,
    write_frame,
)

MAX_SUPERVISED_WORKERS = 128
MAX_WORKER_ARGUMENTS = 128
MAX_WORKER_ARGUMENT_CHARS = 4096
DEFAULT_HANDSHAKE_TIMEOUT_SECONDS = 5.0
DEFAULT_STOP_TIMEOUT_SECONDS = 2.0


class WorkerState(StrEnum):
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    EXITED = "exited"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    """Validated launch and protocol offer for one active plugin."""

    plugin_id: str
    argv: tuple[str, ...]
    minimum_protocol: int = 1
    maximum_protocol: int = 1
    capabilities: frozenset[str] = frozenset()

    def offer(self) -> HandshakeOffer:
        return HandshakeOffer(
            self.plugin_id,
            self.minimum_protocol,
            self.maximum_protocol,
            self.capabilities,
        )


@dataclass(frozen=True, slots=True)
class WorkerStatus:
    plugin_id: str
    state: WorkerState
    pid: int | None
    protocol_version: int | None
    detail: str


class WorkerProcess(Protocol):
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    pid: int

    @property
    def returncode(self) -> int | None: ...

    async def wait(self) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


class WorkerLauncher(Protocol):
    async def launch(self, spec: WorkerSpec) -> WorkerProcess: ...


class _SubprocessWorker:
    def __init__(self, process: asyncio.subprocess.Process) -> None:
        if process.stdout is None or process.stdin is None:
            raise RuntimeError("worker pipes are unavailable")
        self._process = process
        self.reader = process.stdout
        self.writer = process.stdin
        self.pid = process.pid

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    async def wait(self) -> int:
        return await self._process.wait()

    def terminate(self) -> None:
        self._process.terminate()

    def kill(self) -> None:
        self._process.kill()


class AsyncioSubprocessLauncher:
    """Launch workers without a shell, inherited descriptors, or a shared session."""

    async def launch(self, spec: WorkerSpec) -> WorkerProcess:
        process = await asyncio.create_subprocess_exec(
            *spec.argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
        return _SubprocessWorker(process)


@dataclass(slots=True)
class _WorkerSlot:
    spec: WorkerSpec
    process: WorkerProcess
    agreement: HandshakeAgreement
    monitor: asyncio.Task | None = None


class PluginWorkerSupervisor:
    """Own exactly one child process for each successfully started plugin."""

    def __init__(
        self,
        launcher: WorkerLauncher | None = None,
        *,
        handshake_timeout: float = DEFAULT_HANDSHAKE_TIMEOUT_SECONDS,
        stop_timeout: float = DEFAULT_STOP_TIMEOUT_SECONDS,
    ) -> None:
        _validate_timeout("handshake_timeout", handshake_timeout)
        _validate_timeout("stop_timeout", stop_timeout)
        self._launcher = launcher or AsyncioSubprocessLauncher()
        self._handshake_timeout = handshake_timeout
        self._stop_timeout = stop_timeout
        self._slots: dict[str, _WorkerSlot] = {}
        self._statuses: dict[str, WorkerStatus] = {}
        self._startup_order: list[str] = []
        self._running = False
        self._lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self._running

    def statuses(self) -> tuple[WorkerStatus, ...]:
        return tuple(self._statuses[key] for key in sorted(self._statuses))

    async def start(self, specs: Sequence[WorkerSpec]) -> tuple[WorkerStatus, ...]:
        """Launch and authenticate active plugins in stable identity order."""
        ordered = _validate_specs(specs)
        async with self._lock:
            if self._running:
                raise RuntimeError("plugin supervisor is already running")
            self._running = True
            self._statuses.clear()
            self._startup_order.clear()
            for spec in ordered:
                await self._start_one(spec)
            return self.statuses()

    async def stop(self) -> tuple[WorkerStatus, ...]:
        """Request graceful shutdown, then terminate children in reverse order."""
        async with self._lock:
            if not self._running:
                return self.statuses()
            self._running = False
            for plugin_id in reversed(self._startup_order):
                slot = self._slots.pop(plugin_id, None)
                if slot is None:
                    continue
                await _cancel_monitor(slot.monitor)
                await self._stop_process(slot)
                self._statuses[plugin_id] = WorkerStatus(
                    plugin_id,
                    WorkerState.STOPPED,
                    slot.process.pid,
                    slot.agreement.protocol_version,
                    "worker stopped",
                )
            self._startup_order.clear()
            return self.statuses()

    async def _start_one(self, spec: WorkerSpec) -> None:
        self._statuses[spec.plugin_id] = WorkerStatus(
            spec.plugin_id,
            WorkerState.STARTING,
            None,
            None,
            "worker starting",
        )
        try:
            process = await self._launcher.launch(spec)
        except Exception as error:
            self._statuses[spec.plugin_id] = _failed_status(
                spec.plugin_id,
                None,
                f"worker launch failed: {type(error).__name__}",
            )
            return
        try:
            agreement = await asyncio.wait_for(
                perform_service_handshake(process.reader, process.writer, spec.offer()),
                timeout=self._handshake_timeout,
            )
        except Exception as error:
            await _force_stop(process, self._stop_timeout)
            self._statuses[spec.plugin_id] = _failed_status(
                spec.plugin_id,
                process.pid,
                _handshake_failure(error),
            )
            return
        slot = _WorkerSlot(spec, process, agreement)
        self._slots[spec.plugin_id] = slot
        self._startup_order.append(spec.plugin_id)
        self._statuses[spec.plugin_id] = WorkerStatus(
            spec.plugin_id,
            WorkerState.READY,
            process.pid,
            agreement.protocol_version,
            "worker ready",
        )
        slot.monitor = asyncio.create_task(
            self._monitor(spec.plugin_id, process),
            name=f"omnitensor-worker-{spec.plugin_id}",
        )

    async def _monitor(self, plugin_id: str, process: WorkerProcess) -> None:
        returncode = await process.wait()
        async with self._lock:
            slot = self._slots.get(plugin_id)
            if slot is None or slot.process is not process:
                return
            self._slots.pop(plugin_id)
            state = WorkerState.EXITED if returncode == 0 else WorkerState.FAILED
            detail = (
                "worker exited"
                if returncode == 0
                else f"worker exited with status {returncode}"
            )
            self._statuses[plugin_id] = WorkerStatus(
                plugin_id,
                state,
                process.pid,
                slot.agreement.protocol_version,
                detail,
            )

    async def _stop_process(self, slot: _WorkerSlot) -> None:
        if slot.process.returncode is not None:
            return
        with suppress(ConnectionError, IPCProtocolError, RuntimeError):
            await write_frame(
                slot.process.writer,
                IPCFrame(
                    FRAME_FORMAT_VERSION,
                    WorkerMessageType.CANCEL,
                    None,
                    {"reason": "shutdown"},
                ),
            )
        await _close_writer(slot.process.writer)
        await _terminate_after_timeout(slot.process, self._stop_timeout)


def _validate_specs(specs: Sequence[WorkerSpec]) -> tuple[WorkerSpec, ...]:
    if len(specs) > MAX_SUPERVISED_WORKERS:
        raise ValueError(f"at most {MAX_SUPERVISED_WORKERS} workers may be supervised")
    ordered = tuple(sorted(specs, key=lambda spec: spec.plugin_id))
    if len({spec.plugin_id for spec in ordered}) != len(ordered):
        raise ValueError("worker plugin IDs must be unique")
    for spec in ordered:
        if not spec.argv or len(spec.argv) > MAX_WORKER_ARGUMENTS:
            raise ValueError(f"worker argv must contain 1-{MAX_WORKER_ARGUMENTS} arguments")
        if any(
            not isinstance(argument, str)
            or not argument
            or "\0" in argument
            or len(argument) > MAX_WORKER_ARGUMENT_CHARS
            for argument in spec.argv
        ):
            raise ValueError("worker argv contains an invalid argument")
        handshake_frame(spec.offer())
    return ordered


def _validate_timeout(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be positive")


def _failed_status(plugin_id: str, pid: int | None, detail: str) -> WorkerStatus:
    return WorkerStatus(plugin_id, WorkerState.FAILED, pid, None, detail)


def _handshake_failure(error: Exception) -> str:
    if isinstance(error, TimeoutError):
        return "worker handshake timed out"
    if isinstance(error, IPCProtocolError):
        return f"worker handshake rejected: {error.code}"
    return f"worker handshake failed: {type(error).__name__}"


async def _cancel_monitor(task: asyncio.Task | None) -> None:
    if task is None:
        return
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with suppress(ConnectionError, RuntimeError):
        await writer.wait_closed()


async def _force_stop(process: WorkerProcess, timeout: float) -> None:
    await _close_writer(process.writer)
    await _terminate_after_timeout(process, timeout)


async def _terminate_after_timeout(process: WorkerProcess, timeout: float) -> None:
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
        return
    except TimeoutError:
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
        return
    except TimeoutError:
        process.kill()
    await asyncio.wait_for(process.wait(), timeout=timeout)
