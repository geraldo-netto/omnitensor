"""Ordered process supervision for identity-validated plugin workers."""

from __future__ import annotations

import asyncio
import math
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
DEFAULT_MAX_RESTARTS = 3
DEFAULT_RESTART_INITIAL_BACKOFF_SECONDS = 0.1
DEFAULT_RESTART_MAX_BACKOFF_SECONDS = 5.0
DEFAULT_RESTART_BACKOFF_MULTIPLIER = 2.0
MAX_RESTARTS = 16
MAX_WORKER_DIAGNOSTICS = 16


class WorkerState(StrEnum):
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    EXITED = "exited"
    RESTARTING = "restarting"
    EXHAUSTED = "exhausted"
    STOPPED = "stopped"


class WorkerDiagnosticCode(StrEnum):
    EXITED = "worker-exited"
    ORPHAN_CANCEL_FAILED = "orphan-cancel-failed"
    RESTART_FAILED = "restart-failed"
    RECOVERED = "worker-recovered"
    EXHAUSTED = "restart-budget-exhausted"


@dataclass(frozen=True, slots=True)
class WorkerRecoveryPolicy:
    max_restarts: int = DEFAULT_MAX_RESTARTS
    initial_backoff_seconds: float = DEFAULT_RESTART_INITIAL_BACKOFF_SECONDS
    max_backoff_seconds: float = DEFAULT_RESTART_MAX_BACKOFF_SECONDS
    backoff_multiplier: float = DEFAULT_RESTART_BACKOFF_MULTIPLIER

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_restarts, bool)
            or not isinstance(self.max_restarts, int)
            or not 0 <= self.max_restarts <= MAX_RESTARTS
        ):
            raise ValueError(f"max_restarts must be between 0 and {MAX_RESTARTS}")
        _validate_timeout("initial_backoff_seconds", self.initial_backoff_seconds)
        _validate_timeout("max_backoff_seconds", self.max_backoff_seconds)
        if self.initial_backoff_seconds > self.max_backoff_seconds:
            raise ValueError("initial_backoff_seconds must not exceed max_backoff_seconds")
        if (
            isinstance(self.backoff_multiplier, bool)
            or not isinstance(self.backoff_multiplier, (int, float))
            or not math.isfinite(self.backoff_multiplier)
            or self.backoff_multiplier < 1
        ):
            raise ValueError("backoff_multiplier must be finite and at least 1")

    def delay(self, attempt: int) -> float:
        if (
            isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or not 1 <= attempt <= MAX_RESTARTS
        ):
            raise ValueError(f"attempt must be between 1 and {MAX_RESTARTS}")
        return min(
            self.initial_backoff_seconds
            * self.backoff_multiplier ** (attempt - 1),
            self.max_backoff_seconds,
        )


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
    restart_attempts: int = 0


@dataclass(frozen=True, slots=True)
class WorkerDiagnostic:
    plugin_id: str
    code: WorkerDiagnosticCode
    detail: str
    pid: int | None
    returncode: int | None
    restart_attempt: int


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


class WorkerFailureObserver(Protocol):
    async def cancel_orphaned_jobs(self, plugin_id: str, detail: str) -> None: ...


class _NullWorkerFailureObserver:
    async def cancel_orphaned_jobs(self, plugin_id: str, detail: str) -> None:
        return None


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


class _WorkerStartError(Exception):
    def __init__(self, detail: str, pid: int | None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.pid = pid


class PluginWorkerSupervisor:
    """Own exactly one child process for each successfully started plugin."""

    def __init__(
        self,
        launcher: WorkerLauncher | None = None,
        *,
        handshake_timeout: float = DEFAULT_HANDSHAKE_TIMEOUT_SECONDS,
        stop_timeout: float = DEFAULT_STOP_TIMEOUT_SECONDS,
        recovery_policy: WorkerRecoveryPolicy | None = None,
        failure_observer: WorkerFailureObserver | None = None,
    ) -> None:
        _validate_timeout("handshake_timeout", handshake_timeout)
        _validate_timeout("stop_timeout", stop_timeout)
        self._launcher = launcher or AsyncioSubprocessLauncher()
        self._handshake_timeout = handshake_timeout
        self._stop_timeout = stop_timeout
        self._recovery_policy = recovery_policy or WorkerRecoveryPolicy()
        self._failure_observer = failure_observer or _NullWorkerFailureObserver()
        self._slots: dict[str, _WorkerSlot] = {}
        self._recoveries: dict[str, asyncio.Task] = {}
        self._statuses: dict[str, WorkerStatus] = {}
        self._diagnostics: dict[str, list[WorkerDiagnostic]] = {}
        self._startup_order: list[str] = []
        self._running = False
        self._lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self._running

    def statuses(self) -> tuple[WorkerStatus, ...]:
        return tuple(self._statuses[key] for key in sorted(self._statuses))

    def diagnostics(self, plugin_id: str) -> tuple[WorkerDiagnostic, ...]:
        return tuple(self._diagnostics.get(plugin_id, ()))

    async def start(self, specs: Sequence[WorkerSpec]) -> tuple[WorkerStatus, ...]:
        """Launch and authenticate active plugins in stable identity order."""
        ordered = _validate_specs(specs)
        async with self._lock:
            if self._running:
                raise RuntimeError("plugin supervisor is already running")
            self._running = True
            self._statuses.clear()
            self._diagnostics.clear()
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
            recovering_ids = frozenset(self._recoveries)
            recoveries = tuple(self._recoveries.values())
            self._recoveries.clear()
            for recovery in recoveries:
                recovery.cancel()
            await asyncio.gather(*recoveries, return_exceptions=True)
            for plugin_id in reversed(self._startup_order):
                slot = self._slots.pop(plugin_id, None)
                if slot is None:
                    if plugin_id in recovering_ids:
                        previous = self._statuses[plugin_id]
                        self._statuses[plugin_id] = WorkerStatus(
                            plugin_id,
                            WorkerState.STOPPED,
                            previous.pid,
                            previous.protocol_version,
                            "worker recovery stopped",
                            previous.restart_attempts,
                        )
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
            process, agreement = await self._launch_authenticated(spec)
        except _WorkerStartError as error:
            self._statuses[spec.plugin_id] = _failed_status(
                spec.plugin_id,
                error.pid,
                error.detail,
            )
            return
        self._install_ready_slot(spec, process, agreement, 0)

    async def _launch_authenticated(
        self, spec: WorkerSpec
    ) -> tuple[WorkerProcess, HandshakeAgreement]:
        try:
            process = await self._launcher.launch(spec)
        except Exception as error:
            raise _WorkerStartError(
                f"worker launch failed: {type(error).__name__}", None
            ) from error
        try:
            agreement = await asyncio.wait_for(
                perform_service_handshake(process.reader, process.writer, spec.offer()),
                timeout=self._handshake_timeout,
            )
        except asyncio.CancelledError:
            await _force_stop(process, self._stop_timeout)
            raise
        except Exception as error:
            await _force_stop(process, self._stop_timeout)
            raise _WorkerStartError(
                _handshake_failure(error), process.pid
            ) from error
        return process, agreement

    def _install_ready_slot(
        self,
        spec: WorkerSpec,
        process: WorkerProcess,
        agreement: HandshakeAgreement,
        restart_attempts: int,
    ) -> None:
        slot = _WorkerSlot(spec, process, agreement)
        self._slots[spec.plugin_id] = slot
        if spec.plugin_id not in self._startup_order:
            self._startup_order.append(spec.plugin_id)
        detail = (
            "worker ready"
            if restart_attempts == 0
            else f"worker recovered after {restart_attempts} restart attempts"
        )
        self._statuses[spec.plugin_id] = WorkerStatus(
            spec.plugin_id,
            WorkerState.READY,
            process.pid,
            agreement.protocol_version,
            detail,
            restart_attempts,
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
            self._record_diagnostic(
                WorkerDiagnostic(
                    plugin_id,
                    WorkerDiagnosticCode.EXITED,
                    detail,
                    process.pid,
                    returncode,
                    0,
                )
            )
            if self._running and self._recovery_policy.max_restarts:
                recovery = asyncio.create_task(
                    self._recover(slot.spec, detail),
                    name=f"omnitensor-recovery-{plugin_id}",
                )
                self._recoveries[plugin_id] = recovery

    async def _recover(self, spec: WorkerSpec, failure_detail: str) -> None:
        plugin_id = spec.plugin_id
        await self._cancel_orphaned_jobs(plugin_id, failure_detail)
        last_detail = failure_detail
        for attempt in range(1, self._recovery_policy.max_restarts + 1):
            async with self._lock:
                if (
                    not self._running
                    or self._recoveries.get(plugin_id) is not asyncio.current_task()
                ):
                    return
                previous = self._statuses[plugin_id]
                self._statuses[plugin_id] = WorkerStatus(
                    plugin_id,
                    WorkerState.RESTARTING,
                    previous.pid,
                    previous.protocol_version,
                    f"worker restart attempt {attempt}",
                    attempt,
                )
            await asyncio.sleep(self._recovery_policy.delay(attempt))
            try:
                process, agreement = await self._launch_authenticated(spec)
            except _WorkerStartError as error:
                last_detail = error.detail
                self._record_diagnostic(
                    WorkerDiagnostic(
                        plugin_id,
                        WorkerDiagnosticCode.RESTART_FAILED,
                        error.detail,
                        error.pid,
                        None,
                        attempt,
                    )
                )
                continue
            async with self._lock:
                if (
                    not self._running
                    or self._recoveries.get(plugin_id) is not asyncio.current_task()
                ):
                    await _force_stop(process, self._stop_timeout)
                    return
                self._install_ready_slot(spec, process, agreement, attempt)
                self._record_diagnostic(
                    WorkerDiagnostic(
                        plugin_id,
                        WorkerDiagnosticCode.RECOVERED,
                        f"worker recovered after {attempt} restart attempts",
                        process.pid,
                        None,
                        attempt,
                    )
                )
                self._recoveries.pop(plugin_id, None)
                return
        async with self._lock:
            if self._recoveries.get(plugin_id) is not asyncio.current_task():
                return
            previous = self._statuses[plugin_id]
            detail = (
                f"worker restart budget exhausted after "
                f"{self._recovery_policy.max_restarts} attempts: {last_detail}"
            )
            self._statuses[plugin_id] = WorkerStatus(
                plugin_id,
                WorkerState.EXHAUSTED,
                previous.pid,
                previous.protocol_version,
                detail,
                self._recovery_policy.max_restarts,
            )
            self._record_diagnostic(
                WorkerDiagnostic(
                    plugin_id,
                    WorkerDiagnosticCode.EXHAUSTED,
                    detail,
                    previous.pid,
                    None,
                    self._recovery_policy.max_restarts,
                )
            )
            self._recoveries.pop(plugin_id, None)

    async def _cancel_orphaned_jobs(self, plugin_id: str, detail: str) -> None:
        try:
            await asyncio.wait_for(
                self._failure_observer.cancel_orphaned_jobs(plugin_id, detail),
                timeout=self._stop_timeout,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._record_diagnostic(
                WorkerDiagnostic(
                    plugin_id,
                    WorkerDiagnosticCode.ORPHAN_CANCEL_FAILED,
                    f"orphan cancellation failed: {type(error).__name__}",
                    None,
                    None,
                    0,
                )
            )

    def _record_diagnostic(self, diagnostic: WorkerDiagnostic) -> None:
        diagnostics = self._diagnostics.setdefault(diagnostic.plugin_id, [])
        diagnostics.append(diagnostic)
        del diagnostics[:-MAX_WORKER_DIAGNOSTICS]

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
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
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
