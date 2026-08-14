"""Ordered process supervision for identity-validated plugin workers."""

from __future__ import annotations

import asyncio as asyncio
import math as math
from collections.abc import Sequence as Sequence
from contextlib import suppress as suppress
from dataclasses import dataclass as dataclass
from dataclasses import field as field
from enum import StrEnum as StrEnum
from typing import Protocol as Protocol

from . import supervisor_process as _process
from . import supervisor_recovery as _recovery
from . import supervisor_session as _session
from .budgets import WorkerBudgetCode as _WorkerBudgetCode
from .budgets import WorkerBudgetEnforcer as _WorkerBudgetEnforcer
from .budgets import WorkerBudgetExceededError as _WorkerBudgetExceededError
from .ipc import FRAME_FORMAT_VERSION as FRAME_FORMAT_VERSION
from .ipc import HandshakeAgreement as HandshakeAgreement
from .ipc import HandshakeOffer as HandshakeOffer
from .ipc import IPCFrame as IPCFrame
from .ipc import IPCProtocolError as IPCProtocolError
from .ipc import WorkerMessageType as WorkerMessageType
from .ipc import await_worker_ready as await_worker_ready
from .ipc import execute_frame as execute_frame
from .ipc import handshake_frame as handshake_frame
from .ipc import parse_progress as parse_progress
from .ipc import parse_result as parse_result
from .ipc import perform_service_handshake as perform_service_handshake
from .ipc import read_frame as read_frame
from .ipc import write_frame as write_frame
from .protocol import PluginRequest as PluginRequest
from .protocol import PluginResult as PluginResult
from .protocol import ProgressReporter as ProgressReporter
from .sandbox import FilesystemSandbox as FilesystemSandbox
from .supervisor_diagnostics import MAX_WORKER_DIAGNOSTICS as MAX_WORKER_DIAGNOSTICS
from .supervisor_diagnostics import WorkerDiagnostic as WorkerDiagnostic
from .supervisor_diagnostics import WorkerDiagnosticCode as WorkerDiagnosticCode
from .supervisor_diagnostics import WorkerState as WorkerState
from .supervisor_diagnostics import WorkerStatus as WorkerStatus
from .supervisor_diagnostics import diagnostics_for as _diagnostics_for
from .supervisor_diagnostics import failed_status as _failed_status
from .supervisor_diagnostics import record_diagnostic as _record_bounded_diagnostic
from .supervisor_process import (
    DEFAULT_CANCEL_TIMEOUT_SECONDS as DEFAULT_CANCEL_TIMEOUT_SECONDS,
)
from .supervisor_process import (
    DEFAULT_HANDSHAKE_TIMEOUT_SECONDS as DEFAULT_HANDSHAKE_TIMEOUT_SECONDS,
)
from .supervisor_process import (
    DEFAULT_STARTUP_TIMEOUT_SECONDS as DEFAULT_STARTUP_TIMEOUT_SECONDS,
)
from .supervisor_process import DEFAULT_STOP_TIMEOUT_SECONDS as DEFAULT_STOP_TIMEOUT_SECONDS
from .supervisor_process import MAX_SUPERVISED_WORKERS as MAX_SUPERVISED_WORKERS
from .supervisor_process import MAX_WORKER_ARGUMENT_CHARS as MAX_WORKER_ARGUMENT_CHARS
from .supervisor_process import MAX_WORKER_ARGUMENTS as MAX_WORKER_ARGUMENTS
from .supervisor_process import AsyncioSubprocessLauncher as AsyncioSubprocessLauncher
from .supervisor_process import WorkerLauncher as WorkerLauncher
from .supervisor_process import WorkerProcess as WorkerProcess
from .supervisor_process import WorkerSpec as WorkerSpec
from .supervisor_process import _SubprocessWorker as _SubprocessWorker
from .supervisor_process import cancel_monitor as _cancel_monitor
from .supervisor_process import close_writer as _close_writer
from .supervisor_process import terminate_after_timeout as _terminate_after_timeout
from .supervisor_process import validate_specs as _validate_specs_owner
from .supervisor_process import validate_timeout as _validate_timeout
from .supervisor_recovery import DEFAULT_MAX_RESTARTS as DEFAULT_MAX_RESTARTS
from .supervisor_recovery import (
    DEFAULT_RESTART_BACKOFF_MULTIPLIER as DEFAULT_RESTART_BACKOFF_MULTIPLIER,
)
from .supervisor_recovery import DEFAULT_RESTART_DECAY_SECONDS as DEFAULT_RESTART_DECAY_SECONDS
from .supervisor_recovery import (
    DEFAULT_RESTART_INITIAL_BACKOFF_SECONDS as DEFAULT_RESTART_INITIAL_BACKOFF_SECONDS,
)
from .supervisor_recovery import (
    DEFAULT_RESTART_MAX_BACKOFF_SECONDS as DEFAULT_RESTART_MAX_BACKOFF_SECONDS,
)
from .supervisor_recovery import MAX_RESTARTS as MAX_RESTARTS
from .supervisor_recovery import NullWorkerFailureObserver as _NullWorkerFailureObserver
from .supervisor_recovery import WorkerFailureObserver as WorkerFailureObserver
from .supervisor_recovery import WorkerRecoveryPolicy as WorkerRecoveryPolicy
from .supervisor_session import PluginWorkerError as PluginWorkerError
from .supervisor_session import WorkerSlot as _WorkerSlot
from .supervisor_session import WorkerStartError as _WorkerStartError
from .supervisor_session import with_protocol as _with_protocol


def _handshake_failure(error: Exception) -> str:
    return _session.handshake_failure(error, protocol_error_type=IPCProtocolError)


def _startup_failure(error: Exception) -> str:
    return _session.startup_failure(error, protocol_error_type=IPCProtocolError)


async def _force_stop(process: WorkerProcess, timeout: float) -> bool:
    return await _process.force_stop(
        process,
        timeout,
        close_writer_of=_close_writer,
        terminate_of=_terminate_after_timeout,
    )


def _validate_specs(specs: Sequence[WorkerSpec]) -> tuple[WorkerSpec, ...]:
    return _validate_specs_owner(
        specs,
        max_workers=MAX_SUPERVISED_WORKERS,
        max_arguments=MAX_WORKER_ARGUMENTS,
        max_argument_chars=MAX_WORKER_ARGUMENT_CHARS,
        validate_offer=handshake_frame,
    )


class PluginWorkerSupervisor:
    """Own exactly one child process for each successfully started plugin."""

    def __init__(
        self,
        launcher: WorkerLauncher | None = None,
        *,
        handshake_timeout: float = DEFAULT_HANDSHAKE_TIMEOUT_SECONDS,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
        stop_timeout: float = DEFAULT_STOP_TIMEOUT_SECONDS,
        cancel_timeout: float = DEFAULT_CANCEL_TIMEOUT_SECONDS,
        recovery_policy: WorkerRecoveryPolicy | None = None,
        failure_observer: WorkerFailureObserver | None = None,
    ) -> None:
        _validate_timeout("handshake_timeout", handshake_timeout)
        _validate_timeout("startup_timeout", startup_timeout)
        _validate_timeout("stop_timeout", stop_timeout)
        _validate_timeout("cancel_timeout", cancel_timeout)
        self._launcher = launcher or AsyncioSubprocessLauncher()
        self._handshake_timeout = handshake_timeout
        self._startup_timeout = startup_timeout
        self._stop_timeout = stop_timeout
        self._cancel_timeout = cancel_timeout
        self._recovery_policy = recovery_policy or WorkerRecoveryPolicy()
        self._failure_observer = failure_observer or _NullWorkerFailureObserver()
        self._slots: dict[str, _WorkerSlot] = {}
        self._recoveries: dict[str, asyncio.Task] = {}
        self._restart_attempts: dict[str, int] = {}
        self._ready_since: dict[str, float] = {}
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
        return _diagnostics_for(self._diagnostics, plugin_id)

    async def execute(
        self,
        request: PluginRequest,
        progress: ProgressReporter | None = None,
    ) -> PluginResult:
        """Execute one request over its authenticated worker channel."""
        slot = self._slots.get(request.plugin_id)
        status = self._statuses.get(request.plugin_id)
        if slot is None or status is None or status.state is not WorkerState.READY:
            raise PluginWorkerError("worker-unavailable", "plugin worker is not ready")
        if "execute" not in slot.agreement.capabilities:
            raise PluginWorkerError("worker-incompatible", "plugin worker cannot execute jobs")
        try:
            return await self._run_request(slot, request, progress)
        except asyncio.CancelledError:
            await asyncio.shield(self._cancel_request(slot, request.job_id))
            raise
        except _WorkerBudgetExceededError as error:
            if error.code is not _WorkerBudgetCode.CONCURRENCY:
                await asyncio.shield(_force_stop(slot.process, self._stop_timeout))
            raise PluginWorkerError(error.code.value, error.detail) from error
        except PluginWorkerError:
            raise
        except Exception as error:
            raise PluginWorkerError(
                "worker-protocol-failed",
                f"worker channel failed: {type(error).__name__}",
            ) from error

    async def _run_request(
        self,
        slot: _WorkerSlot,
        request: PluginRequest,
        progress: ProgressReporter | None,
    ) -> PluginResult:
        async def operation():
            async with slot.request_lock:
                await write_frame(
                    slot.process.writer,
                    _with_protocol(execute_frame(request), slot.agreement.protocol_version),
                )
                return await self._read_result(slot, request.job_id, progress)

        if slot.budget is None:
            return await operation()
        return await slot.budget.run(operation)

    async def _read_result(
        self,
        slot: _WorkerSlot,
        request_id: str,
        progress: ProgressReporter | None,
    ) -> PluginResult:
        return await _session.read_result(
            slot,
            request_id,
            progress,
            read=read_frame,
            progress_parser=parse_progress,
            result_parser=parse_result,
        )

    async def _cancel_request(self, slot: _WorkerSlot, request_id: str) -> None:
        await _session.cancel_request(
            slot,
            request_id,
            cancel_timeout=self._cancel_timeout,
            stop_timeout=self._stop_timeout,
            read_result_of=self._read_result,
            write=write_frame,
            force_stop=_force_stop,
        )

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
            self._restart_attempts.clear()
            self._ready_since.clear()
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
                reaped = await self._stop_process(slot)
                detail = "worker stopped" if reaped else "worker survived kill"
                if not reaped:
                    self._record_diagnostic(
                        WorkerDiagnostic(
                            plugin_id,
                            WorkerDiagnosticCode.STOP_FAILED,
                            detail,
                            slot.process.pid,
                            None,
                            0,
                        )
                    )
                self._statuses[plugin_id] = WorkerStatus(
                    plugin_id,
                    WorkerState.STOPPED,
                    slot.process.pid,
                    slot.agreement.protocol_version,
                    detail,
                    self._restart_attempts.get(plugin_id, 0),
                )
            self._startup_order.clear()
            self._restart_attempts.clear()
            self._ready_since.clear()
            return self.statuses()

    async def revoke(self, plugin_id: str, detail: str) -> WorkerStatus | None:
        """Permanently stop one worker whose immutable sandbox grant changed."""
        async with self._lock:
            recovery = self._recoveries.pop(plugin_id, None)
            if recovery is not None:
                recovery.cancel()
                await asyncio.gather(recovery, return_exceptions=True)
            slot = self._slots.pop(plugin_id, None)
            previous = self._statuses.get(plugin_id)
            if slot is None:
                self._restart_attempts.pop(plugin_id, None)
                self._ready_since.pop(plugin_id, None)
                return previous
            await _cancel_monitor(slot.monitor)
            await self._stop_process(slot)
            status = WorkerStatus(
                plugin_id,
                WorkerState.STOPPED,
                slot.process.pid,
                slot.agreement.protocol_version,
                detail,
                previous.restart_attempts if previous is not None else 0,
            )
            self._statuses[plugin_id] = status
            self._restart_attempts.pop(plugin_id, None)
            self._ready_since.pop(plugin_id, None)
        await self._cancel_orphaned_jobs(plugin_id, detail)
        return status

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
        self,
        spec: WorkerSpec,
    ) -> tuple[WorkerProcess, HandshakeAgreement]:
        return await _session.launch_authenticated(
            self._launcher,
            spec,
            handshake_timeout=self._handshake_timeout,
            startup_timeout=self._startup_timeout,
            stop_timeout=self._stop_timeout,
            handshake=perform_service_handshake,
            await_ready=await_worker_ready,
            force_stop=_force_stop,
            handshake_failure_of=_handshake_failure,
            startup_failure_of=_startup_failure,
        )

    def _install_ready_slot(
        self,
        spec: WorkerSpec,
        process: WorkerProcess,
        agreement: HandshakeAgreement,
        restart_attempts: int,
    ) -> None:
        usage_probe = getattr(process, "usage_probe", None)
        budget = (
            _WorkerBudgetEnforcer(spec.budget_limits, usage_probe)
            if callable(usage_probe)
            else None
        )
        slot = _WorkerSlot(spec, process, agreement, budget=budget)
        self._slots[spec.plugin_id] = slot
        self._ready_since[spec.plugin_id] = asyncio.get_running_loop().time()
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
        await _recovery.monitor_worker(self, plugin_id, process)

    async def _fail_startup(
        self,
        plugin_id: str,
        error: _WorkerStartError,
        attempt: int,
    ) -> None:
        await _recovery.fail_startup(self, plugin_id, error, attempt)

    async def _recover(self, spec: WorkerSpec, failure_detail: str) -> None:
        await _recovery.recover_worker(
            self,
            spec,
            failure_detail,
            force_stop=_force_stop,
        )

    async def _cancel_orphaned_jobs(self, plugin_id: str, detail: str) -> None:
        await _recovery.cancel_orphaned_jobs(self, plugin_id, detail)

    def _record_diagnostic(self, diagnostic: WorkerDiagnostic) -> None:
        _record_bounded_diagnostic(
            self._diagnostics,
            diagnostic,
            MAX_WORKER_DIAGNOSTICS,
        )

    async def _stop_process(self, slot: _WorkerSlot) -> bool:
        """Ask the worker to exit, escalating if it will not; never raise."""
        return await _session.stop_process(
            slot,
            timeout=self._stop_timeout,
            write=write_frame,
            close_writer=_close_writer,
            terminate=_terminate_after_timeout,
        )


PluginWorkerSupervisor.__module__ = "omnitensor.plugins.supervisor"
