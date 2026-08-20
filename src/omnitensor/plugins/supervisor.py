from __future__ import annotations

import asyncio
from collections.abc import Sequence

from . import supervisor_process as _process
from . import supervisor_recovery as _recovery
from . import supervisor_session as _session
from .budgets import WorkerBudgetCode as _WorkerBudgetCode
from .budgets import WorkerBudgetEnforcer as _WorkerBudgetEnforcer
from .budgets import WorkerBudgetExceededError as _WorkerBudgetExceededError
from .ipc import (
    HandshakeAgreement,
    IPCProtocolError,
    await_worker_ready,
    execute_frame,
    handshake_frame,
    parse_progress,
    parse_result,
    perform_service_handshake,
    read_frame,
    write_frame,
)
from .liveness import CallHeartbeat, report
from .protocol import PluginRequest, PluginResult, ProgressReporter
from .supervisor_diagnostics import (
    IDLE_WORKER_DETAIL,
    WorkerDiagnostic,
    WorkerDiagnosticCode,
    WorkerState,
    WorkerStatus,
)
from .supervisor_diagnostics import diagnostics_for as _diagnostics_for
from .supervisor_diagnostics import failed_status as _failed_status
from .supervisor_diagnostics import forget_diagnostics as _forget_diagnostics
from .supervisor_diagnostics import record_diagnostic as _record_journal_diagnostic
from .supervisor_process import (
    DEFAULT_CANCEL_TIMEOUT_SECONDS,
    DEFAULT_HANDSHAKE_TIMEOUT_SECONDS,
    DEFAULT_STARTUP_TIMEOUT_SECONDS,
    DEFAULT_STOP_TIMEOUT_SECONDS,
    MAX_SUPERVISED_WORKERS,
    MAX_WORKER_ARGUMENT_CHARS,
    MAX_WORKER_ARGUMENTS,
    AsyncioSubprocessLauncher,
    WorkerLauncher,
    WorkerProcess,
    WorkerSpec,
)
from .supervisor_process import cancel_monitor as _cancel_monitor
from .supervisor_process import close_writer as _close_writer
from .supervisor_process import terminate_after_timeout as _terminate_after_timeout
from .supervisor_process import validate_specs as _validate_specs_owner
from .supervisor_process import validate_timeout as _validate_timeout
from .supervisor_recovery import NullWorkerFailureObserver as _NullWorkerFailureObserver
from .supervisor_recovery import WorkerFailureObserver, WorkerRecoveryPolicy
from .supervisor_session import PluginWorkerError
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


class _BeatingReporter:
    """The caller's progress reporter, plus the two clocks' sign of life.

    A wrapper rather than a change to `read_result`: the transport's job is to
    read frames and hand them on, and whether anything is timing the call is
    not its business. Beating the job-wide heartbeat as well is what keeps the
    flow controller outside this call from giving up on a job that is working.
    """

    __slots__ = ("_heartbeat", "_reporter")

    def __init__(self, reporter: ProgressReporter | None, heartbeat: CallHeartbeat) -> None:
        self._reporter = reporter
        self._heartbeat = heartbeat

    async def report(self, progress) -> None:
        self._heartbeat.beat()
        report()
        if self._reporter is not None:
            await self._reporter.report(progress)


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
        self._specs: dict[str, WorkerSpec] = {}
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
        except PluginWorkerError as error:
            await self._stop_protocol_failure(slot, error)
            raise
        except Exception as error:
            await asyncio.shield(_force_stop(slot.process, self._stop_timeout))
            raise PluginWorkerError(
                "worker-protocol-failed",
                f"worker channel failed: {type(error).__name__}",
            ) from error

    async def _stop_protocol_failure(
        self,
        slot: _WorkerSlot,
        error: PluginWorkerError,
    ) -> None:
        if error.code == "worker-protocol-failed":
            await asyncio.shield(_force_stop(slot.process, self._stop_timeout))

    async def _run_request(
        self,
        slot: _WorkerSlot,
        request: PluginRequest,
        progress: ProgressReporter | None,
    ) -> PluginResult:
        # Every progress frame this call reads is a sign of life, for the
        # worker's own budget and for the flow controller waiting outside it.
        # Without it a call was cut off at its deadline however steadily the
        # worker was reporting, which is how a long transcription lost its
        # answer at the ten-minute mark.
        heartbeat = CallHeartbeat()
        beating_progress = _BeatingReporter(progress, heartbeat)

        async def operation():
            async with slot.request_lock:
                await write_frame(
                    slot.process.writer,
                    _with_protocol(execute_frame(request), slot.agreement.protocol_version),
                )
                return await self._read_result(slot, request.job_id, beating_progress)

        if slot.budget is None:
            return await operation()
        return await slot.budget.run(operation, heartbeat)

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
        async with self._lock:
            ordered = self._admit_specs(specs)
            for spec in ordered:
                await self._start_one(spec)
            return self.statuses()

    async def register(self, specs: Sequence[WorkerSpec]) -> tuple[WorkerStatus, ...]:
        """Accept the plugins that may run, without launching any of them.

        A worker launched at startup is a process holding a model for someone
        who has not asked for anything.  Registration keeps every guarantee
        :meth:`start` gives — validated specs, stable identity order, one slot
        per plugin — and leaves the launch to the first job routed there.
        """
        async with self._lock:
            ordered = self._admit_specs(specs)
            for spec in ordered:
                self._statuses[spec.plugin_id] = WorkerStatus(
                    spec.plugin_id,
                    WorkerState.IDLE,
                    None,
                    None,
                    IDLE_WORKER_DETAIL,
                )
            return self.statuses()

    def _admit_specs(self, specs: Sequence[WorkerSpec]) -> tuple[WorkerSpec, ...]:
        """Validate and adopt one generation of specs; the caller holds the lock."""
        ordered = _validate_specs(specs)
        if self._running:
            raise RuntimeError("plugin supervisor is already running")
        self._running = True
        self._statuses.clear()
        self._diagnostics.clear()
        self._startup_order.clear()
        self._restart_attempts.clear()
        self._ready_since.clear()
        self._specs = {spec.plugin_id: spec for spec in ordered}
        return ordered

    async def ensure_ready(self, plugin_id: str) -> WorkerStatus | None:
        """Launch this plugin's worker unless it is already serving.

        The caller pays the start-up cost of the first request after an idle
        exit; it is never refused one.  A worker whose restart budget ran out
        is tried again here, because a fresh request is a new demand rather
        than another turn of a restart loop.
        """
        async with self._lock:
            if not self._running:
                return None
            status = self._statuses.get(plugin_id)
            if plugin_id in self._slots or plugin_id in self._recoveries:
                # Serving, or a recovery already owns the launch.
                return status
            spec = self._specs.get(plugin_id)
            if spec is None:
                return status
            self._restart_attempts[plugin_id] = 0
            await self._start_one(spec)
            return self._statuses.get(plugin_id)

    async def stop_worker(
        self,
        plugin_id: str,
        detail: str = IDLE_WORKER_DETAIL,
    ) -> WorkerStatus | None:
        """Stop one worker that is not needed, leaving it startable again.

        Unlike :meth:`revoke` this is not a verdict on the plugin: the process
        goes away, its accelerator lease and model memory with it, and the
        next request starts it back up.
        """
        async with self._lock:
            recovery = self._recoveries.pop(plugin_id, None)
            if recovery is not None:
                recovery.cancel()
                await asyncio.gather(recovery, return_exceptions=True)
            slot = self._slots.pop(plugin_id, None)
            previous = self._statuses.get(plugin_id)
            if slot is None:
                return previous
            await _cancel_monitor(slot.monitor)
            await self._stop_process(slot)
            status = WorkerStatus(plugin_id, WorkerState.IDLE, None, None, detail)
            self._statuses[plugin_id] = status
            self._restart_attempts.pop(plugin_id, None)
            self._ready_since.pop(plugin_id, None)
            return status

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
            self._specs.clear()
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
                _forget_diagnostics(self._diagnostics, plugin_id)
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
            # A revoked plugin never runs again under this generation of
            # specs, so its journal has no reader left and must not outlive it.
            _forget_diagnostics(self._diagnostics, plugin_id)
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
        # Every ready worker gets its budget. It used to depend on the process
        # exposing a usage probe, so a launcher that offered none left the call
        # deadline and the concurrency limit unenforced for that worker.
        slot = _WorkerSlot(
            spec, process, agreement, budget=_WorkerBudgetEnforcer(spec.budget_limits)
        )
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
        _record_journal_diagnostic(
            self._diagnostics,
            diagnostic,
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
