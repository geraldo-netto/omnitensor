"""Worker restart policy and recovery state transitions."""

from __future__ import annotations

import asyncio
import math
import sys
from dataclasses import dataclass
from typing import Protocol

from .supervisor_diagnostics import (
    WorkerDiagnostic,
    WorkerDiagnosticCode,
    WorkerState,
    WorkerStatus,
)
from .supervisor_process import WorkerProcess, WorkerSpec, validate_timeout
from .supervisor_session import WorkerStartError

DEFAULT_MAX_RESTARTS = 3
DEFAULT_RESTART_INITIAL_BACKOFF_SECONDS = 0.1
DEFAULT_RESTART_MAX_BACKOFF_SECONDS = 5.0
DEFAULT_RESTART_BACKOFF_MULTIPLIER = 2.0
MAX_RESTARTS = 16


def _facade_value(name: str, fallback):
    facade = sys.modules.get("omnitensor.plugins.supervisor")
    return getattr(facade, name, fallback) if facade is not None else fallback


@dataclass(frozen=True, slots=True)
class WorkerRecoveryPolicy:
    max_restarts: int = DEFAULT_MAX_RESTARTS
    initial_backoff_seconds: float = DEFAULT_RESTART_INITIAL_BACKOFF_SECONDS
    max_backoff_seconds: float = DEFAULT_RESTART_MAX_BACKOFF_SECONDS
    backoff_multiplier: float = DEFAULT_RESTART_BACKOFF_MULTIPLIER

    def __post_init__(self) -> None:
        maximum = _facade_value("MAX_RESTARTS", MAX_RESTARTS)
        if (
            isinstance(self.max_restarts, bool)
            or not isinstance(self.max_restarts, int)
            or not 0 <= self.max_restarts <= maximum
        ):
            raise ValueError(f"max_restarts must be between 0 and {maximum}")
        validator = _facade_value("_validate_timeout", validate_timeout)
        validator("initial_backoff_seconds", self.initial_backoff_seconds)
        validator("max_backoff_seconds", self.max_backoff_seconds)
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
        maximum = _facade_value("MAX_RESTARTS", MAX_RESTARTS)
        if (
            isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or not 1 <= attempt <= maximum
        ):
            raise ValueError(f"attempt must be between 1 and {maximum}")
        return min(
            self.initial_backoff_seconds * self.backoff_multiplier ** (attempt - 1),
            self.max_backoff_seconds,
        )


class WorkerFailureObserver(Protocol):
    async def cancel_orphaned_jobs(self, plugin_id: str, detail: str) -> None: ...


class NullWorkerFailureObserver:
    async def cancel_orphaned_jobs(self, plugin_id: str, detail: str) -> None:
        return None


async def monitor_worker(host, plugin_id: str, process: WorkerProcess) -> None:
    returncode = await process.wait()
    async with host._lock:
        slot = host._slots.get(plugin_id)
        if slot is None or slot.process is not process:
            return
        host._slots.pop(plugin_id)
        state = WorkerState.EXITED if returncode == 0 else WorkerState.FAILED
        detail = "worker exited" if returncode == 0 else f"worker exited with status {returncode}"
        host._statuses[plugin_id] = WorkerStatus(
            plugin_id,
            state,
            process.pid,
            slot.agreement.protocol_version,
            detail,
        )
        host._record_diagnostic(
            WorkerDiagnostic(
                plugin_id,
                WorkerDiagnosticCode.EXITED,
                detail,
                process.pid,
                returncode,
                0,
            )
        )
        if host._running and host._recovery_policy.max_restarts:
            recovery = asyncio.create_task(
                host._recover(slot.spec, detail),
                name=f"omnitensor-recovery-{plugin_id}",
            )
            host._recoveries[plugin_id] = recovery


async def fail_startup(
    host,
    plugin_id: str,
    error: WorkerStartError,
    attempt: int,
) -> None:
    async with host._lock:
        if host._recoveries.get(plugin_id) is not asyncio.current_task():
            return
        previous = host._statuses[plugin_id]
        host._statuses[plugin_id] = WorkerStatus(
            plugin_id,
            WorkerState.FAILED,
            error.pid,
            previous.protocol_version,
            error.detail,
            attempt - 1,
        )
        host._recoveries.pop(plugin_id, None)


async def recover_worker(
    host,
    spec: WorkerSpec,
    failure_detail: str,
    *,
    force_stop,
) -> None:
    plugin_id = spec.plugin_id
    await host._cancel_orphaned_jobs(plugin_id, failure_detail)
    last_detail = failure_detail
    for attempt in range(1, host._recovery_policy.max_restarts + 1):
        async with host._lock:
            if (
                not host._running
                or host._recoveries.get(plugin_id) is not asyncio.current_task()
            ):
                return
            previous = host._statuses[plugin_id]
            host._statuses[plugin_id] = WorkerStatus(
                plugin_id,
                WorkerState.RESTARTING,
                previous.pid,
                previous.protocol_version,
                f"worker restart attempt {attempt}",
                attempt,
            )
        await asyncio.sleep(host._recovery_policy.delay(attempt))
        start_error_type = _facade_value("_WorkerStartError", WorkerStartError)
        try:
            process, agreement = await host._launch_authenticated(spec)
        except start_error_type as error:
            last_detail = error.detail
            host._record_diagnostic(
                WorkerDiagnostic(
                    plugin_id,
                    WorkerDiagnosticCode.RESTART_FAILED
                    if error.charges_restart
                    else WorkerDiagnosticCode.STARTUP_TIMEOUT,
                    error.detail,
                    error.pid,
                    None,
                    attempt,
                )
            )
            if not error.charges_restart:
                await host._fail_startup(plugin_id, error, attempt)
                return
            continue
        installed = False
        try:
            async with host._lock:
                if (
                    not host._running
                    or host._recoveries.get(plugin_id) is not asyncio.current_task()
                ):
                    return
                host._install_ready_slot(spec, process, agreement, attempt)
                host._record_diagnostic(
                    WorkerDiagnostic(
                        plugin_id,
                        WorkerDiagnosticCode.RECOVERED,
                        f"worker recovered after {attempt} restart attempts",
                        process.pid,
                        None,
                        attempt,
                    )
                )
                host._recoveries.pop(plugin_id, None)
                installed = True
                return
        finally:
            if not installed:
                await force_stop(process, host._stop_timeout)
    async with host._lock:
        if host._recoveries.get(plugin_id) is not asyncio.current_task():
            return
        previous = host._statuses[plugin_id]
        detail = (
            f"worker restart budget exhausted after "
            f"{host._recovery_policy.max_restarts} attempts: {last_detail}"
        )
        host._statuses[plugin_id] = WorkerStatus(
            plugin_id,
            WorkerState.EXHAUSTED,
            previous.pid,
            previous.protocol_version,
            detail,
            host._recovery_policy.max_restarts,
        )
        host._record_diagnostic(
            WorkerDiagnostic(
                plugin_id,
                WorkerDiagnosticCode.EXHAUSTED,
                detail,
                previous.pid,
                None,
                host._recovery_policy.max_restarts,
            )
        )
        host._recoveries.pop(plugin_id, None)


async def cancel_orphaned_jobs(host, plugin_id: str, detail: str) -> None:
    try:
        await asyncio.wait_for(
            host._failure_observer.cancel_orphaned_jobs(plugin_id, detail),
            timeout=host._stop_timeout,
        )
    except asyncio.CancelledError:
        raise
    except Exception as error:
        host._record_diagnostic(
            WorkerDiagnostic(
                plugin_id,
                WorkerDiagnosticCode.ORPHAN_CANCEL_FAILED,
                f"orphan cancellation failed: {type(error).__name__}",
                None,
                None,
                0,
            )
        )


for _legacy_type in (WorkerRecoveryPolicy, WorkerFailureObserver):
    _legacy_type.__module__ = "omnitensor.plugins.supervisor"


__all__ = [
    "DEFAULT_MAX_RESTARTS",
    "DEFAULT_RESTART_BACKOFF_MULTIPLIER",
    "DEFAULT_RESTART_INITIAL_BACKOFF_SECONDS",
    "DEFAULT_RESTART_MAX_BACKOFF_SECONDS",
    "MAX_RESTARTS",
    "NullWorkerFailureObserver",
    "WorkerFailureObserver",
    "WorkerRecoveryPolicy",
    "cancel_orphaned_jobs",
    "fail_startup",
    "monitor_worker",
    "recover_worker",
]
