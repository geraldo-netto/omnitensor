"""Stable worker states and bounded supervisor diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

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
    STOP_FAILED = "worker-stop-failed"
    STARTUP_TIMEOUT = "worker-startup-timeout"


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


def failed_status(plugin_id: str, pid: int | None, detail: str) -> WorkerStatus:
    return WorkerStatus(plugin_id, WorkerState.FAILED, pid, None, detail)


def diagnostics_for(
    diagnostics: dict[str, list[WorkerDiagnostic]],
    plugin_id: str,
) -> tuple[WorkerDiagnostic, ...]:
    return tuple(diagnostics.get(plugin_id, ()))


def record_diagnostic(
    diagnostics: dict[str, list[WorkerDiagnostic]],
    diagnostic: WorkerDiagnostic,
    maximum: int,
) -> None:
    journal = diagnostics.setdefault(diagnostic.plugin_id, [])
    journal.append(diagnostic)
    del journal[:-maximum]



__all__ = [
    "MAX_WORKER_DIAGNOSTICS",
    "WorkerDiagnostic",
    "WorkerDiagnosticCode",
    "WorkerState",
    "WorkerStatus",
    "diagnostics_for",
    "failed_status",
    "record_diagnostic",
]
