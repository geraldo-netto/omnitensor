"""Stable worker states and supervisor diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# What a registered worker that is deliberately not running has to say for
# itself. Published as the worker detail so a reader can tell "nobody has asked
# for anything" from "it would not start".
IDLE_WORKER_DETAIL = "worker idle; starts on demand"


class WorkerState(StrEnum):
    # Registered and startable, with no process running: the state of a plugin
    # nobody has asked for anything yet, and the one a worker returns to after
    # its idle exit. It is not a failure and not an outage.
    IDLE = "idle"
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
    maximum: int | None = None,
) -> None:
    """Append one diagnostic to its plugin's journal.

    The journal is kept whole by default: a worker in a restart loop records
    its root cause first, and a bound that drops the oldest entries is exactly
    the one that throws that entry away. A caller that must bound the journal
    passes a `maximum` of at least one; anything smaller is a caller defect,
    not a request to keep nothing.
    """
    if maximum is not None and (isinstance(maximum, bool) or not isinstance(maximum, int)):
        raise TypeError("diagnostic journal bound must be an integer")
    if maximum is not None and maximum < 1:
        raise ValueError("diagnostic journal bound must be at least 1")
    journal = diagnostics.setdefault(diagnostic.plugin_id, [])
    journal.append(diagnostic)
    if maximum is not None:
        del journal[:-maximum]


def forget_diagnostics(
    diagnostics: dict[str, list[WorkerDiagnostic]],
    plugin_id: str,
) -> None:
    """Drop one plugin's journal once nothing can consult it again."""
    diagnostics.pop(plugin_id, None)


__all__ = [
    "IDLE_WORKER_DETAIL",
    "WorkerDiagnostic",
    "WorkerDiagnosticCode",
    "WorkerState",
    "WorkerStatus",
    "diagnostics_for",
    "failed_status",
    "forget_diagnostics",
    "record_diagnostic",
]
