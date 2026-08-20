"""Leaf contracts for runtime job policy, dispatch, and observation."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol, runtime_checkable


class JobAuthorizer(Protocol):
    def allows(self, action: str, workload_id: str) -> bool: ...


@runtime_checkable
class JobDispatcher(Protocol):
    def dispatch(self, job_id: str, workload_id: str, payload: dict) -> Awaitable[object]: ...


@runtime_checkable
class JobAdmission(Protocol):
    """Everything about a job that can be refused before it becomes active.

    Separate from :class:`JobDispatcher` because dispatching is asynchronous by
    the time a pipeline runs it: whatever the runner's stages decide is decided
    after this service has already answered ``accepted``. A refusal a caller
    can act on has to be reachable synchronously, in the submitting call, which
    is what this port is for. ``runtime_checkable`` lets wiring offer a
    dispatcher that also admits without a second registration.
    """

    def admit(self, workload_id: str, payload: dict) -> None:
        """Return normally, or raise :class:`JobDispatchError` with a reason."""


@runtime_checkable
class PreparedLaneDispatcher(Protocol):
    """A dispatcher that freezes the lane before the job is handed to it.

    Named here rather than probed with ``getattr(..., "prepare_lane")`` at four
    call sites: the capability is a contract, and an adapter that implements
    half of it should fail where it is wired, not degrade to the backend name
    as a lane at the first job.
    """

    def prepare_lane(self, workload_id: str) -> tuple: ...

    def dispatch_prepared(self, job_id: str, workload_id: str, payload: dict, lane) -> object: ...


class JobDispatchError(RuntimeError):
    """Stable admission failure raised before a job becomes active."""

    def __init__(self, code: str, message: str) -> None:
        _validate_code(code)
        if not isinstance(message, str) or not message:
            raise ValueError("job dispatch message must be non-empty")
        super().__init__(message)
        self.code = code
        self.message = message


class JobLifecycleObserver(Protocol):
    """Observe job starts and outcomes without changing lifecycle behavior."""

    def job_started(self, workload_id: str) -> None: ...

    def job_finished(self, workload_id: str, status: str, detail: str) -> None: ...


def _validate_code(code: object) -> None:
    if (
        not isinstance(code, str)
        or not 1 <= len(code) <= 80
        or code.strip("abcdefghijklmnopqrstuvwxyz0123456789-")
        or code.startswith("-")
        or code.endswith("-")
        or "--" in code
    ):
        raise ValueError("job dispatch code is invalid")
