"""Transport-neutral runtime API facade.

A guard refusal is *not* a method result.  The guard runs before the method
does, so no method-specific acknowledgement exists yet: returning the refusal
document from the call put a ``runtime-refusal`` where a client validating
against ``runtime-job-acknowledgement`` or ``plugin-inventory`` expected the
method's own schema, and left no envelope error to branch on.
:class:`omnitensor.guard.GuardRefusedError` therefore propagates out of every
method here, and the transport renders it as an envelope error carrying the
guard's stable code.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from .callers import CallerIdentityResolver
from .contract import RUNTIME_METHODS, contract_document_text
from .control import ControlService
from .guard import ControlGuard, guarded
from .inspection import PLUGIN_INVENTORY_VERSION
from .jobs import JobSubmissionService


def no_inventory() -> str:
    """Fail closed with the exact empty plugin inventory document."""
    return json.dumps(
        {"version": PLUGIN_INVENTORY_VERSION, "generatedAt": 1, "plugins": []},
        separators=(",", ":"),
    )


class RuntimeAPI:
    """Transport-neutral facade retaining policy control while adding jobs."""

    def __init__(
        self,
        control: ControlService,
        jobs: JobSubmissionService,
        inspector: Callable[[], str] | None = None,
        callers: CallerIdentityResolver | None = None,
        guard: ControlGuard | None = None,
    ) -> None:
        self._control = control
        self._jobs = jobs
        self._inspector = inspector or no_inventory
        self._callers = callers or CallerIdentityResolver()
        self._guard = guard or ControlGuard()

    async def apply_command_text(self, text: str) -> str:
        return await self._guarded("apply-command", text, self._control.apply_command_text)

    async def submit_job_text(self, text: str) -> str:
        owner = self._callers.owner_token()
        return await self._guarded(
            "submit-job",
            text,
            lambda request: self._jobs.submit_job_text(request, owner=owner),
            owner=owner,
        )

    async def cancel_job_text(self, text: str) -> str:
        owner = self._callers.owner_token()
        return await self._guarded(
            "cancel-job",
            text,
            lambda request: self._jobs.cancel_job_text(request, owner=owner),
            owner=owner,
        )

    async def job_result_text(self, text: str) -> str:
        owner = self._callers.owner_token()
        return await self._guarded(
            "get-job-result",
            text,
            lambda request: self._jobs.job_result_text(request, owner=owner),
            owner=owner,
        )

    def describe_plugins_text(self) -> str:
        owner = self._callers.owner_token()
        with guarded(self._guard, "describe-plugins", owner):
            return self._inspector()

    def describe_contract_text(self) -> str:
        owner = self._callers.owner_token()
        with guarded(self._guard, "describe-contract", owner):
            return contract_document_text(RUNTIME_METHODS)

    async def _guarded(
        self,
        method: str,
        text: str,
        call,
        *,
        owner: str | None = None,
    ) -> str:
        resolved = owner if owner is not None else self._callers.owner_token()
        with guarded(self._guard, method, resolved, text):
            return await call(text)


__all__ = ["RuntimeAPI", "no_inventory"]
