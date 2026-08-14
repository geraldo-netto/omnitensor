"""Transport-neutral runtime API facade."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable

from .callers import CallerIdentityResolver
from .contract import RUNTIME_METHODS, contract_document_text
from .control import ControlService
from .guard import BusGuard, GuardRefusedError, guarded
from .inspection import PLUGIN_INVENTORY_VERSION
from .jobs import JobSubmissionService


def _runtime_methods() -> tuple[str, ...]:
    facade = sys.modules.get("omnitensor.service")
    return (
        getattr(facade, "BUS_METHODS", RUNTIME_METHODS)
        if facade is not None
        else RUNTIME_METHODS
    )


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
        guard: BusGuard | None = None,
    ) -> None:
        self._control = control
        self._jobs = jobs
        self._inspector = inspector or no_inventory
        self._callers = callers or CallerIdentityResolver()
        self._guard = guard or BusGuard()

    async def apply_command_text(self, text: str) -> str:
        return await self._guarded("ApplyCommand", text, self._control.apply_command_text)

    async def submit_job_text(self, text: str) -> str:
        owner = await self._callers.owner_token()
        return await self._guarded(
            "SubmitJob",
            text,
            lambda request: self._jobs.submit_job_text(request, owner=owner),
            owner=owner,
        )

    async def cancel_job_text(self, text: str) -> str:
        owner = await self._callers.owner_token()
        return await self._guarded(
            "CancelJob",
            text,
            lambda request: self._jobs.cancel_job_text(request, owner=owner),
            owner=owner,
        )

    async def job_result_text(self, text: str) -> str:
        owner = await self._callers.owner_token()
        return await self._guarded(
            "GetJobResult",
            text,
            lambda request: self._jobs.job_result_text(request, owner=owner),
            owner=owner,
        )

    def describe_plugins_text(self) -> str:
        owner = self._callers.cached_owner_token()
        try:
            with guarded(self._guard, "DescribePlugins", owner):
                return self._inspector()
        except GuardRefusedError as refusal:
            return refusal.text()

    def describe_contract_text(self) -> str:
        owner = self._callers.cached_owner_token()
        try:
            with guarded(self._guard, "DescribeContract", owner):
                return contract_document_text(_runtime_methods())
        except GuardRefusedError as refusal:
            return refusal.text()

    async def _guarded(
        self,
        method: str,
        text: str,
        call,
        *,
        owner: str | None = None,
    ) -> str:
        resolved = owner if owner is not None else await self._callers.owner_token()
        try:
            with guarded(self._guard, method, resolved, text):
                return await call(text)
        except GuardRefusedError as refusal:
            return refusal.text()


RuntimeAPI.__module__ = "omnitensor.service"

__all__ = ["RuntimeAPI", "no_inventory"]
