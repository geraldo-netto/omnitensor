"""Consent that keeps applying after work has already started.

A permission set captured when a plugin starts is a photograph: it cannot show
a grant the user withdraws a minute later.  Everything already running would
keep collecting, inferring, and delivering on consent that no longer exists,
and the withdrawal would only take effect at the next restart.

:class:`LiveGrantView` re-reads the ledger for every check, so a revocation
committed by any process is visible immediately.  :class:`ConsentGuard` turns
that into prompt cancellation: it watches the grants an operation depends on
while the operation runs, and cancels it the moment one disappears.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Collection

from ..stable_error import StableError
from .grants import GrantLedger

DEFAULT_REVOCATION_POLL_SECONDS = 0.25
MAX_REVOCATION_POLL_SECONDS = 60.0


class RevocationError(StableError, RuntimeError):
    """Work stopped because consent for it was withdrawn or never granted."""


class LiveGrantView:
    """Fail-closed permission gate that re-reads the ledger on every check.

    Satisfies the same ``allows``/``require`` surface as the SDK's static
    permission view, so a collector or pipeline stage written against that
    contract observes revocation without knowing anything about ledgers.
    """

    def __init__(
        self,
        ledger: GrantLedger,
        plugin_id: str,
        declared_permissions: Collection[str],
    ) -> None:
        if not isinstance(ledger, GrantLedger):
            raise TypeError("ledger must be a GrantLedger")
        self._ledger = ledger
        self._plugin_id = plugin_id
        self._declared = set(declared_permissions)

    @property
    def declared(self) -> frozenset[str]:
        return frozenset(self._declared)

    @property
    def granted(self) -> frozenset[str]:
        """The currently active grants, re-read from the ledger."""
        self._ledger.reload()
        return self._ledger.active_permissions(self._plugin_id, self._declared)

    def allows(self, permission: str) -> bool:
        # Undeclared permissions can never be granted, so they are rejected
        # without touching the ledger at all.
        return permission in self._declared and permission in self.granted

    def require(self, permission: str) -> None:
        if not self.allows(permission):
            raise RevocationError(
                "permission-denied",
                f"{self._plugin_id} has no active grant for {permission}",
            )


class ConsentGuard:
    """Run one operation only while every permission it needs stays granted."""

    def __init__(
        self,
        view: LiveGrantView,
        *,
        poll_seconds: float = DEFAULT_REVOCATION_POLL_SECONDS,
    ) -> None:
        if (
            isinstance(poll_seconds, bool)
            or not isinstance(poll_seconds, (int, float))
            or not 0 < poll_seconds <= MAX_REVOCATION_POLL_SECONDS
        ):
            raise ValueError(f"poll_seconds must be a number in (0, {MAX_REVOCATION_POLL_SECONDS}]")
        self._view = view
        self._poll_seconds = poll_seconds

    def revoked(self, permissions: Collection[str]) -> tuple[str, ...]:
        """Which of ``permissions`` are not currently granted."""
        return tuple(
            permission for permission in sorted(permissions) if not self._view.allows(permission)
        )

    async def revoked_async(self, permissions: Collection[str]) -> tuple[str, ...]:
        """Refresh grants without blocking the caller's event loop."""
        required = tuple(permissions)
        return await asyncio.to_thread(self.revoked, required)

    async def run(
        self,
        operation: Callable[[], Awaitable],
        permissions: Collection[str],
    ) -> object:
        """Await ``operation``, cancelling it as soon as a grant disappears.

        The check runs before the operation starts as well: consent withdrawn
        between queueing and dispatch must stop the work, not merely stop the
        next one.
        """
        if not callable(operation):
            raise TypeError("operation must be callable")
        required = set(permissions)
        await self._raise_if_revoked(required)
        task = asyncio.ensure_future(operation())
        try:
            while True:
                done, _pending = await asyncio.wait((task,), timeout=self._poll_seconds)
                if done:
                    return await task
                missing = await self.revoked_async(required)
                if missing:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    raise RevocationError(
                        "consent-revoked",
                        f"stopped after {missing[0]} was revoked",
                    )
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def _raise_if_revoked(self, permissions: Collection[str]) -> None:
        missing = await self.revoked_async(permissions)
        if missing:
            raise RevocationError(
                "permission-denied",
                f"no active grant for {missing[0]}",
            )
