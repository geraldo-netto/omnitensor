"""Everything that starts, stops, or re-mounts a plugin worker, in one place.

OMNI-0411 is the rule that a worker's device mount is immutable: once policy
points a profile at another accelerator, the running worker cannot be told —
it has to be replaced, and new jobs must wait until the replacement owns that
exact lease. A replacement that fails leaves the profile unavailable rather
than quietly serving from the previous device.

That rule needs three pieces of state to hold: one lock, so a start, a stop and
a reload never overlap; the profiles whose reload is still in flight, which
admission must refuse; and the profiles whose reload failed outright, which are
published as unavailable so a person sees why nothing is being accepted. All
three were fields on `OmniTensorService`, read from six methods spread across
the class, which made the rule something a reader had to reassemble. Here it is
a local invariant.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

# A device reload that fails is usually transient — a worker still shutting
# down, a lease not yet released — so it is retried a few times with growing
# delay. The attempts are bounded: an idle runtime must not keep waking to
# retry forever, and the next policy change schedules a reload anyway.
DEVICE_RELOAD_RETRY_ATTEMPTS = 3
DEVICE_RELOAD_RETRY_DELAY_S = 0.5


class PluginLifecycle:
    """Serialise plugin start, stop and accelerator reload; own what is in flight."""

    def __init__(
        self,
        runtime: Callable[[], object],
        *,
        plugin_ids: Callable[[], frozenset[str]],
        reloadable: Callable[[object], bool],
        on_reloaded: Callable[[object], Awaitable[None]],
        logger: logging.Logger | None = None,
        retry_attempts: int = DEVICE_RELOAD_RETRY_ATTEMPTS,
        retry_delay_s: float = DEVICE_RELOAD_RETRY_DELAY_S,
    ) -> None:
        # Late-bound: the runtime is replaced on a live service by tests and
        # by anything that swaps the adapter, and a captured reference would
        # keep driving the object that was there at construction.
        self._runtime = runtime
        self._plugin_ids = plugin_ids
        self._reloadable = reloadable
        self._on_reloaded = on_reloaded
        self._logger = logger or logging.getLogger(__name__)
        self._retry_attempts = retry_attempts
        self._retry_delay_s = retry_delay_s
        self.lock = asyncio.Lock()
        self.pending: set[str] = set()
        # Profiles whose accelerator reload failed outright. They stay refused
        # — their worker may still hold the previous device — but unlike the
        # pending set they are published as unavailable, so a person sees why
        # nothing is being accepted instead of a profile that reports 'Ready'.
        self.unavailable: set[str] = set()
        self.reload_task: asyncio.Task | None = None
        self._requested = False

    async def start(self):
        async with self.lock:
            return await self._runtime().start()

    async def stop(self) -> None:
        async with self.lock:
            await self._runtime().stop()

    def cancel_reload(self) -> asyncio.Task | None:
        """Stop an in-flight reload, for a service that is shutting down."""
        task = self.reload_task
        if task is not None:
            task.cancel()
        return task

    def schedule_reload(self, profile_ids: set[str] | frozenset[str] | None = None) -> None:
        if not self._reloadable(self._runtime()):
            return
        affected = self._plugin_ids() if profile_ids is None else frozenset(profile_ids)
        if not affected:
            return
        # A worker's device mount is immutable. Once policy points elsewhere,
        # new jobs must wait until the replacement worker owns that exact
        # lease; a failed replacement remains unavailable rather than using
        # the old GPU.
        self.pending.update(affected)
        self._requested = True
        if self.reload_task is None or self.reload_task.done():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            self.reload_task = loop.create_task(
                self.reload_accelerators(),
                name="omnitensor-plugin-accelerator-reload",
            )

    async def reload_accelerators(self) -> None:
        try:
            while self._requested:
                affected = frozenset(self.pending)
                self._requested = False
                plugin_snapshot = await self._reload_devices(affected)
                if plugin_snapshot is None:
                    continue
                self.unavailable.difference_update(affected)
                await self._on_reloaded(plugin_snapshot)
        finally:
            # Whatever happened, admission must not stay wedged on a set that
            # only this loop can clear: a profile that is genuinely unusable is
            # held in `unavailable`, where it is published.
            self.pending.clear()

    async def _reload_devices(self, affected: frozenset[str]):
        """Reload plugin device leases, retrying a transient failure a few times."""
        delay = self._retry_delay_s
        for attempt in range(self._retry_attempts):
            try:
                async with self.lock:
                    return await self._runtime().reload_accelerator_devices()
            except Exception:  # noqa: BLE001 - plugin adapters are external
                self._logger.exception(
                    "Could not reload plugin accelerator leases (attempt %d of %d)",
                    attempt + 1,
                    self._retry_attempts,
                )
                if attempt + 1 < self._retry_attempts:
                    await asyncio.sleep(delay)
                    delay *= 2
        self.unavailable.update(affected)
        return None


__all__ = [
    "DEVICE_RELOAD_RETRY_ATTEMPTS",
    "DEVICE_RELOAD_RETRY_DELAY_S",
    "PluginLifecycle",
]
