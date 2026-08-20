"""Publishing a snapshot when it changes, and proving life when it does not.

This was service state and service methods: `_idle`, `_last_published_snapshot`,
`_next_snapshot_heartbeat` and `_snapshot_retracted`, plus the six methods that
read them. None of it is about being a service — it is one policy, stated
here, about when a document is worth writing:

* a document that says nothing the last one did not is not written again, so an
  idle machine stops waking every reader watching the file to say nothing
  changed;
* a reader treats a snapshot older than 15 s as stale, so an unchanged document
  is still written on the heartbeat interval — proof of life, not a change;
* with no accelerator devices there is no schema-valid snapshot to publish at
  all (the contract requires at least one), so the last one is retracted and
  readers observe honest absence rather than staleness;
* an idle runtime waits far longer between ticks than a busy one, and anything
  that changes state says so through `request_publish` rather than being
  polled for it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

# An idle runtime rebuilds and revalidates a document that says nothing new —
# about 6 ms of CPU to conclude that nothing happened. State changes wake the
# loop through `request_publish`, so backing an idle machine off this far costs
# no responsiveness.
_PUBLISH_FAILURE = "Could not publish the runtime snapshot; retrying next tick"
IDLE_PUBLISH_INTERVAL_S = 10.0
# Readers treat a snapshot older than 15 s as stale, so an idle runtime still
# owes them proof of life — but only that, not a fresh copy of an unchanged
# document twice a second.
SNAPSHOT_HEARTBEAT_INTERVAL_S = 10.0


class SnapshotPublishLoop:
    """Own what has been published, and decide when to publish again."""

    def __init__(
        self,
        build: Callable[[object], dict],
        publisher,
        *,
        stopping: asyncio.Event,
        interval_s: float,
        observe: Callable[[], object],
        devices_present: Callable[[], bool],
        before_tick: Callable[[], Awaitable[None]] | None = None,
        off_loop: Callable[..., Awaitable] | None = None,
        logger: logging.Logger | None = None,
        idle_interval_s: float = IDLE_PUBLISH_INTERVAL_S,
        heartbeat_interval_s: float = SNAPSHOT_HEARTBEAT_INTERVAL_S,
    ) -> None:
        self._build = build
        self._publisher = publisher
        self._stopping = stopping
        self._interval_s = interval_s
        self._observe = observe
        self._devices_present = devices_present
        self._before_tick = before_tick
        self._off_loop = off_loop
        self._logger = logger or logging.getLogger(__name__)
        self._idle_interval_s = idle_interval_s
        self._heartbeat_interval_s = heartbeat_interval_s
        self.idle = False
        self.retracted = False
        self.last_published: dict | None = None
        self.next_heartbeat = 0.0
        self.wake = asyncio.Event()
        self.loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop | None) -> None:
        """Name the loop `request_publish` may wake from another thread."""
        self.loop = loop

    def request_publish(self) -> None:
        """Ask for a tick now rather than at the next deadline.

        Backing an idle runtime off to a slow tick would otherwise delay every
        state change by that interval. Whatever changed something says so
        instead of being polled for it.
        """
        loop = self.loop
        if loop is None or self.wake.is_set():
            return
        try:
            # Runners deliver progress from their own threads, and `Event.set`
            # is not safe to call from one.
            loop.call_soon_threadsafe(self.wake.set)
        except RuntimeError:
            # The loop is closed; there is no publisher left to wake.
            return

    def publish_once(self) -> dict:
        """Synchronously build and publish one snapshot (test/adapter API)."""
        snapshot = self._build(self._observe())
        self._publisher.publish(snapshot)
        self.record_published(snapshot)
        return snapshot

    @staticmethod
    def content_of(snapshot: dict) -> dict:
        """The snapshot's content, apart from when it was built."""
        return {key: value for key, value in snapshot.items() if key != "generatedAt"}

    def changed(self, snapshot: dict) -> bool:
        """Whether this document says anything the last published one did not.

        `generatedAt` moves on every build, so it is compared out: it is the
        proof of life the heartbeat carries, never a change in its own right.
        """
        if self.last_published is None:
            return True
        return self.content_of(snapshot) != self.content_of(self.last_published)

    def heartbeat_due(self) -> bool:
        return self.monotonic() >= self.next_heartbeat

    def record_published(self, snapshot: dict) -> None:
        self.last_published = snapshot
        self.next_heartbeat = self.monotonic() + self._heartbeat_interval_s

    @staticmethod
    def monotonic() -> float:
        try:
            return asyncio.get_running_loop().time()
        except RuntimeError:
            # `publish_once` is a synchronous test/adapter entry point that can
            # run with no loop; the heartbeat still needs a clock.
            return time.monotonic()

    async def _run_off_loop(self, call, *arguments):
        if self._off_loop is None:
            return call(*arguments)
        return await self._off_loop(call, *arguments)

    async def run(self) -> None:
        while not self._stopping.is_set():
            # A publish failure is an outage of one output, not of the service.
            # Letting an OSError escape here propagates through the gather that
            # supervises the loops and takes down job dispatch and D-Bus with
            # it, so a full disk would stop far more than snapshot publishing.
            try:
                if self._before_tick is not None:
                    await self._before_tick()
                if self._devices_present():
                    await self._publish_tick()
                elif not self.retracted:
                    await self._retract()
            except (OSError, ValueError):
                self._logger.exception(_PUBLISH_FAILURE)
                self.idle = False
            await self.await_next()

    async def _publish_tick(self) -> None:
        if self.retracted:
            self._logger.info("Accelerator devices returned; publishing snapshots again")
            self.retracted = False
        # Ticked and read on the event loop, then handed to the worker thread
        # as an immutable value. Reading the live scheduler off-loop walked
        # mappings the loop was deleting from, and `tick()` reset busy time a
        # running job was still adding to.
        snapshot = await self._run_off_loop(self._build, self._observe())
        # An idle runtime rebuilds the same document every tick. Rewriting it
        # anyway costs a disk write and wakes every reader watching the file,
        # forever, to say nothing changed.
        changed = self.changed(snapshot)
        if changed or self.heartbeat_due():
            await self._run_off_loop(self._publisher.publish, snapshot)
            self.record_published(snapshot)
        self.idle = not changed

    async def _retract(self) -> None:
        # Honest-absence decision (OMNI-0011): with zero devices no
        # schema-valid snapshot exists, so rather than letting the last
        # document go silently stale it is retracted — readers observe
        # absence, exactly as when the service is down — and the outage is
        # logged once. Publishing resumes when devices return.
        await self._run_off_loop(self._publisher.retract)
        self.retracted = True
        self.last_published = None
        self.next_heartbeat = 0.0
        self._logger.warning(
            "No accelerator devices present; retracted the runtime snapshot"
            " so readers observe absence instead of stale data",
        )

    async def await_next(self) -> None:
        """Sleep until the next deadline, a state change, or shutdown.

        An idle runtime has nothing to say, so it waits far longer between
        ticks than a busy one — rebuilding, revalidating, and rewriting an
        unchanged document costs real power on a machine doing nothing.
        """
        interval = self._idle_interval_s if self.idle else self._interval_s
        waits = {
            asyncio.ensure_future(self._stopping.wait()),
            asyncio.ensure_future(self.wake.wait()),
        }
        try:
            await asyncio.wait(waits, timeout=interval, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for wait in waits:
                wait.cancel()
            await asyncio.gather(*waits, return_exceptions=True)
        self.wake.clear()


__all__ = [
    "IDLE_PUBLISH_INTERVAL_S",
    "SNAPSHOT_HEARTBEAT_INTERVAL_S",
    "SnapshotPublishLoop",
]
