"""OMNI-0516: the immutable-device-mount rule, reachable without a service.

Three fields read from six methods spread across `OmniTensorService` was what
OMNI-0411 amounted to. These are the branches that had no direct route in:
a runtime that cannot reload at all, a reload asked for with no running loop,
and a reload asked for again while one is in flight.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from omnitensor.plugin_lifecycle import PluginLifecycle


class Runtime:
    def __init__(self, *, fail=False):
        self.reloads = 0
        self.started = 0
        self.stopped = 0
        self.fail = fail

    async def start(self):
        self.started += 1
        return "started-snapshot"

    async def stop(self):
        self.stopped += 1

    async def reload_accelerator_devices(self):
        self.reloads += 1
        if self.fail:
            raise RuntimeError("lease is still held")
        return f"snapshot-{self.reloads}"


def lifecycle(runtime, *, reloadable=True, reloaded=None, **kwargs):
    async def note(snapshot):
        if reloaded is not None:
            reloaded.append(snapshot)

    return PluginLifecycle(
        lambda: runtime,
        plugin_ids=lambda: frozenset({"external-plugin"}),
        reloadable=lambda _runtime: reloadable,
        on_reloaded=note,
        logger=logging.getLogger("lifecycle-test"),
        retry_delay_s=0.0,
        **kwargs,
    )


def test_a_runtime_that_cannot_reload_is_never_asked_to():
    """The bundled runtime has no accelerator leases to re-mount."""
    runtime = Runtime()
    subject = lifecycle(runtime, reloadable=False)

    subject.schedule_reload({"external-plugin"})

    assert subject.pending == set()
    assert subject.reload_task is None
    assert runtime.reloads == 0


def test_a_reload_for_no_profile_is_not_a_reload():
    subject = PluginLifecycle(
        lambda: Runtime(),
        plugin_ids=frozenset,
        reloadable=lambda _runtime: True,
        on_reloaded=_unreachable,
    )

    subject.schedule_reload(None)

    assert subject.pending == set()
    assert subject.reload_task is None


async def _unreachable(_snapshot):  # pragma: no cover - asserted by not running
    raise AssertionError("nothing was reloaded")


def test_a_reload_asked_for_with_no_running_loop_records_it_and_starts_nothing():
    """Policy can change before `run()`; the profile still has to be refused."""
    subject = lifecycle(Runtime())

    subject.schedule_reload({"external-plugin"})

    assert subject.pending == {"external-plugin"}
    assert subject.reload_task is None


@pytest.mark.asyncio
async def test_a_second_request_is_served_by_the_same_task_rather_than_a_new_one():
    """Two policy changes in a row must not race two replacements of one worker."""
    started = asyncio.Event()
    release = asyncio.Event()

    class Blocking(Runtime):
        async def reload_accelerator_devices(self):
            self.reloads += 1
            started.set()
            await release.wait()
            release.clear()
            return f"snapshot-{self.reloads}"

    runtime = Blocking()
    reloaded: list[str] = []
    subject = lifecycle(runtime, reloaded=reloaded)

    # Both requests arrive before anything runs: they coalesce into one
    # replacement rather than two.
    subject.schedule_reload({"external-plugin"})
    first = subject.reload_task
    subject.schedule_reload({"external-plugin"})
    assert subject.reload_task is first

    await asyncio.wait_for(started.wait(), timeout=1)
    # This one arrives while the reload is running, so the loop goes round
    # again on the same task.
    subject.schedule_reload({"external-plugin"})
    assert subject.reload_task is first
    started.clear()
    release.set()
    await asyncio.wait_for(started.wait(), timeout=1)
    release.set()

    await asyncio.wait_for(first, timeout=1)
    assert runtime.reloads == 2
    assert reloaded == ["snapshot-1", "snapshot-2"]
    assert subject.pending == set()


@pytest.mark.asyncio
async def test_a_reload_that_keeps_failing_marks_the_profile_unavailable(caplog):
    runtime = Runtime(fail=True)
    subject = lifecycle(runtime, retry_attempts=2)

    with caplog.at_level("ERROR", logger="lifecycle-test"):
        subject.schedule_reload({"external-plugin"})
        await asyncio.wait_for(subject.reload_task, timeout=1)

    assert runtime.reloads == 2
    # Not left pending: admission would wedge on a set only this loop clears.
    assert subject.pending == set()
    assert subject.unavailable == {"external-plugin"}
    assert "attempt 2 of 2" in caplog.text


@pytest.mark.asyncio
async def test_a_recovered_reload_clears_what_the_failure_left_behind():
    runtime = Runtime(fail=True)
    subject = lifecycle(runtime, retry_attempts=1)
    subject.schedule_reload({"external-plugin"})
    await asyncio.wait_for(subject.reload_task, timeout=1)
    assert subject.unavailable == {"external-plugin"}

    runtime.fail = False
    subject.schedule_reload({"external-plugin"})
    await asyncio.wait_for(subject.reload_task, timeout=1)

    assert subject.unavailable == set()


@pytest.mark.asyncio
async def test_start_and_stop_take_the_same_lock_the_reload_takes():
    runtime = Runtime()
    subject = lifecycle(runtime)

    assert await subject.start() == "started-snapshot"
    await subject.stop()

    assert (runtime.started, runtime.stopped) == (1, 1)
    assert subject.lock.locked() is False
    # Nothing to cancel when nothing is in flight.
    assert subject.cancel_reload() is None


@pytest.mark.asyncio
async def test_a_shutdown_cancels_a_reload_that_is_still_in_flight():
    release = asyncio.Event()

    class Blocking(Runtime):
        async def reload_accelerator_devices(self):
            await release.wait()
            return "never"

    subject = lifecycle(Blocking())
    subject.schedule_reload({"external-plugin"})
    await asyncio.sleep(0)

    task = subject.cancel_reload()

    assert task is subject.reload_task
    await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()
