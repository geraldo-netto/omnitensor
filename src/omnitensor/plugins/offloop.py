"""Cancellation-responsive bridge for bounded blocking runtime operations."""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

_Result = TypeVar("_Result")

THREAD_NAME_PREFIX = "omnitensor-plugin-io"

# One executor per event loop, installed as that loop's default so asyncio owns
# its shutdown: `asyncio.run` (and `Runner.close`) awaits
# `loop.shutdown_default_executor()`, which joins the threads instead of letting
# the interpreter kill one mid-`rmtree` or mid-`publish`. The keys are weak, so
# a loop that is closed and dropped takes its entry with it.
_installed: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, ThreadPoolExecutor] = (
    weakref.WeakKeyDictionary()
)


def _executor(loop: asyncio.AbstractEventLoop) -> ThreadPoolExecutor:
    existing = _installed.get(loop)
    if existing is not None:
        return existing
    # No worker ceiling: the pool grows only as far as concurrent callers take
    # it, and each thread is idle-blocked on the work queue, costing nothing.
    executor = ThreadPoolExecutor(thread_name_prefix=THREAD_NAME_PREFIX)
    loop.set_default_executor(executor)
    _installed[loop] = executor
    return executor


async def run_off_loop(callback: Callable[..., _Result], *args: object) -> _Result:
    """Run one blocking operation on this loop's long-lived worker pool.

    The waiter parks on a future the executor resolves through
    ``call_soon_threadsafe``, so an idle caller costs zero wakeups — the same
    property a hand-rolled thread had, without a thread per call. That mattered:
    this is called once per received IPC frame and once every
    ``GRANT_REFRESH_SECONDS``, which was hundreds of thousands of thread
    creations a day, and a caller cancelled mid-call abandoned a daemon thread
    still inside ``shutil.rmtree`` or ``publisher.publish()`` for the
    interpreter to kill mid-write at exit. Pooled threads are joined at loop
    shutdown instead, so the write finishes.

    Cancelling the waiter still does not unblock the callback — nothing can
    interrupt a blocking syscall — but the thread returns to the pool when the
    call ends rather than being leaked.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor(loop), callback, *args)
