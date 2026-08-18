"""Cancellation-responsive bridge for bounded blocking runtime operations."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import TypeVar

_Result = TypeVar("_Result")


def _settle(future: asyncio.Future, result: object, error: BaseException | None) -> None:
    # The waiter may already have been cancelled while the thread was still
    # inside the blocking call; completing a resolved future would raise
    # InvalidStateError on the event loop, not on the caller.
    if future.done():
        return
    if error is not None:
        future.set_exception(error)
    else:
        future.set_result(result)


async def run_off_loop(callback: Callable[..., _Result], *args: object) -> _Result:
    """Run one blocking operation without asyncio's shared executor.

    The waiter parks on a future that the worker thread resolves through
    ``call_soon_threadsafe``, so an idle caller costs zero wakeups. Polling for
    completion instead would keep every plugin worker — and the CPU package —
    out of its idle state for as long as the process lives.
    """
    loop = asyncio.get_running_loop()
    completion: asyncio.Future = loop.create_future()

    def invoke() -> None:
        result: object = None
        error: BaseException | None = None
        try:
            result = callback(*args)
        except BaseException as caught:
            error = caught
        try:
            loop.call_soon_threadsafe(_settle, completion, result, error)
        except RuntimeError:
            # The caller was cancelled and its loop closed while this thread
            # was still blocked; nobody is left to receive the outcome.
            return

    worker = threading.Thread(
        target=invoke,
        name="omnitensor-plugin-io",
        daemon=True,
    )
    worker.start()
    try:
        return await completion
    finally:
        if not worker.is_alive():
            worker.join()
