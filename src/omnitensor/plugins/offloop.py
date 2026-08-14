"""Cancellation-responsive bridge for bounded blocking runtime operations."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import TypeVar

OFF_LOOP_POLL_SECONDS = 0.001

_Result = TypeVar("_Result")


async def run_off_loop(callback: Callable[..., _Result], *args: object) -> _Result:
    """Run one blocking operation without asyncio's shared executor."""
    completed = threading.Event()
    outcome: list[_Result] = []
    errors: list[BaseException] = []

    def invoke() -> None:
        try:
            outcome.append(callback(*args))
        except BaseException as caught:
            errors.append(caught)
        finally:
            completed.set()

    worker = threading.Thread(
        target=invoke,
        name="omnitensor-plugin-io",
        daemon=True,
    )
    worker.start()
    try:
        while not completed.is_set():
            await asyncio.sleep(OFF_LOOP_POLL_SECONDS)
    finally:
        if not worker.is_alive():
            worker.join()
    if errors:
        raise errors[0]
    return outcome[0]
