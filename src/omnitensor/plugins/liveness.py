"""Whether a call that has not answered yet is still doing anything.

Two clocks bound a worker call: the flow controller that admitted it and the
worker's own budget. Both used to measure total work, so a job was killed at
the deadline however much of it was done — a long recording, a large document,
a slow model — and the answer was thrown away with `deadline-exceeded`. A
ceiling on how long an answer may take is a wrong answer waiting for a big
enough input.

What the deadline is actually worth having for is a worker that has stopped:
crashed mid-call, wedged on a lock, waiting on a device that will not answer.
That is distinguishable, and the workers already say so — every one of them
streams `PROGRESS` frames while it works, and `supervisor_session.read_result`
reads them on the way past.

So a call carries one of these. Each progress frame beats it, and a clock that
finds a beat since it last looked waits again instead of refusing. Nothing is
timed here: whether a beat arrived is all either clock needs, and a timestamp
would be a second opinion about a clock that already has one.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar


class CallHeartbeat:
    """One call's sign of life, set by the transport and read by the clocks.

    Deliberately not thread-safe and deliberately not a lock: it is written
    and read from the one event loop that runs the call, and a missed beat
    costs at most one more wait of the same length.
    """

    __slots__ = ("_beaten",)

    def __init__(self) -> None:
        self._beaten = False

    def beat(self) -> None:
        """Say that the worker reported something. Called per progress frame."""
        self._beaten = True

    def consume(self) -> bool:
        """Whether anything was reported since this was last asked, and reset."""
        beaten = self._beaten
        self._beaten = False
        return beaten


_CURRENT: ContextVar[CallHeartbeat | None] = ContextVar("omnitensor_call_heartbeat", default=None)


@contextlib.contextmanager
def beating(heartbeat: CallHeartbeat):
    """Make this the heartbeat that `report()` beats, for one job.

    A context variable rather than an argument threaded through six stages:
    the thing that knows a job is alive is the transport reading its progress
    frames, and the clocks that care are the flow controller and the worker
    budget — with the whole pipeline in between, none of which has any use for
    it. This is the same seam `_current_job` already uses in `orchestration`.
    """
    token = _CURRENT.set(heartbeat)
    try:
        yield heartbeat
    finally:
        _CURRENT.reset(token)


def report() -> None:
    """Say that the job running here reported something, if anything is listening."""
    heartbeat = _CURRENT.get()
    if heartbeat is not None:
        heartbeat.beat()


__all__ = ["CallHeartbeat", "beating", "report"]
