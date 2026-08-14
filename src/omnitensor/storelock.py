"""Advisory cross-process locking for the service's on-disk stores.

Atomic replacement (:mod:`omnitensor.atomicio`) makes a single write
indivisible, but it cannot make a *read-modify-write* indivisible: two
processes can both read revision 4, both decide they may proceed, and both
write revision 5.  Every store that implements compare-and-swap therefore has
to hold this lock across the whole check-and-replace cycle, not merely around
the write.

``flock`` is advisory and per-open-file-description, so it serializes only
processes that take it, and it is released when the descriptor closes even if
the holder is killed.  That is exactly the guarantee these stores need.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import stat
import time
from collections.abc import Iterator
from pathlib import Path


@contextlib.contextmanager
def store_lock(
    root: Path,
    name: str,
    *,
    timeout_seconds: float | None = None,
) -> Iterator[None]:
    """Hold an exclusive advisory lock on ``root/name`` for the whole block.

    ``root`` is created when absent.  The lock file is opened ``O_NOFOLLOW``
    and required to be a regular file, so a symlink planted in a
    world-writable store cannot redirect the lock somewhere it does not
    serialize against.
    """
    store = Path(root).resolve()
    store.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(store / name, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f"store lock is not a regular file: {name}")
        if timeout_seconds is None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        else:
            deadline = time.monotonic() + max(0.0, timeout_seconds)
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"timed out acquiring store lock: {name}"
                        ) from None
                    time.sleep(min(0.01, remaining))
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
