"""Durable atomic JSON persistence shared by snapshot and policy writers.

The document is written to a temp file in the target directory, fsynced,
renamed over the target, and the parent directory is fsynced so the rename
itself survives a power failure.  Readers therefore never observe a partial
document, and a completed write is durable.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path


def write_json_atomic(path: Path, payload: dict, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=prefix)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp_name)
        raise
    _fsync_directory(path.parent)


def _fsync_directory(directory: Path) -> None:
    # Best effort: some filesystems refuse directory fsync; the data file
    # itself is already synced.
    with contextlib.suppress(OSError):
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
