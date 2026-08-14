"""Durable atomic JSON persistence shared by snapshot and policy writers.

The document is written to a temp file in the target directory, fsynced,
renamed over the target, and the parent directory is fsynced so the rename
itself survives a power failure.  Readers therefore never observe a partial
document, and a completed write is durable.

:func:`read_json_bounded` is the reading counterpart: every store file has a
declared size ceiling, and enforcing it on the same descriptor that supplies
the bytes is what keeps a grown or replaced file from being read unbounded.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path


class JsonTooLargeError(ValueError):
    """A stored document exceeds the byte ceiling its reader declared."""


def read_json_bounded(path: Path, max_bytes: int) -> object:
    """Parse the JSON document at ``path`` without ever reading past ``max_bytes``.

    A separate ``stat`` would only describe the file as it was before the read;
    bounding the read itself is what makes the ceiling hold when the file grows
    or is replaced concurrently.  Raises :class:`JsonTooLargeError` when the
    document is oversized, ``OSError`` when it cannot be read, and
    ``ValueError`` when its bytes are not valid UTF-8 JSON.
    """
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    with path.open("rb") as stream:
        payload = stream.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise JsonTooLargeError(f"{path}: document exceeds {max_bytes} bytes")
    return json.loads(payload)


def fsync_directory(directory: Path) -> None:
    """Best-effort durability for a directory entry update."""
    with contextlib.suppress(OSError):
        descriptor = os.open(
            directory,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def write_bytes_atomic(
    path: Path,
    payload: bytes,
    mode: int,
    *,
    replace: bool = True,
    prefix: str | None = None,
) -> None:
    """Durably publish bytes from a synced same-directory staging file.

    Replacement uses :func:`os.replace`. Exclusive publication hard-links the
    stage into place, so an existing file is refused atomically and never
    overwritten.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=prefix if prefix is not None else f".{target.name}.",
    )
    try:
        os.fchmod(handle, mode)
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary_name, target)
        else:
            os.link(temporary_name, target, follow_symlinks=False)
            with contextlib.suppress(OSError):
                os.unlink(temporary_name)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(handle)
        with contextlib.suppress(OSError):
            os.unlink(temporary_name)
        raise
    fsync_directory(target.parent)


def write_json_atomic(path: Path, payload: dict, prefix: str) -> None:
    serialized = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    write_bytes_atomic(path, serialized, 0o600, prefix=prefix)


def remove_durable(path: Path) -> None:
    """Remove ``path`` (a no-op when absent) and fsync the parent directory
    so the removal itself survives a power failure."""
    with contextlib.suppress(FileNotFoundError):
        path.unlink()
        fsync_directory(path.parent)
