"""Private selected-file staging for isolated plugin workers."""

from __future__ import annotations

import errno
import hashlib
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

from .supervisor_session import PluginWorkerError

# Source cardinality belongs to each plugin's published input schema.  The
# broker used to impose a second, hidden ceiling of thirty-two after schema
# validation, so a directory workload accepting every selected media file was
# still refused here.  ``None`` means the broker adds no policy of its own.
MAX_SELECTED_SOURCES: int | None = None
MAX_SELECTED_SOURCE_BYTES = 128 * 1024 * 1024
_STAGING_COMPONENT = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def prepare_selected_files_root(root: Path | None) -> Path | None:
    if root is None:
        return None
    candidate = Path(root)
    if not candidate.is_absolute():
        raise ValueError("selected_files_root must be absolute")
    candidate.mkdir(parents=True, exist_ok=True)
    candidate.chmod(0o700)
    return candidate.resolve()


def stage_selected_sources(
    root: Path,
    plugin_id: str,
    job_id: str,
    payload: Mapping[str, object],
    *,
    copier: Callable[[str, Path, int, dict[tuple[int, int], Path]], Path] | None = None,
    maximum_sources: int | None = MAX_SELECTED_SOURCES,
) -> tuple[dict, Path]:
    sources = payload.get("sources")
    if (
        not isinstance(sources, list)
        or not sources
        or (maximum_sources is not None and len(sources) > maximum_sources)
        or not all(isinstance(source, str) and source for source in sources)
        or len(set(sources)) != len(sources)
    ):
        limit = f"1-{maximum_sources}" if maximum_sources is not None else "one or more"
        raise PluginWorkerError(
            "selected-files-invalid", f"sources must name {limit} selected files"
        )
    if (
        _STAGING_COMPONENT.fullmatch(plugin_id) is None
        or _STAGING_COMPONENT.fullmatch(job_id) is None
    ):
        raise PluginWorkerError("selected-files-invalid", "request identity is invalid")
    plugin_root = root / plugin_id
    plugin_root.mkdir(mode=0o700, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f"{job_id}-", dir=plugin_root))
    observed: dict[tuple[int, int], Path] = {}
    copy_source = copier or copy_selected_source
    try:
        staged_sources = [
            str(copy_source(source, staged, index, observed))
            for index, source in enumerate(sources)
        ]
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    rewritten = dict(payload)
    rewritten["sources"] = staged_sources
    return rewritten, staged


def copy_selected_source(
    source: str,
    staged: Path,
    index: int,
    observed: dict[tuple[int, int], Path],
    *,
    change_detector: Callable[[os.stat_result, os.stat_result, Path], bool] | None = None,
    maximum_bytes: int = MAX_SELECTED_SOURCE_BYTES,
) -> Path:
    candidate = Path(source)
    try:
        selected_status = candidate.stat(follow_symlinks=False)
    except OSError as error:
        raise PluginWorkerError(
            "selected-file-unavailable", _unreachable_selected_source(candidate, error)
        ) from error
    resolved = canonical_selected_source(candidate)
    # Everything that can raise happens before the descriptor exists: the name
    # is derived from an attacker-influenced filename, and until `os.fdopen`
    # adopts it below there is nobody to close what `open` returned.
    destination = staged / f"{index:02d}" / staged_source_name(candidate)
    changed = change_detector or selected_source_changed
    descriptor = open_selected_source(resolved)
    try:
        with os.fdopen(descriptor, "rb") as reader:
            before = os.fstat(reader.fileno())
            identity = (before.st_dev, before.st_ino)
            validate_selected_source_stat(before, maximum_bytes=maximum_bytes)
            if identity != (selected_status.st_dev, selected_status.st_ino):
                raise PluginWorkerError(
                    "selected-file-changed", "selected source changed while it was copied"
                )
            prior = observed.get(identity)
            if prior is not None:
                final_status = candidate.stat(follow_symlinks=False)
                if changed(before, final_status, prior):
                    raise PluginWorkerError(
                        "selected-file-changed", "selected source changed while it was copied"
                    )
                return prior
            destination.parent.mkdir(mode=0o700)
            initial_digest = _stream_digest(reader, before.st_size)
            reader.seek(0)
            with destination.open("xb") as writer:
                copied_bytes = _copy_bounded(reader, writer, before.st_size)
            reader.seek(0)
            final_digest = _stream_digest(reader, before.st_size)
            after = os.fstat(reader.fileno())
            with destination.open("rb") as copied:
                copied_digest = _stream_digest(copied, before.st_size)
            try:
                final_status = candidate.stat(follow_symlinks=False)
                final_resolved = candidate.resolve(strict=True)
            except (OSError, RuntimeError) as error:
                raise PluginWorkerError(
                    "selected-file-changed", "selected source changed while it was copied"
                ) from error
            if (
                changed(before, after, destination)
                or (final_status.st_dev, final_status.st_ino) != identity
                or final_resolved != resolved
                or copied_bytes != before.st_size
                or initial_digest != copied_digest
                or final_digest != copied_digest
            ):
                raise PluginWorkerError(
                    "selected-file-changed", "selected source changed while it was copied"
                )
            observed[identity] = destination
    except OSError as error:
        raise PluginWorkerError(
            "selected-file-unavailable", "selected source cannot be copied"
        ) from error
    return destination


class _BoundedReader:
    def __init__(self, handle, maximum_bytes: int) -> None:
        self._handle = handle
        self._remaining = maximum_bytes
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        if self._remaining == 0:
            return b""
        if size < 0 or size > self._remaining:
            size = self._remaining
        chunk = self._handle.read(size)
        self._remaining -= len(chunk)
        self.bytes_read += len(chunk)
        return chunk


def _copy_bounded(reader, writer, expected_size: int) -> int:
    bounded = _BoundedReader(reader, expected_size + 1)
    shutil.copyfileobj(bounded, writer, length=1024 * 1024)
    return bounded.bytes_read


def _stream_digest(handle, expected_size: int) -> bytes:
    digest = hashlib.sha256()
    remaining = expected_size
    while remaining:
        chunk = handle.read(min(1024 * 1024, remaining))
        if not chunk:
            raise PluginWorkerError(
                "selected-file-changed", "selected source changed while it was copied"
            )
        digest.update(chunk)
        remaining -= len(chunk)
    if handle.read(1):
        raise PluginWorkerError(
            "selected-file-changed", "selected source changed while it was copied"
        )
    return digest.digest()


def staged_source_name(candidate: Path) -> str:
    """Keep a safe selected basename without allowing staging-path control."""
    name = candidate.name
    if name and len(name) <= 255 and "/" not in name and "\\" not in name:
        return name
    suffix = candidate.suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
        suffix = ".bin"
    return f"selected-file{suffix}"


def _unreachable_selected_source(candidate: Path, error: OSError) -> str:
    """Say why a path the caller could see is not there for the worker.

    Staging runs inside the sandbox, where only the brokered input root is
    mounted.  A file the person picked in their own file manager and can read
    perfectly well is simply absent here if it lives anywhere else, and the
    kernel reports that as ENOENT - indistinguishable, in the old wording, from
    a file that was deleted.  The caller cannot act on "cannot be opened"; it
    can act on being told the selection has to live under a configured input
    root.
    """
    if error.errno == errno.ENOENT:
        return (
            f"selected source is not reachable here: {candidate}. A selected file must "
            "live under one of the service's configured input roots "
            "(OMNITENSOR_INPUT_ROOTS); nothing outside them is visible to the worker."
        )
    if error.errno in (errno.EACCES, errno.EPERM):
        return f"selected source cannot be read: {candidate} ({os.strerror(error.errno)})"
    return f"selected source cannot be opened: {candidate} ({error.strerror or error})"


def canonical_selected_source(candidate: Path) -> Path:
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        # A relative path never had a chance of naming a real selection, so the
        # input-root explanation would be a wrong guess there; only an absolute
        # path that is simply not present gets it.
        detail = (
            _unreachable_selected_source(candidate, error)
            if candidate.is_absolute() and isinstance(error, OSError)
            else "selected source cannot be opened"
        )
        raise PluginWorkerError("selected-file-unavailable", detail) from error
    if not candidate.is_absolute() or resolved != candidate:
        raise PluginWorkerError(
            "selected-file-invalid", "selected source must be a canonical absolute path"
        )
    return resolved


def open_selected_source(candidate: Path) -> int:
    try:
        return os.open(candidate, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise PluginWorkerError(
            "selected-file-unavailable", _unreachable_selected_source(candidate, error)
        ) from error


def validate_selected_source_stat(
    status: os.stat_result,
    *,
    maximum_bytes: int = MAX_SELECTED_SOURCE_BYTES,
) -> None:
    # Three unrelated things a person can do wrong, and one sentence covering
    # all of them told none of them apart: an empty file is a regular file and
    # is bounded, so "must be a bounded regular file" read as a contradiction
    # of what the user could see in their own file manager.
    if not stat.S_ISREG(status.st_mode):
        raise PluginWorkerError("selected-file-invalid", "selected source is not a regular file")
    if status.st_size == 0:
        raise PluginWorkerError("selected-file-invalid", "selected source is empty")
    if status.st_size > maximum_bytes:
        raise PluginWorkerError(
            "selected-file-invalid",
            f"selected source is {status.st_size} bytes, over the {maximum_bytes} "
            "a selected file may be staged at",
        )


def selected_source_changed(
    before: os.stat_result,
    after: os.stat_result,
    destination: Path,
) -> bool:
    return (
        (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or destination.stat().st_size != before.st_size
    )


__all__ = [
    "canonical_selected_source",
    "copy_selected_source",
    "open_selected_source",
    "prepare_selected_files_root",
    "selected_source_changed",
    "stage_selected_sources",
    "staged_source_name",
    "validate_selected_source_stat",
]
