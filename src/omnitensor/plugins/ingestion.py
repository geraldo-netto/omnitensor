"""Enumerate opted-in roots safely: bounds first, content never by default.

Both file-facing profiles need the same guarantees and both would be dangerous
without them.  A root is opted into explicitly and never inferred; a path that
escapes its root through a symlink is refused rather than followed; size, type,
and depth are checked *before* anything is opened; and identity is the content
digest, so a rename is a rename rather than a delete plus an add.

Nothing here decodes or parses.  Decoding untrusted media and parsing untrusted
documents are where the memory bombs live, so those are injected ports the
profile supplies and this module bounds — it decides what is worth opening, not
what is inside.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

DEFAULT_MAX_FILE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_FILES = 10_000
DEFAULT_MAX_DEPTH = 8
MAX_ROOTS = 32


class IngestionRejection(StrEnum):
    """Why a candidate was not ingested, in terms a user can act on."""

    OUTSIDE_ROOT = "outside-root"
    NOT_A_FILE = "not-a-file"
    TOO_LARGE = "too-large"
    TOO_DEEP = "too-deep"
    UNSUPPORTED_TYPE = "unsupported-type"
    UNREADABLE = "unreadable"


class IngestionError(ValueError):
    """Stable ingestion configuration or scan failure."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class IngestedFile:
    """One accepted file, identified by content rather than by path."""

    path: str
    size_bytes: int
    modified_at_ms: int
    suffix: str
    digest: str


@dataclass(frozen=True, slots=True)
class RejectedFile:
    """One candidate that was not accepted, and why."""

    path: str
    reason: IngestionRejection
    detail: str


@dataclass(frozen=True, slots=True)
class IngestionScan:
    """One bounded pass over the opted-in roots."""

    files: tuple[IngestedFile, ...]
    rejected: tuple[RejectedFile, ...]
    truncated: bool

    @property
    def duplicates(self) -> dict[str, tuple[str, ...]]:
        """Paths grouped by digest, for every digest seen more than once."""
        seen: dict[str, list[str]] = {}
        for item in self.files:
            seen.setdefault(item.digest, []).append(item.path)
        return {
            digest: tuple(sorted(paths))
            for digest, paths in seen.items()
            if len(paths) > 1
        }


class OptedInRootScanner:
    """Walk only explicitly opted-in roots under hard bounds."""

    def __init__(
        self,
        roots: Sequence[Path | str],
        *,
        suffixes: Sequence[str] = (),
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_files: int = DEFAULT_MAX_FILES,
        max_depth: int = DEFAULT_MAX_DEPTH,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if isinstance(roots, (str, bytes)) or not isinstance(roots, Sequence):
            raise IngestionError("roots-invalid", "roots must be a sequence of paths")
        if not roots:
            raise IngestionError("roots-invalid", "at least one opted-in root is required")
        if len(roots) > MAX_ROOTS:
            raise IngestionError("roots-invalid", f"at most {MAX_ROOTS} roots may be watched")
        for name, value in (
            ("max_file_bytes", max_file_bytes),
            ("max_files", max_files),
            ("max_depth", max_depth),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise IngestionError("bounds-invalid", f"{name} must be a positive integer")
        self._roots = tuple(Path(root).resolve() for root in roots)
        self._suffixes = frozenset(suffix.lower() for suffix in suffixes)
        self._max_file_bytes = max_file_bytes
        self._max_files = max_files
        self._max_depth = max_depth
        self._clock_ms = clock_ms

    @property
    def roots(self) -> tuple[Path, ...]:
        return self._roots

    def scan(self) -> IngestionScan:
        """One bounded pass; a candidate is judged before it is opened."""
        files: list[IngestedFile] = []
        rejected: list[RejectedFile] = []
        truncated = False
        for candidate in self._candidates():
            if len(files) >= self._max_files:
                truncated = True
                break
            accepted, rejection = self._accept(candidate)
            if accepted is not None:
                files.append(accepted)
            elif rejection is not None:
                rejected.append(rejection)
        return IngestionScan(tuple(files), tuple(rejected), truncated)

    def _candidates(self) -> Iterator[Path]:
        for root in self._roots:
            if not root.is_dir():
                continue
            for directory, subdirectories, names in os.walk(root, followlinks=False):
                current = Path(directory)
                depth = len(current.relative_to(root).parts)
                if depth >= self._max_depth:
                    # Pruning here rather than rejecting each leaf keeps a
                    # deliberately deep tree from costing a full walk.
                    subdirectories[:] = []
                subdirectories.sort()
                for name in sorted(names):
                    yield current / name

    def _accept(self, path: Path) -> tuple[IngestedFile | None, RejectedFile | None]:
        # Local import avoids loading the artifact installer through
        # preparation while the plugin package is still initializing.
        from omnitensor.preparation import FileDigestTooLargeError, file_digest

        try:
            if not self._within_roots(path):
                return None, RejectedFile(
                    str(path), IngestionRejection.OUTSIDE_ROOT, "path escapes its opted-in root"
                )
            status = path.lstat()
            if not path.is_file() or path.is_symlink():
                return None, RejectedFile(
                    str(path), IngestionRejection.NOT_A_FILE, "not a regular file"
                )
            if status.st_size > self._max_file_bytes:
                return None, RejectedFile(
                    str(path),
                    IngestionRejection.TOO_LARGE,
                    f"{status.st_size} bytes exceeds {self._max_file_bytes}",
                )
            suffix = path.suffix.lower()
            if self._suffixes and suffix not in self._suffixes:
                return None, RejectedFile(
                    str(path), IngestionRejection.UNSUPPORTED_TYPE, f"unsupported type: {suffix}"
                )
            digest = file_digest(path, max_bytes=self._max_file_bytes)
        except FileDigestTooLargeError as error:
            return None, RejectedFile(
                str(path),
                IngestionRejection.TOO_LARGE,
                f"{error.observed_bytes} bytes exceeds {self._max_file_bytes}",
            )
        except OSError as error:
            # A file deleted or renamed mid-scan is ordinary, not a failure of
            # the scan: it is reported and the pass continues.
            return None, RejectedFile(str(path), IngestionRejection.UNREADABLE, str(error))
        return (
            IngestedFile(
                str(path),
                status.st_size,
                int(status.st_mtime * 1000),
                suffix,
                digest,
            ),
            None,
        )

    def _within_roots(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
        except OSError:
            return False
        return any(resolved == root or root in resolved.parents for root in self._roots)
