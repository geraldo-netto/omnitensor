"""Bounded artifact cache accounting and deterministic garbage collection."""

from __future__ import annotations

import contextlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from ..atomicio import read_json_bounded
from .artifact_installation import (
    ARTIFACT_STORE_LOCK_FILE,
    ArtifactActivation,
    ArtifactInstallationError,
    ArtifactInstaller,
    artifact_reference_from_document,
    artifact_store_lock,
)
from .artifacts import ArtifactReference, artifact_filename, artifact_reference_error

_ARTIFACT_METADATA_VERSION = 1
# Artifact metadata is one reference document; a larger file is corruption.
MAX_ARTIFACT_METADATA_BYTES = 64 * 1024


class ArtifactCacheError(RuntimeError):
    """Stable cache failure safe to surface through orchestration."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class ArtifactCacheAccounting:
    """Current storage consumption split by collection eligibility.

    ``total_*`` counts installed versions.  ``reclaimable_*`` counts installer
    leftovers, which occupy real bytes but are not versions anyone can resolve.
    Reporting them separately keeps them from hiding from the operator without
    letting them masquerade as installed state.
    """

    total_items: int
    total_bytes: int
    protected_items: int
    protected_bytes: int
    unreferenced_items: int
    unreferenced_bytes: int
    reclaimable_items: int = 0
    reclaimable_bytes: int = 0


@dataclass(frozen=True, slots=True)
class ArtifactCacheCollection:
    """One deterministic quota-enforcement result."""

    before: ArtifactCacheAccounting
    after: ArtifactCacheAccounting
    removed: tuple[ArtifactReference, ...]
    reclaimed: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    reference: ArtifactReference
    path: Path
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _Leftover:
    """A dot-prefixed installer temporary that survived a killed install."""

    path: Path
    size_bytes: int


class ArtifactCache:
    """Account and collect immutable versions under one installer store."""

    def __init__(self, root: Path, *, max_bytes: int, max_items: int):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer")
        if type(max_items) is not int or max_items < 0:
            raise ValueError("max_items must be a non-negative integer")
        self._root = Path(root).resolve()
        self._max_bytes = max_bytes
        self._max_items = max_items

    def accounting(self) -> ArtifactCacheAccounting:
        """Report exact primary and sidecar bytes for installed versions."""
        with artifact_store_lock(self._root):
            entries, protected, leftovers = self._scan()
            return _accounting(entries, protected, leftovers)

    def enforce(self) -> ArtifactCacheCollection:
        """Reclaim installer leftovers, then remove unreferenced versions.

        Installation holds the same store lock for its whole duration, so any
        leftover still present while enforcement holds the lock belongs to an
        install that is no longer running.  No age heuristic is needed to tell
        an abandoned stage directory from an in-flight one.
        """
        with artifact_store_lock(self._root):
            entries, protected, leftovers = self._scan()
            before = _accounting(entries, protected, leftovers)
            reclaimed = tuple(self._reclaim(leftover) for leftover in leftovers)
            protected_entries = [entry for entry in entries if _entry_key(entry) in protected]
            protected_usage = _accounting(protected_entries, protected)
            if (
                protected_usage.total_items > self._max_items
                or protected_usage.total_bytes > self._max_bytes
            ):
                raise ArtifactCacheError(
                    "quota-unsatisfiable",
                    "active and rollback artifacts exceed cache quotas",
                )
            kept = list(entries)
            removed: list[ArtifactReference] = []
            candidates = sorted(
                (entry for entry in entries if _entry_key(entry) not in protected),
                key=lambda entry: (entry.reference.id, entry.reference.version),
            )
            for entry in candidates:
                usage = _accounting(kept, protected)
                if usage.total_items <= self._max_items and usage.total_bytes <= self._max_bytes:
                    break
                self._remove_entry(entry)
                kept.remove(entry)
                removed.append(entry.reference)
            after = _accounting(kept, protected)
            if after.total_items > self._max_items or after.total_bytes > self._max_bytes:
                raise ArtifactCacheError("quota-unsatisfiable", "cache quotas cannot be satisfied")
            return ArtifactCacheCollection(before, after, tuple(removed), reclaimed)

    def _scan(self) -> tuple[list[_CacheEntry], set[tuple[str, str]], list[_Leftover]]:
        entries: list[_CacheEntry] = []
        protected: set[tuple[str, str]] = set()
        leftovers: list[_Leftover] = []
        self._root.mkdir(parents=True, exist_ok=True)
        for artifact_root in sorted(self._root.iterdir(), key=lambda path: path.name):
            if artifact_root.name == ARTIFACT_STORE_LOCK_FILE:
                continue
            self._validate_artifact_root(artifact_root)
            activation = self._activation(artifact_root.name)
            protected.update(_activation_keys(activation))
            found, stale = self._scan_artifact(artifact_root)
            entries.extend(found)
            leftovers.extend(stale)
        return entries, protected, leftovers

    def _validate_artifact_root(self, path: Path) -> None:
        probe = ArtifactReference(path.name, "0.0.0", "onnx", "0" * 64)
        invalid = artifact_reference_error(probe)
        if invalid or path.is_symlink() or not path.is_dir() or path.resolve() != path:
            raise ArtifactCacheError(
                "cache-state-invalid",
                f"invalid artifact cache directory: {path.name}",
            )

    def _activation(self, artifact_id: str) -> ArtifactActivation:
        activation_path = self._root / artifact_id / "activation.json"
        if activation_path.is_symlink():
            raise ArtifactCacheError("cache-state-invalid", "artifact activation path is a symlink")
        try:
            return ArtifactInstaller(self._root).activation(artifact_id)
        except ArtifactInstallationError as error:
            raise ArtifactCacheError("cache-state-invalid", error.detail) from error

    def _scan_artifact(self, artifact_root: Path) -> tuple[list[_CacheEntry], list[_Leftover]]:
        entries: list[_CacheEntry] = []
        leftovers: list[_Leftover] = []
        for path in sorted(artifact_root.iterdir(), key=lambda value: value.name):
            if path.name == "activation.json":
                continue
            if path.name.startswith("."):
                # An interrupted install leaves .install-* and .activation-*
                # temporaries behind.  A resolvable version can never be
                # dot-prefixed, so these are not cache state to validate;
                # rejecting them would disable accounting, enforcement, and
                # garbage collection for the entire store until an operator
                # cleaned up by hand.  They are reported as reclaimable so
                # their bytes stay visible rather than silently ignored.
                leftovers.append(_Leftover(path, _leftover_size(path)))
                continue
            if path.is_symlink() or not path.is_dir() or path.resolve() != path:
                raise ArtifactCacheError(
                    "cache-state-invalid",
                    f"invalid artifact version directory: {artifact_root.name}/{path.name}",
                )
            entries.append(self._read_entry(artifact_root.name, path))
        return entries, leftovers

    def _read_entry(self, artifact_id: str, version_root: Path) -> _CacheEntry:
        metadata_path = version_root / "artifact.json"
        try:
            document = read_json_bounded(metadata_path, MAX_ARTIFACT_METADATA_BYTES)
        except (OSError, ValueError) as error:
            raise ArtifactCacheError(
                "cache-state-invalid", f"cannot read {metadata_path}: {error}"
            ) from error
        if not isinstance(document, dict) or set(document) != {"version", "artifact"}:
            raise ArtifactCacheError("cache-state-invalid", "artifact metadata has invalid fields")
        if document["version"] != _ARTIFACT_METADATA_VERSION:
            raise ArtifactCacheError("cache-state-invalid", "artifact metadata version is invalid")
        try:
            reference = artifact_reference_from_document(document["artifact"])
        except ArtifactInstallationError as error:
            raise ArtifactCacheError("cache-state-invalid", error.detail) from error
        if reference.id != artifact_id or reference.version != version_root.name:
            raise ArtifactCacheError(
                "cache-state-invalid", "artifact cache identity does not match"
            )
        primary = version_root / artifact_filename(reference.format)
        if primary.is_symlink() or not primary.is_file():
            raise ArtifactCacheError(
                "cache-state-invalid", "artifact cache primary file is invalid"
            )
        return _CacheEntry(reference, version_root, _version_size(version_root))

    def _reclaim(self, leftover: _Leftover) -> str:
        """Delete one abandoned installer temporary, following no symlink."""
        path = leftover.path
        label = f"{path.parent.name}/{path.name}"
        try:
            if path.is_symlink() or not path.is_dir():
                path.unlink()
            else:
                for directory, _names, files in os.walk(path, topdown=False, followlinks=False):
                    current = Path(directory)
                    for name in files:
                        (current / name).unlink()
                    for name in _names:
                        child = current / name
                        if child.is_symlink():
                            child.unlink()
                        else:
                            child.rmdir()
                path.rmdir()
            _fsync_directory(path.parent)
        except OSError as error:
            raise ArtifactCacheError(
                "cache-reclaim-failed",
                f"cannot reclaim {label}: {error}",
            ) from error
        return label

    def _remove_entry(self, entry: _CacheEntry) -> None:
        path = entry.path
        if path.is_symlink() or path.resolve() != path:
            raise ArtifactCacheError("cache-delete-failed", "artifact version path changed")
        try:
            for child in path.iterdir():
                child_stat = child.lstat()
                if not stat.S_ISREG(child_stat.st_mode):
                    raise ArtifactCacheError(
                        "cache-delete-failed",
                        f"artifact version contains unsafe entry: {child.name}",
                    )
            for child in sorted(path.iterdir(), key=lambda value: value.name):
                child.unlink()
            path.rmdir()
            _fsync_directory(path.parent)
        except ArtifactCacheError:
            raise
        except OSError as error:
            raise ArtifactCacheError(
                "cache-delete-failed",
                f"cannot remove {entry.reference.id}@{entry.reference.version}: {error}",
            ) from error


def _activation_keys(activation: ArtifactActivation) -> set[tuple[str, str]]:
    return {
        (reference.id, reference.version)
        for reference in (activation.active, activation.rollback)
        if reference is not None
    }


def _entry_key(entry: _CacheEntry) -> tuple[str, str]:
    return entry.reference.id, entry.reference.version


def _accounting(
    entries: list[_CacheEntry],
    protected: set[tuple[str, str]],
    leftovers: list[_Leftover] = (),
) -> ArtifactCacheAccounting:
    protected_entries = [entry for entry in entries if _entry_key(entry) in protected]
    protected_bytes = sum(entry.size_bytes for entry in protected_entries)
    total_bytes = sum(entry.size_bytes for entry in entries)
    return ArtifactCacheAccounting(
        total_items=len(entries),
        total_bytes=total_bytes,
        protected_items=len(protected_entries),
        protected_bytes=protected_bytes,
        unreferenced_items=len(entries) - len(protected_entries),
        unreferenced_bytes=total_bytes - protected_bytes,
        reclaimable_items=len(leftovers),
        reclaimable_bytes=sum(leftover.size_bytes for leftover in leftovers),
    )


def _leftover_size(path: Path) -> int:
    """Bytes held by one installer temporary, following no symlink."""
    try:
        entry_stat = path.lstat()
        if stat.S_ISREG(entry_stat.st_mode):
            return entry_stat.st_size
        if not stat.S_ISDIR(entry_stat.st_mode):
            return 0
        total = 0
        for directory, _names, files in os.walk(path, followlinks=False):
            for name in files:
                with contextlib.suppress(OSError):
                    total += os.lstat(Path(directory) / name).st_size
        return total
    except OSError as error:
        raise ArtifactCacheError(
            "cache-state-invalid", f"cannot account artifact leftover: {error}"
        ) from error


def _version_size(version_root: Path) -> int:
    total = 0
    try:
        for path in version_root.iterdir():
            path_stat = path.lstat()
            if not stat.S_ISREG(path_stat.st_mode):
                raise ArtifactCacheError(
                    "cache-state-invalid",
                    f"artifact version contains unsafe entry: {path.name}",
                )
            total += path_stat.st_size
    except OSError as error:
        raise ArtifactCacheError(
            "cache-state-invalid", f"cannot account artifact version: {error}"
        ) from error
    return total


def _fsync_directory(directory: Path) -> None:
    with contextlib.suppress(OSError):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
