"""Read repository and build metadata without ever running project code.

The tempting implementations here are all dangerous.  Asking a build system
what it would do means running its configuration, and a repository's config is
attacker-controlled the moment anyone clones an untrusted project.  Reading
file *content* by default turns a build advisor into a source-code exfiltrator.
Following a symlink out of the repository reads whatever it points at.

So this layer reads exported build history and file metadata only: names,
sizes, timestamps, and outcomes.  There is no evaluation, no hook execution,
and no content — and the tests assert the module imports nothing that could run
a subprocess.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .collection import CollectionPermissionGate
from .ingestion import IngestionError, IngestionScan, OptedInRootScanner

BUILD_PLUGIN_ID = "build-advisor"
BUILD_METADATA_PERMISSION = "read:build-metadata"
DEFAULT_MAX_BUILD_RECORDS = 5_000
MAX_DURATION_MS = 30 * 24 * 60 * 60 * 1000


class BuildOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class BuildRecord:
    """One exported build result; never a log body, never source content."""

    build_id: str
    outcome: BuildOutcome
    duration_ms: int
    started_at_ms: int
    changed_paths: tuple[str, ...]
    failed_checks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RepositoryProfile:
    """What a repository looks like from its metadata alone."""

    root: str
    file_count: int
    total_bytes: int
    suffix_counts: dict[str, int]
    truncated: bool


def build_record_error(record: object) -> str:
    """One stable validation failure for an untrusted exported record."""
    if not isinstance(record, BuildRecord):
        return "build record has an invalid type"
    if not isinstance(record.outcome, BuildOutcome):
        return "build outcome is invalid"
    if not isinstance(record.build_id, str) or not 1 <= len(record.build_id) <= 120:
        return "build identity is invalid"
    timing = _timing_error(record)
    if timing:
        return timing
    for field, name in (
        (record.changed_paths, "changed paths"),
        (record.failed_checks, "failed checks"),
    ):
        if not isinstance(field, tuple) or any(not isinstance(item, str) for item in field):
            return f"build {name} are invalid"
    return ""


def _timing_error(record: BuildRecord) -> str:
    for value, name in ((record.duration_ms, "duration"), (record.started_at_ms, "start")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return f"build {name} is invalid"
    if record.duration_ms > MAX_DURATION_MS:
        return "build duration is implausible"
    return ""


class BuildMetadataIngestor:
    """Profile configured repositories and validate exported build history."""

    def __init__(
        self,
        roots: Sequence[Path | str],
        permissions: CollectionPermissionGate,
        *,
        max_records: int = DEFAULT_MAX_BUILD_RECORDS,
        **scanner_options,
    ) -> None:
        if not isinstance(permissions, CollectionPermissionGate):
            raise TypeError("permissions must implement CollectionPermissionGate")
        if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 1:
            raise IngestionError("bounds-invalid", "max_records must be a positive integer")
        self._scanner = OptedInRootScanner(roots, **scanner_options)
        self._permissions = permissions
        self._max_records = max_records

    @property
    def roots(self) -> tuple[Path, ...]:
        return self._scanner.roots

    def profile(self) -> tuple[RepositoryProfile, ...]:
        """Describe each configured repository from metadata alone."""
        if not self._permissions.allows(BUILD_METADATA_PERMISSION):
            raise IngestionError(
                "permission-denied", "build metadata permission is not granted"
            )
        scan: IngestionScan = self._scanner.scan()
        by_root: dict[str, list] = {str(root): [] for root in self._scanner.roots}
        for item in scan.files:
            for root in self._scanner.roots:
                if item.path.startswith(f"{root}/"):
                    by_root[str(root)].append(item)
                    break
        profiles = []
        for root, items in sorted(by_root.items()):
            suffixes: dict[str, int] = {}
            for item in items:
                suffixes[item.suffix or "(none)"] = suffixes.get(item.suffix or "(none)", 0) + 1
            profiles.append(
                RepositoryProfile(
                    root,
                    len(items),
                    sum(item.size_bytes for item in items),
                    dict(sorted(suffixes.items())),
                    scan.truncated,
                )
            )
        return tuple(profiles)

    def accept_history(
        self, records: Sequence[object]
    ) -> tuple[tuple[BuildRecord, ...], tuple[str, ...]]:
        """Validate an exported history, returning the accepted and the reasons."""
        accepted: list[BuildRecord] = []
        rejected: list[str] = []
        for record in records[: self._max_records]:
            error = build_record_error(record)
            if error:
                rejected.append(error)
            else:
                accepted.append(record)
        if len(records) > self._max_records:
            rejected.append(
                f"history truncated at {self._max_records} records"
            )
        return tuple(accepted), tuple(rejected)
