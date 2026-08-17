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
from pathlib import Path

import omnitensor.telemetry_types as _telemetry_types

from .collection import CollectionPermissionGate
from .ingestion import (
    IngestionError,
    IngestionScan,
    OptedInRootScanner,
    positive_integer_bound,
)

BUILD_PLUGIN_ID = "build-advisor"
BUILD_METADATA_PERMISSION = "read:build-metadata"
DEFAULT_MAX_BUILD_RECORDS = _telemetry_types.DEFAULT_MAX_BUILD_RECORDS
MAX_DURATION_MS = _telemetry_types.MAX_BUILD_DURATION_MS
BuildOutcome = _telemetry_types.BuildOutcome
BuildRecord = _telemetry_types.BuildRecord
build_record_error = _telemetry_types.build_record_error


@dataclass(frozen=True, slots=True)
class RepositoryProfile:
    """What a repository looks like from its metadata alone."""

    root: str
    file_count: int
    total_bytes: int
    # Pairs rather than a dict, in suffix order. `frozen=True` promises a value
    # that can be hashed and shared, and a dict field silently withdraws both:
    # the counts stayed mutable through the "immutable" profile, and `hash()`
    # raised on a type declared frozen. This is the same shape
    # `ArtifactReference.companions` already uses for the same reason.
    suffix_counts: tuple[tuple[str, int], ...]
    truncated: bool

    @property
    def counts_by_suffix(self) -> dict[str, int]:
        """A fresh mapping for callers that want to look one up."""
        return dict(self.suffix_counts)


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
        positive_integer_bound(max_records, "max_records", error_code="bounds-invalid")
        self._scanner = OptedInRootScanner(roots, **scanner_options)
        self._permissions = permissions
        self._max_records = max_records

    @property
    def roots(self) -> tuple[Path, ...]:
        return self._scanner.roots

    def profile(self) -> tuple[RepositoryProfile, ...]:
        """Describe each configured repository from metadata alone."""
        if not self._permissions.allows(BUILD_METADATA_PERMISSION):
            raise IngestionError("permission-denied", "build metadata permission is not granted")
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
                    tuple(sorted(suffixes.items())),
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
            rejected.append(f"history truncated at {self._max_records} records")
        return tuple(accepted), tuple(rejected)
