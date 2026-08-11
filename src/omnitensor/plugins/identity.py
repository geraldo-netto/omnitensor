"""Fail-closed plugin identity validation over metadata-only candidates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from ..registry import MAX_MANIFEST_BYTES, validate_workload_document
from .discovery import PluginMetadata, PluginSource

_DISTRIBUTION_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_ENTRY_POINT_TARGET_PATTERN = re.compile(
    r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?::[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)?$"
)
MAX_DISTRIBUTION_NAME_LENGTH = 200
MAX_DISTRIBUTION_VERSION_LENGTH = 128
MAX_ENTRY_POINT_TARGET_LENGTH = 240


class PluginRejectionCode(StrEnum):
    """Stable machine-readable reasons for rejecting one candidate."""

    METADATA_UNAVAILABLE = "metadata-unavailable"
    MANIFEST_COUNT = "manifest-count"
    MANIFEST_IO = "manifest-io"
    MANIFEST_INVALID = "manifest-invalid"
    INCOMPATIBLE = "incompatible"
    IDENTITY_MISMATCH = "identity-mismatch"
    DUPLICATE_ID = "duplicate-id"


@dataclass(frozen=True, slots=True)
class ResolvedPlugin:
    """Validated identity safe to hand to compatibility and policy layers."""

    plugin_id: str
    version: str
    source: PluginSource
    distribution_name: str
    distribution_version: str | None
    entry_point_name: str
    entry_point_value: str | None
    manifest_path: Path
    manifest: dict


@dataclass(frozen=True, slots=True)
class PluginRejection:
    """Bounded rejection that cannot abort validation of other candidates."""

    entry_point_name: str
    distribution_name: str | None
    code: PluginRejectionCode
    detail: str


@dataclass(frozen=True, slots=True)
class PluginCatalog:
    """Accepted identities and independently rejected candidates."""

    plugins: tuple[ResolvedPlugin, ...]
    rejections: tuple[PluginRejection, ...]


class _CandidateRejectedError(ValueError):
    def __init__(self, code: PluginRejectionCode, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def resolve_plugin_identities(candidates: tuple[PluginMetadata, ...]) -> PluginCatalog:
    """Validate candidates independently and reject ambiguous external IDs."""
    validated: list[ResolvedPlugin] = []
    rejected: list[PluginRejection] = []
    for candidate in candidates:
        try:
            validated.append(_validate_candidate(candidate))
        except _CandidateRejectedError as error:
            rejected.append(_rejection(candidate, error.code, error.detail))

    accepted: list[ResolvedPlugin] = []
    groups: dict[str, list[ResolvedPlugin]] = {}
    for plugin in validated:
        groups.setdefault(plugin.plugin_id, []).append(plugin)
    for plugin_id in sorted(groups):
        group = groups[plugin_id]
        bundled = [plugin for plugin in group if plugin.source is PluginSource.BUNDLED]
        if len(group) == 1:
            accepted.extend(group)
        elif len(bundled) == 1:
            accepted.extend(bundled)
            rejected.extend(
                _identity_rejection(
                    plugin,
                    PluginRejectionCode.DUPLICATE_ID,
                    "plugin ID conflicts with a bundled plugin",
                )
                for plugin in group
                if plugin.source is PluginSource.EXTERNAL
            )
        else:
            rejected.extend(
                _identity_rejection(
                    plugin,
                    PluginRejectionCode.DUPLICATE_ID,
                    "plugin ID is declared by multiple candidates",
                )
                for plugin in group
            )

    accepted.sort(key=lambda plugin: (plugin.source, plugin.plugin_id))
    rejected.sort(
        key=lambda item: (
            item.entry_point_name,
            item.distribution_name or "",
            item.code,
        )
    )
    return PluginCatalog(tuple(accepted), tuple(rejected))


def _validate_candidate(candidate: PluginMetadata) -> ResolvedPlugin:
    if candidate.metadata_error:
        raise _CandidateRejectedError(
            PluginRejectionCode.METADATA_UNAVAILABLE,
            candidate.metadata_error,
        )
    if len(candidate.manifest_paths) != 1:
        raise _CandidateRejectedError(
            PluginRejectionCode.MANIFEST_COUNT,
            "candidate must provide exactly one manifest",
        )
    manifest = _candidate_manifest(candidate)
    _validate_manifest_identity(candidate, manifest)
    return ResolvedPlugin(
        plugin_id=manifest["id"],
        version=manifest["version"],
        source=candidate.source,
        distribution_name=candidate.distribution_name or "omnitensor",
        distribution_version=candidate.distribution_version,
        entry_point_name=candidate.entry_point_name,
        entry_point_value=candidate.entry_point_value,
        manifest_path=candidate.manifest_paths[0],
        manifest=manifest,
    )


def _candidate_manifest(candidate: PluginMetadata) -> dict:
    if candidate.source is PluginSource.EXTERNAL:
        _validate_external_metadata(candidate)
        manifest = _read_manifest(candidate.manifest_paths[0])
        if manifest.get("manifestVersion") != 2:
            raise _CandidateRejectedError(
                PluginRejectionCode.INCOMPATIBLE,
                "external plugins require manifest version 2",
            )
        return manifest
    if candidate.distribution_name != "omnitensor" or candidate.bundled_manifest is None:
        raise _CandidateRejectedError(
            PluginRejectionCode.IDENTITY_MISMATCH,
            "bundled candidate does not belong to OmniTensor",
        )
    return candidate.bundled_manifest


def _validate_manifest_identity(candidate: PluginMetadata, manifest: dict) -> None:
    if validate_workload_document(manifest):
        raise _CandidateRejectedError(
            PluginRejectionCode.MANIFEST_INVALID,
            "manifest violates the workload schema",
        )
    plugin_id = manifest["id"]
    if candidate.entry_point_name != plugin_id:
        raise _CandidateRejectedError(
            PluginRejectionCode.IDENTITY_MISMATCH,
            "manifest ID and entry-point name differ",
        )
    if manifest["manifestVersion"] == 2:
        declared_entry_point = manifest["plugin"]["entryPoint"]
        if declared_entry_point != candidate.entry_point_name:
            raise _CandidateRejectedError(
                PluginRejectionCode.IDENTITY_MISMATCH,
                "manifest and installed entry-point identities differ",
            )


def _validate_external_metadata(candidate: PluginMetadata) -> None:
    name = candidate.distribution_name
    version = candidate.distribution_version
    target = candidate.entry_point_value
    if (
        not name
        or len(name) > MAX_DISTRIBUTION_NAME_LENGTH
        or _DISTRIBUTION_PATTERN.fullmatch(name) is None
        or not version
        or len(version) > MAX_DISTRIBUTION_VERSION_LENGTH
    ):
        raise _CandidateRejectedError(
            PluginRejectionCode.METADATA_UNAVAILABLE,
            "distribution identity is missing or invalid",
        )
    if (
        not target
        or len(target) > MAX_ENTRY_POINT_TARGET_LENGTH
        or _ENTRY_POINT_TARGET_PATTERN.fullmatch(target) is None
    ):
        raise _CandidateRejectedError(
            PluginRejectionCode.IDENTITY_MISMATCH,
            "entry-point target is missing or invalid",
        )


def _read_manifest(path: Path) -> dict:
    try:
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            raise _CandidateRejectedError(
                PluginRejectionCode.MANIFEST_INVALID,
                f"manifest exceeds {MAX_MANIFEST_BYTES} bytes",
            )
        document = json.loads(path.read_text(encoding="utf-8"))
    except _CandidateRejectedError:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise _CandidateRejectedError(
            PluginRejectionCode.MANIFEST_IO,
            f"manifest cannot be read: {type(error).__name__}",
        ) from error
    if not isinstance(document, dict):
        raise _CandidateRejectedError(
            PluginRejectionCode.MANIFEST_INVALID,
            "manifest root must be an object",
        )
    return document


def _rejection(
    candidate: PluginMetadata,
    code: PluginRejectionCode,
    detail: str,
) -> PluginRejection:
    return PluginRejection(
        entry_point_name=candidate.entry_point_name,
        distribution_name=candidate.distribution_name,
        code=code,
        detail=detail,
    )


def _identity_rejection(
    plugin: ResolvedPlugin,
    code: PluginRejectionCode,
    detail: str,
) -> PluginRejection:
    return PluginRejection(
        entry_point_name=plugin.entry_point_name,
        distribution_name=plugin.distribution_name,
        code=code,
        detail=detail,
    )
