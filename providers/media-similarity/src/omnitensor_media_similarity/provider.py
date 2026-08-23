"""OmniTensor plugin facade for one directory-wide media scan."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path, PurePosixPath

from omnitensor.sdk import (
    ManagedPlugin,
    PluginHealth,
    PluginHealthStatus,
    PluginProgress,
    PluginRequest,
    succeeded_result,
    workload_result,
)

from .fingerprints import MediaFingerprint, MediaFingerprintError, fingerprint_file
from .similarity import similarity_groups

PLUGIN_ID = "media-similarity"
READ_PERMISSION = "files:read-selected"
DEFAULT_MINIMUM_SIMILARITY = 0.8
_SAFE_COMPONENT = re.compile(r"^[^/\\\x00]+$")


class MediaSimilarityError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(detail)


class MediaSimilarityPlugin(ManagedPlugin):
    plugin_id = PLUGIN_ID

    def __init__(
        self,
        *,
        extractor: Callable[[Path, str, object], MediaFingerprint] = fingerprint_file,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        super().__init__()
        self._extractor = extractor
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    async def on_start(self) -> None:
        self.permissions.require(READ_PERMISSION)

    async def on_health(self) -> PluginHealth:
        return PluginHealth(
            PluginHealthStatus.READY,
            "ready for directory-scale audio and video similarity",
            self._clock_ms(),
        )

    async def execute(self, request, cancellation, progress):
        return await workload_result(
            request,
            lambda: self._scan(request, cancellation, progress),
            completed_at_ms=self._clock_ms,
            cancelled_detail="media similarity scan cancelled",
            failures=(MediaSimilarityError, Exception),
            detail_of=lambda error: str(getattr(error, "code", "media-similarity-failed")),
        )

    async def _scan(self, request: PluginRequest, cancellation, progress):
        sources, relative_paths, minimum = _validated_request(request)
        found: list[MediaFingerprint] = []
        extracted: dict[Path, MediaFingerprint | MediaFingerprintError] = {}
        failures = []
        total = len(sources)
        await self._report(request, progress, "fingerprint", 0.0, f"0 of {total} files")
        for index, (source, relative_path) in enumerate(
            zip(sources, relative_paths, strict=True), start=1
        ):
            cancellation.raise_if_cancelled()
            try:
                selected = Path(source)
                cached = extracted.get(selected)
                if isinstance(cached, MediaFingerprintError):
                    raise cached
                fingerprint = cached
                if fingerprint is None:
                    try:
                        fingerprint = await asyncio.to_thread(
                            self._extractor, selected, relative_path, cancellation
                        )
                    except MediaFingerprintError as error:
                        extracted[selected] = error
                        raise
                    extracted[selected] = fingerprint
                elif fingerprint.relative_path != relative_path:
                    fingerprint = replace(
                        fingerprint,
                        relative_path=relative_path,
                        file_name=Path(relative_path).name,
                    )
                found.append(fingerprint)
            except MediaFingerprintError as error:
                failures.append({"relativePath": relative_path, "code": error.code})
            cancellation.raise_if_cancelled()
            await self._report(
                request,
                progress,
                "fingerprint",
                0.9 * index / total,
                f"{index} of {total} files",
            )
        await self._report(request, progress, "group", 0.95, "grouping similar media")
        groups = await asyncio.to_thread(similarity_groups, tuple(found), minimum)
        grouped = {file["relativePath"] for group in groups for file in group.get("files", ())}
        output = {
            "version": 1,
            "scannedFileCount": total,
            "fingerprintedFileCount": len(found),
            "similarFileCount": len(grouped),
            "unmatchedFileCount": len(found) - len(grouped),
            "minimumSimilarity": minimum,
            "groups": list(groups),
            "failures": failures,
        }
        await self._report(request, progress, "terminal", 1.0, "similarity groups ready")
        return succeeded_result(
            request,
            output,
            completed_at_ms=self._clock_ms(),
            detail="media similarity groups ready",
        )

    async def _report(self, request, progress, stage: str, fraction: float, detail: str) -> None:
        await progress.report(
            PluginProgress(request.job_id, stage, fraction, detail, self._clock_ms())
        )


def _validated_request(request: PluginRequest) -> tuple[tuple[str, ...], tuple[str, ...], float]:
    if not isinstance(request, PluginRequest) or request.plugin_id != PLUGIN_ID:
        raise MediaSimilarityError("request-invalid", "media similarity request is invalid")
    sources = request.payload.get("sources")
    paths = request.payload.get("relativePaths")
    if (
        not isinstance(sources, list)
        or not sources
        or not all(isinstance(item, str) for item in sources)
    ):
        raise MediaSimilarityError("sources-invalid", "sources must name selected media files")
    if (
        not isinstance(paths, list)
        or len(paths) != len(sources)
        or not all(_safe_relative_path(item) for item in paths)
        or len(set(paths)) != len(paths)
    ):
        raise MediaSimilarityError(
            "paths-invalid", "relativePaths must uniquely name every selected media file"
        )
    minimum = request.payload.get("minimumSimilarity", DEFAULT_MINIMUM_SIMILARITY)
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or not 0 <= minimum <= 1:
        raise MediaSimilarityError(
            "similarity-invalid", "minimumSimilarity must be between zero and one"
        )
    return tuple(sources), tuple(paths), float(minimum)


def _safe_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and all(
        part not in {"", ".", ".."} and _SAFE_COMPONENT.fullmatch(part) for part in path.parts
    )


def create() -> MediaSimilarityPlugin:
    return MediaSimilarityPlugin()


__all__ = ["MediaSimilarityError", "MediaSimilarityPlugin", "create"]
