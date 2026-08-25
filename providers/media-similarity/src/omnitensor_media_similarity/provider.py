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
    current_plugin_bootstrap,
    succeeded_result,
    workload_result,
)

from .checkpoints import FingerprintCheckpointStore, selection_key
from .fingerprints import (
    FingerprintProgress,
    MediaFingerprint,
    MediaFingerprintError,
    file_digest,
    fingerprint_file,
)
from .similarity import similarity_groups

PLUGIN_ID = "media-similarity"
READ_PERMISSION = "files:read-selected"
DEFAULT_MINIMUM_SIMILARITY = 0.8
DEFAULT_PROGRESS_INTERVAL_SECONDS = 5.0
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
        extractor: Callable[[Path, str, object], MediaFingerprint] | None = None,
        checkpoints: FingerprintCheckpointStore | None = None,
        clock_ms: Callable[[], int] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        progress_interval_seconds: float = DEFAULT_PROGRESS_INTERVAL_SECONDS,
    ) -> None:
        super().__init__()
        self._extractor = extractor
        self._checkpoints = checkpoints
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._monotonic = monotonic
        self._progress_interval_seconds = progress_interval_seconds

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
        selection = selection_key(relative_paths)
        await self._report(request, progress, "fingerprint", 0.0, f"0 of {total} files")
        for index, (source, relative_path) in enumerate(
            zip(sources, relative_paths, strict=True), start=1
        ):
            cancellation.raise_if_cancelled()
            try:
                found.append(
                    await self._extract_one(
                        request,
                        progress,
                        selection,
                        Path(source),
                        relative_path,
                        cancellation,
                        index,
                        total,
                        extracted,
                    )
                )
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
        if self._checkpoints is not None:
            await asyncio.to_thread(self._checkpoints.clear, selection)
        await self._report(request, progress, "terminal", 1.0, "similarity groups ready")
        return succeeded_result(
            request,
            output,
            completed_at_ms=self._clock_ms(),
            detail="media similarity groups ready",
        )

    async def _extract_one(
        self,
        request: PluginRequest,
        progress,
        selection: str,
        selected: Path,
        relative_path: str,
        cancellation,
        index: int,
        total: int,
        extracted: dict[Path, MediaFingerprint | MediaFingerprintError],
    ) -> MediaFingerprint:
        cached = extracted.get(selected)
        if isinstance(cached, MediaFingerprintError):
            raise cached
        fingerprint = cached
        if fingerprint is None:
            try:
                if self._extractor is not None:
                    fingerprint = await asyncio.to_thread(
                        self._extractor, selected, relative_path, cancellation
                    )
                else:
                    observer = _FingerprintProgressRelay(
                        request,
                        progress,
                        index,
                        total,
                        self._clock_ms,
                        self._monotonic,
                        self._progress_interval_seconds,
                    )
                    fingerprint = await asyncio.to_thread(
                        self._resumable_fingerprint,
                        selection,
                        selected,
                        relative_path,
                        cancellation,
                        observer,
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
        return fingerprint

    def _resumable_fingerprint(
        self,
        selection: str,
        source: Path,
        relative_path: str,
        cancellation,
        observer,
    ) -> MediaFingerprint:
        digest = file_digest(source, cancellation, observer)
        if self._checkpoints is not None:
            cached = self._checkpoints.load(selection, relative_path, digest)
            if cached is not None:
                observer(FingerprintProgress("checkpoint", 1, 1, "files"))
                return cached
        fingerprint = fingerprint_file(
            source,
            relative_path,
            cancellation,
            observer,
            digest=digest,
        )
        if self._checkpoints is not None:
            self._checkpoints.save(selection, fingerprint)
        return fingerprint

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


class _FingerprintProgressRelay:
    """Publish throttled worker-thread work units on the request's event loop."""

    def __init__(
        self,
        request: PluginRequest,
        progress,
        file_index: int,
        file_count: int,
        clock_ms: Callable[[], int],
        monotonic: Callable[[], float],
        interval_seconds: float,
    ) -> None:
        self._request = request
        self._progress = progress
        self._file_index = file_index
        self._file_count = file_count
        self._clock_ms = clock_ms
        self._monotonic = monotonic
        self._interval = interval_seconds
        self._loop = asyncio.get_running_loop()
        self._last_stage: str | None = None
        self._last_at = float("-inf")

    def __call__(self, update: FingerprintProgress) -> None:
        now = self._monotonic()
        completed = update.total is not None and update.completed == update.total
        if (
            update.stage == self._last_stage
            and not completed
            and now - self._last_at < self._interval
        ):
            return
        self._last_stage = update.stage
        self._last_at = now
        future = asyncio.run_coroutine_threadsafe(self._publish(update), self._loop)
        future.result()

    async def _publish(self, update: FingerprintProgress) -> None:
        extent = f" of {update.total}" if update.total is not None else ""
        await self._progress.report(
            PluginProgress(
                self._request.job_id,
                f"fingerprint-{update.stage}",
                0.9 * (self._file_index - 1) / self._file_count,
                f"file {self._file_index} of {self._file_count}: {update.completed}"
                f"{extent} {update.unit}",
                self._clock_ms(),
            )
        )


def create() -> MediaSimilarityPlugin:
    state_path = current_plugin_bootstrap(PLUGIN_ID).state_path
    checkpoints = None if state_path is None else FingerprintCheckpointStore(state_path)
    return MediaSimilarityPlugin(checkpoints=checkpoints)


__all__ = ["MediaSimilarityError", "MediaSimilarityPlugin", "create"]
