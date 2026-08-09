"""Bounded helpers for plugin authors."""

from __future__ import annotations

import asyncio
import copy
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from ..plugins.artifacts import (
    ArtifactReference,
    ArtifactResolution,
    artifact_reference_error,
)
from ..plugins.protocol import (
    JsonObject,
    PluginProgress,
    PluginRequest,
    PluginResult,
    PluginResultStatus,
)
from .contracts import ProgressSink

MAX_PROGRESS_DETAIL_CHARS = 1024
MAX_RESULT_DETAIL_CHARS = 2048
MAX_PROGRESS_STAGE_CHARS = 80
_MAX_JOB_ID_CHARS = 128
_MISSING = object()


class SDKContractError(ValueError):
    """Stable plugin-author error raised before a host boundary is crossed."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class PluginCancelledError(asyncio.CancelledError):
    """Cooperative cancellation raised by :class:`CancellationController`."""


class CancellationController:
    """Small cancellation token for SDK tests and composed plugin operations."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason = ""

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def cancel(self, reason: str = "cancelled") -> bool:
        _bounded_text(reason, "cancellation reason", MAX_RESULT_DETAIL_CHARS)
        if self.cancelled:
            return False
        self._reason = reason
        self._event.set()
        return True

    async def wait(self) -> None:
        await self._event.wait()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise PluginCancelledError(self._reason)


class ProgressEmitter:
    """Validate monotonic bounded progress before publishing it."""

    def __init__(self, job_id: str, sink: ProgressSink) -> None:
        _bounded_text(job_id, "job_id", _MAX_JOB_ID_CHARS)
        if not isinstance(sink, ProgressSink):
            raise TypeError("sink must implement publish")
        self._job_id = job_id
        self._sink = sink
        self._fraction = 0.0

    async def report(
        self,
        stage: str,
        fraction: float,
        detail: str,
        *,
        observed_at_ms: int,
    ) -> PluginProgress:
        _bounded_text(stage, "progress stage", MAX_PROGRESS_STAGE_CHARS)
        _bounded_text(detail, "progress detail", MAX_PROGRESS_DETAIL_CHARS, allow_empty=True)
        _timestamp(observed_at_ms)
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction)
            or fraction < self._fraction
            or fraction > 1
        ):
            raise SDKContractError(
                "invalid-progress",
                "fraction must be finite, monotonic, and between 0 and 1",
            )
        self._fraction = float(fraction)
        progress = PluginProgress(
            self._job_id,
            stage,
            self._fraction,
            detail,
            observed_at_ms,
        )
        await self._sink.publish(progress)
        return progress


class ConfigurationView:
    """Read a defensive configuration copy with explicit required types."""

    def __init__(self, configuration: JsonObject) -> None:
        if not isinstance(configuration, Mapping):
            raise TypeError("configuration must be a mapping")
        self._configuration = copy.deepcopy(dict(configuration))

    def required(self, key: str, expected_type: type):
        return self._read(key, expected_type, _MISSING)

    def optional(self, key: str, expected_type: type, default=None):
        return self._read(key, expected_type, default)

    def as_dict(self) -> dict:
        return copy.deepcopy(self._configuration)

    def _read(self, key: str, expected_type: type, default):
        _bounded_text(key, "configuration key", 120)
        if not isinstance(expected_type, type):
            raise TypeError("expected_type must be a type")
        value = self._configuration.get(key, _MISSING)
        if value is _MISSING:
            if default is _MISSING:
                raise SDKContractError(
                    "configuration-missing",
                    f"required configuration key is absent: {key}",
                )
            # The default is the caller's own answer for an absent key, so it
            # is not theirs to fail: type-checking it made optional(key, str)
            # raise on the implicit None it was asked to return.
            return copy.deepcopy(default)
        if not _matches_type(value, expected_type):
            raise SDKContractError(
                "configuration-type",
                f"configuration key {key} must be {expected_type.__name__}",
            )
        return copy.deepcopy(value)


class PermissionView:
    """Fail closed over the declared/granted permission intersection."""

    def __init__(self, declared: frozenset[str], granted: frozenset[str]) -> None:
        if not all(isinstance(value, str) and value for value in declared | granted):
            raise SDKContractError("invalid-permissions", "permissions must be non-empty strings")
        undeclared = granted - declared
        if undeclared:
            raise SDKContractError(
                "undeclared-grant",
                f"host supplied undeclared grant: {sorted(undeclared)[0]}",
            )
        self._declared = declared
        self._granted = granted

    @property
    def granted(self) -> frozenset[str]:
        return self._granted

    def allows(self, permission: str) -> bool:
        return permission in self._declared and permission in self._granted

    def require(self, permission: str) -> None:
        if not self.allows(permission):
            raise SDKContractError("permission-denied", f"permission is not granted: {permission}")


class ArtifactView:
    """Expose only declared artifact resolutions and verified ready paths."""

    def __init__(
        self,
        resolutions: Sequence[tuple[ArtifactReference, ArtifactResolution]],
    ) -> None:
        entries: dict[tuple[str, str], tuple[ArtifactReference, ArtifactResolution]] = {}
        for reference, resolution in resolutions:
            problem = artifact_reference_error(reference)
            if problem:
                raise SDKContractError("invalid-artifact", problem)
            key = (reference.id, reference.version)
            if key in entries:
                raise SDKContractError(
                    "duplicate-artifact",
                    f"artifact is declared more than once: {reference.id}@{reference.version}",
                )
            entries[key] = (reference, resolution)
        self._entries = entries

    def resolution(self, reference: ArtifactReference) -> ArtifactResolution:
        key = (reference.id, reference.version)
        try:
            declared, resolution = self._entries[key]
        except KeyError as error:
            raise SDKContractError(
                "artifact-undeclared",
                f"artifact is not declared: {reference.id}@{reference.version}",
            ) from error
        if declared != reference:
            raise SDKContractError(
                "artifact-identity-mismatch",
                f"artifact metadata differs: {reference.id}@{reference.version}",
            )
        return resolution

    def require_ready(self, reference: ArtifactReference) -> Path:
        resolution = self.resolution(reference)
        if not resolution.ready or resolution.path is None:
            raise SDKContractError(
                "artifact-unavailable",
                resolution.reason or "artifact is not ready",
            )
        return resolution.path


def succeeded_result(
    request: PluginRequest,
    output: JsonObject,
    *,
    completed_at_ms: int,
    detail: str = "",
) -> PluginResult:
    return _result(request, PluginResultStatus.SUCCEEDED, output, detail, completed_at_ms)


def failed_result(
    request: PluginRequest,
    detail: str,
    *,
    completed_at_ms: int,
) -> PluginResult:
    return _result(request, PluginResultStatus.FAILED, {}, detail, completed_at_ms)


def cancelled_result(
    request: PluginRequest,
    detail: str,
    *,
    completed_at_ms: int,
) -> PluginResult:
    return _result(request, PluginResultStatus.CANCELLED, {}, detail, completed_at_ms)


def _result(
    request: PluginRequest,
    status: PluginResultStatus,
    output: JsonObject,
    detail: str,
    completed_at_ms: int,
) -> PluginResult:
    if not isinstance(request, PluginRequest):
        raise TypeError("request must be a PluginRequest")
    if not isinstance(output, Mapping):
        raise TypeError("result output must be a mapping")
    _bounded_text(detail, "result detail", MAX_RESULT_DETAIL_CHARS, allow_empty=True)
    _timestamp(completed_at_ms)
    return PluginResult(
        request.job_id,
        status,
        copy.deepcopy(dict(output)),
        detail,
        completed_at_ms,
    )


def _bounded_text(value: str, name: str, maximum: int, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str) or (not value and not allow_empty) or len(value) > maximum:
        qualifier = f"0-{maximum}" if allow_empty else f"1-{maximum}"
        raise SDKContractError("invalid-text", f"{name} must contain {qualifier} characters")


def _timestamp(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SDKContractError("invalid-timestamp", "timestamp must be a non-negative integer")


def _matches_type(value, expected_type: type) -> bool:
    if expected_type is int and isinstance(value, bool):
        return False
    return isinstance(value, expected_type)
