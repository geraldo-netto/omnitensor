"""Deterministic local plugin runner and accelerator-free test doubles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..plugins.artifacts import ArtifactReference, ArtifactResolution
from ..plugins.pipeline import CollectedOutput, InferenceOutput
from ..plugins.protocol import (
    CancellationToken,
    JsonObject,
    PluginProgress,
    PluginRequest,
    PluginResult,
)
from ..plugins.triggers import (
    Collector,
    CollectorReadiness,
    SourceStatus,
    Trigger,
    TriggerKind,
)
from .contracts import ArtifactProvider, PipelineHooks, ProgressSink, ResultSink
from .helpers import (
    ArtifactView,
    CancellationController,
    PermissionView,
    PluginCancelledError,
    ProgressEmitter,
    SDKContractError,
    cancelled_result,
    failed_result,
    succeeded_result,
)

DEFAULT_MAX_RECORDED_RESULTS = 128
DEFAULT_MAX_RECORDED_PROGRESS = 512


class FakeClock:
    """Manually advanced monotonic millisecond clock."""

    def __init__(self, now_ms: int = 0) -> None:
        _non_negative_integer(now_ms, "now_ms")
        self._now_ms = now_ms

    @property
    def now_ms(self) -> int:
        return self._now_ms

    def advance(self, milliseconds: int) -> int:
        _non_negative_integer(milliseconds, "milliseconds")
        self._now_ms += milliseconds
        return self._now_ms


class FakeTriggerFactory:
    """Create stable trigger identities against a fake clock."""

    def __init__(self, plugin_id: str, clock: FakeClock) -> None:
        if not isinstance(plugin_id, str) or not plugin_id:
            raise ValueError("plugin_id must be a non-empty string")
        self._plugin_id = plugin_id
        self._clock = clock
        self._sequence = 0

    def create(self, kind: TriggerKind, payload: JsonObject) -> Trigger:
        if not isinstance(kind, TriggerKind):
            raise TypeError("kind must be a TriggerKind")
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        self._sequence += 1
        return Trigger(
            self._plugin_id,
            f"{kind}-{self._sequence}",
            kind,
            dict(payload),
            self._clock.now_ms,
        )

    def manual(self, payload: JsonObject) -> Trigger:
        return self.create(TriggerKind.MANUAL, payload)

    def event(self, payload: JsonObject) -> Trigger:
        return self.create(TriggerKind.EVENT, payload)

    def periodic(self, payload: JsonObject) -> Trigger:
        return self.create(TriggerKind.PERIODIC, payload)


class FakeCollector:
    """Replay collected outputs or exceptions by trigger identity."""

    def __init__(
        self,
        outputs: Mapping[str, CollectedOutput | Exception],
        *,
        readiness: CollectorReadiness | None = None,
    ) -> None:
        self._outputs = dict(outputs)
        self._readiness = readiness or CollectorReadiness(SourceStatus.READY, "fixture ready", 0)
        self.calls: list[Trigger] = []

    async def readiness(self) -> CollectorReadiness:
        return self._readiness

    async def collect(self, trigger: Trigger) -> CollectedOutput:
        self.calls.append(trigger)
        try:
            output = self._outputs[trigger.trigger_id]
        except KeyError as error:
            raise SDKContractError(
                "fixture-missing",
                f"no collected fixture for trigger: {trigger.trigger_id}",
            ) from error
        if isinstance(output, Exception):
            raise output
        return output


class FakeArtifactProvider:
    """Resolve immutable artifact fixtures by their complete identity."""

    def __init__(
        self,
        resolutions: Sequence[tuple[ArtifactReference, ArtifactResolution]],
    ) -> None:
        self._resolutions = {
            (reference.id, reference.version, reference.format, reference.sha256): resolution
            for reference, resolution in resolutions
        }
        if len(self._resolutions) != len(resolutions):
            raise ValueError("artifact fixtures must have unique identities")
        self.calls: list[ArtifactReference] = []

    def resolve(self, reference: ArtifactReference) -> ArtifactResolution:
        self.calls.append(reference)
        try:
            return self._resolutions[
                (reference.id, reference.version, reference.format, reference.sha256)
            ]
        except KeyError:
            return ArtifactResolution(False, None, "fixture artifact unavailable", 0)


@runtime_checkable
class FakeExecutorContract(Protocol):
    async def infer(
        self,
        artifact_path: Path,
        tensors: JsonObject,
        cancellation: CancellationToken,
    ) -> InferenceOutput: ...


class FakeExecutor:
    """Replay inference outputs without loading accelerator libraries."""

    def __init__(self, outputs: Sequence[InferenceOutput | Exception]) -> None:
        self._outputs = list(outputs)
        self.calls: list[tuple[Path, JsonObject]] = []

    async def infer(
        self,
        artifact_path: Path,
        tensors: JsonObject,
        cancellation: CancellationToken,
    ) -> InferenceOutput:
        cancellation.raise_if_cancelled()
        self.calls.append((artifact_path, tensors))
        if not self._outputs:
            raise SDKContractError("fixture-missing", "no inference fixture remains")
        output = self._outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


class FakePermissions:
    """Mutable local grant fixture that yields immutable permission views."""

    def __init__(self, declared: frozenset[str], granted: frozenset[str] = frozenset()) -> None:
        self._declared = declared
        self._granted = granted
        PermissionView(declared, granted)

    @property
    def view(self) -> PermissionView:
        return PermissionView(self._declared, self._granted)

    def grant(self, permission: str) -> PermissionView:
        if permission not in self._declared:
            raise SDKContractError(
                "permission-undeclared",
                f"permission is undeclared: {permission}",
            )
        self._granted = self._granted | {permission}
        return self.view

    def revoke(self, permission: str) -> PermissionView:
        self._granted = self._granted - {permission}
        return self.view


class RecordingResultSink:
    def __init__(self, *, limit: int = DEFAULT_MAX_RECORDED_RESULTS) -> None:
        _positive_integer(limit, "limit")
        self._limit = limit
        self.items: list[PluginResult] = []

    async def publish(self, result: PluginResult) -> None:
        if not isinstance(result, PluginResult):
            raise TypeError("result must be a PluginResult")
        if len(self.items) >= self._limit:
            raise SDKContractError("result-limit", f"at most {self._limit} results are recorded")
        self.items.append(result)


class RecordingProgressSink:
    def __init__(self, *, limit: int = DEFAULT_MAX_RECORDED_PROGRESS) -> None:
        _positive_integer(limit, "limit")
        self._limit = limit
        self.items: list[PluginProgress] = []

    async def publish(self, progress: PluginProgress) -> None:
        if not isinstance(progress, PluginProgress):
            raise TypeError("progress must be PluginProgress")
        if len(self.items) >= self._limit:
            raise SDKContractError(
                "progress-limit",
                f"at most {self._limit} progress updates are recorded",
            )
        self.items.append(progress)


@dataclass(frozen=True, slots=True)
class LocalRun:
    request: PluginRequest
    result: PluginResult
    progress: tuple[PluginProgress, ...]


class LocalPipelineRunner:
    """Execute collector-to-delivery behavior with deterministic local doubles."""

    def __init__(
        self,
        *,
        collector: Collector,
        pipeline: PipelineHooks,
        artifacts: ArtifactProvider,
        executor: FakeExecutorContract,
        permissions: PermissionView,
        clock: FakeClock,
        results: ResultSink,
        progress: ProgressSink | None = None,
    ) -> None:
        for value, contract, name in (
            (collector, Collector, "collector"),
            (pipeline, PipelineHooks, "pipeline"),
            (artifacts, ArtifactProvider, "artifacts"),
            (executor, FakeExecutorContract, "executor"),
            (results, ResultSink, "results"),
        ):
            if not isinstance(value, contract):
                raise TypeError(f"{name} does not implement its SDK contract")
        if progress is not None and not isinstance(progress, ProgressSink):
            raise TypeError("progress does not implement its SDK contract")
        self._collector = collector
        self._pipeline = pipeline
        self._artifacts = artifacts
        self._executor = executor
        self._permissions = permissions
        self._clock = clock
        self._results = results
        self._progress = progress

    async def run(
        self,
        trigger: Trigger,
        artifact: ArtifactReference,
        *,
        required_permissions: frozenset[str] = frozenset(),
        deadline_ms: int = 30_000,
        cancellation: CancellationController | None = None,
    ) -> LocalRun:
        _positive_integer(deadline_ms, "deadline_ms")
        token = cancellation or CancellationController()
        request = PluginRequest(
            trigger.trigger_id,
            trigger.plugin_id,
            str(trigger.kind),
            trigger.payload,
            trigger.created_at_ms,
            trigger.created_at_ms + deadline_ms,
        )
        capture = RecordingProgressSink()
        progress = ProgressEmitter(request.job_id, _TeeProgressSink(capture, self._progress))
        try:
            for permission in sorted(required_permissions):
                self._permissions.require(permission)
            await self._checkpoint(token, request, progress, "collect", 0.0)
            collected = await self._collector.collect(trigger)
            await self._checkpoint(token, request, progress, "preprocess", 0.2)
            preprocessed = await self._pipeline.preprocess(collected, token)
            await self._checkpoint(token, request, progress, "resolve", 0.4)
            resolution = self._artifacts.resolve(artifact)
            artifact_path = ArtifactView([(artifact, resolution)]).require_ready(artifact)
            await self._checkpoint(token, request, progress, "infer", 0.6)
            inference = await self._executor.infer(artifact_path, preprocessed.tensors, token)
            await self._checkpoint(token, request, progress, "postprocess", 0.8)
            postprocessed = await self._pipeline.postprocess(inference, token)
            await self._checkpoint(token, request, progress, "deliver", 0.9)
            delivered = await self._pipeline.deliver(postprocessed, token)
            result = succeeded_result(
                request,
                delivered.output,
                detail=delivered.detail,
                completed_at_ms=self._clock.now_ms,
            )
            await progress.report("terminal", 1.0, "succeeded", observed_at_ms=self._clock.now_ms)
        except PluginCancelledError as error:
            result = cancelled_result(
                request,
                str(error) or "cancelled",
                completed_at_ms=self._clock.now_ms,
            )
        except Exception as error:
            result = failed_result(
                request,
                f"local pipeline failed: {type(error).__name__}",
                completed_at_ms=self._clock.now_ms,
            )
        await self._results.publish(result)
        return LocalRun(request, result, tuple(capture.items))

    async def _checkpoint(
        self,
        cancellation: CancellationController,
        request: PluginRequest,
        progress: ProgressEmitter,
        stage: str,
        fraction: float,
    ) -> None:
        if request.deadline_at_ms is not None and self._clock.now_ms > request.deadline_at_ms:
            cancellation.cancel("deadline exceeded")
        cancellation.raise_if_cancelled()
        await progress.report(stage, fraction, "", observed_at_ms=self._clock.now_ms)


class _TeeProgressSink:
    def __init__(self, capture: ProgressSink, downstream: ProgressSink | None) -> None:
        self._capture = capture
        self._downstream = downstream

    async def publish(self, progress: PluginProgress) -> None:
        await self._capture.publish(progress)
        if self._downstream is not None:
            await self._downstream.publish(progress)


def _non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
