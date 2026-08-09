"""Deterministic workload pipeline state and typed stage boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

from .artifacts import ArtifactReference
from .protocol import JsonObject, PluginResult, PluginResultStatus

_MAX_JOB_ID_CHARS = 128


class PipelineStage(StrEnum):
    """Ordered stages; terminal status lives in the single plugin result."""

    QUEUED = "queued"
    COLLECT = "collect"
    PREPROCESS = "preprocess"
    RESOLVE = "resolve"
    INFER = "infer"
    POSTPROCESS = "postprocess"
    DELIVER = "deliver"
    TERMINAL = "terminal"


class PipelineTransitionError(RuntimeError):
    """Stable state-transition rejection for orchestration diagnostics."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class CollectedOutput:
    """Bounded source material emitted by collection."""

    payload: JsonObject


@dataclass(frozen=True, slots=True)
class PreprocessedOutput:
    """Executor-ready tensors and non-secret inference metadata."""

    tensors: JsonObject
    metadata: JsonObject


@dataclass(frozen=True, slots=True)
class ResolvedOutput:
    """Verified artifact selected for this immutable job revision."""

    artifact: ArtifactReference
    path: Path


@dataclass(frozen=True, slots=True)
class InferenceOutput:
    """Raw bounded executor outputs."""

    tensors: JsonObject


@dataclass(frozen=True, slots=True)
class PostprocessedOutput:
    """Validated domain result ready for an authorized consumer."""

    output: JsonObject


@dataclass(frozen=True, slots=True)
class DeliveredOutput:
    """Final public result after the delivery side effect succeeds."""

    output: JsonObject
    detail: str = ""


StageOutput: TypeAlias = (
    CollectedOutput
    | PreprocessedOutput
    | ResolvedOutput
    | InferenceOutput
    | PostprocessedOutput
    | DeliveredOutput
)


@dataclass(frozen=True, slots=True)
class PipelineSnapshot:
    """Current state without retaining unbounded transition history."""

    job_id: str
    stage: PipelineStage
    revision: int
    last_output: StageOutput | None
    terminal_result: PluginResult | None


_TRANSITIONS: dict[PipelineStage, tuple[type[StageOutput], PipelineStage]] = {
    PipelineStage.COLLECT: (CollectedOutput, PipelineStage.PREPROCESS),
    PipelineStage.PREPROCESS: (PreprocessedOutput, PipelineStage.RESOLVE),
    PipelineStage.RESOLVE: (ResolvedOutput, PipelineStage.INFER),
    PipelineStage.INFER: (InferenceOutput, PipelineStage.POSTPROCESS),
    PipelineStage.POSTPROCESS: (PostprocessedOutput, PipelineStage.DELIVER),
    PipelineStage.DELIVER: (DeliveredOutput, PipelineStage.TERMINAL),
}


class PipelineStateMachine:
    """Enforce ordered stage outputs and publish exactly one terminal result."""

    def __init__(self, job_id: str):
        if not isinstance(job_id, str) or not job_id or len(job_id) > _MAX_JOB_ID_CHARS:
            raise ValueError(f"job_id must contain 1-{_MAX_JOB_ID_CHARS} characters")
        self._job_id = job_id
        self._stage = PipelineStage.QUEUED
        self._revision = 0
        self._last_output: StageOutput | None = None
        self._terminal_result: PluginResult | None = None

    @property
    def snapshot(self) -> PipelineSnapshot:
        return PipelineSnapshot(
            self._job_id,
            self._stage,
            self._revision,
            self._last_output,
            self._terminal_result,
        )

    @property
    def terminal_result(self) -> PluginResult | None:
        return self._terminal_result

    def start(self) -> PipelineSnapshot:
        """Move a new job to collection; repeated starts are rejected."""
        if self._stage is not PipelineStage.QUEUED:
            raise PipelineTransitionError(
                "already-started",
                f"job is already at {self._stage}",
            )
        self._stage = PipelineStage.COLLECT
        self._revision += 1
        return self.snapshot

    def advance(self, output: StageOutput, *, completed_at_ms: int) -> PipelineSnapshot:
        """Accept only the output type belonging to the current stage."""
        _validate_timestamp(completed_at_ms)
        if self._stage in {PipelineStage.QUEUED, PipelineStage.TERMINAL}:
            raise PipelineTransitionError(
                "stage-not-runnable",
                f"cannot advance from {self._stage}",
            )
        expected_type, next_stage = _TRANSITIONS[self._stage]
        if not isinstance(output, expected_type):
            received_type = type(output).__name__
            raise PipelineTransitionError(
                "stage-output-mismatch",
                f"{self._stage} requires {expected_type.__name__}; received {received_type}",
            )
        self._last_output = output
        if next_stage is PipelineStage.TERMINAL:
            delivered = output
            return self._finish(
                PluginResultStatus.SUCCEEDED,
                delivered.output,
                delivered.detail,
                completed_at_ms,
            )
        self._stage = next_stage
        self._revision += 1
        return self.snapshot

    def fail(self, detail: str, *, completed_at_ms: int) -> PipelineSnapshot:
        """Publish one failed result; later terminal signals return it unchanged."""
        return self._finish(PluginResultStatus.FAILED, {}, detail, completed_at_ms)

    def cancel(self, detail: str, *, completed_at_ms: int) -> PipelineSnapshot:
        """Publish one cancelled result; later terminal signals return it unchanged."""
        return self._finish(PluginResultStatus.CANCELLED, {}, detail, completed_at_ms)

    def _finish(
        self,
        status: PluginResultStatus,
        output: JsonObject,
        detail: str,
        completed_at_ms: int,
    ) -> PipelineSnapshot:
        if self._terminal_result is not None:
            return self.snapshot
        _validate_timestamp(completed_at_ms)
        if not isinstance(detail, str):
            raise TypeError("terminal detail must be a string")
        self._terminal_result = PluginResult(
            self._job_id,
            status,
            output,
            detail,
            completed_at_ms,
        )
        self._stage = PipelineStage.TERMINAL
        self._revision += 1
        return self.snapshot


def _validate_timestamp(completed_at_ms: int) -> None:
    if (
        isinstance(completed_at_ms, bool)
        or not isinstance(completed_at_ms, int)
        or completed_at_ms < 0
    ):
        raise ValueError("completed_at_ms must be a non-negative integer")
