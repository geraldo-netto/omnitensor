"""Stable value and port contracts shared by the generation provider modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from .acceptance_kit import (
    NativeLoadReport,
    require_integer,
    require_mapping,
    require_sequence,
    require_text,
)
from .generation import GenerationRequest, GenerationTask
from .protocol import CancellationToken, ProgressReporter


class GenerationProviderError(ValueError):
    """Stable provider configuration or qualification refusal."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class ModelSource:
    role: str
    uri: str
    revision: str
    filename: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ProviderTemplate:
    """One declared lane: who runs the model, on what, and whether by default.

    Five positional fields used to travel as a bare tuple, so `item[3]` was
    "is it the default" only in the reader's head, and a test asserted
    `providers[0][4]` to reach a status string.
    """

    provider_id: str
    accelerator: str
    runtime: str
    default: bool
    status: str


@dataclass(frozen=True, slots=True)
class ModelCatalog:
    model_id: str
    version: str
    upstream_revision: str
    license_spdx: str
    sources: tuple[ModelSource, ...]
    providers: tuple[ProviderTemplate, ...]
    evaluation: ModelEvaluation


@runtime_checkable
class NativeGenerationRuntime(Protocol):
    """Native SDK adapter hosted inside a killable plugin worker."""

    async def load(self, artifacts: tuple[Path, ...], accelerator: str) -> NativeLoadReport: ...

    async def generate(
        self,
        task: GenerationTask,
        request: GenerationRequest,
        cancellation: CancellationToken,
        progress: ProgressReporter,
    ) -> str: ...

    async def terminate(self, request_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class FrozenEventCase:
    case_id: str
    modality: str
    content: str
    prompt_injection_probe: bool
    expected: tuple[tuple[str, str, str, str], ...]


@dataclass(frozen=True, slots=True)
class FrozenEventCorpus:
    corpus_id: str
    sha256: str
    cases: tuple[FrozenEventCase, ...]


@dataclass(frozen=True, slots=True)
class EventProviderObservation:
    case_id: str
    result: Mapping[str, object]
    latency_ms: int
    peak_memory_bytes: int


@dataclass(frozen=True, slots=True)
class EventProviderEvidence:
    provider_id: str
    corpus_sha256: str
    runtime_version: str
    device_name: str
    load: NativeLoadReport
    observations: tuple[EventProviderObservation, ...]
    cancellation_latency_ms: int


@dataclass(frozen=True, slots=True)
class EventQualificationPolicy:
    minimum_precision: float = 0.9
    minimum_recall: float = 0.9
    maximum_p95_latency_ms: int = 30_000
    maximum_peak_memory_bytes: int = 12 * 1024 * 1024 * 1024
    maximum_cancellation_latency_ms: int = 2_000


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    corpus: str
    policy: EventQualificationPolicy


@dataclass(frozen=True, slots=True)
class EventQualificationReport:
    provider_id: str
    accelerator: str
    corpus_sha256: str
    runtime_version: str
    device_name: str
    precision: float
    recall: float
    p95_latency_ms: int
    peak_memory_bytes: int
    cancellation_latency_ms: int
    qualified: bool


def bounded_mapping(value: object, label: str, code: str) -> Mapping[str, object]:
    return require_mapping(
        value,
        error_type=GenerationProviderError,
        code=code,
        detail=f"{label} must be an object",
    )


def bounded_sequence(value: object, label: str, code: str) -> Sequence[object]:
    return require_sequence(
        value,
        error_type=GenerationProviderError,
        code=code,
        detail=f"{label} must be a non-empty sequence",
        allow_empty=False,
    )


def bounded_positive_integer(value: object, label: str, code: str) -> int:
    return require_integer(
        value,
        error_type=GenerationProviderError,
        code=code,
        detail=f"{label} must be a positive integer",
        minimum=1,
    )


def bounded_text(value: object, label: str, code: str, limit: int) -> str:
    return require_text(
        value,
        error_type=GenerationProviderError,
        code=code,
        detail=f"{label} must be bounded text",
        maximum=limit,
        allow_whitespace=True,
    )


def validate_event_policy(policy: EventQualificationPolicy) -> None:
    if not isinstance(policy, EventQualificationPolicy):
        raise GenerationProviderError("policy-invalid", "qualification policy is invalid")
    for name in ("minimum_precision", "minimum_recall"):
        value = getattr(policy, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise GenerationProviderError("policy-invalid", f"{name} must be in [0, 1]")
    for name in (
        "maximum_p95_latency_ms",
        "maximum_peak_memory_bytes",
        "maximum_cancellation_latency_ms",
    ):
        bounded_positive_integer(getattr(policy, name), name, "policy-invalid")


__all__ = [
    "EventProviderEvidence",
    "EventProviderObservation",
    "EventQualificationPolicy",
    "EventQualificationReport",
    "FrozenEventCase",
    "FrozenEventCorpus",
    "NativeGenerationRuntime",
    "ModelCatalog",
    "ProviderTemplate",
    "ModelEvaluation",
    "GenerationProviderError",
    "ModelSource",
]
