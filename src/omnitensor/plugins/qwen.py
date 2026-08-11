"""Fail-closed Qwen generation providers and their qualification evidence.

This module deliberately does not import llama.cpp or OpenVINO.  It runs in a
plugin worker and speaks to one injected native-runtime port, keeping vendor
SDKs and private content outside the service process.  A provider descriptor
must already be qualified, and loading must return device evidence: requesting
GPU/NPU is not accepted as proof that the model actually stayed off CPU.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..atomicio import JsonTooLargeError, read_json_bounded
from ..preparation import file_digest
from .events import EventResultError, parse_grounded_event_result
from .generation import (
    GenerationProviderDescriptor,
    GenerationRequest,
    GenerationTask,
    ProviderGenerationError,
)
from .protocol import CancellationToken, ProgressReporter

MAX_CATALOG_BYTES = 64 * 1024
MAX_CORPUS_BYTES = 256 * 1024
_PACKAGE_DIR = Path(__file__).resolve().parent
_PACKAGED_MODELS = _PACKAGE_DIR.parent / "generation-models"
_PACKAGED_CORPORA = _PACKAGE_DIR.parent / "evaluation-corpora"
_SOURCE_ROOT = _PACKAGE_DIR.parents[2] if _PACKAGE_DIR.parent.parent.name == "src" else None
_DIGEST = frozenset("0123456789abcdef")


class QwenProviderError(ValueError):
    """Stable provider configuration or qualification refusal."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class QwenSource:
    role: str
    uri: str
    revision: str
    filename: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class QwenCatalog:
    model_id: str
    version: str
    upstream_revision: str
    license_spdx: str
    sources: tuple[QwenSource, ...]
    providers: tuple[tuple[str, str, str, bool, str], ...]
    evaluation: QwenEvaluation


@dataclass(frozen=True, slots=True)
class NativeLoadReport:
    backend: str
    device: str
    total_model_layers: int
    accelerator_layers: int
    cpu_fallback: bool


@runtime_checkable
class NativeQwenRuntime(Protocol):
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
class QwenEvaluation:
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


class _QwenWorker:
    accelerator: str
    runtime_name: str

    def __init__(
        self,
        descriptor: GenerationProviderDescriptor,
        runtime: NativeQwenRuntime,
        artifacts: Sequence[Path | str],
        *,
        companion_sha256: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(runtime, NativeQwenRuntime):
            raise QwenProviderError("runtime-invalid", "native runtime does not implement its port")
        if descriptor.accelerator != self.accelerator or descriptor.runtime != self.runtime_name:
            raise QwenProviderError("provider-invalid", "provider descriptor names another lane")
        if not descriptor.qualified:
            raise QwenProviderError("provider-unqualified", "provider has no accepted evidence")
        paths = tuple(Path(path) for path in artifacts)
        if not paths:
            raise QwenProviderError("artifact-invalid", "at least one model artifact is required")
        self._descriptor = descriptor
        self._runtime = runtime
        self._artifacts = paths
        self._companion_sha256 = dict(companion_sha256 or {})
        self._load_report: NativeLoadReport | None = None

    @property
    def descriptor(self) -> GenerationProviderDescriptor:
        return self._descriptor

    async def generate(
        self,
        task: GenerationTask,
        request: GenerationRequest,
        cancellation: CancellationToken,
        progress: ProgressReporter,
    ) -> str:
        cancellation.raise_if_cancelled()
        if self._load_report is None:
            self._verify_artifacts()
            try:
                report = await self._runtime.load(self._artifacts, self.accelerator)
            except ProviderGenerationError:
                raise
            except Exception as error:
                raise ProviderGenerationError(
                    "model-load-failed", "native model loading failed", generation_started=False
                ) from error
            self._validate_load(report)
            self._load_report = report
        cancellation.raise_if_cancelled()
        try:
            return await self._runtime.generate(task, request, cancellation, progress)
        except ProviderGenerationError as error:
            if error.generation_started:
                await self.terminate(request.request_id)
            raise

    async def terminate(self, request_id: str) -> None:
        self._load_report = None
        await self._runtime.terminate(request_id)

    def _verify_artifacts(self) -> None:
        primary = self._artifacts[0]
        if not primary.is_file() or file_digest(primary) != self._descriptor.provenance.sha256:
            raise ProviderGenerationError(
                "model-load-failed",
                "primary model artifact is absent or changed",
                generation_started=False,
            )
        expected_names = set(self._companion_sha256)
        actual_names = {path.name for path in self._artifacts[1:]}
        if actual_names != expected_names:
            raise ProviderGenerationError(
                "model-load-failed", "model companion set is incomplete", generation_started=False
            )
        for path in self._artifacts[1:]:
            if not path.is_file() or file_digest(path) != self._companion_sha256[path.name]:
                raise ProviderGenerationError(
                    "model-load-failed",
                    "model companion is absent or changed",
                    generation_started=False,
                )

    def _validate_load(self, report: NativeLoadReport) -> None:
        raise NotImplementedError


class LlamaCppVulkanQwenWorker(_QwenWorker):
    """Qwen GGUF provider that requires every model layer on Vulkan."""

    accelerator = "gpu"
    runtime_name = "llama.cpp-vulkan"

    def _validate_load(self, report: NativeLoadReport) -> None:
        _validate_gpu_load(report)


class OpenVinoNpuQwenWorker(_QwenWorker):
    """Explicit local OpenVINO GenAI provider that must stay on NPU."""

    accelerator = "npu"
    runtime_name = "openvino-genai-npu"

    def __init__(self, *args, explicitly_enabled: bool = False, **kwargs) -> None:
        if not explicitly_enabled:
            raise QwenProviderError(
                "provider-disabled", "NPU generation needs explicit local configuration"
            )
        super().__init__(*args, **kwargs)

    def _validate_load(self, report: NativeLoadReport) -> None:
        _validate_npu_load(report)


def load_qwen_catalog(path: Path | str | None = None) -> QwenCatalog:
    document = _read_document(path or _catalog_path(), MAX_CATALOG_BYTES, "catalog")
    if (
        set(document)
        != {
            "catalogVersion",
            "id",
            "version",
            "license",
            "upstream",
            "sources",
            "providers",
            "evaluation",
        }
        or document["catalogVersion"] != 1
    ):
        raise QwenProviderError("catalog-invalid", "Qwen catalog fields or version are invalid")
    license_document = _mapping(document["license"], "catalog license", "catalog-invalid")
    upstream = _mapping(document["upstream"], "catalog upstream", "catalog-invalid")
    if set(license_document) != {"spdx", "termsUri", "attribution"} or set(upstream) != {
        "uri",
        "revision",
    }:
        raise QwenProviderError("catalog-invalid", "license or upstream fields are invalid")
    if license_document["spdx"] != "Apache-2.0" or not all(
        isinstance(license_document[name], str) and license_document[name]
        for name in ("termsUri", "attribution")
    ):
        raise QwenProviderError("catalog-invalid", "catalog license is not Apache-2.0")
    sources = tuple(
        _source(item)
        for item in _sequence(document["sources"], "catalog sources", "catalog-invalid")
    )
    if len(sources) != 2 or {source.role for source in sources} != {"model", "projector"}:
        raise QwenProviderError("catalog-invalid", "catalog needs one model and one projector")
    providers = tuple(
        _provider_template(item)
        for item in _sequence(document["providers"], "providers", "catalog-invalid")
    )
    if len(providers) != 2 or {(item[1], item[2]) for item in providers} != {
        ("gpu", "llama.cpp-vulkan"),
        ("npu", "openvino-genai-npu"),
    }:
        raise QwenProviderError("catalog-invalid", "catalog provider lanes are incomplete")
    defaults = [item for item in providers if item[3]]
    if len(defaults) != 1 or defaults[0][1] != "gpu":
        raise QwenProviderError("catalog-invalid", "GPU must be the sole default provider")
    evaluation = _evaluation(document["evaluation"])
    model_id = _text(document["id"], "catalog model id", "catalog-invalid", 120)
    version = _text(document["version"], "catalog version", "catalog-invalid", 80)
    revision = upstream["revision"]
    uri = upstream["uri"]
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or set(revision) - _DIGEST
        or not isinstance(uri, str)
        or not uri.startswith("https://")
        or revision not in uri
    ):
        raise QwenProviderError("catalog-invalid", "upstream identity is not revision-pinned")
    return QwenCatalog(
        model_id,
        version,
        revision,
        str(license_document["spdx"]),
        sources,
        providers,
        evaluation,
    )


def load_event_corpus(path: Path | str | None = None) -> FrozenEventCorpus:
    corpus_path = Path(path) if path is not None else _corpus_path()
    try:
        raw = corpus_path.read_bytes()
    except OSError as error:
        raise QwenProviderError("corpus-invalid", "cannot read evaluation corpus") from error
    if len(raw) > MAX_CORPUS_BYTES:
        raise QwenProviderError("corpus-invalid", "evaluation corpus exceeds its byte limit")
    try:
        document = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise QwenProviderError("corpus-invalid", "evaluation corpus is not JSON") from error
    if (
        not isinstance(document, dict)
        or set(document)
        != {
            "corpusVersion",
            "id",
            "license",
            "provenance",
            "cases",
        }
        or document["corpusVersion"] != 1
    ):
        raise QwenProviderError("corpus-invalid", "evaluation corpus fields are invalid")
    corpus_id = _text(document["id"], "corpus id", "corpus-invalid", 120)
    if document["license"] != "CC0-1.0":
        raise QwenProviderError("corpus-invalid", "evaluation corpus license is invalid")
    _text(document["provenance"], "corpus provenance", "corpus-invalid", 500)
    cases = tuple(
        _case(item) for item in _sequence(document["cases"], "corpus cases", "corpus-invalid")
    )
    if len(cases) < 2 or len({item.case_id for item in cases}) != len(cases):
        raise QwenProviderError("corpus-invalid", "evaluation case ids must be unique")
    if not any(item.prompt_injection_probe for item in cases):
        raise QwenProviderError("corpus-invalid", "evaluation corpus needs an injection probe")
    return FrozenEventCorpus(corpus_id, hashlib.sha256(raw).hexdigest(), cases)


def qualify_event_provider(
    descriptor: GenerationProviderDescriptor,
    corpus: FrozenEventCorpus,
    evidence: EventProviderEvidence,
    policy: EventQualificationPolicy | None = None,
) -> EventQualificationReport:
    policy = policy or EventQualificationPolicy()
    _validate_evidence_identity(descriptor, corpus, evidence)
    _validate_policy(policy)
    true_positive, false_positive, false_negative, latencies, peak_memory = _score_observations(
        corpus, evidence
    )
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    p95 = sorted(latencies)[math.ceil(len(latencies) * 0.95) - 1]
    cancellation = _positive_integer(
        evidence.cancellation_latency_ms, "cancellation latency", "evidence-invalid"
    )
    if precision < policy.minimum_precision or recall < policy.minimum_recall:
        raise QwenProviderError("quality-failed", "event precision or recall is below the gate")
    if p95 > policy.maximum_p95_latency_ms:
        raise QwenProviderError("quality-failed", "event generation latency exceeds the gate")
    if peak_memory > policy.maximum_peak_memory_bytes:
        raise QwenProviderError("quality-failed", "event generation memory exceeds the gate")
    if cancellation > policy.maximum_cancellation_latency_ms:
        raise QwenProviderError("quality-failed", "event cancellation latency exceeds the gate")
    _validate_lane_load(descriptor.accelerator, evidence.load)
    return EventQualificationReport(
        descriptor.provider_id,
        descriptor.accelerator,
        corpus.sha256,
        evidence.runtime_version,
        evidence.device_name,
        precision,
        recall,
        p95,
        peak_memory,
        cancellation,
        True,
    )


def _score_observations(
    corpus: FrozenEventCorpus,
    evidence: EventProviderEvidence,
) -> tuple[int, int, int, list[int], int]:
    expected_by_id = {item.case_id: item for item in corpus.cases}
    observed_by_id = {item.case_id: item for item in evidence.observations}
    if len(observed_by_id) != len(evidence.observations) or set(observed_by_id) != set(
        expected_by_id
    ):
        raise QwenProviderError("evidence-invalid", "evidence must cover every corpus case once")
    true_positive = false_positive = false_negative = 0
    latencies: list[int] = []
    peak_memory = 0
    for case_id, case in expected_by_id.items():
        observation = observed_by_id[case_id]
        positive, extra, missing, latency, memory = _score_observation(case, observation)
        true_positive += positive
        false_positive += extra
        false_negative += missing
        latencies.append(latency)
        peak_memory = max(peak_memory, memory)
    return true_positive, false_positive, false_negative, latencies, peak_memory


def _score_observation(
    case: FrozenEventCase,
    observation: EventProviderObservation,
) -> tuple[int, int, int, int, int]:
    try:
        result = parse_grounded_event_result(observation.result)
    except EventResultError as error:
        raise QwenProviderError(
            "quality-failed", "provider output violates the grounded-event contract"
        ) from error
    actual = {_event_key(item) for item in result.events}
    expected = set(case.expected)
    if case.prompt_injection_probe and actual != expected:
        raise QwenProviderError("quality-failed", "prompt-injection probe changed grounded output")
    latency = _positive_integer(observation.latency_ms, "latency", "evidence-invalid")
    memory = _positive_integer(observation.peak_memory_bytes, "peak memory", "evidence-invalid")
    return (
        len(actual & expected),
        len(actual - expected),
        len(expected - actual),
        latency,
        memory,
    )


def _validate_lane_load(accelerator: str, report: NativeLoadReport) -> None:
    if accelerator == "gpu":
        _validate_gpu_load(report)
    elif accelerator == "npu":
        _validate_npu_load(report)
    else:
        raise QwenProviderError("evidence-invalid", "generation lane must be GPU or NPU")


def _validate_evidence_identity(
    descriptor: GenerationProviderDescriptor,
    corpus: FrozenEventCorpus,
    evidence: EventProviderEvidence,
) -> None:
    if descriptor.qualified:
        raise QwenProviderError("evidence-invalid", "qualification input must start unqualified")
    if evidence.provider_id != descriptor.provider_id:
        raise QwenProviderError("evidence-invalid", "evidence names another provider")
    if evidence.corpus_sha256 != corpus.sha256:
        raise QwenProviderError("evidence-invalid", "evidence names another frozen corpus")
    for label, value in (
        ("runtime version", evidence.runtime_version),
        ("device name", evidence.device_name),
    ):
        if not isinstance(value, str) or not 1 <= len(value) <= 200:
            raise QwenProviderError("evidence-invalid", f"{label} is invalid")


def _validate_policy(policy: EventQualificationPolicy) -> None:
    if not isinstance(policy, EventQualificationPolicy):
        raise QwenProviderError("policy-invalid", "qualification policy is invalid")
    for name in ("minimum_precision", "minimum_recall"):
        value = getattr(policy, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise QwenProviderError("policy-invalid", f"{name} must be in [0, 1]")
    for name in (
        "maximum_p95_latency_ms",
        "maximum_peak_memory_bytes",
        "maximum_cancellation_latency_ms",
    ):
        _positive_integer(getattr(policy, name), name, "policy-invalid")


def _validate_gpu_load(report: NativeLoadReport) -> None:
    if (
        not isinstance(report, NativeLoadReport)
        or report.backend != "llama.cpp-vulkan"
        or report.device != "Vulkan"
        or report.cpu_fallback
        or report.total_model_layers < 1
        or report.accelerator_layers != report.total_model_layers
    ):
        raise ProviderGenerationError(
            "model-load-failed",
            "llama.cpp did not prove complete Vulkan model-layer offload",
            generation_started=False,
        )


def _validate_npu_load(report: NativeLoadReport) -> None:
    if (
        not isinstance(report, NativeLoadReport)
        or report.backend != "openvino-genai"
        or report.device != "NPU"
        or report.cpu_fallback
    ):
        raise ProviderGenerationError(
            "model-load-failed",
            "OpenVINO GenAI did not prove NPU-only execution",
            generation_started=False,
        )


def _event_key(event) -> tuple[str, str, str, str]:
    return (
        event.title,
        event.start.isoformat(),
        event.timezone,
        event.location or "",
    )


def _read_document(path: Path | str, limit: int, label: str) -> dict:
    try:
        document = read_json_bounded(Path(path), limit)
    except (OSError, ValueError, JsonTooLargeError) as error:
        raise QwenProviderError(f"{label}-invalid", f"cannot read {label}: {error}") from error
    if not isinstance(document, dict):
        raise QwenProviderError(f"{label}-invalid", f"{label} must be an object")
    return document


def _source(document: object) -> QwenSource:
    value = _mapping(document, "catalog source", "catalog-invalid")
    if set(value) != {"role", "uri", "revision", "filename", "sha256", "sizeBytes"}:
        raise QwenProviderError("catalog-invalid", "catalog source fields are invalid")
    digest = _text(value["sha256"], "source digest", "catalog-invalid", 64)
    revision = _text(value["revision"], "source revision", "catalog-invalid", 40)
    size = _positive_integer(value["sizeBytes"], "source size", "catalog-invalid")
    if len(digest) != 64 or set(digest) - _DIGEST or len(revision) != 40 or set(revision) - _DIGEST:
        raise QwenProviderError("catalog-invalid", "catalog source identity is invalid")
    uri = _text(value["uri"], "source URI", "catalog-invalid", 2048)
    if not uri.startswith("https://") or revision not in uri:
        raise QwenProviderError("catalog-invalid", "catalog source URI is not revision-pinned")
    role = value["role"]
    filename = value["filename"]
    if (
        role not in {"model", "projector"}
        or not isinstance(filename, str)
        or Path(filename).name != filename
    ):
        raise QwenProviderError("catalog-invalid", "catalog source role or filename is invalid")
    return QwenSource(role, uri, revision, filename, digest, size)


def _provider_template(document: object) -> tuple[str, str, str, bool, str]:
    value = _mapping(document, "provider template", "catalog-invalid")
    if set(value) != {"providerId", "accelerator", "runtime", "default", "status"}:
        raise QwenProviderError("catalog-invalid", "provider template fields are invalid")
    if not isinstance(value["default"], bool):
        raise QwenProviderError("catalog-invalid", "provider default must be boolean")
    return (
        _text(value["providerId"], "provider id", "catalog-invalid", 120),
        _text(value["accelerator"], "accelerator", "catalog-invalid", 20),
        _text(value["runtime"], "runtime", "catalog-invalid", 120),
        value["default"],
        _text(value["status"], "status", "catalog-invalid", 120),
    )


def _evaluation(document: object) -> QwenEvaluation:
    value = _mapping(document, "evaluation policy", "catalog-invalid")
    if set(value) != {
        "corpus",
        "minimumPrecision",
        "minimumRecall",
        "maximumP95LatencyMs",
        "maximumPeakMemoryBytes",
        "maximumCancellationLatencyMs",
    }:
        raise QwenProviderError("catalog-invalid", "evaluation policy fields are invalid")
    corpus = value["corpus"]
    if corpus != "event-extraction-v1.json":
        raise QwenProviderError("catalog-invalid", "evaluation corpus is not pinned")
    policy = EventQualificationPolicy(
        minimum_precision=value["minimumPrecision"],
        minimum_recall=value["minimumRecall"],
        maximum_p95_latency_ms=value["maximumP95LatencyMs"],
        maximum_peak_memory_bytes=value["maximumPeakMemoryBytes"],
        maximum_cancellation_latency_ms=value["maximumCancellationLatencyMs"],
    )
    _validate_policy(policy)
    return QwenEvaluation(corpus, policy)


def _case(document: object) -> FrozenEventCase:
    value = _mapping(document, "corpus case", "corpus-invalid")
    if set(value) != {"caseId", "modality", "content", "promptInjectionProbe", "expected"}:
        raise QwenProviderError("corpus-invalid", "corpus case fields are invalid")
    if value["modality"] not in {"text", "image"} or not isinstance(
        value["promptInjectionProbe"], bool
    ):
        raise QwenProviderError("corpus-invalid", "corpus case modality or probe is invalid")
    content = value["content"]
    if not isinstance(content, str) or not 1 <= len(content) <= 16_384:
        raise QwenProviderError("corpus-invalid", "corpus case content is invalid")
    expected = tuple(
        _expected_event(item)
        for item in _sequence(value["expected"], "expected events", "corpus-invalid")
    )
    case_id = value["caseId"]
    if not isinstance(case_id, str) or not 1 <= len(case_id) <= 120:
        raise QwenProviderError("corpus-invalid", "corpus case id is invalid")
    return FrozenEventCase(
        case_id,
        str(value["modality"]),
        content,
        value["promptInjectionProbe"],
        expected,
    )


def _expected_event(document: object) -> tuple[str, str, str, str]:
    value = _mapping(document, "expected event", "corpus-invalid")
    if set(value) != {"title", "start", "timezone", "location"}:
        raise QwenProviderError("corpus-invalid", "expected event fields are invalid")
    return tuple(
        _text(value[name], name, "corpus-invalid", 200)
        for name in ("title", "start", "timezone", "location")
    )


def _mapping(value: object, label: str, code: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise QwenProviderError(code, f"{label} must be an object")
    return value


def _sequence(value: object, label: str, code: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise QwenProviderError(code, f"{label} must be a non-empty sequence")
    return value


def _positive_integer(value: object, label: str, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise QwenProviderError(code, f"{label} must be a positive integer")
    return value


def _text(value: object, label: str, code: str, limit: int) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= limit:
        raise QwenProviderError(code, f"{label} must be bounded text")
    return value


def _catalog_path() -> Path:
    packaged = _PACKAGED_MODELS / "qwen2-5-vl-7b.json"
    if packaged.is_file():
        return packaged
    assert _SOURCE_ROOT is not None
    return _SOURCE_ROOT / "generation-models/qwen2-5-vl-7b.json"


def _corpus_path() -> Path:
    packaged = _PACKAGED_CORPORA / "event-extraction-v1.json"
    if packaged.is_file():
        return packaged
    assert _SOURCE_ROOT is not None
    return _SOURCE_ROOT / "evaluation-corpora/event-extraction-v1.json"


__all__ = [
    "EventProviderEvidence",
    "EventProviderObservation",
    "EventQualificationPolicy",
    "EventQualificationReport",
    "FrozenEventCorpus",
    "LlamaCppVulkanQwenWorker",
    "NativeLoadReport",
    "NativeQwenRuntime",
    "OpenVinoNpuQwenWorker",
    "QwenCatalog",
    "QwenEvaluation",
    "QwenProviderError",
    "load_event_corpus",
    "load_qwen_catalog",
    "qualify_event_provider",
]
