"""Frozen-corpus scoring and qualification for Qwen event providers."""

from __future__ import annotations

import math
from pathlib import Path

from ..atomicio import JsonTooLargeError
from .acceptance_kit import (
    NativeLoadReport,
    discover_resource,
    read_bounded_json,
    validate_gpu_load,
    validate_npu_load,
)
from .events import EventResultError, parse_grounded_event_result
from .generation import GenerationProviderDescriptor
from .qwen_contracts import (
    EventProviderEvidence,
    EventProviderObservation,
    EventQualificationPolicy,
    EventQualificationReport,
    FrozenEventCase,
    FrozenEventCorpus,
    QwenProviderError,
    legacy_qwen_callback,
    qwen_mapping,
    qwen_positive_integer,
    qwen_sequence,
    qwen_text,
    validate_event_policy,
)

MAX_CORPUS_BYTES = 256 * 1024
_PACKAGE_DIR = Path(__file__).resolve().parent
_PACKAGED_CORPORA = _PACKAGE_DIR.parent / "evaluation-corpora"
_SOURCE_ROOT = _PACKAGE_DIR.parents[2] if _PACKAGE_DIR.parent.parent.name == "src" else None


def load_event_corpus(path: Path | str | None = None) -> FrozenEventCorpus:
    corpus_path = Path(path) if path is not None else _corpus_path()
    try:
        snapshot = read_bounded_json(corpus_path, MAX_CORPUS_BYTES)
    except JsonTooLargeError as error:
        raise QwenProviderError(
            "corpus-invalid", "evaluation corpus exceeds its byte limit"
        ) from error
    except OSError as error:
        raise QwenProviderError("corpus-invalid", "cannot read evaluation corpus") from error
    except (UnicodeError, ValueError) as error:
        raise QwenProviderError("corpus-invalid", "evaluation corpus is not JSON") from error
    document = snapshot.document
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
    corpus_id = qwen_text(document["id"], "corpus id", "corpus-invalid", 120)
    if document["license"] != "CC0-1.0":
        raise QwenProviderError("corpus-invalid", "evaluation corpus license is invalid")
    qwen_text(document["provenance"], "corpus provenance", "corpus-invalid", 500)
    cases = tuple(
        _case(item) for item in qwen_sequence(document["cases"], "corpus cases", "corpus-invalid")
    )
    if len(cases) < 2 or len({item.case_id for item in cases}) != len(cases):
        raise QwenProviderError("corpus-invalid", "evaluation case ids must be unique")
    if not any(item.prompt_injection_probe for item in cases):
        raise QwenProviderError("corpus-invalid", "evaluation corpus needs an injection probe")
    return FrozenEventCorpus(corpus_id, snapshot.sha256, cases)


def qualify_event_provider(
    descriptor: GenerationProviderDescriptor,
    corpus: FrozenEventCorpus,
    evidence: EventProviderEvidence,
    policy: EventQualificationPolicy | None = None,
) -> EventQualificationReport:
    policy = policy or EventQualificationPolicy()
    _validate_evidence_identity(descriptor, corpus, evidence)
    legacy_qwen_callback("_validate_policy", validate_event_policy)(policy)
    true_positive, false_positive, false_negative, latencies, peak_memory = _score_observations(
        corpus, evidence
    )
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    p95 = sorted(latencies)[math.ceil(len(latencies) * 0.95) - 1]
    cancellation = qwen_positive_integer(
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


def corpus_path(packaged_corpora: Path, source_root: Path | None) -> Path:
    """Resolve the package-first event corpus path for an injected install layout."""
    packaged = packaged_corpora / "event-extraction-v1.json"
    if source_root is None:
        if packaged.is_file():
            return packaged
        assert source_root is not None
    return discover_resource(packaged, source_root / "evaluation-corpora/event-extraction-v1.json")


def _corpus_path() -> Path:
    return corpus_path(_PACKAGED_CORPORA, _SOURCE_ROOT)


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
    latency = qwen_positive_integer(observation.latency_ms, "latency", "evidence-invalid")
    memory = qwen_positive_integer(observation.peak_memory_bytes, "peak memory", "evidence-invalid")
    return (
        len(actual & expected),
        len(actual - expected),
        len(expected - actual),
        latency,
        memory,
    )


def _validate_lane_load(accelerator: str, report: NativeLoadReport) -> None:
    if accelerator == "gpu":
        legacy_qwen_callback("_validate_gpu_load", validate_gpu_load)(report)
    elif accelerator == "npu":
        legacy_qwen_callback("_validate_npu_load", validate_npu_load)(report)
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


def _event_key(event) -> tuple[str, str, str, str]:
    """What the corpus compares: the label, when it starts, and where.

    Date and time are separate now and either may be absent, so an expectation
    states what the source states — a corpus case for a banner with no year
    compares an empty date rather than being impossible to write.
    """
    place = ""
    if event.where is not None:
        stated = event.where.address.full if event.where.address else event.where.venue
        place = stated or event.where.url or ""
    return (
        event.label.text,
        event.when.date or "",
        event.when.time or "",
        place,
    )


def _case(document: object) -> FrozenEventCase:
    value = qwen_mapping(document, "corpus case", "corpus-invalid")
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
        for item in qwen_sequence(value["expected"], "expected events", "corpus-invalid")
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
    value = qwen_mapping(document, "expected event", "corpus-invalid")
    if set(value) != {"label", "date", "time", "place"}:
        raise QwenProviderError("corpus-invalid", "expected event fields are invalid")
    # A corpus case may state an empty date, time or place: that is what a
    # banner without a year actually says, and version 2 records it rather than
    # refusing the event.
    if not isinstance(value["label"], str) or not value["label"]:
        raise QwenProviderError("corpus-invalid", "expected event needs a label")
    for name in ("date", "time", "place"):
        if value[name] is not None and not isinstance(value[name], str):
            raise QwenProviderError("corpus-invalid", f"expected event {name} is invalid")
    return tuple(
        qwen_text(value[name], name, "corpus-invalid", 200) if value[name] else ""
        for name in ("label", "date", "time", "place")
    )


_validate_policy = validate_event_policy


__all__ = [
    "MAX_CORPUS_BYTES",
    "corpus_path",
    "load_event_corpus",
    "qualify_event_provider",
]
