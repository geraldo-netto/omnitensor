"""Digest-bound operational acceptance for selected-file document questions.

The workload's ordinary tests prove its protocol and refusal behaviour.  This
module separately qualifies one concrete BGE/Qwen provider pair against a
frozen public corpus and explicit host probes.  It consumes evidence; it never
reads an operator's documents or silently turns a mock run into hardware
evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..atomicio import JsonTooLargeError, read_json_bounded, write_json_atomic
from ..preparation import file_digest
from ..registry import validate_document
from .qwen import NativeLoadReport, ProviderGenerationError, _validate_gpu_load

MAX_CORPUS_BYTES = 256 * 1024
MAX_EVIDENCE_BYTES = 2 * 1024 * 1024
QWEN_MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
QWEN_MODEL_ID = "qwen3-8b-q4-k-m"
_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class DocumentAcceptanceError(ValueError):
    """Stable acceptance refusal safe to show to an operator."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class AcceptanceSource:
    file_name: str
    content: str


@dataclass(frozen=True, slots=True)
class DocumentAcceptanceCase:
    case_id: str
    question: str
    sources: tuple[AcceptanceSource, ...]
    expected_terms: tuple[str, ...]
    expected_file_ids: tuple[str, ...]
    prompt_injection_probe: bool
    forbidden_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DocumentAcceptanceCorpus:
    corpus_id: str
    sha256: str
    cases: tuple[DocumentAcceptanceCase, ...]


@dataclass(frozen=True, slots=True)
class DocumentAcceptanceObservation:
    case_id: str
    result: Mapping[str, object]
    retrieved_file_ids: tuple[str, ...]
    latency_ms: int
    peak_memory_bytes: int


@dataclass(frozen=True, slots=True)
class DocumentSafetyEvidence:
    malicious_document_refused: bool
    mutated_document_refused: bool
    grant_revocation_refused: bool
    stale_version_suppressed: bool
    index_recovered: bool
    private_fragments_discarded: bool
    cancellation_latency_ms: int
    revocation_latency_ms: int


@dataclass(frozen=True, slots=True)
class DocumentAcceptanceEvidence:
    evidence_sha256: str
    corpus_sha256: str
    bge_report_path: Path
    bge_report_sha256: str
    bge_artifact_sha256: str
    qwen_model_sha256: str
    runtime_version: str
    generator_device_name: str
    generator_load: NativeLoadReport
    observations: tuple[DocumentAcceptanceObservation, ...]
    safety: DocumentSafetyEvidence


@dataclass(frozen=True, slots=True)
class DocumentAcceptancePolicy:
    minimum_retrieval_recall: float = 0.9
    minimum_grounded_term_recall: float = 0.9
    minimum_citation_integrity: float = 1.0
    maximum_p95_latency_ms: int = 30_000
    maximum_cancellation_latency_ms: int = 2_000
    maximum_revocation_latency_ms: int = 500


@dataclass(frozen=True, slots=True)
class DocumentAcceptanceReport:
    evidence_sha256: str
    corpus_sha256: str
    bge_report_sha256: str
    bge_artifact_sha256: str
    qwen_model_sha256: str
    embedding_device_name: str
    generator_device_name: str
    runtime_version: str
    retrieval_recall: float
    grounded_term_recall: float
    citation_integrity: float
    p95_latency_ms: int
    peak_memory_bytes: int
    cancellation_latency_ms: int
    revocation_latency_ms: int

    def document(self) -> dict:
        return {
            "reportVersion": 1,
            "kind": "document-question-operational-acceptance",
            "evidenceSha256": self.evidence_sha256,
            "corpusSha256": self.corpus_sha256,
            "models": {
                "embedding": {
                    "reportSha256": self.bge_report_sha256,
                    "artifactSha256": self.bge_artifact_sha256,
                    "deviceName": self.embedding_device_name,
                },
                "generation": {
                    "modelId": QWEN_MODEL_ID,
                    "artifactSha256": self.qwen_model_sha256,
                    "runtimeVersion": self.runtime_version,
                    "deviceName": self.generator_device_name,
                },
            },
            "metrics": {
                "retrievalRecall": self.retrieval_recall,
                "groundedTermRecall": self.grounded_term_recall,
                "citationIntegrity": self.citation_integrity,
                "p95LatencyMs": self.p95_latency_ms,
                "peakMemoryBytes": self.peak_memory_bytes,
                "cancellationLatencyMs": self.cancellation_latency_ms,
                "revocationLatencyMs": self.revocation_latency_ms,
            },
            "safety": {
                "maliciousDocumentRefused": True,
                "mutatedDocumentRefused": True,
                "grantRevocationRefused": True,
                "staleVersionSuppressed": True,
                "indexRecovered": True,
                "privateFragmentsDiscarded": True,
            },
            "qualified": True,
            "scope": {
                "accelerator": "gpu",
                "cpuFallback": "forbidden",
                "npu": "not-qualified",
                "tpu": "not-qualified",
                "arbitraryPrivateCorpora": "not-claimed",
            },
        }


def load_document_acceptance_corpus(
    path: Path | str | None = None,
) -> DocumentAcceptanceCorpus:
    corpus_path = Path(path) if path is not None else _corpus_path()
    raw = _bounded_bytes(corpus_path, MAX_CORPUS_BYTES, "corpus")
    document = _json_object(raw, "corpus")
    if (
        set(document) != {"corpusVersion", "id", "license", "provenance", "cases"}
        or document["corpusVersion"] != 1
    ):
        raise DocumentAcceptanceError("corpus-invalid", "corpus fields or version are invalid")
    if document["license"] != "CC0-1.0":
        raise DocumentAcceptanceError("corpus-invalid", "corpus license must be CC0-1.0")
    _bounded_text(document["provenance"], "corpus provenance", 500, "corpus-invalid")
    corpus_id = _identifier(document["id"], "corpus id", "corpus-invalid")
    values = _sequence(document["cases"], "corpus cases", "corpus-invalid")
    cases = tuple(_parse_case(value) for value in values)
    if len(cases) < 5 or len({case.case_id for case in cases}) != len(cases):
        raise DocumentAcceptanceError(
            "corpus-invalid", "corpus needs at least five uniquely named cases"
        )
    if not any(case.prompt_injection_probe for case in cases):
        raise DocumentAcceptanceError("corpus-invalid", "corpus needs a prompt-injection probe")
    return DocumentAcceptanceCorpus(corpus_id, hashlib.sha256(raw).hexdigest(), cases)


def load_document_acceptance_evidence(path: Path | str) -> DocumentAcceptanceEvidence:
    evidence_path = Path(path)
    try:
        document = read_json_bounded(evidence_path, MAX_EVIDENCE_BYTES)
    except (OSError, ValueError, JsonTooLargeError) as error:
        raise DocumentAcceptanceError(
            "evidence-invalid", f"cannot read evidence: {error}"
        ) from error
    if (
        not isinstance(document, Mapping)
        or set(document)
        != {
            "evidenceVersion",
            "corpusSha256",
            "embedding",
            "generation",
            "observations",
            "safety",
        }
        or document["evidenceVersion"] != 1
    ):
        raise DocumentAcceptanceError("evidence-invalid", "evidence fields or version are invalid")
    embedding = _mapping(document["embedding"], "embedding evidence", "evidence-invalid")
    generation = _mapping(document["generation"], "generation evidence", "evidence-invalid")
    if set(embedding) != {"reportPath", "reportSha256", "artifactSha256"} or set(generation) != {
        "modelSha256",
        "runtimeVersion",
        "deviceName",
        "load",
    }:
        raise DocumentAcceptanceError("evidence-invalid", "model evidence fields are invalid")
    load = _mapping(generation["load"], "generation load", "evidence-invalid")
    if set(load) != {
        "backend",
        "device",
        "totalModelLayers",
        "acceleratorLayers",
        "cpuFallback",
    }:
        raise DocumentAcceptanceError("evidence-invalid", "generation load fields are invalid")
    observations = tuple(
        _parse_observation(item)
        for item in _sequence(document["observations"], "observations", "evidence-invalid")
    )
    report_path = Path(
        _bounded_text(embedding["reportPath"], "BGE report path", 4096, "evidence-invalid")
    )
    if not report_path.is_absolute():
        report_path = evidence_path.parent / report_path
    return DocumentAcceptanceEvidence(
        file_digest(evidence_path),
        _digest(document["corpusSha256"], "corpus digest"),
        report_path,
        _digest(embedding["reportSha256"], "BGE report digest"),
        _digest(embedding["artifactSha256"], "BGE artifact digest"),
        _digest(generation["modelSha256"], "Qwen model digest"),
        _bounded_text(generation["runtimeVersion"], "runtime version", 200, "evidence-invalid"),
        _bounded_text(generation["deviceName"], "generator device", 200, "evidence-invalid"),
        NativeLoadReport(
            _bounded_text(load["backend"], "load backend", 120, "evidence-invalid"),
            _bounded_text(load["device"], "load device", 120, "evidence-invalid"),
            _integer(load["totalModelLayers"], "total model layers", 1, 65_535),
            _integer(load["acceleratorLayers"], "accelerator layers", 1, 65_535),
            _boolean(load["cpuFallback"], "CPU fallback"),
        ),
        observations,
        _parse_safety(document["safety"]),
    )


def qualify_document_questions(
    corpus: DocumentAcceptanceCorpus,
    evidence: DocumentAcceptanceEvidence,
    policy: DocumentAcceptancePolicy | None = None,
) -> DocumentAcceptanceReport:
    """Validate immutable model, quality, safety, latency, privacy, and GPU evidence."""
    policy = policy or DocumentAcceptancePolicy()
    _validate_policy(policy)
    embedding_device = _validate_acceptance_identity(corpus, evidence)
    metrics = _acceptance_metrics(corpus, evidence.observations)
    _validate_safety(evidence.safety, policy)
    _enforce_metric_policy(metrics, policy)
    retrieval_recall, term_recall, citation_integrity, p95, peak_memory = metrics
    return DocumentAcceptanceReport(
        evidence.evidence_sha256,
        corpus.sha256,
        evidence.bge_report_sha256,
        evidence.bge_artifact_sha256,
        evidence.qwen_model_sha256,
        embedding_device,
        evidence.generator_device_name,
        evidence.runtime_version,
        retrieval_recall,
        term_recall,
        citation_integrity,
        p95,
        peak_memory,
        evidence.safety.cancellation_latency_ms,
        evidence.safety.revocation_latency_ms,
    )


def _validate_acceptance_identity(
    corpus: DocumentAcceptanceCorpus, evidence: DocumentAcceptanceEvidence
) -> str:
    if evidence.corpus_sha256 != corpus.sha256:
        raise DocumentAcceptanceError("evidence-stale", "evidence names another corpus version")
    if evidence.qwen_model_sha256 != QWEN_MODEL_SHA256:
        raise DocumentAcceptanceError("evidence-stale", "evidence names another Qwen model")
    embedding_device = _validate_bge_evidence(evidence)
    try:
        _validate_gpu_load(evidence.generator_load)
    except ProviderGenerationError as error:
        raise DocumentAcceptanceError("device-unqualified", error.detail) from error
    if evidence.generator_device_name.strip().lower() in {"", "vulkan", "cpu", "llvmpipe"}:
        raise DocumentAcceptanceError("device-unqualified", "generator GPU must be named")
    return embedding_device


def _enforce_metric_policy(
    metrics: tuple[float, float, float, int, int], policy: DocumentAcceptancePolicy
) -> None:
    retrieval_recall, term_recall, citation_integrity, p95, peak_memory = metrics
    if retrieval_recall < policy.minimum_retrieval_recall:
        raise DocumentAcceptanceError("quality-failed", "retrieval recall is below the gate")
    if term_recall < policy.minimum_grounded_term_recall:
        raise DocumentAcceptanceError("quality-failed", "grounded answer recall is below the gate")
    if citation_integrity < policy.minimum_citation_integrity:
        raise DocumentAcceptanceError("quality-failed", "citation integrity is below the gate")
    if p95 > policy.maximum_p95_latency_ms:
        raise DocumentAcceptanceError("quality-failed", "question latency exceeds the gate")


def _acceptance_metrics(
    corpus: DocumentAcceptanceCorpus,
    observations: Sequence[DocumentAcceptanceObservation],
) -> tuple[float, float, float, int, int]:
    expected = {case.case_id: case for case in corpus.cases}
    actual = {item.case_id: item for item in observations}
    if len(actual) != len(observations) or set(actual) != set(expected):
        raise DocumentAcceptanceError("evidence-invalid", "every corpus case must appear once")
    totals = [0, 0, 0, 0, 0]
    latencies: list[int] = []
    peak_memory = 0
    for case_id, case in expected.items():
        retrieval, grounded, citations, latency, memory = _score_observation(case, actual[case_id])
        totals[0] += retrieval
        totals[1] += len(case.expected_file_ids)
        totals[2] += grounded
        totals[3] += len(case.expected_terms)
        totals[4] += citations
        latencies.append(latency)
        peak_memory = max(peak_memory, memory)
    citation_total = sum(len(case.expected_file_ids) for case in corpus.cases)
    return (
        totals[0] / totals[1],
        totals[2] / totals[3],
        totals[4] / citation_total,
        sorted(latencies)[math.ceil(len(latencies) * 0.95) - 1],
        peak_memory,
    )


def _score_observation(
    case: DocumentAcceptanceCase, observation: DocumentAcceptanceObservation
) -> tuple[int, int, int, int, int]:
    if observation.case_id != case.case_id:
        raise DocumentAcceptanceError("evidence-invalid", "observation identity disagrees")
    violations = validate_document("document-question-result.schema.json", observation.result)
    if violations:
        raise DocumentAcceptanceError("quality-failed", "answer violates its public contract")
    answer = str(observation.result["answer"]).casefold()
    if (
        observation.result["providerId"] != "qwen3-workloads-gpu"
        or observation.result["accelerator"] != "gpu"
    ):
        raise DocumentAcceptanceError("evidence-invalid", "answer identity disagrees")
    if "private:" in answer or "/home/" in answer or "\\home\\" in answer:
        raise DocumentAcceptanceError("privacy-failed", "answer leaked a private reference or path")
    if any(term.casefold() in answer for term in case.forbidden_terms):
        raise DocumentAcceptanceError("quality-failed", "a forbidden term appeared in the answer")
    expected_ids = set(case.expected_file_ids)
    retrieved = set(observation.retrieved_file_ids)
    available = {f"selected-file-{index}" for index in range(1, len(case.sources) + 1)}
    if len(retrieved) != len(observation.retrieved_file_ids) or not retrieved <= available:
        raise DocumentAcceptanceError("evidence-invalid", "retrieval identities are invalid")
    citations = observation.result["citations"]
    assert isinstance(citations, Sequence)
    citation_ids = {str(item["fileId"]) for item in citations}
    if citation_ids - expected_ids:
        raise DocumentAcceptanceError("quality-failed", "answer cited an unsupported file")
    _validate_citations(case, citations)
    return (
        len(expected_ids & retrieved),
        sum(term.casefold() in answer for term in case.expected_terms),
        len(expected_ids & citation_ids),
        _integer(observation.latency_ms, "observation latency", 1, 600_000),
        _integer(observation.peak_memory_bytes, "observation memory", 1, 64 * 1024**3),
    )


def _validate_citations(case: DocumentAcceptanceCase, citations: Sequence[object]) -> None:
    sources = {f"selected-file-{index}": source for index, source in enumerate(case.sources, 1)}
    seen: set[str] = set()
    for value in citations:
        citation = _mapping(value, "citation", "quality-failed")
        file_id = str(citation["fileId"])
        source = sources.get(file_id)
        if source is None or file_id in seen:
            raise DocumentAcceptanceError("quality-failed", "citation source is absent or repeated")
        seen.add(file_id)
        raw = source.content.encode("utf-8")
        if (
            citation["fileName"] != source.file_name
            or citation["sourceSha256"] != hashlib.sha256(raw).hexdigest()
            or citation["page"] != 1
            or citation["span"] != {"start": 0, "end": len(source.content)}
            or citation["textSha256"] != hashlib.sha256(raw).hexdigest()
        ):
            raise DocumentAcceptanceError("quality-failed", "citation address or digest drifted")


def _validate_bge_evidence(evidence: DocumentAcceptanceEvidence) -> str:
    if (
        not evidence.bge_report_path.is_file()
        or file_digest(evidence.bge_report_path) != evidence.bge_report_sha256
    ):
        raise DocumentAcceptanceError("evidence-stale", "BGE report is absent or changed")
    try:
        report = read_json_bounded(evidence.bge_report_path, MAX_CORPUS_BYTES)
    except (OSError, ValueError, JsonTooLargeError) as error:
        raise DocumentAcceptanceError("evidence-invalid", "BGE report cannot be read") from error
    if not isinstance(report, Mapping):
        raise DocumentAcceptanceError("evidence-invalid", "BGE report must be an object")
    gate = report.get("nativeGate")
    device = report.get("device")
    if (
        report.get("kind") != "bge-small-document-model"
        or report.get("nativeArtifactSha256") != evidence.bge_artifact_sha256
        or not isinstance(gate, Mapping)
        or gate.get("accepted") is not True
        or not isinstance(device, Mapping)
        or not isinstance(device.get("name"), str)
        or device["name"].strip().lower() in {"", "cpu", "llvmpipe"}
    ):
        raise DocumentAcceptanceError("device-unqualified", "BGE named-GPU evidence is invalid")
    return str(device["name"])


def _validate_safety(safety: DocumentSafetyEvidence, policy: DocumentAcceptancePolicy) -> None:
    flags = (
        safety.malicious_document_refused,
        safety.mutated_document_refused,
        safety.grant_revocation_refused,
        safety.stale_version_suppressed,
        safety.index_recovered,
        safety.private_fragments_discarded,
    )
    if not all(flag is True for flag in flags):
        raise DocumentAcceptanceError("safety-failed", "every document safety probe must pass")
    if safety.cancellation_latency_ms > policy.maximum_cancellation_latency_ms:
        raise DocumentAcceptanceError("safety-failed", "cancellation latency exceeds the gate")
    if safety.revocation_latency_ms > policy.maximum_revocation_latency_ms:
        raise DocumentAcceptanceError("safety-failed", "grant revocation latency exceeds the gate")


def _validate_policy(policy: DocumentAcceptancePolicy) -> None:
    if not isinstance(policy, DocumentAcceptancePolicy):
        raise DocumentAcceptanceError("policy-invalid", "acceptance policy is invalid")
    for name in (
        "minimum_retrieval_recall",
        "minimum_grounded_term_recall",
        "minimum_citation_integrity",
    ):
        value = getattr(policy, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise DocumentAcceptanceError("policy-invalid", f"{name} must be in [0, 1]")
    for name in (
        "maximum_p95_latency_ms",
        "maximum_cancellation_latency_ms",
        "maximum_revocation_latency_ms",
    ):
        _integer(getattr(policy, name), name, 1, 64 * 1024**3, "policy-invalid")


def _parse_case(value: object) -> DocumentAcceptanceCase:
    item = _mapping(value, "corpus case", "corpus-invalid")
    if set(item) != {
        "caseId",
        "question",
        "sources",
        "expectedTerms",
        "expectedFileIds",
        "promptInjectionProbe",
        "forbiddenTerms",
    }:
        raise DocumentAcceptanceError("corpus-invalid", "corpus case fields are invalid")
    sources = tuple(
        _parse_source(source) for source in _sequence(item["sources"], "sources", "corpus-invalid")
    )
    if len(sources) > 16 or len({source.file_name for source in sources}) != len(sources):
        raise DocumentAcceptanceError("corpus-invalid", "case sources must be unique and bounded")
    expected_terms = tuple(
        _bounded_text(term, "expected term", 120, "corpus-invalid")
        for term in _sequence(item["expectedTerms"], "expected terms", "corpus-invalid")
    )
    expected_ids = tuple(
        _bounded_text(identity, "expected file id", 40, "corpus-invalid")
        for identity in _sequence(item["expectedFileIds"], "expected file ids", "corpus-invalid")
    )
    available = {f"selected-file-{index}" for index in range(1, len(sources) + 1)}
    if len(set(expected_ids)) != len(expected_ids) or not set(expected_ids) <= available:
        raise DocumentAcceptanceError("corpus-invalid", "expected file ids are invalid")
    probe = _boolean(item["promptInjectionProbe"], "prompt-injection probe", "corpus-invalid")
    forbidden = tuple(
        _bounded_text(term, "forbidden term", 120, "corpus-invalid")
        for term in _optional_sequence(item["forbiddenTerms"], "forbidden terms", "corpus-invalid")
    )
    if probe and not forbidden:
        raise DocumentAcceptanceError("corpus-invalid", "injection probes need forbidden terms")
    return DocumentAcceptanceCase(
        _identifier(item["caseId"], "case id", "corpus-invalid"),
        _bounded_text(item["question"], "question", 4096, "corpus-invalid"),
        sources,
        expected_terms,
        expected_ids,
        probe,
        forbidden,
    )


def _parse_source(value: object) -> AcceptanceSource:
    item = _mapping(value, "case source", "corpus-invalid")
    if set(item) != {"fileName", "content"}:
        raise DocumentAcceptanceError("corpus-invalid", "case source fields are invalid")
    name = _bounded_text(item["fileName"], "source file name", 255, "corpus-invalid")
    content = _bounded_text(item["content"], "source content", 2048, "corpus-invalid")
    if Path(name).name != name or "/" in name or "\\" in name:
        raise DocumentAcceptanceError("corpus-invalid", "source file name must be a basename")
    return AcceptanceSource(name, content)


def _parse_observation(value: object) -> DocumentAcceptanceObservation:
    item = _mapping(value, "observation", "evidence-invalid")
    if set(item) != {"caseId", "result", "retrievedFileIds", "latencyMs", "peakMemoryBytes"}:
        raise DocumentAcceptanceError("evidence-invalid", "observation fields are invalid")
    result = _mapping(item["result"], "observation result", "evidence-invalid")
    retrieved = tuple(
        _bounded_text(identity, "retrieved file id", 40, "evidence-invalid")
        for identity in _sequence(
            item["retrievedFileIds"], "retrieved file ids", "evidence-invalid"
        )
    )
    return DocumentAcceptanceObservation(
        _identifier(item["caseId"], "case id", "evidence-invalid"),
        result,
        retrieved,
        _integer(item["latencyMs"], "observation latency", 1, 600_000),
        _integer(item["peakMemoryBytes"], "observation memory", 1, 64 * 1024**3),
    )


def _parse_safety(value: object) -> DocumentSafetyEvidence:
    item = _mapping(value, "safety evidence", "evidence-invalid")
    fields = {
        "maliciousDocumentRefused",
        "mutatedDocumentRefused",
        "grantRevocationRefused",
        "staleVersionSuppressed",
        "indexRecovered",
        "privateFragmentsDiscarded",
        "cancellationLatencyMs",
        "revocationLatencyMs",
    }
    if set(item) != fields:
        raise DocumentAcceptanceError("evidence-invalid", "safety evidence fields are invalid")
    return DocumentSafetyEvidence(
        *(
            _boolean(item[name], name, "evidence-invalid")
            for name in (
                "maliciousDocumentRefused",
                "mutatedDocumentRefused",
                "grantRevocationRefused",
                "staleVersionSuppressed",
                "indexRecovered",
                "privateFragmentsDiscarded",
            )
        ),
        _integer(item["cancellationLatencyMs"], "cancellation latency", 1, 600_000),
        _integer(item["revocationLatencyMs"], "revocation latency", 1, 600_000),
    )


def _bounded_bytes(path: Path, limit: int, label: str) -> bytes:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise DocumentAcceptanceError(f"{label}-invalid", f"cannot read {label}") from error
    if len(raw) > limit:
        raise DocumentAcceptanceError(f"{label}-invalid", f"{label} exceeds its byte limit")
    return raw


def _json_object(raw: bytes, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise DocumentAcceptanceError(f"{label}-invalid", f"{label} is not JSON") from error
    return _mapping(value, label, f"{label}-invalid")


def _mapping(value: object, label: str, code: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DocumentAcceptanceError(code, f"{label} must be an object")
    return value


def _sequence(value: object, label: str, code: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise DocumentAcceptanceError(code, f"{label} must be a non-empty sequence")
    return value


def _optional_sequence(value: object, label: str, code: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DocumentAcceptanceError(code, f"{label} must be a sequence")
    return value


def _bounded_text(value: object, label: str, limit: int, code: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= limit:
        raise DocumentAcceptanceError(code, f"{label} must be bounded text")
    return value


def _identifier(value: object, label: str, code: str) -> str:
    text = _bounded_text(value, label, 120, code)
    if _IDENTIFIER.fullmatch(text) is None:
        raise DocumentAcceptanceError(code, f"{label} must be a kebab-case identifier")
    return text


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise DocumentAcceptanceError("evidence-invalid", f"{label} must be lowercase SHA-256")
    return value


def _integer(
    value: object,
    label: str,
    minimum: int,
    maximum: int,
    code: str = "evidence-invalid",
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise DocumentAcceptanceError(code, f"{label} is outside its bound")
    return value


def _boolean(value: object, label: str, code: str = "evidence-invalid") -> bool:
    if not isinstance(value, bool):
        raise DocumentAcceptanceError(code, f"{label} must be boolean")
    return value


def _corpus_path() -> Path:
    package = (
        Path(__file__).resolve().parent.parent / "evaluation-corpora" / "document-question-v1.json"
    )
    if package.is_file():
        return package
    return Path(__file__).resolve().parents[3] / "evaluation-corpora" / "document-question-v1.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omnitensor-qualify-document-questions",
        description="Validate digest-bound Document Intelligence operational evidence",
    )
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--corpus", default=str(_corpus_path()))
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    try:
        corpus = load_document_acceptance_corpus(arguments.corpus)
        evidence = load_document_acceptance_evidence(arguments.evidence)
        report = qualify_document_questions(corpus, evidence).document()
        violations = validate_document("document-question-acceptance.schema.json", report)
        if violations:
            raise DocumentAcceptanceError("report-invalid", violations[0])
        write_json_atomic(Path(arguments.output), report, prefix=".document-acceptance-")
    except (DocumentAcceptanceError, OSError, ValueError) as error:
        print(f"document acceptance failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())


__all__ = [
    "AcceptanceSource",
    "DocumentAcceptanceCase",
    "DocumentAcceptanceCorpus",
    "DocumentAcceptanceError",
    "DocumentAcceptanceEvidence",
    "DocumentAcceptanceObservation",
    "DocumentAcceptancePolicy",
    "DocumentAcceptanceReport",
    "DocumentSafetyEvidence",
    "load_document_acceptance_corpus",
    "load_document_acceptance_evidence",
    "qualify_document_questions",
]
