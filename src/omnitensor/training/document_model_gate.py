"""Parity metrics and measured report publication for the document model."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from jsonschema.exceptions import SchemaError

from omnitensor.preparation import file_digest

from ..atomicio import write_json_atomic
from ..registry import validate_document
from .document_model_types import (
    DOCUMENT_MODEL_REPORT_SCHEMA,
    DocumentModelError,
    native_tensor_contract,
)
from .embedding_production import l2_normalize


def maximum_embedding_error(holdout, portable, native) -> float:
    maximum = 0.0
    for text in holdout.queries + holdout.documents:
        left = portable.embed(text)
        right = native.embed(text)
        maximum = max(
            maximum,
            max(abs(a - b) for a, b in zip(left, right, strict=True)),
        )
    return maximum


def expected_retrieval_hits(holdout, expected, runner) -> int:
    documents = tuple(l2_normalize(runner.embed(text)) for text in holdout.documents)
    hits = 0
    for query, expected_index in zip(holdout.queries, expected, strict=True):
        vector = l2_normalize(runner.embed(query))
        scores = [sum(a * b for a, b in zip(vector, item, strict=True)) for item in documents]
        if max(range(len(scores)), key=lambda index: (scores[index], -index)) == expected_index:
            hits += 1
    return hits


def report_document(
    source,
    portable_sha,
    native_sha,
    holdout,
    evidence,
    *,
    digest: Callable[[Path], str] = file_digest,
) -> dict:
    return {
        "reportVersion": 1,
        "kind": "bge-small-document-model",
        "recipeId": source.recipe.id,
        "recipeVersion": source.recipe.version,
        "recipeSha256": source.recipe.document_sha256,
        "sourceReceiptSha256": digest(source.receipt_path),
        "portableReferenceSha256": portable_sha,
        "nativeArtifactSha256": native_sha,
        "tensorContract": native_tensor_contract(source.recipe),
        "outputContract": {"kind": "embedding"},
        "holdout": {
            "licenseId": holdout.license_id,
            "corpusSha256": holdout.corpus_sha256,
            "queryCount": evidence.samples // 2 - len(holdout.documents),
            "documentCount": len(holdout.documents),
        },
        "device": {"index": evidence.device_index, "name": evidence.device_name},
        "nativeGate": {
            "samples": evidence.samples,
            "minimumCosineSimilarity": evidence.minimum_cosine_similarity,
            "minimumTop10Overlap": evidence.minimum_top10_overlap,
            "maximumAbsoluteError": evidence.maximum_absolute_error,
            "expectedTopHits": evidence.expected_top_hits,
            "accepted": True,
        },
        "cpuFallback": False,
        "limitations": {
            "productionDomainQuality": "not-claimed",
            "namedDeviceProfileAcceptance": False,
            "otherProfiles": "disabled",
        },
    }


def write_document_model_report(
    path: Path,
    document: dict,
    *,
    validator: Callable[[str, dict], list[str]] = validate_document,
    writer: Callable = write_json_atomic,
) -> None:
    try:
        violations = validator(DOCUMENT_MODEL_REPORT_SCHEMA, document)
    except (OSError, ValueError, SchemaError) as error:
        raise DocumentModelError(
            "report-invalid", f"cannot validate document model report: {error}"
        ) from error
    if violations:
        raise DocumentModelError(
            "report-invalid",
            f"document model report violates schema: {violations[0]}",
        )
    writer(path, document, prefix=".document-model-report-")
