"""Pinned Qwen source catalog parsing and discovery."""

from __future__ import annotations

from pathlib import Path

from ..atomicio import JsonTooLargeError
from .acceptance_kit import discover_resource, read_bounded_json
from .qwen_contracts import (
    EventQualificationPolicy,
    QwenCatalog,
    QwenEvaluation,
    QwenProviderError,
    QwenSource,
    legacy_qwen_callback,
    qwen_mapping,
    qwen_positive_integer,
    qwen_sequence,
    qwen_text,
    validate_event_policy,
)

MAX_CATALOG_BYTES = 64 * 1024
_PACKAGE_DIR = Path(__file__).resolve().parent
_PACKAGED_MODELS = _PACKAGE_DIR.parent / "generation-models"
_SOURCE_ROOT = _PACKAGE_DIR.parents[2] if _PACKAGE_DIR.parent.parent.name == "src" else None
_DIGEST = frozenset("0123456789abcdef")


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
    license_document = qwen_mapping(document["license"], "catalog license", "catalog-invalid")
    upstream = qwen_mapping(document["upstream"], "catalog upstream", "catalog-invalid")
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
        for item in qwen_sequence(document["sources"], "catalog sources", "catalog-invalid")
    )
    if len(sources) != 2 or {source.role for source in sources} != {"model", "projector"}:
        raise QwenProviderError("catalog-invalid", "catalog needs one model and one projector")
    providers = tuple(
        _provider_template(item)
        for item in qwen_sequence(document["providers"], "providers", "catalog-invalid")
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
    model_id = qwen_text(document["id"], "catalog model id", "catalog-invalid", 120)
    version = qwen_text(document["version"], "catalog version", "catalog-invalid", 80)
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


def catalog_path(packaged_models: Path, source_root: Path | None) -> Path:
    """Resolve the package-first catalog path for an injected install layout."""
    packaged = packaged_models / "qwen2-5-vl-7b.json"
    if source_root is None:
        if packaged.is_file():
            return packaged
        assert source_root is not None
    return discover_resource(packaged, source_root / "generation-models/qwen2-5-vl-7b.json")


def _catalog_path() -> Path:
    return catalog_path(_PACKAGED_MODELS, _SOURCE_ROOT)


def _read_document(path: Path | str, limit: int, label: str) -> dict:
    try:
        document = read_bounded_json(Path(path), limit).document
    except (OSError, ValueError, JsonTooLargeError) as error:
        raise QwenProviderError(f"{label}-invalid", f"cannot read {label}: {error}") from error
    if not isinstance(document, dict):
        raise QwenProviderError(f"{label}-invalid", f"{label} must be an object")
    return document


def _source(document: object) -> QwenSource:
    value = qwen_mapping(document, "catalog source", "catalog-invalid")
    if set(value) != {"role", "uri", "revision", "filename", "sha256", "sizeBytes"}:
        raise QwenProviderError("catalog-invalid", "catalog source fields are invalid")
    digest = qwen_text(value["sha256"], "source digest", "catalog-invalid", 64)
    revision = qwen_text(value["revision"], "source revision", "catalog-invalid", 40)
    size = qwen_positive_integer(value["sizeBytes"], "source size", "catalog-invalid")
    if len(digest) != 64 or set(digest) - _DIGEST or len(revision) != 40 or set(revision) - _DIGEST:
        raise QwenProviderError("catalog-invalid", "catalog source identity is invalid")
    uri = qwen_text(value["uri"], "source URI", "catalog-invalid", 2048)
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
    value = qwen_mapping(document, "provider template", "catalog-invalid")
    if set(value) != {"providerId", "accelerator", "runtime", "default", "status"}:
        raise QwenProviderError("catalog-invalid", "provider template fields are invalid")
    if not isinstance(value["default"], bool):
        raise QwenProviderError("catalog-invalid", "provider default must be boolean")
    return (
        qwen_text(value["providerId"], "provider id", "catalog-invalid", 120),
        qwen_text(value["accelerator"], "accelerator", "catalog-invalid", 20),
        qwen_text(value["runtime"], "runtime", "catalog-invalid", 120),
        value["default"],
        qwen_text(value["status"], "status", "catalog-invalid", 120),
    )


def _evaluation(document: object) -> QwenEvaluation:
    value = qwen_mapping(document, "evaluation policy", "catalog-invalid")
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
    legacy_qwen_callback("_validate_policy", validate_event_policy)(policy)
    return QwenEvaluation(corpus, policy)


__all__ = [
    "MAX_CATALOG_BYTES",
    "catalog_path",
    "load_qwen_catalog",
]
