"""Compatibility facade for split Qwen provider and qualification modules.

New code should import from :mod:`qwen_providers`, :mod:`qwen_catalog`,
:mod:`qwen_qualification`, or :mod:`qwen_contracts`.  This facade preserves
the original public and live private names for one compatibility release.
"""

from __future__ import annotations

from pathlib import Path

from . import acceptance_kit as _acceptance_kit
from . import generation as _generation
from . import qwen_catalog as _catalog
from . import qwen_contracts as _contracts
from . import qwen_providers as _providers
from . import qwen_qualification as _qualification

NativeLoadReport = _acceptance_kit.NativeLoadReport
validate_gpu_load = _acceptance_kit.validate_gpu_load
validate_npu_load = _acceptance_kit.validate_npu_load
ProviderGenerationError = _generation.ProviderGenerationError

EventProviderEvidence = _contracts.EventProviderEvidence
EventProviderObservation = _contracts.EventProviderObservation
EventQualificationPolicy = _contracts.EventQualificationPolicy
EventQualificationReport = _contracts.EventQualificationReport
FrozenEventCase = _contracts.FrozenEventCase
FrozenEventCorpus = _contracts.FrozenEventCorpus
NativeQwenRuntime = _contracts.NativeQwenRuntime
QwenCatalog = _contracts.QwenCatalog
QwenEvaluation = _contracts.QwenEvaluation
QwenProviderError = _contracts.QwenProviderError
QwenSource = _contracts.QwenSource

LlamaCppVulkanQwenWorker = _providers.LlamaCppVulkanQwenWorker
OpenVinoNpuQwenWorker = _providers.OpenVinoNpuQwenWorker
_QwenWorker = _providers._QwenWorker

qualify_event_provider = _qualification.qualify_event_provider
MAX_CATALOG_BYTES = _catalog.MAX_CATALOG_BYTES
MAX_CORPUS_BYTES = _qualification.MAX_CORPUS_BYTES

_evaluation = _catalog._evaluation
_provider_template = _catalog._provider_template
_read_document = _catalog._read_document
_source = _catalog._source
_case = _qualification._case
_event_key = _qualification._event_key
_expected_event = _qualification._expected_event
_score_observation = _qualification._score_observation
_score_observations = _qualification._score_observations
_validate_evidence_identity = _qualification._validate_evidence_identity
_validate_lane_load = _qualification._validate_lane_load
_mapping = _contracts.qwen_mapping
_positive_integer = _contracts.qwen_positive_integer
_sequence = _contracts.qwen_sequence
_text = _contracts.qwen_text
_validate_policy = _contracts.validate_event_policy

_PACKAGE_DIR = Path(__file__).resolve().parent
_PACKAGED_MODELS = _PACKAGE_DIR.parent / "generation-models"
_PACKAGED_CORPORA = _PACKAGE_DIR.parent / "evaluation-corpora"
_SOURCE_ROOT = _PACKAGE_DIR.parents[2] if _PACKAGE_DIR.parent.parent.name == "src" else None
_DIGEST = frozenset("0123456789abcdef")

# These aliases moved to the public acceptance kit in the previous release.
_validate_gpu_load = validate_gpu_load
_validate_npu_load = validate_npu_load


def load_qwen_catalog(path: Path | str | None = None) -> QwenCatalog:
    """Load through the facade-owned discovery seam used by existing callers."""
    return _catalog.load_qwen_catalog(path or _catalog_path())


def load_event_corpus(path: Path | str | None = None) -> FrozenEventCorpus:
    """Load through the facade-owned discovery seam used by existing callers."""
    corpus_path = Path(path) if path is not None else _corpus_path()
    return _qualification.load_event_corpus(corpus_path)


def _catalog_path() -> Path:
    return _catalog.catalog_path(_PACKAGED_MODELS, _SOURCE_ROOT)


def _corpus_path() -> Path:
    return _qualification.corpus_path(_PACKAGED_CORPORA, _SOURCE_ROOT)


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
