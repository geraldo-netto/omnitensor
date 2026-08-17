"""Stable facade for pinned BGE document-model production."""

# ruff: noqa: F401 - the facade deliberately preserves legacy imports

from __future__ import annotations

from pathlib import Path

from omnitensor.preparation import file_digest, install_prepared, prepare_artifact

from ..atomicio import JsonTooLargeError, read_json_bounded, write_json_atomic
from ..document_model_runners import (
    BgeTokenizer,
    VulkanBgeRunner,
    select_document_vulkan_device,
)
from ..document_model_types import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_BINDINGS_ROOT,
    DEFAULT_BUILD_ROOT,
    DEFAULT_SOURCE_ROOT,
    DOCUMENT_MODEL_REPORT_SCHEMA,
    MAX_CORPUS_BYTES,
    MAX_NATIVE_ABSOLUTE_ERROR,
    MAX_TEXT_BYTES,
    MIN_NATIVE_COSINE,
    MIN_RETRIEVAL_OVERLAP,
    PRODUCER_CPU_REFERENCE_BACKEND,
    PROFILE_ID,
    QUERY_PREFIX,
    RECIPE_ID,
    SEQUENCE_LENGTH,
    DocumentModelError,
    DocumentModelEvidence,
    InstalledDocumentModel,
    TokenizedText,
    native_tensor_contract,
)
from ..registry import (
    bundled_workloads_path,
    load_workloads,
    validate_document,
    validate_workload_document,
)
from .document_model_cli import main as _main
from .document_model_contracts import (
    bundled_corpus_path,
    source_path,
)
from .document_model_contracts import (
    load_bge_holdout as _load_bge_holdout,
)
from .document_model_cpu_reference import (
    PortableBgeRunner,
    ProducerCpuBgeReferenceRunner,
)
from .document_model_cpu_reference import (
    _producer_cpu_reference_runner as _canonical_cpu_reference_factory,
)
from .document_model_export import (
    append_ncnn_l2_normalization,
    document_dependencies,
    fixed_bge_model,
)
from .document_model_export import (
    export_bge_ncnn as _export_bge_ncnn,
)
from .document_model_gate import (
    expected_retrieval_hits,
    maximum_embedding_error,
    report_document,
    write_document_model_report,
)
from .document_model_installation import (
    DocumentModelInstallationDependencies,
    binding_document,
)
from .document_model_installation import (
    install_document_model as _install_document_model,
)
from .embedding_production import (
    EmbeddingHoldout,
    SentenceEmbeddingOnnxExporter,
    evaluate_embedding_gate,
    l2_normalize,
)
from .recipe_fetch import fetch_model_sources
from .recipe_model import FetchedModelSource, ModelRecipeError
from .recipe_registry import load_model_recipe as _load_model_recipe
from .recipe_registry import resolve_model_recipe_path

_LEGACY_NATIVE_CONTRACT = "NATIVE_TENSOR_CONTRACT"


def __getattr__(name: str):
    if name != _LEGACY_NATIVE_CONTRACT:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    contract = native_tensor_contract(_load_model_recipe(resolve_model_recipe_path(RECIPE_ID)))
    globals()[name] = contract
    return contract


def __dir__() -> list[str]:
    return sorted({*globals(), _LEGACY_NATIVE_CONTRACT})


def _source_path(source: FetchedModelSource, role: str) -> Path:
    return source_path(source, role)


def load_bge_holdout(path: Path | str):
    return _load_bge_holdout(path, reader=read_json_bounded)


def _select_vulkan_device(runtime, requested: int | None) -> tuple[int, str]:
    return select_document_vulkan_device(runtime, requested)


def _document_dependencies():
    return document_dependencies()


def _fixed_bge_model(torch, encoder):
    return fixed_bge_model(torch, encoder)


def _append_ncnn_l2_normalization(param: Path) -> None:
    append_ncnn_l2_normalization(param)


def export_bge_ncnn(source, destination: Path, sample: TokenizedText) -> Path:
    return _export_bge_ncnn(
        source,
        destination,
        sample,
        dependencies=_document_dependencies,
        source_resolver=_source_path,
        model_factory=_fixed_bge_model,
        graph_normalizer=_append_ncnn_l2_normalization,
    )


def _maximum_embedding_error(holdout, portable, native) -> float:
    return maximum_embedding_error(holdout, portable, native)


def _expected_retrieval_hits(holdout, expected, runner) -> int:
    return expected_retrieval_hits(holdout, expected, runner)


def _report_document(source, portable_sha, native_sha, holdout, evidence) -> dict:
    return report_document(
        source,
        portable_sha,
        native_sha,
        holdout,
        evidence,
        digest=file_digest,
    )


def _write_document_model_report(path: Path, document: dict) -> None:
    write_document_model_report(
        path,
        document,
        validator=validate_document,
        writer=write_json_atomic,
    )


def _binding_document(source, artifact, portable_sha, report_sha, evidence, bundled_root) -> dict:
    return binding_document(
        source,
        artifact,
        portable_sha,
        report_sha,
        evidence,
        bundled_root,
        root_provider=bundled_workloads_path,
        workload_loader=load_workloads,
        workload_validator=validate_workload_document,
    )


def _portable_reference_factory(model: Path, tokenizer: BgeTokenizer):
    if PortableBgeRunner is ProducerCpuBgeReferenceRunner:
        return _canonical_cpu_reference_factory(model, tokenizer)
    return PortableBgeRunner(model, tokenizer)


def install_document_model(
    source,
    *,
    corpus_path: Path | str,
    build_root: Path | str,
    artifact_root: Path | str,
    bindings_root: Path | str,
    device_index: int | None = None,
    bundled_root: Path | None = None,
) -> InstalledDocumentModel:
    dependencies = DocumentModelInstallationDependencies(
        tokenizer_factory=BgeTokenizer,
        source_resolver=_source_path,
        holdout_loader=load_bge_holdout,
        portable_exporter_type=SentenceEmbeddingOnnxExporter,
        native_exporter=export_bge_ncnn,
        portable_runner_factory=_portable_reference_factory,
        native_runner_factory=VulkanBgeRunner,
        gate_evaluator=evaluate_embedding_gate,
        maximum_error=_maximum_embedding_error,
        expected_hits=_expected_retrieval_hits,
        preparer=prepare_artifact,
        digest=file_digest,
        report_builder=_report_document,
        report_writer=_write_document_model_report,
        installer=install_prepared,
        binding_builder=_binding_document,
        atomic_writer=write_json_atomic,
    )
    return _install_document_model(
        source,
        corpus_path=corpus_path,
        build_root=build_root,
        artifact_root=artifact_root,
        bindings_root=bindings_root,
        device_index=device_index,
        bundled_root=bundled_root,
        dependencies=dependencies,
    )


def _bundled_corpus_path() -> Path:
    return bundled_corpus_path(__file__)


def main(argv: list[str] | None = None) -> int:
    return _main(
        argv,
        module_file=__file__,
        resolver=resolve_model_recipe_path,
        fetcher=fetch_model_sources,
        installer=install_document_model,
        corpus_provider=_bundled_corpus_path,
    )


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
