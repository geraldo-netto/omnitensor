"""Build, qualify, install, and bind the pinned document model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omnitensor.preparation import file_digest, install_prepared, prepare_artifact

from ..atomicio import write_json_atomic
from ..document_model_runners import BgeTokenizer, VulkanBgeRunner
from ..document_model_types import (
    MAX_NATIVE_ABSOLUTE_ERROR,
    MIN_NATIVE_COSINE,
    MIN_RETRIEVAL_OVERLAP,
    PROFILE_ID,
    RECIPE_ID,
    DocumentModelError,
    DocumentModelEvidence,
    InstalledDocumentModel,
    native_tensor_contract,
)
from ..registry import (
    bundled_workloads_path,
    load_workloads,
    validate_workload_document,
)
from .binding import binding_manifest, model_fragment, native_evidence, publish_binding
from .document_model_contracts import load_bge_holdout, source_path
from .document_model_cpu_reference import _producer_cpu_reference_runner
from .document_model_export import export_bge_ncnn
from .document_model_gate import (
    expected_retrieval_hits,
    maximum_embedding_error,
    report_document,
    write_document_model_report,
)
from .embedding_production import SentenceEmbeddingOnnxExporter, evaluate_embedding_gate


@dataclass(frozen=True, slots=True)
class DocumentModelInstallationDependencies:
    tokenizer_factory: object = BgeTokenizer
    source_resolver: object = source_path
    holdout_loader: object = load_bge_holdout
    portable_exporter_type: object = SentenceEmbeddingOnnxExporter
    native_exporter: object = export_bge_ncnn
    portable_runner_factory: object = _producer_cpu_reference_runner
    native_runner_factory: object = VulkanBgeRunner
    gate_evaluator: object = evaluate_embedding_gate
    maximum_error: object = maximum_embedding_error
    expected_hits: object = expected_retrieval_hits
    preparer: object = prepare_artifact
    digest: object = file_digest
    report_builder: object = report_document
    report_writer: object = write_document_model_report
    installer: object = install_prepared
    binding_builder: object = None
    atomic_writer: object = write_json_atomic


def install_document_model(
    source,
    *,
    corpus_path: Path | str,
    build_root: Path | str,
    artifact_root: Path | str,
    bindings_root: Path | str,
    device_index: int | None = None,
    bundled_root: Path | None = None,
    dependencies: DocumentModelInstallationDependencies | None = None,
) -> InstalledDocumentModel:
    """Build, gate, digest-install, and bind one GPU BGE-small variant."""
    deps = dependencies or DocumentModelInstallationDependencies()
    if source.recipe.id != RECIPE_ID or source.recipe.profile_ids != (PROFILE_ID,):
        raise DocumentModelError("recipe-incompatible", "recipe is not the BGE document model")
    output = Path(build_root)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = deps.tokenizer_factory(deps.source_resolver(source, "tokenizer"))
    holdout, expected = deps.holdout_loader(corpus_path)
    sample = tokenizer.encode(holdout.queries[0])
    portable = output / "model.reference.onnx"
    if portable.exists():
        raise DocumentModelError("producer-conflict", "portable reference output already exists")
    deps.portable_exporter_type().export(source, portable)
    native = deps.native_exporter(source, output, sample)
    portable_runner = deps.portable_runner_factory(portable, tokenizer)
    native_runner = deps.native_runner_factory(native, tokenizer, device_index=device_index)
    gate = deps.gate_evaluator(holdout, portable_runner, native_runner)
    maximum_error = deps.maximum_error(holdout, portable_runner, native_runner)
    expected_hits = deps.expected_hits(holdout, expected, native_runner)
    if (
        not gate.accepted
        or gate.minimum_cosine_similarity < MIN_NATIVE_COSINE
        or gate.minimum_top10_overlap < MIN_RETRIEVAL_OVERLAP
        or maximum_error > MAX_NATIVE_ABSOLUTE_ERROR
        or expected_hits != len(expected)
    ):
        raise DocumentModelError("native-parity-failed", "BGE native retrieval gate did not pass")
    prepared = deps.preparer(
        native,
        artifact_id=f"{RECIPE_ID}-gpu",
        version=source.recipe.version,
        model_format="ncnn",
    )
    evidence = DocumentModelEvidence(
        native_runner.device_index,
        native_runner.device_name,
        gate.vector_count,
        gate.minimum_cosine_similarity,
        gate.minimum_top10_overlap,
        maximum_error,
        expected_hits,
    )
    report_path = output / "document-model-report.json"
    portable_sha = deps.digest(portable)
    report = deps.report_builder(source, portable_sha, prepared.reference.sha256, holdout, evidence)
    deps.report_writer(report_path, report)
    report_sha = deps.digest(report_path)
    installed = deps.installer(prepared, artifact_root)
    builder = deps.binding_builder or binding_document
    binding = builder(
        source,
        prepared.manifest_fragment(),
        portable_sha,
        report_sha,
        evidence,
        bundled_root,
    )
    binding_path = Path(bindings_root) / PROFILE_ID / "manifest.json"
    publish_binding(
        binding_path,
        binding,
        prefix=".document-model-binding-",
        writer=deps.atomic_writer,
    )
    return InstalledDocumentModel(installed, binding_path, report_path, evidence)


def binding_document(
    source,
    artifact,
    portable_sha,
    report_sha,
    evidence,
    bundled_root,
    *,
    root_provider=bundled_workloads_path,
    workload_loader=load_workloads,
    workload_validator=validate_workload_document,
) -> dict:
    def variants() -> dict:
        model = model_fragment(
            artifact,
            fully_quantized=False,
            minimum_compiler_version="20260526",
            minimum_runtime_version="1.0.20260526",
            tensor_contract=native_tensor_contract(source.recipe),
            output_contract={"kind": "embedding"},
            training_contract={
                "version": 1,
                "profileId": PROFILE_ID,
                "recipe": RECIPE_ID,
                "reportSha256": report_sha,
                "taskSemanticsSha256": source.recipe.document_sha256,
            },
            evidence=native_evidence(
                portable_sha256=portable_sha,
                native_sha256=artifact["sha256"],
                report_sha256=report_sha,
                samples=evidence.samples,
                maximum_absolute_error=evidence.maximum_absolute_error,
                tolerance=MAX_NATIVE_ABSOLUTE_ERROR,
                compiler_report_sha256=None,
                named_device_accepted=False,
            ),
        )
        return {"gpu": model}

    return binding_manifest(
        PROFILE_ID,
        variants,
        lane_order=("gpu",),
        plural=False,
        root=bundled_root or root_provider(),
        load=workload_loader,
        validate=workload_validator,
        error_type=DocumentModelError,
    )
