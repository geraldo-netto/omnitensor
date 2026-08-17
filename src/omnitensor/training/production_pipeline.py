"""Shared fail-closed machinery for portable model source production."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypeVar

from .recipe_model import FetchedModelSource, ModelRecipeError

_GENERIC_NATIVE_REASON = "no target compiler, native parity, or named-device evidence was run"

Evidence = TypeVar("Evidence")
Result = TypeVar("Result")


class ReportWriter(Protocol):
    """Atomic JSON writer accepted by the shared production runner."""

    def __call__(self, path: Path, payload: dict, *, prefix: str) -> None: ...


def run_production_pipeline(
    model_path: Path,
    report_path: Path,
    *,
    conflict_detail: str,
    export: Callable[[], None] | None,
    evaluate: Callable[[], Evidence],
    accepted: Callable[[Evidence], bool],
    rejection_code: str,
    rejection_detail: str,
    report: Callable[[Evidence], dict],
    report_prefix: str,
    result: Callable[[Path, Path, Evidence], Result],
    writer: ReportWriter,
    prepare_export: Callable[[], Callable[[], None]] | None = None,
) -> Result:
    """Run one producer, removing every partial output on any failure."""
    if model_path.exists() or report_path.exists():
        raise ModelRecipeError("producer-conflict", conflict_detail)
    export_operation = prepare_export() if prepare_export is not None else export
    if export_operation is None:
        raise TypeError("production export operation is required")
    try:
        export_operation()
        evidence = evaluate()
        if not accepted(evidence):
            raise ModelRecipeError(rejection_code, rejection_detail)
        writer(report_path, report(evidence), prefix=report_prefix)
    except BaseException:
        model_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        raise
    return result(model_path, report_path, evidence)


def production_report(
    source: FetchedModelSource,
    model_path: Path,
    *,
    kind: str,
    source_identity: Mapping[str, object],
    holdout: Mapping[str, object],
    portable_source_gate: Mapping[str, object],
    native_targets: Mapping[str, object] | None = None,
) -> dict:
    """Build the ordered report envelope shared by all source producers."""
    report = {
        "reportVersion": 1,
        "kind": kind,
        "recipeId": source.recipe.id,
        "recipeVersion": source.recipe.version,
        "recipeSha256": source.recipe.document_sha256,
        "sourceReceiptSha256": sha256_file(source.receipt_path),
    }
    report.update(source_identity)
    report.update(
        {
            "portableModel": {
                "format": "onnx",
                "sha256": sha256_file(model_path),
                "tensorContract": source.recipe.tensor_contract,
                "outputContract": source.recipe.output_contract,
                "producer": source.recipe.producer,
            },
            "holdout": dict(holdout),
            "portableSourceGate": dict(portable_source_gate),
            "nativeTargets": (
                dict(native_targets) if native_targets is not None else _native_targets()
            ),
            "cpuFallback": "forbidden",
        }
    )
    return report


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one local production input or artifact."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def width_checked_dot(
    left: Sequence[float],
    right: Sequence[float],
    *,
    mismatch_detail: str,
) -> float:
    """Return a dot product only when both vectors have exactly equal widths."""
    if len(left) != len(right):
        raise ValueError(mismatch_detail)
    return sum(a * b for a, b in zip(left, right, strict=True))


def validate_onnx_io(
    model: object,
    *,
    input_name: str,
    input_shape: Sequence[int],
    input_detail: str,
    output_name: str,
    output_shape: Sequence[int],
    output_detail: str,
    shape_detail: str,
) -> None:
    """Validate one fixed-input/fixed-output ONNX tensor contract."""
    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    if len(inputs) != 1 or inputs[0].name != input_name:
        raise ModelRecipeError("producer-invalid", input_detail)
    if len(outputs) != 1 or outputs[0].name != output_name:
        raise ModelRecipeError("producer-invalid", output_detail)
    actual_input = [dimension.dim_value for dimension in inputs[0].type.tensor_type.shape.dim]
    actual_output = [dimension.dim_value for dimension in outputs[0].type.tensor_type.shape.dim]
    if actual_input != list(input_shape) or actual_output != list(output_shape):
        raise ModelRecipeError("producer-invalid", shape_detail)


def export_torch_onnx_atomic(
    destination: Path,
    *,
    prefix: str,
    build_arguments: Callable[[], tuple[object, object]],
    input_names: Sequence[str],
    output_names: Sequence[str],
    validate: Callable[[object], None],
    torch: object,
    onnx: object,
) -> None:
    """Export and validate Torch ONNX through a private sibling file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged_name = tempfile.mkstemp(
        prefix=prefix, suffix=".onnx", dir=destination.parent
    )
    os.close(descriptor)
    staged = Path(staged_name)
    try:
        exportable, dummy = build_arguments()
        torch.onnx.export(
            exportable,
            dummy,
            staged,
            input_names=list(input_names),
            output_names=list(output_names),
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        graph = onnx.load(staged, load_external_data=False)
        onnx.checker.check_model(graph)
        validate(graph)
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)


def _native_targets() -> dict:
    return {
        target: {"status": "unqualified", "reason": _GENERIC_NATIVE_REASON}
        for target in ("gpu", "npu", "tpu")
    }
