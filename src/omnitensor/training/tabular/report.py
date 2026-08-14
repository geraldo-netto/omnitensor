"""Common report tail and publication flow for numeric trainers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeVar

from ..contracts import TrainingError

Model = TypeVar("Model")


class NumericExporter(Protocol[Model]):
    def export(self, model: Model, destination: Path) -> None: ...


def numeric_training_report(
    family_fields: dict,
    model_path: Path,
    input_width: int,
    *,
    digest: Callable[[Path], str],
) -> dict:
    """Append the canonical model/tensor/output/target report tail in order."""
    report = dict(family_fields)
    report.update(
        {
            "model": {
                "format": "onnx",
                "filename": "model.onnx",
                "sha256": digest(model_path),
            },
            "tensorContract": {
                "inputs": [{"shape": [1, input_width], "dtype": "float32", "layout": "NC"}]
            },
            "outputContract": {"kind": "raw"},
            "targets": {"tpu": "uncompiled", "npu": "uncompiled", "gpu": "uncompiled"},
        }
    )
    return report


def publish_numeric_training(
    output_dir: Path | str,
    model: Model,
    exporter: NumericExporter[Model],
    *,
    report_name: str,
    report_prefix: str,
    report: Callable[[Path], dict],
    writer: Callable[..., None],
) -> dict:
    """Export, verify, build, validate, and atomically publish one numeric report."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "model.onnx"
    exporter.export(model, model_path)
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise TrainingError("export-failed", "exporter produced no portable model")
    payload = report(model_path)
    writer(
        destination / report_name,
        payload,
        prefix=report_prefix,
        numeric=True,
    )
    return payload
