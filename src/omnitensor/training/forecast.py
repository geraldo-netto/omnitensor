"""Self-supervised local forecast recipe and portable ONNX export."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from omnitensor.forecasting import fit_forecaster
from omnitensor.preparation import file_digest
from omnitensor.telemetry_recorder import TelemetryRecorder
from omnitensor.telemetry_types import FeatureRow, LinearForecaster

from .contracts import TrainingError, TrainingReport, TrainingSpec, write_training_report

SUPPORTED_PROFILES = frozenset({"resource-scheduler"})


class PortableModelExporter(Protocol):
    """Export one fitted logical model without knowing any target device."""

    def export(self, model: LinearForecaster, destination: Path) -> None: ...


@dataclass(frozen=True, slots=True)
class ForecastDataset:
    inputs: tuple[tuple[float, ...], ...]
    targets: tuple[float, ...]
    corpus_sha256: str


class ForecastTrainer:
    """Fit local history, reject a useless fit, and freeze portable evidence."""

    def __init__(self, exporter: PortableModelExporter | None = None) -> None:
        self._exporter = exporter or OnnxLinearExporter()

    async def train_async(
        self,
        spec: TrainingSpec,
        records_root: Path | str,
        output_dir: Path | str,
    ) -> TrainingReport:
        """:meth:`train` off the event loop.

        The fit is a synchronous least-squares solve over every recorded
        window, and the export writes a file; both stall a loop they are
        called on.  An async caller must come through here.
        """
        return await asyncio.to_thread(self.train, spec, records_root, output_dir)

    def train(
        self,
        spec: TrainingSpec,
        records_root: Path | str,
        output_dir: Path | str,
    ) -> TrainingReport:
        if spec.profile_id not in SUPPORTED_PROFILES:
            raise TrainingError(
                "recipe-unsupported",
                f"forecast-v1 does not support profile {spec.profile_id}",
            )
        dataset = forecast_dataset(spec, TelemetryRecorder(records_root).rows(spec.profile_id))
        model = fit_forecaster(
            dataset.inputs,
            dataset.targets,
            spec.feature_names,
            spec.window,
        )
        if not model.quality.useful:
            raise TrainingError(
                "model-not-useful",
                "held-out fit does not beat predicting the latest observation",
            )
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        model_path = destination / "model.onnx"
        self._exporter.export(model, model_path)
        if not model_path.is_file() or model_path.stat().st_size == 0:
            raise TrainingError("export-failed", "exporter produced no portable model")
        report = TrainingReport(
            spec,
            len(dataset.inputs),
            dataset.corpus_sha256,
            file_digest(model_path),
            model.quality.document(),
        )
        write_training_report(
            destination / "training-report.json",
            report.document(),
            prefix=".training-report-",
        )
        return report


def forecast_dataset(spec: TrainingSpec, rows: tuple[FeatureRow, ...]) -> ForecastDataset:
    """Build time-ordered windows, refusing schema drift instead of zero-filling."""
    needed = spec.window + spec.horizon
    if len(rows) < needed:
        raise TrainingError(
            "insufficient-history",
            f"at least {needed} recorded rows are required, got {len(rows)}",
        )
    for previous, current in zip(rows, rows[1:], strict=False):
        if current.observed_at_ms <= previous.observed_at_ms:
            raise TrainingError(
                "observations-unordered",
                "recorded row timestamps must increase strictly",
            )
    for row in rows:
        missing = [name for name in spec.feature_names if name not in row.features]
        if missing:
            raise TrainingError(
                "features-missing",
                f"recorded row {row.observed_at_ms} has no {missing[0]}",
            )
    inputs: list[tuple[float, ...]] = []
    targets: list[float] = []
    for start in range(len(rows) - spec.window - spec.horizon + 1):
        history = rows[start : start + spec.window]
        target = rows[start + spec.window + spec.horizon - 1]
        inputs.append(
            tuple(
                float(observation.features[name])
                for observation in history
                for name in spec.feature_names
            )
        )
        targets.append(float(target.features[spec.target_feature]))
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row.document(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return ForecastDataset(tuple(inputs), tuple(targets), digest.hexdigest())


class OnnxLinearExporter:
    """Write the common MatMul/Add subset consumed by every planned lane."""

    def export(self, model: LinearForecaster, destination: Path) -> None:
        try:
            import onnx  # noqa: PLC0415 - optional producer-only dependency
            from onnx import TensorProto, helper  # noqa: PLC0415
        except ImportError as error:
            raise TrainingError(
                "exporter-missing", "onnx is not installed; install the [train] extra"
            ) from error
        width = model.input_width
        weights = helper.make_tensor(
            "weights",
            TensorProto.FLOAT,
            [width, 1],
            [float(value) for value in model.weights],
        )
        bias = helper.make_tensor("bias", TensorProto.FLOAT, [1], [float(model.intercept)])
        graph = helper.make_graph(
            [
                helper.make_node("MatMul", ["input", "weights"], ["linear"]),
                helper.make_node("Add", ["linear", "bias"], ["output"]),
            ],
            "omnitensor-local-forecast",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, width])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])],
            [weights, bias],
        )
        portable = helper.make_model(
            graph,
            producer_name="omnitensor",
            opset_imports=[helper.make_opsetid("", 13)],
        )
        portable.ir_version = min(portable.ir_version, 8)
        try:
            onnx.checker.check_model(portable)
        except Exception as error:  # noqa: BLE001 - exporter diagnostics vary by release
            raise TrainingError("export-failed", f"ONNX validation failed: {error}") from error
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".model-", dir=destination.parent)
        os.close(handle)
        try:
            onnx.save_model(portable, temporary)
            os.replace(temporary, destination)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
