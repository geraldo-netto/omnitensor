"""Local labelled hardware-health window training with portable output."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..plugins.collection import STABLE_ID
from ..plugins.hardware_collection import (
    HardwareSample,
    SensorHealth,
    SensorKind,
    hardware_sample_error,
)
from ..plugins.triggers import SourceStatus
from ..preparation import file_digest
from .build import BuildAdvisorModel, BuildExample, _auc, _fit_output, _normalization
from .contracts import MAX_INPUT_WIDTH, TrainingError, write_training_report

HARDWARE_RECIPE = "labelled-sensor-window-v1"
HARDWARE_LABEL_CONFIRMATION = "I-confirm-hardware-labels-are-reviewed"
MAX_HISTORY_BYTES = 64 * 1024 * 1024
MAX_HISTORY_LINE_BYTES = 1024 * 1024
MAX_HISTORY_SNAPSHOTS = 5_000
MAX_TRAINING_SENSORS = 32
MAX_ROLE_LENGTH = 64
MAX_WINDOW = 128
DEFAULT_WINDOW = 8
DEFAULT_MIN_CLASS_EXAMPLES = 8
DEFAULT_MIN_AUC = 0.6
DEFAULT_MIN_RECALL = 0.5
DEFAULT_MAX_FALSE_POSITIVE_RATE = 0.1
_ROLE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_WRAPPER_KEYS = {"label", "labelSource", "snapshot"}
_SNAPSHOT_KEYS = {
    "schemaVersion",
    "source",
    "sourceHealth",
    "observedAtMs",
    "items",
    "churn",
    "truncatedItems",
}
_ITEM_KEYS = {"id", "kind", "health", "value", "unit", "label", "observedAtMs"}


@dataclass(frozen=True, slots=True)
class SensorBinding:
    role: str
    stable_id: str


@dataclass(frozen=True, slots=True)
class HardwareObservation:
    observed_at_ms: int
    label: int
    label_source: str
    features: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class HardwareDataset:
    observations: tuple[HardwareObservation, ...]
    history_sha256: str
    roles: tuple[str, ...]
    kinds: tuple[str, ...]
    units: tuple[str, ...]


class HardwareModelExporter(Protocol):
    def export(self, model: BuildAdvisorModel, destination: Path) -> None: ...


def parse_sensor_bindings(values: Sequence[str]) -> tuple[SensorBinding, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("sensor bindings must be a sequence")
    bindings = [_binding_from_text(value) for value in values]
    if not 1 <= len(bindings) <= MAX_TRAINING_SENSORS:
        raise ValueError(f"sensor bindings must contain 1 to {MAX_TRAINING_SENSORS} entries")
    if len({item.role for item in bindings}) != len(bindings):
        raise ValueError("sensor roles must be unique")
    if len({item.stable_id for item in bindings}) != len(bindings):
        raise ValueError("sensor stable identities must be unique")
    return tuple(bindings)


def _binding_from_text(value: object) -> SensorBinding:
    if not isinstance(value, str) or value.count("=") != 1:
        raise ValueError("sensor binding must be ROLE=STABLE_ID")
    role, stable_id = value.split("=", 1)
    if len(role) > MAX_ROLE_LENGTH or not _ROLE.fullmatch(role):
        raise ValueError("sensor role is invalid")
    if not STABLE_ID.fullmatch(stable_id):
        raise ValueError("sensor stable identity is invalid")
    return SensorBinding(role, stable_id)


def load_hardware_history(
    path: Path | str,
    *,
    bindings: Sequence[SensorBinding],
    confirm_labels: str,
) -> HardwareDataset:
    if confirm_labels != HARDWARE_LABEL_CONFIRMATION:
        raise TrainingError(
            "labels-not-confirmed",
            f"review the labels and pass exactly {HARDWARE_LABEL_CONFIRMATION}",
        )
    selected = _validated_bindings(bindings)
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise TrainingError("history-invalid", "hardware history is not a safe regular file")
    if source.stat().st_size > MAX_HISTORY_BYTES:
        raise TrainingError("history-too-large", "hardware history byte limit exceeded")
    observations, semantics, digest = _read_history(source, selected)
    if not observations:
        raise TrainingError("insufficient-history", "hardware history is empty")
    _strict_observation_times(observations)
    kinds, units = zip(*semantics, strict=True)
    return HardwareDataset(
        observations,
        digest,
        tuple(item.role for item in selected),
        tuple(str(kind) for kind in kinds),
        tuple(units),
    )


def _read_history(
    source: Path, bindings: tuple[SensorBinding, ...]
) -> tuple[
    tuple[HardwareObservation, ...],
    tuple[tuple[SensorKind, str], ...],
    str,
]:
    digest = hashlib.sha256()
    observations = []
    semantics = None
    with source.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            digest.update(raw)
            if line_number > MAX_HISTORY_SNAPSHOTS:
                raise TrainingError("history-too-large", "hardware snapshot limit exceeded")
            observation, current_semantics = _history_line(raw, line_number, bindings)
            if semantics is None:
                semantics = current_semantics
            elif semantics != current_semantics:
                raise TrainingError("sensor-drift", "hardware sensor kind or unit changed")
            observations.append(observation)
    return (
        tuple(observations),
        semantics or (),
        digest.hexdigest(),
    )


def _validated_bindings(bindings: Sequence[SensorBinding]) -> tuple[SensorBinding, ...]:
    if isinstance(bindings, (str, bytes)) or not isinstance(bindings, Sequence):
        raise TrainingError("bindings-invalid", "sensor bindings must be a sequence")
    values = tuple(bindings)
    try:
        rendered = tuple(f"{item.role}={item.stable_id}" for item in values)
    except AttributeError as error:
        raise TrainingError("bindings-invalid", "sensor bindings are not typed") from error
    try:
        return parse_sensor_bindings(rendered)
    except ValueError as error:
        raise TrainingError("bindings-invalid", str(error)) from error


def _history_line(
    raw: bytes,
    line_number: int,
    bindings: tuple[SensorBinding, ...],
) -> tuple[HardwareObservation, tuple[tuple[SensorKind, str], ...]]:
    if not raw.strip() or len(raw) > MAX_HISTORY_LINE_BYTES:
        raise TrainingError(
            "history-invalid", f"hardware history line {line_number} is invalid"
        )
    try:
        return _observation_from_document(json.loads(raw), bindings)
    except (UnicodeDecodeError, json.JSONDecodeError, TrainingError) as error:
        detail = error.detail if isinstance(error, TrainingError) else "invalid JSON"
        raise TrainingError(
            "history-invalid", f"hardware history line {line_number}: {detail}"
        ) from error


def _observation_from_document(
    document: object,
    bindings: tuple[SensorBinding, ...],
) -> tuple[HardwareObservation, tuple[tuple[SensorKind, str], ...]]:
    if not isinstance(document, dict) or set(document) != _WRAPPER_KEYS:
        raise TrainingError("snapshot-invalid", "labelled hardware fields are invalid")
    label = document.get("label")
    source = document.get("labelSource")
    if label not in {"baseline", "fault"} or source not in {"observed", "injected"}:
        raise TrainingError("label-invalid", "hardware label or source is invalid")
    if source == "injected" and label != "fault":
        raise TrainingError("label-invalid", "only a fault may have an injected label")
    snapshot = document.get("snapshot")
    if not isinstance(snapshot, dict) or set(snapshot) != _SNAPSHOT_KEYS:
        raise TrainingError("snapshot-invalid", "hardware snapshot fields are invalid")
    _validate_snapshot_header(snapshot)
    items = snapshot["items"]
    if not isinstance(items, list):
        raise TrainingError("snapshot-invalid", "hardware items must be an array")
    parsed = tuple(_hardware_item(item, snapshot["observedAtMs"]) for item in items)
    by_id = {item.stable_id: item for item in parsed}
    expected = {item.stable_id for item in bindings}
    if len(by_id) != len(parsed) or set(by_id) != expected:
        raise TrainingError("sensor-set-mismatch", "hardware snapshot sensor set disagrees")
    ordered = tuple(by_id[item.stable_id] for item in bindings)
    features = tuple(value for item in ordered for value in _sensor_features(item))
    semantics = tuple((item.kind, item.unit) for item in ordered)
    return (
        HardwareObservation(snapshot["observedAtMs"], int(label == "fault"), source, features),
        semantics,
    )


def _validate_snapshot_header(snapshot: dict) -> None:
    if (
        type(snapshot.get("schemaVersion")) is not int
        or snapshot["schemaVersion"] != 1
        or snapshot.get("source") != "hwmon-edac-power-service"
        or snapshot.get("sourceHealth") != str(SourceStatus.READY)
    ):
        raise TrainingError("snapshot-invalid", "hardware snapshot identity is invalid")
    observed = snapshot.get("observedAtMs")
    if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
        raise TrainingError("snapshot-invalid", "hardware snapshot time is invalid")
    if type(snapshot.get("truncatedItems")) is not int or snapshot["truncatedItems"] != 0:
        raise TrainingError("snapshot-invalid", "hardware snapshot must contain every sensor")
    churn = snapshot.get("churn")
    if not isinstance(churn, dict) or set(churn) != {"added", "removed", "changed"}:
        raise TrainingError("snapshot-invalid", "hardware churn fields are invalid")
    if any(not isinstance(churn[name], list) for name in churn):
        raise TrainingError("snapshot-invalid", "hardware churn values are invalid")


def _hardware_item(document: object, snapshot_time: int) -> HardwareSample:
    if not isinstance(document, dict) or set(document) != _ITEM_KEYS:
        raise TrainingError("snapshot-invalid", "hardware item fields are invalid")
    try:
        sample = HardwareSample(
            document["id"],
            SensorKind(document["kind"]),
            SensorHealth(document["health"]),
            document["value"],
            document["unit"],
            document["label"],
            document["observedAtMs"],
        )
    except (TypeError, ValueError) as error:
        raise TrainingError("snapshot-invalid", "hardware item values are invalid") from error
    detail = hardware_sample_error(sample)
    if detail:
        raise TrainingError("snapshot-invalid", detail)
    if (
        isinstance(sample.observed_at_ms, bool)
        or not isinstance(sample.observed_at_ms, int)
        or not 0 <= sample.observed_at_ms <= snapshot_time
    ):
        raise TrainingError("snapshot-invalid", "hardware item time is invalid")
    return sample


def _sensor_features(sample: HardwareSample) -> tuple[float, ...]:
    value = 0.0 if sample.value is None else math.log1p(sample.value)
    return (
        value,
        float(sample.health is SensorHealth.DEGRADED),
        float(sample.health is SensorHealth.CRITICAL),
        float(sample.health is SensorHealth.MISSING),
    )


def _strict_observation_times(observations: Sequence[HardwareObservation]) -> None:
    for previous, current in zip(observations, observations[1:], strict=False):
        if current.observed_at_ms <= previous.observed_at_ms:
            raise TrainingError(
                "observations-unordered", "hardware observation times must increase"
            )


class HardwareHealthTrainer:
    def __init__(
        self,
        exporter: HardwareModelExporter | None = None,
        *,
        window: int = DEFAULT_WINDOW,
        minimum_class_examples: int = DEFAULT_MIN_CLASS_EXAMPLES,
        minimum_auc: float = DEFAULT_MIN_AUC,
        minimum_recall: float = DEFAULT_MIN_RECALL,
        maximum_false_positive_rate: float = DEFAULT_MAX_FALSE_POSITIVE_RATE,
    ) -> None:
        if isinstance(window, bool) or not isinstance(window, int) or not 1 <= window <= MAX_WINDOW:
            raise ValueError(f"window must be an integer from 1 to {MAX_WINDOW}")
        if (
            isinstance(minimum_class_examples, bool)
            or not isinstance(minimum_class_examples, int)
            or minimum_class_examples < 1
        ):
            raise ValueError("minimum_class_examples must be a positive integer")
        self._minimum_auc = _unit_metric(minimum_auc, "minimum_auc")
        self._minimum_recall = _unit_metric(minimum_recall, "minimum_recall")
        self._maximum_fpr = _unit_metric(
            maximum_false_positive_rate, "maximum_false_positive_rate"
        )
        self._window = window
        self._minimum_class_examples = minimum_class_examples
        self._exporter = exporter or OnnxHardwareExporter()

    def train(self, dataset: HardwareDataset, output_dir: Path | str) -> dict:
        width = len(dataset.roles) * 4 * self._window
        if width > MAX_INPUT_WIDTH:
            raise TrainingError("features-too-wide", "hardware window exceeds tensor width")
        examples = _window_examples(dataset.observations, self._window)
        training, holdout = _time_split(examples)
        _require_classes(training, self._minimum_class_examples, "training")
        _require_classes(holdout, self._minimum_class_examples, "holdout")
        means, scales = _normalization(training)
        weights, intercept = _fit_output(
            training, lambda item: item.build_failed, means, scales
        )
        model = BuildAdvisorModel(
            means,
            scales,
            tuple((weight,) for weight in weights),
            (intercept,),
        )
        quality = _quality(model, holdout)
        if quality["auc"] < self._minimum_auc:
            raise TrainingError("model-not-useful", "held-out hardware AUC is below the gate")
        if quality["recallAtHalf"] < self._minimum_recall:
            raise TrainingError("model-not-useful", "held-out hardware recall is below the gate")
        if quality["falsePositiveRateAtHalf"] > self._maximum_fpr:
            raise TrainingError(
                "model-not-stable", "held-out hardware false-positive rate exceeds the gate"
            )
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        model_path = destination / "model.onnx"
        self._exporter.export(model, model_path)
        if not model_path.is_file() or model_path.stat().st_size == 0:
            raise TrainingError("export-failed", "exporter produced no portable model")
        report = _report(dataset, training, holdout, self._window, quality, model_path)
        write_training_report(
            destination / "hardware-training-report.json",
            report,
            prefix=".hardware-training-report-",
            numeric=True,
        )
        return report


class OnnxHardwareExporter:
    def export(self, model: BuildAdvisorModel, destination: Path) -> None:
        try:
            import onnx  # noqa: PLC0415 - optional producer dependency
            from onnx import TensorProto, helper  # noqa: PLC0415
        except ImportError as error:
            raise TrainingError(
                "exporter-missing", "onnx is not installed; install the [train] extra"
            ) from error
        width = len(model.means)
        graph = helper.make_graph(
            [
                helper.make_node("Sub", ["input", "means"], ["centered"]),
                helper.make_node("Div", ["centered", "scales"], ["normalized"]),
                helper.make_node("MatMul", ["normalized", "weights"], ["linear"]),
                helper.make_node("Add", ["linear", "intercept"], ["logit"]),
                helper.make_node("Sigmoid", ["logit"], ["risk"]),
            ],
            "omnitensor-hardware-health-risk",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, width])],
            [helper.make_tensor_value_info("risk", TensorProto.FLOAT, [1, 1])],
            [
                helper.make_tensor("means", TensorProto.FLOAT, [width], model.means),
                helper.make_tensor("scales", TensorProto.FLOAT, [width], model.scales),
                helper.make_tensor(
                    "weights",
                    TensorProto.FLOAT,
                    [width, 1],
                    [row[0] for row in model.weights],
                ),
                helper.make_tensor("intercept", TensorProto.FLOAT, [1], model.intercepts),
            ],
        )
        portable = helper.make_model(
            graph,
            producer_name="omnitensor",
            opset_imports=[helper.make_opsetid("", 13)],
        )
        portable.ir_version = min(portable.ir_version, 8)
        try:
            onnx.checker.check_model(portable)
        except Exception as error:  # noqa: BLE001
            raise TrainingError("export-failed", f"ONNX validation failed: {error}") from error
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".hardware-model-", dir=destination.parent)
        os.close(handle)
        try:
            onnx.save_model(portable, temporary)
            os.replace(temporary, destination)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise


def _unit_metric(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be in [0, 1]")
    return float(value)


def _window_examples(
    observations: Sequence[HardwareObservation], window: int
) -> tuple[BuildExample, ...]:
    if len(observations) < window:
        raise TrainingError("insufficient-history", "hardware history is shorter than the window")
    return tuple(
        BuildExample(
            observations[end].observed_at_ms,
            tuple(
                value
                for observation in observations[end - window + 1 : end + 1]
                for value in observation.features
            ),
            observations[end].label,
            (),
            (),
        )
        for end in range(window - 1, len(observations))
    )


def _time_split(
    examples: Sequence[BuildExample],
) -> tuple[tuple[BuildExample, ...], tuple[BuildExample, ...]]:
    if len(examples) < 2:
        raise TrainingError("insufficient-history", "hardware windows cannot be split")
    split = min(len(examples) - 1, max(1, int(len(examples) * 0.8)))
    return tuple(examples[:split]), tuple(examples[split:])


def _require_classes(examples: Sequence[BuildExample], minimum: int, label: str) -> None:
    positives = sum(item.build_failed for item in examples)
    negatives = len(examples) - positives
    if positives < minimum or negatives < minimum:
        raise TrainingError(
            "class-imbalance",
            f"{label} split needs at least {minimum} baseline and fault examples",
        )


def _quality(model: BuildAdvisorModel, examples: Sequence[BuildExample]) -> dict:
    scored = tuple((model.predict(item.features)[0], item.build_failed) for item in examples)
    positives = sum(label for _score, label in scored)
    negatives = len(scored) - positives
    true_positives = sum(score >= 0.5 and label == 1 for score, label in scored)
    false_positives = sum(score >= 0.5 and label == 0 for score, label in scored)
    return {
        "auc": _auc(scored),
        "recallAtHalf": true_positives / positives,
        "falsePositiveRateAtHalf": false_positives / negatives,
        "faultExamples": positives,
        "baselineExamples": negatives,
    }


def _report(
    dataset: HardwareDataset,
    training: Sequence[BuildExample],
    holdout: Sequence[BuildExample],
    window: int,
    quality: dict,
    model_path: Path,
) -> dict:
    label_sources = {"observed": 0, "injected": 0}
    for item in dataset.observations:
        label_sources[item.label_source] += item.label
    roles = [
        {"role": role, "kind": kind, "unit": unit}
        for role, kind, unit in zip(dataset.roles, dataset.kinds, dataset.units, strict=True)
    ]
    width = len(dataset.roles) * 4 * window
    return {
        "version": 1,
        "recipe": HARDWARE_RECIPE,
        "dataset": {
            "historySha256": dataset.history_sha256,
            "observations": len(dataset.observations),
            "observedFaultLabels": label_sources["observed"],
            "injectedFaultLabels": label_sources["injected"],
            "stableIdsPersisted": False,
            "labelsConfirmed": True,
        },
        "split": {
            "kind": "chronological",
            "training": len(training),
            "holdout": len(holdout),
            "holdoutFromMs": holdout[0].started_at_ms,
        },
        "featureContract": {
            "version": 1,
            "roles": roles,
            "perRoleFeatures": ["logValue", "degraded", "critical", "missing"],
            "window": window,
            "observationOrder": "oldest-first",
            "flattenOrder": "observations-then-roles-then-features",
        },
        "resultContract": {
            "version": 1,
            "kind": "hardware-anomaly-risk",
            "threshold": 0.5,
            "evidence": {
                "kind": "largest-normalized-feature-deviations",
                "maximumItems": 4,
                "rolesOnly": True,
            },
            "modelControlsSafetyActions": False,
            "allowedActions": [],
        },
        "quality": quality,
        "model": {
            "format": "onnx",
            "filename": "model.onnx",
            "sha256": file_digest(model_path),
        },
        "tensorContract": {
            "inputs": [{"shape": [1, width], "dtype": "float32", "layout": "NC"}]
        },
        "outputContract": {"kind": "raw"},
        "targets": {"tpu": "uncompiled", "npu": "uncompiled", "gpu": "uncompiled"},
    }
