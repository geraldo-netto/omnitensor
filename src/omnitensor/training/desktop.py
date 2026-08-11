"""Opt-in content-free desktop suggestion source training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..atomicio import write_json_atomic
from ..preparation import file_digest
from ..storelock import store_lock
from .build import BuildAdvisorModel, BuildExample, _fit_output, _normalization
from .contracts import TrainingError

DESKTOP_RECIPE = "confirmed-layout-suggestion-v1"
DESKTOP_CONFIRMATION = "user-confirmed"
DESKTOP_REVOCATION_CONFIRMATION = "I-confirm-delete-desktop-training-history"
MAX_HISTORY_BYTES = 64 * 1024 * 1024
MAX_HISTORY_LINE_BYTES = 64 * 1024
MAX_HISTORY_EXAMPLES = 100_000
MAX_WINDOWS = 64
MAX_WORKSPACES = 64
DEFAULT_MIN_CLASS_EXAMPLES = 8
DEFAULT_MIN_ACCURACY = 0.6
DEFAULT_MIN_MACRO_RECALL = 0.5
SUGGESTIONS = (
    "keep",
    "tile-left",
    "tile-right",
    "maximize",
    "move-next-workspace",
)
FOCUSED_ROLES = ("normal", "dialog", "utility", "notification", "unknown")
SIZE_BUCKETS = ("none", "narrow", "medium", "wide")
DESKTOP_FEATURES = (
    "workspaceCountLog",
    "activeWorkspaceFraction",
    "windowCountLog",
    "visibleFraction",
    "dialogFraction",
    "utilityFraction",
    "notificationFraction",
    *(f"focusedRole:{name}" for name in FOCUSED_ROLES),
    *(f"focusedWidth:{name}" for name in SIZE_BUCKETS),
    *(f"focusedHeight:{name}" for name in SIZE_BUCKETS),
)
_RECORD_KEYS = {
    "schemaVersion",
    "observedAtMs",
    "workspaceCount",
    "activeWorkspace",
    "windowCount",
    "visibleWindows",
    "dialogWindows",
    "utilityWindows",
    "notificationWindows",
    "focusedRole",
    "focusedWidthBucket",
    "focusedHeightBucket",
    "confirmedSuggestion",
    "confirmation",
}


@dataclass(frozen=True, slots=True)
class DesktopExample:
    observed_at_ms: int
    features: tuple[float, ...]
    suggestion: str


@dataclass(frozen=True, slots=True)
class DesktopDataset:
    examples: tuple[DesktopExample, ...]
    history_sha256: str


class DesktopModelExporter(Protocol):
    def export(self, model: BuildAdvisorModel, destination: Path) -> None: ...


def load_desktop_history(path: Path | str) -> DesktopDataset:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise TrainingError("history-invalid", "desktop history is not a safe regular file")
    if source.stat().st_size > MAX_HISTORY_BYTES:
        raise TrainingError("history-too-large", "desktop history byte limit exceeded")
    digest = hashlib.sha256()
    examples = []
    previous = -1
    with source.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            digest.update(raw)
            if line_number > MAX_HISTORY_EXAMPLES:
                raise TrainingError("history-too-large", "desktop example limit exceeded")
            example = _history_line(raw, line_number)
            if example.observed_at_ms <= previous:
                raise TrainingError(
                    "observations-unordered", "desktop example times must increase"
                )
            previous = example.observed_at_ms
            examples.append(example)
    if not examples:
        raise TrainingError("insufficient-history", "desktop history is empty")
    return DesktopDataset(tuple(examples), digest.hexdigest())


def _history_line(raw: bytes, line_number: int) -> DesktopExample:
    if not raw.strip() or len(raw) > MAX_HISTORY_LINE_BYTES:
        raise TrainingError("history-invalid", f"desktop history line {line_number} is invalid")
    try:
        return _example_from_document(json.loads(raw))
    except (UnicodeDecodeError, json.JSONDecodeError, TrainingError) as error:
        detail = error.detail if isinstance(error, TrainingError) else "invalid JSON"
        raise TrainingError(
            "history-invalid", f"desktop history line {line_number}: {detail}"
        ) from error


def _example_from_document(document: object) -> DesktopExample:
    if not isinstance(document, dict) or set(document) != _RECORD_KEYS:
        raise TrainingError("record-invalid", "desktop example fields are invalid")
    if type(document.get("schemaVersion")) is not int or document["schemaVersion"] != 1:
        raise TrainingError("record-invalid", "desktop example version is invalid")
    observed = _bounded_integer(document.get("observedAtMs"), 0, 2**63 - 1, "time")
    workspaces = _bounded_integer(
        document.get("workspaceCount"), 1, MAX_WORKSPACES, "workspace count"
    )
    active = _bounded_integer(
        document.get("activeWorkspace"), 0, workspaces - 1, "active workspace"
    )
    windows = _bounded_integer(document.get("windowCount"), 0, MAX_WINDOWS, "window count")
    counts = tuple(
        _bounded_integer(document.get(name), 0, windows, name)
        for name in (
            "visibleWindows",
            "dialogWindows",
            "utilityWindows",
            "notificationWindows",
        )
    )
    if sum(counts[1:]) > windows:
        raise TrainingError("record-invalid", "desktop role counts exceed window count")
    role = document.get("focusedRole")
    width = document.get("focusedWidthBucket")
    height = document.get("focusedHeightBucket")
    if (
        role not in (*FOCUSED_ROLES, "none")
        or width not in SIZE_BUCKETS
        or height not in SIZE_BUCKETS
    ):
        raise TrainingError("record-invalid", "focused desktop metadata is invalid")
    if (role == "none") != (windows == 0) or (width == "none") != (windows == 0):
        raise TrainingError("record-invalid", "focused metadata disagrees with window count")
    if (height == "none") != (windows == 0):
        raise TrainingError("record-invalid", "focused metadata disagrees with window count")
    suggestion = document.get("confirmedSuggestion")
    if suggestion not in SUGGESTIONS or document.get("confirmation") != DESKTOP_CONFIRMATION:
        raise TrainingError("label-invalid", "desktop suggestion is not user-confirmed")
    denominator = max(1, windows)
    features = (
        math.log1p(workspaces),
        active / max(1, workspaces - 1),
        math.log1p(windows),
        *(value / denominator for value in counts),
        *(float(role == name) for name in FOCUSED_ROLES),
        *(float(width == name) for name in SIZE_BUCKETS),
        *(float(height == name) for name in SIZE_BUCKETS),
    )
    return DesktopExample(observed, features, suggestion)


def _bounded_integer(value: object, minimum: int, maximum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise TrainingError("record-invalid", f"desktop {label} is out of range")
    return value


class DesktopSuggestionTrainer:
    def __init__(
        self,
        exporter: DesktopModelExporter | None = None,
        *,
        minimum_class_examples: int = DEFAULT_MIN_CLASS_EXAMPLES,
        minimum_accuracy: float = DEFAULT_MIN_ACCURACY,
        minimum_macro_recall: float = DEFAULT_MIN_MACRO_RECALL,
    ) -> None:
        if (
            isinstance(minimum_class_examples, bool)
            or not isinstance(minimum_class_examples, int)
            or minimum_class_examples < 1
        ):
            raise ValueError("minimum_class_examples must be a positive integer")
        self._minimum_class_examples = minimum_class_examples
        self._minimum_accuracy = _unit_metric(minimum_accuracy, "minimum_accuracy")
        self._minimum_macro_recall = _unit_metric(
            minimum_macro_recall, "minimum_macro_recall"
        )
        self._exporter = exporter or OnnxDesktopExporter()

    def train(self, dataset: DesktopDataset, output_dir: Path | str) -> dict:
        suggestions = tuple(value for value in SUGGESTIONS if value in {
            item.suggestion for item in dataset.examples
        })
        if len(suggestions) < 2:
            raise TrainingError("class-imbalance", "desktop history needs two suggestions")
        training, holdout = _time_split(dataset.examples)
        _require_suggestions(training, suggestions, self._minimum_class_examples, "training")
        _require_suggestions(holdout, suggestions, self._minimum_class_examples, "holdout")
        build_examples = tuple(_build_example(item, suggestions) for item in training)
        means, scales = _normalization(build_examples)
        fitted = tuple(
            _fit_output(
                build_examples,
                lambda item, index=index: int(item.executed_optional[index] == "1"),
                means,
                scales,
            )
            for index in range(len(suggestions))
        )
        model = BuildAdvisorModel(
            means,
            scales,
            tuple(tuple(output[0][row] for output in fitted) for row in range(len(means))),
            tuple(output[1] for output in fitted),
        )
        quality = _quality(model, holdout, suggestions)
        if quality["accuracy"] < self._minimum_accuracy:
            raise TrainingError("model-not-useful", "held-out desktop accuracy is below the gate")
        if quality["macroRecall"] < self._minimum_macro_recall:
            raise TrainingError(
                "model-not-useful", "held-out desktop macro recall is below the gate"
            )
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        model_path = destination / "model.onnx"
        self._exporter.export(model, model_path)
        if not model_path.is_file() or model_path.stat().st_size == 0:
            raise TrainingError("export-failed", "exporter produced no portable model")
        report = _report(dataset, training, holdout, suggestions, quality, model_path)
        write_json_atomic(
            destination / "desktop-training-report.json",
            report,
            prefix=".desktop-training-report-",
        )
        return report


class OnnxDesktopExporter:
    def export(self, model: BuildAdvisorModel, destination: Path) -> None:
        try:
            import onnx  # noqa: PLC0415 - optional producer dependency
            from onnx import TensorProto, helper  # noqa: PLC0415
        except ImportError as error:
            raise TrainingError(
                "exporter-missing", "onnx is not installed; install the [train] extra"
            ) from error
        width = len(model.means)
        outputs = len(model.intercepts)
        graph = helper.make_graph(
            [
                helper.make_node("Sub", ["input", "means"], ["centered"]),
                helper.make_node("Div", ["centered", "scales"], ["normalized"]),
                helper.make_node("MatMul", ["normalized", "weights"], ["linear"]),
                helper.make_node("Add", ["linear", "intercepts"], ["logits"]),
                helper.make_node("Sigmoid", ["logits"], ["scores"]),
            ],
            "omnitensor-desktop-layout-suggestions",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, width])],
            [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, outputs])],
            [
                helper.make_tensor("means", TensorProto.FLOAT, [width], model.means),
                helper.make_tensor("scales", TensorProto.FLOAT, [width], model.scales),
                helper.make_tensor(
                    "weights",
                    TensorProto.FLOAT,
                    [width, outputs],
                    [value for row in model.weights for value in row],
                ),
                helper.make_tensor("intercepts", TensorProto.FLOAT, [outputs], model.intercepts),
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
        handle, temporary = tempfile.mkstemp(prefix=".desktop-model-", dir=destination.parent)
        os.close(handle)
        try:
            onnx.save_model(portable, temporary)
            os.replace(temporary, destination)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise


def revoke_desktop_history(path: Path | str, confirmation: str) -> bool:
    if confirmation != DESKTOP_REVOCATION_CONFIRMATION:
        raise TrainingError(
            "revocation-not-confirmed",
            f"pass exactly {DESKTOP_REVOCATION_CONFIRMATION}",
        )
    source = Path(path).expanduser()
    if source.is_symlink() or (source.exists() and not source.is_file()):
        raise TrainingError("history-invalid", "desktop history is not a safe regular file")
    with store_lock(source.parent, ".desktop-training.lock"):
        if not source.exists():
            return False
        if source.is_symlink() or not source.is_file():
            raise TrainingError("history-invalid", "desktop history changed during revocation")
        source.unlink()
    return True


def _unit_metric(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be in [0, 1]")
    return float(value)


def _time_split(
    examples: Sequence[DesktopExample],
) -> tuple[tuple[DesktopExample, ...], tuple[DesktopExample, ...]]:
    if len(examples) < 2:
        raise TrainingError("insufficient-history", "desktop history cannot be split")
    split = min(len(examples) - 1, max(1, int(len(examples) * 0.8)))
    return tuple(examples[:split]), tuple(examples[split:])


def _require_suggestions(
    examples: Sequence[DesktopExample],
    suggestions: Sequence[str],
    minimum: int,
    label: str,
) -> None:
    for suggestion in suggestions:
        if sum(item.suggestion == suggestion for item in examples) < minimum:
            raise TrainingError(
                "class-imbalance",
                f"{label} needs at least {minimum} confirmations for {suggestion}",
            )


def _build_example(item: DesktopExample, suggestions: Sequence[str]) -> BuildExample:
    labels = tuple("1" if item.suggestion == suggestion else "0" for suggestion in suggestions)
    return BuildExample(item.observed_at_ms, item.features, 0, labels, ())


def _quality(
    model: BuildAdvisorModel,
    examples: Sequence[DesktopExample],
    suggestions: Sequence[str],
) -> dict:
    scores = tuple(model.predict(item.features) for item in examples)
    predictions = tuple(
        suggestions[max(range(len(suggestions)), key=values.__getitem__)] for values in scores
    )
    correct = sum(
        prediction == item.suggestion
        for prediction, item in zip(predictions, examples, strict=True)
    )
    recalls = tuple(
        sum(
            prediction == suggestion and item.suggestion == suggestion
            for prediction, item in zip(predictions, examples, strict=True)
        )
        / sum(item.suggestion == suggestion for item in examples)
        for suggestion in suggestions
    )
    return {
        "accuracy": correct / len(examples),
        "macroRecall": sum(recalls) / len(recalls),
        "examples": len(examples),
    }


def _report(
    dataset: DesktopDataset,
    training: Sequence[DesktopExample],
    holdout: Sequence[DesktopExample],
    suggestions: Sequence[str],
    quality: dict,
    model_path: Path,
) -> dict:
    return {
        "version": 1,
        "recipe": DESKTOP_RECIPE,
        "dataset": {
            "historySha256": dataset.history_sha256,
            "examples": len(dataset.examples),
            "contentCaptured": False,
            "applicationIdsPersisted": False,
            "windowIdsPersisted": False,
            "labels": "explicit-user-confirmation",
        },
        "split": {
            "kind": "chronological",
            "training": len(training),
            "holdout": len(holdout),
            "holdoutFromMs": holdout[0].observed_at_ms,
        },
        "featureContract": {
            "version": 1,
            "features": list(DESKTOP_FEATURES),
            "contentFree": True,
        },
        "resultContract": {
            "version": 1,
            "kind": "desktop-layout-suggestion-scores",
            "suggestions": list(suggestions),
            "requiresUserConfirmation": True,
            "automaticWindowActions": False,
        },
        "quality": quality,
        "model": {
            "format": "onnx",
            "filename": "model.onnx",
            "sha256": file_digest(model_path),
        },
        "tensorContract": {
            "inputs": [{"shape": [1, len(DESKTOP_FEATURES)], "dtype": "float32", "layout": "NC"}]
        },
        "outputContract": {"kind": "raw"},
        "targets": {"tpu": "uncompiled", "npu": "uncompiled", "gpu": "uncompiled"},
    }
