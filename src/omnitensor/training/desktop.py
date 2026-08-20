"""Opt-in content-free desktop suggestion source training."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import omnitensor.training.desktop_history as _desktop_history
from omnitensor.preparation import file_digest

from .contracts import TrainingError, write_training_report
from .tabular import (
    BuildAdvisorModel,
    BuildExample,
    JsonlPolicy,
    chronological_split,
    fit_output,
    load_bounded_jsonl,
    load_onnx_dependency,
    normalized_logistic_model,
    numeric_training_report,
    parse_jsonl_line,
    publish_numeric_training,
    save_onnx_atomic,
)
from .tabular import normalization as _normalization
from .tabular import unit_interval as _unit_metric

DESKTOP_RECIPE = "confirmed-layout-suggestion-v1"
DESKTOP_CONFIRMATION = "user-confirmed"
DESKTOP_REVOCATION_CONFIRMATION = _desktop_history.DESKTOP_REVOCATION_CONFIRMATION
revoke_desktop_history = _desktop_history.revoke_desktop_history
store_lock = _desktop_history.store_lock
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


def _history_policy() -> JsonlPolicy:
    return JsonlPolicy(
        MAX_HISTORY_BYTES,
        MAX_HISTORY_EXAMPLES,
        MAX_HISTORY_LINE_BYTES,
        "history-invalid",
        "desktop history is not a safe regular file",
        "history-too-large",
        "desktop history byte limit exceeded",
        "desktop example limit exceeded",
        "history-invalid",
        "desktop history line {line_number} is invalid",
        "desktop history line {line_number}: {detail}",
    )


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
    previous = -1

    def on_item(example: DesktopExample) -> None:
        nonlocal previous
        if example.observed_at_ms <= previous:
            raise TrainingError("observations-unordered", "desktop example times must increase")
        previous = example.observed_at_ms

    examples, digest = load_bounded_jsonl(
        path,
        _history_policy(),
        _example_from_document,
        on_item=on_item,
    )
    if not examples:
        raise TrainingError("insufficient-history", "desktop history is empty")
    return DesktopDataset(examples, digest)


def _history_line(raw: bytes, line_number: int) -> DesktopExample:
    return parse_jsonl_line(raw, line_number, _history_policy(), _example_from_document)


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
        self._minimum_macro_recall = _unit_metric(minimum_macro_recall, "minimum_macro_recall")
        self._exporter = exporter or OnnxDesktopExporter()

    def train(self, dataset: DesktopDataset, output_dir: Path | str) -> dict:
        suggestions = tuple(
            value
            for value in SUGGESTIONS
            if value in {item.suggestion for item in dataset.examples}
        )
        if len(suggestions) < 2:
            raise TrainingError("class-imbalance", "desktop history needs two suggestions")
        training, holdout = _time_split(dataset.examples)
        _require_suggestions(training, suggestions, self._minimum_class_examples, "training")
        _require_suggestions(holdout, suggestions, self._minimum_class_examples, "holdout")
        build_examples = tuple(_build_example(item, suggestions) for item in training)
        means, scales = _normalization(build_examples)
        fitted = tuple(
            fit_output(
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
        return publish_numeric_training(
            output_dir,
            model,
            self._exporter,
            report_name="desktop-training-report.json",
            report_prefix=".desktop-training-report-",
            report=lambda model_path: _report(
                dataset,
                training,
                holdout,
                suggestions,
                quality,
                model_path,
            ),
            writer=write_training_report,
        )


class OnnxDesktopExporter:
    def export(self, model: BuildAdvisorModel, destination: Path) -> None:
        onnx, tensor_proto, helper = load_onnx_dependency()
        portable = normalized_logistic_model(
            model,
            graph_name="omnitensor-desktop-layout-suggestions",
            intercept_name="intercepts",
            logit_name="logits",
            output_name="scores",
            onnx=onnx,
            tensor_proto=tensor_proto,
            helper=helper,
        )
        save_onnx_atomic(portable, destination, onnx, prefix=".desktop-model-")


def _time_split(
    examples: Sequence[DesktopExample],
) -> tuple[tuple[DesktopExample, ...], tuple[DesktopExample, ...]]:
    return chronological_split(
        examples,
        insufficient_detail="desktop history cannot be split",
    )


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
    family_fields = {
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
    }
    return numeric_training_report(
        family_fields,
        model_path,
        len(DESKTOP_FEATURES),
        digest=file_digest,
    )
