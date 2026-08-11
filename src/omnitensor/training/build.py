"""Provenance-safe local build-risk and optional-check ranking training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from ..atomicio import write_json_atomic
from ..plugins.build_ingestion import (
    DEFAULT_MAX_BUILD_RECORDS,
    BuildOutcome,
    BuildRecord,
    build_record_error,
)
from ..preparation import file_digest
from .contracts import TrainingError

BUILD_RECIPE = "metadata-risk-ranking-v1"
BUILD_PROVENANCE_CONFIRMATION = "I-confirm-build-metadata-is-approved"
BUILD_FEATURES = (
    "logChangedPathCount",
    "extensionDiversity",
    "topLevelDiversity",
    "maximumPathDepth",
    "sourceFraction",
    "testFraction",
    "documentationFraction",
    "configurationFraction",
    "dependencyFraction",
    "automationFraction",
    "assetFraction",
    "rootFileFraction",
)
MAX_HISTORY_BYTES = 64 * 1024 * 1024
MAX_HISTORY_LINE_BYTES = 1024 * 1024
MAX_CHANGED_PATHS = 256
MAX_PATH_LENGTH = 512
MAX_CHECKS = 128
MAX_CHECK_NAME = 120
DEFAULT_MIN_CLASS_EXAMPLES = 8
DEFAULT_MIN_CHECK_CLASS_EXAMPLES = 4
DEFAULT_MIN_RISK_AUC = 0.55
DEFAULT_MIN_RANKING_MRR = 0.25
_RECORD_KEYS = {
    "schemaVersion",
    "buildId",
    "outcome",
    "durationMs",
    "startedAtMs",
    "changedPaths",
    "executedChecks",
    "failedChecks",
}
_CHECK_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9 ._:/-]{0,118}[A-Za-z0-9])?")
_SOURCE_SUFFIXES = frozenset(
    {".c", ".cc", ".cpp", ".go", ".h", ".hpp", ".java", ".js", ".py", ".rs", ".ts"}
)
_DOC_SUFFIXES = frozenset({".md", ".rst", ".txt", ".adoc"})
_CONFIG_SUFFIXES = frozenset({".json", ".toml", ".yaml", ".yml", ".ini", ".cfg"})
_ASSET_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico"})
_DEPENDENCY_FILES = frozenset(
    {"package-lock.json", "pnpm-lock.yaml", "yarn.lock", "poetry.lock", "uv.lock", "cargo.lock"}
)


@dataclass(frozen=True, slots=True)
class BuildTrainingRecord:
    record: BuildRecord
    executed_checks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BuildDataset:
    records: tuple[BuildTrainingRecord, ...]
    history_sha256: str


@dataclass(frozen=True, slots=True)
class BuildExample:
    started_at_ms: int
    features: tuple[float, ...]
    build_failed: int
    executed_optional: tuple[str, ...]
    failed_optional: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BuildAdvisorModel:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]
    intercepts: tuple[float, ...]

    def predict(self, features: Sequence[float]) -> tuple[float, ...]:
        normalized = _normalized_features(features, self.means, self.scales)
        return tuple(
            _sigmoid(
                intercept
                + sum(
                    normalized[row] * self.weights[row][column]
                    for row in range(len(normalized))
                )
            )
            for column, intercept in enumerate(self.intercepts)
        )


class BuildModelExporter(Protocol):
    def export(self, model: BuildAdvisorModel, destination: Path) -> None: ...


def load_build_history(
    path: Path | str,
    *,
    accept_provenance: str,
) -> BuildDataset:
    """Load bounded exported metadata while retaining no content or logs."""
    if accept_provenance != BUILD_PROVENANCE_CONFIRMATION:
        raise TrainingError(
            "provenance-not-confirmed",
            f"review the export and pass exactly {BUILD_PROVENANCE_CONFIRMATION}",
        )
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise TrainingError("history-invalid", "build history is not a safe regular file")
    if source.stat().st_size > MAX_HISTORY_BYTES:
        raise TrainingError("history-too-large", "build history byte limit exceeded")
    digest = hashlib.sha256()
    records = []
    with source.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            digest.update(raw)
            if line_number > DEFAULT_MAX_BUILD_RECORDS:
                raise TrainingError("history-too-large", "build record limit exceeded")
            records.append(_history_line(raw, line_number))
    if not records:
        raise TrainingError("insufficient-history", "build history is empty")
    return BuildDataset(tuple(records), digest.hexdigest())


def _history_line(raw: bytes, line_number: int) -> BuildTrainingRecord:
    if not raw.strip() or len(raw) > MAX_HISTORY_LINE_BYTES:
        raise TrainingError(
            "history-invalid", f"build history line {line_number} is invalid"
        )
    try:
        return _record_from_document(json.loads(raw))
    except (UnicodeDecodeError, json.JSONDecodeError, TrainingError) as error:
        detail = error.detail if isinstance(error, TrainingError) else "invalid JSON"
        raise TrainingError(
            "history-invalid", f"build history line {line_number}: {detail}"
        ) from error


class BuildAdvisorTrainer:
    """Fit one build-risk output plus one score per optional check."""

    def __init__(
        self,
        *,
        mandatory_checks: Sequence[str],
        optional_checks: Sequence[str],
        exporter: BuildModelExporter | None = None,
        minimum_class_examples: int = DEFAULT_MIN_CLASS_EXAMPLES,
        minimum_check_class_examples: int = DEFAULT_MIN_CHECK_CLASS_EXAMPLES,
        minimum_risk_auc: float = DEFAULT_MIN_RISK_AUC,
        minimum_ranking_mrr: float = DEFAULT_MIN_RANKING_MRR,
    ) -> None:
        mandatory = _check_catalog(mandatory_checks, "mandatory")
        optional = _check_catalog(optional_checks, "optional")
        if set(mandatory) & set(optional):
            raise ValueError("mandatory and optional check catalogs overlap")
        _positive_integer(minimum_class_examples, "minimum_class_examples")
        _positive_integer(minimum_check_class_examples, "minimum_check_class_examples")
        minimum_risk_auc = _unit_interval(minimum_risk_auc, "minimum_risk_auc")
        minimum_ranking_mrr = _unit_interval(minimum_ranking_mrr, "minimum_ranking_mrr")
        self._mandatory = mandatory
        self._optional = optional
        self._exporter = exporter or OnnxBuildExporter()
        self._minimum_class_examples = minimum_class_examples
        self._minimum_check_class_examples = minimum_check_class_examples
        self._minimum_risk_auc = minimum_risk_auc
        self._minimum_ranking_mrr = minimum_ranking_mrr

    def train(self, dataset: BuildDataset, output_dir: Path | str) -> dict:
        examples, ignored = _build_examples(
            dataset.records,
            self._mandatory,
            self._optional,
        )
        training, holdout = _time_split(examples)
        _require_binary_classes(training, self._minimum_class_examples, "training build risk")
        _require_binary_classes(holdout, self._minimum_class_examples, "holdout build risk")
        means, scales = _normalization(training)
        outputs = [_fit_output(training, lambda item: item.build_failed, means, scales)]
        for check in self._optional:
            check_training = tuple(
                item for item in training if check in item.executed_optional
            )
            _require_check_classes(
                check_training,
                check,
                self._minimum_check_class_examples,
            )
            outputs.append(
                _fit_output(
                    check_training,
                    lambda item, selected=check: int(selected in item.failed_optional),
                    means,
                    scales,
                )
            )
        model = BuildAdvisorModel(
            means,
            scales,
            tuple(tuple(output[0][row] for output in outputs) for row in range(len(means))),
            tuple(output[1] for output in outputs),
        )
        risk_auc = _risk_auc(model, holdout)
        ranking_mrr, ranked_records = _ranking_mrr(model, holdout, self._optional)
        if risk_auc < self._minimum_risk_auc:
            raise TrainingError("model-not-useful", "held-out build-risk AUC is below the gate")
        if ranked_records == 0:
            raise TrainingError(
                "ranking-unmeasured", "holdout has no failed optional check to rank"
            )
        if ranking_mrr < self._minimum_ranking_mrr:
            raise TrainingError("model-not-useful", "held-out optional-check MRR is below the gate")
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        model_path = destination / "model.onnx"
        self._exporter.export(model, model_path)
        if not model_path.is_file() or model_path.stat().st_size == 0:
            raise TrainingError("export-failed", "exporter produced no portable model")
        report = _training_report(
            dataset,
            training,
            holdout,
            ignored,
            self._mandatory,
            self._optional,
            risk_auc,
            ranking_mrr,
            ranked_records,
            model_path,
        )
        write_json_atomic(
            destination / "build-training-report.json",
            report,
            prefix=".build-training-report-",
        )
        return report


class OnnxBuildExporter:
    """Export shared normalization and bounded multi-label logistic outputs."""

    def export(self, model: BuildAdvisorModel, destination: Path) -> None:
        try:
            import onnx  # noqa: PLC0415 - optional producer dependency
            from onnx import TensorProto, helper  # noqa: PLC0415
        except ImportError as error:
            raise TrainingError(
                "exporter-missing", "onnx is not installed; install the [train] extra"
            ) from error
        width = len(BUILD_FEATURES)
        outputs = len(model.intercepts)
        graph = helper.make_graph(
            [
                helper.make_node("Sub", ["input", "means"], ["centered"]),
                helper.make_node("Div", ["centered", "scales"], ["normalized"]),
                helper.make_node("MatMul", ["normalized", "weights"], ["linear"]),
                helper.make_node("Add", ["linear", "intercepts"], ["logits"]),
                helper.make_node("Sigmoid", ["logits"], ["scores"]),
            ],
            "omnitensor-build-risk-ranking",
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
                helper.make_tensor(
                    "intercepts", TensorProto.FLOAT, [outputs], model.intercepts
                ),
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
        except Exception as error:  # noqa: BLE001 - exporter diagnostics vary
            raise TrainingError("export-failed", f"ONNX validation failed: {error}") from error
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".build-model-", dir=destination.parent)
        os.close(handle)
        try:
            onnx.save_model(portable, temporary)
            os.replace(temporary, destination)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise


def _record_from_document(document: object) -> BuildTrainingRecord:
    if not isinstance(document, dict) or set(document) != _RECORD_KEYS:
        raise TrainingError("record-invalid", "build record fields are invalid")
    if type(document.get("schemaVersion")) is not int or document["schemaVersion"] != 1:
        raise TrainingError("record-invalid", "build record version is invalid")
    changed_paths = _list_field(document, "changedPaths")
    executed_checks = _list_field(document, "executedChecks")
    failed_checks = _list_field(document, "failedChecks")
    try:
        record = BuildRecord(
            document["buildId"],
            BuildOutcome(document["outcome"]),
            document["durationMs"],
            document["startedAtMs"],
            changed_paths,
            failed_checks,
        )
    except (TypeError, ValueError) as error:
        raise TrainingError("record-invalid", "build record values are invalid") from error
    detail = build_record_error(record)
    if detail:
        raise TrainingError("record-invalid", detail)
    _validate_paths(changed_paths)
    _validate_checks(executed_checks, "executed")
    _validate_checks(failed_checks, "failed")
    if not set(failed_checks) <= set(executed_checks):
        raise TrainingError("record-invalid", "failed checks were not executed")
    return BuildTrainingRecord(record, executed_checks)


def _list_field(document: dict, name: str) -> tuple:
    value = document[name]
    if not isinstance(value, list):
        raise TrainingError("record-invalid", f"{name} must be an array")
    return tuple(value)


def _validate_paths(paths: tuple[str, ...]) -> None:
    if not 1 <= len(paths) <= MAX_CHANGED_PATHS or len(paths) != len(set(paths)):
        raise TrainingError("record-invalid", "changed paths are empty, duplicated, or excessive")
    for value in paths:
        if not isinstance(value, str) or not 1 <= len(value) <= MAX_PATH_LENGTH:
            raise TrainingError("record-invalid", "changed path is invalid")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or value != path.as_posix()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise TrainingError("record-invalid", "changed path is not safe relative metadata")
        if any(ord(character) < 32 for character in value) or "\\" in value:
            raise TrainingError("record-invalid", "changed path contains unsupported characters")


def _validate_checks(checks: tuple[str, ...], label: str) -> None:
    if len(checks) > MAX_CHECKS or len(checks) != len(set(checks)):
        raise TrainingError("record-invalid", f"{label} checks are duplicated or excessive")
    if any(not isinstance(check, str) or not _CHECK_NAME.fullmatch(check) for check in checks):
        raise TrainingError("record-invalid", f"{label} check name is invalid")


def _check_catalog(checks: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(checks, (str, bytes)) or not isinstance(checks, Sequence):
        raise ValueError(f"{label} checks must be a sequence")
    selected = tuple(checks)
    if not selected:
        raise ValueError(f"{label} checks must not be empty")
    try:
        _validate_checks(selected, label)
    except TrainingError as error:
        raise ValueError(error.detail) from error
    return selected


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _unit_interval(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be in [0, 1]")
    return float(value)


def _build_examples(
    records: Sequence[BuildTrainingRecord],
    mandatory: tuple[str, ...],
    optional: tuple[str, ...],
) -> tuple[tuple[BuildExample, ...], int]:
    catalog = set(mandatory) | set(optional)
    examples = []
    ignored = 0
    previous_time = -1
    for item in records:
        if item.record.started_at_ms <= previous_time:
            raise TrainingError("observations-unordered", "build start times must increase")
        previous_time = item.record.started_at_ms
        unknown = set(item.executed_checks) - catalog
        if unknown:
            raise TrainingError(
                "check-catalog-mismatch", f"unlisted executed check: {min(unknown)}"
            )
        if not set(mandatory) <= set(item.executed_checks):
            raise TrainingError("mandatory-check-missing", "a mandatory check was not executed")
        if item.record.outcome not in {BuildOutcome.SUCCEEDED, BuildOutcome.FAILED}:
            ignored += 1
            continue
        examples.append(
            BuildExample(
                item.record.started_at_ms,
                _path_features(item.record.changed_paths),
                int(item.record.outcome is BuildOutcome.FAILED),
                tuple(check for check in optional if check in item.executed_checks),
                tuple(check for check in optional if check in item.record.failed_checks),
            )
        )
    return tuple(examples), ignored


def _path_features(paths: Sequence[str]) -> tuple[float, ...]:
    parsed = tuple(PurePosixPath(path) for path in paths)
    count = len(parsed)
    suffixes = {path.suffix.lower() for path in parsed}
    top_levels = {path.parts[0] for path in parsed}
    source = sum(path.suffix.lower() in _SOURCE_SUFFIXES for path in parsed)
    tests = sum(
        any(part.lower() in {"test", "tests", "spec", "specs"} for part in path.parts)
        for path in parsed
    )
    docs = sum(
        path.suffix.lower() in _DOC_SUFFIXES
        or "docs" in {part.lower() for part in path.parts}
        for path in parsed
    )
    config = sum(path.suffix.lower() in _CONFIG_SUFFIXES for path in parsed)
    dependency = sum(path.name.lower() in _DEPENDENCY_FILES for path in parsed)
    automation = sum(path.parts[0].lower() in {".github", ".gitlab", "ci"} for path in parsed)
    assets = sum(path.suffix.lower() in _ASSET_SUFFIXES for path in parsed)
    roots = sum(len(path.parts) == 1 for path in parsed)
    return (
        math.log1p(count),
        float(len(suffixes)),
        float(len(top_levels)),
        float(max(len(path.parts) for path in parsed)),
        source / count,
        tests / count,
        docs / count,
        config / count,
        dependency / count,
        automation / count,
        assets / count,
        roots / count,
    )


def _time_split(
    examples: Sequence[BuildExample],
) -> tuple[tuple[BuildExample, ...], tuple[BuildExample, ...]]:
    if len(examples) < 2:
        raise TrainingError("insufficient-history", "build history needs at least two outcomes")
    split = min(len(examples) - 1, max(1, int(len(examples) * 0.8)))
    return tuple(examples[:split]), tuple(examples[split:])


def _require_binary_classes(
    examples: Sequence[BuildExample], minimum: int, label: str
) -> None:
    positives = sum(item.build_failed for item in examples)
    negatives = len(examples) - positives
    if positives < minimum or negatives < minimum:
        raise TrainingError(
            "class-imbalance",
            f"{label} needs at least {minimum} positive and negative examples",
        )


def _require_check_classes(
    examples: Sequence[BuildExample], check: str, minimum: int
) -> None:
    positives = sum(check in item.failed_optional for item in examples)
    negatives = len(examples) - positives
    if positives < minimum or negatives < minimum:
        raise TrainingError(
            "class-imbalance",
            f"optional check {check} needs at least {minimum} failed and passed examples",
        )


def _normalization(
    examples: Sequence[BuildExample],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    width = len(BUILD_FEATURES)
    means = tuple(
        sum(item.features[index] for item in examples) / len(examples)
        for index in range(width)
    )
    scales = tuple(
        max(
            1e-9,
            math.sqrt(
                sum((item.features[index] - means[index]) ** 2 for item in examples)
                / len(examples)
            ),
        )
        for index in range(width)
    )
    return means, scales


def _fit_output(
    examples: Sequence[BuildExample],
    label_of,
    means: tuple[float, ...],
    scales: tuple[float, ...],
) -> tuple[tuple[float, ...], float]:
    labels = tuple(label_of(item) for item in examples)
    positives = sum(labels)
    positive_weight = (len(labels) - positives) / positives
    total_weight = 2 * (len(labels) - positives)
    weights = [0.0] * len(BUILD_FEATURES)
    intercept = 0.0
    for step in range(240):
        gradients = [0.0] * len(weights)
        intercept_gradient = 0.0
        for item, label in zip(examples, labels, strict=True):
            features = _normalized_features(item.features, means, scales)
            probability = _sigmoid(
                intercept
                + sum(
                    weight * value
                    for weight, value in zip(weights, features, strict=True)
                )
            )
            sample_weight = positive_weight if label else 1.0
            error = sample_weight * (probability - label)
            intercept_gradient += error
            for index, value in enumerate(features):
                gradients[index] += error * value
        rate = 0.2 / math.sqrt(step + 1)
        intercept -= rate * intercept_gradient / total_weight
        for index in range(len(weights)):
            weights[index] -= rate * (gradients[index] / total_weight + 1e-4 * weights[index])
    return tuple(weights), intercept


def _normalized_features(
    features: Sequence[float], means: Sequence[float], scales: Sequence[float]
) -> tuple[float, ...]:
    if len(features) != len(BUILD_FEATURES):
        raise TrainingError("features-invalid", "build feature width disagrees")
    values = []
    for value, mean, scale in zip(features, means, scales, strict=True):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TrainingError("features-invalid", "build features must be numbers")
        if not math.isfinite(value):
            raise TrainingError("features-invalid", "build features must be finite")
        values.append((float(value) - mean) / scale)
    return tuple(values)


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-min(value, 700.0))
        return 1 / (1 + inverse)
    exponential = math.exp(max(value, -700.0))
    return exponential / (1 + exponential)


def _risk_auc(model: BuildAdvisorModel, examples: Sequence[BuildExample]) -> float:
    return _auc((model.predict(item.features)[0], item.build_failed) for item in examples)


def _ranking_mrr(
    model: BuildAdvisorModel,
    examples: Sequence[BuildExample],
    optional: tuple[str, ...],
) -> tuple[float, int]:
    reciprocal_ranks = []
    for item in examples:
        if not item.failed_optional:
            continue
        scores = model.predict(item.features)[1:]
        ranked = sorted(
            (
                (scores[index], check)
                for index, check in enumerate(optional)
                if check in item.executed_optional
            ),
            key=lambda value: (-value[0], value[1]),
        )
        first_failed = next(
            index
            for index, (_score, check) in enumerate(ranked, start=1)
            if check in item.failed_optional
        )
        reciprocal_ranks.append(1 / first_failed)
    if not reciprocal_ranks:
        return 0.0, 0
    return sum(reciprocal_ranks) / len(reciprocal_ranks), len(reciprocal_ranks)


def _auc(scored: Iterable[tuple[float, int]]) -> float:
    ordered = sorted(scored)
    positives = sum(label for _score, label in ordered)
    negatives = len(ordered) - positives
    if positives == 0 or negatives == 0:
        raise TrainingError("class-imbalance", "AUC requires both build outcomes")
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        rank_sum += average_rank * sum(label for _score, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _training_report(
    dataset: BuildDataset,
    training: Sequence[BuildExample],
    holdout: Sequence[BuildExample],
    ignored: int,
    mandatory: tuple[str, ...],
    optional: tuple[str, ...],
    risk_auc: float,
    ranking_mrr: float,
    ranked_records: int,
    model_path: Path,
) -> dict:
    return {
        "version": 1,
        "recipe": BUILD_RECIPE,
        "dataset": {
            "historySha256": dataset.history_sha256,
            "recordsRead": len(dataset.records),
            "ignoredCancelledOrUnknown": ignored,
            "pathsPersisted": False,
            "buildIdsPersisted": False,
            "logsOrSourceContentIngested": False,
            "provenanceConfirmed": True,
        },
        "split": {
            "kind": "chronological",
            "training": len(training),
            "holdout": len(holdout),
            "holdoutFromMs": holdout[0].started_at_ms,
        },
        "features": list(BUILD_FEATURES),
        "outputs": {
            "buildFailureRiskIndex": 0,
            "optionalCheckRisk": [
                {"check": check, "index": index}
                for index, check in enumerate(optional, start=1)
            ],
            "mandatoryChecks": list(mandatory),
            "mandatoryChecksAlwaysIncluded": True,
            "modelMayOnlyReorderOptionalChecks": True,
        },
        "featureContract": {
            "version": 1,
            "recipe": BUILD_RECIPE,
            "featureNames": list(BUILD_FEATURES),
            "pathContentRead": False,
        },
        "resultContract": {
            "version": 1,
            "kind": "build-advice-scores",
            "shape": [1, 1 + len(optional)],
            "mandatorySource": "deterministic-report-catalog",
            "mandatoryOmissionAllowed": False,
        },
        "quality": {
            "holdoutBuildRiskAuc": risk_auc,
            "holdoutOptionalCheckMrr": ranking_mrr,
            "rankedHoldoutRecords": ranked_records,
        },
        "model": {
            "format": "onnx",
            "filename": "model.onnx",
            "sha256": file_digest(model_path),
        },
        "tensorContract": {
            "inputs": [{"shape": [1, len(BUILD_FEATURES)], "dtype": "float32", "layout": "NC"}]
        },
        "outputContract": {"kind": "raw"},
        "targets": {"tpu": "uncompiled", "npu": "uncompiled", "gpu": "uncompiled"},
    }
