"""Provenance-safe local build-risk and optional-check ranking training."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from ..preparation import file_digest
from ..telemetry_types import (
    DEFAULT_MAX_BUILD_RECORDS,
    BuildOutcome,
    BuildRecord,
    build_record_error,
)
from .contracts import TrainingError, write_training_report
from .tabular import (
    BuildAdvisorModel,
    BuildExample,
    JsonlPolicy,
    chronological_split,
    fit_balanced_logistic,
    load_bounded_jsonl,
    load_onnx_dependency,
    normalized_logistic_model,
    numeric_training_report,
    parse_jsonl_line,
    publish_numeric_training,
    require_binary_class_floor,
    save_onnx_atomic,
)
from .tabular import binary_auc as _auc
from .tabular import normalization as _normalization
from .tabular import normalized_features as _normalized_features  # noqa: F401
from .tabular import stable_sigmoid as _sigmoid  # noqa: F401
from .tabular import unit_interval as _unit_interval


def _model_normalized_features(features, means, scales):
    return _normalized_features(features, means, scales)


def _model_probability(value):
    return _sigmoid(value)


BuildAdvisorModel._normalize = staticmethod(_model_normalized_features)
BuildAdvisorModel._probability = staticmethod(_model_probability)

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


def _history_policy() -> JsonlPolicy:
    return JsonlPolicy(
        MAX_HISTORY_BYTES,
        DEFAULT_MAX_BUILD_RECORDS,
        MAX_HISTORY_LINE_BYTES,
        "history-invalid",
        "build history is not a safe regular file",
        "history-too-large",
        "build history byte limit exceeded",
        "build record limit exceeded",
        "history-invalid",
        "build history line {line_number} is invalid",
        "build history line {line_number}: {detail}",
    )


@dataclass(frozen=True, slots=True)
class BuildTrainingRecord:
    record: BuildRecord
    executed_checks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BuildDataset:
    records: tuple[BuildTrainingRecord, ...]
    history_sha256: str


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
    records, digest = load_bounded_jsonl(path, _history_policy(), _record_from_document)
    if not records:
        raise TrainingError("insufficient-history", "build history is empty")
    return BuildDataset(records, digest)


def _history_line(raw: bytes, line_number: int) -> BuildTrainingRecord:
    return parse_jsonl_line(raw, line_number, _history_policy(), _record_from_document)


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
            check_training = tuple(item for item in training if check in item.executed_optional)
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
        return publish_numeric_training(
            output_dir,
            model,
            self._exporter,
            report_name="build-training-report.json",
            report_prefix=".build-training-report-",
            report=lambda model_path: _training_report(
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
            ),
            writer=write_training_report,
        )


class OnnxBuildExporter:
    """Export shared normalization and bounded multi-label logistic outputs."""

    def export(self, model: BuildAdvisorModel, destination: Path) -> None:
        onnx, tensor_proto, helper = load_onnx_dependency()
        portable = normalized_logistic_model(
            model,
            graph_name="omnitensor-build-risk-ranking",
            intercept_name="intercepts",
            logit_name="logits",
            output_name="scores",
            onnx=onnx,
            tensor_proto=tensor_proto,
            helper=helper,
            input_width=len(BUILD_FEATURES),
        )
        save_onnx_atomic(portable, destination, onnx, prefix=".build-model-")


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
        path.suffix.lower() in _DOC_SUFFIXES or "docs" in {part.lower() for part in path.parts}
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
    return chronological_split(
        examples,
        insufficient_detail="build history needs at least two outcomes",
    )


def _require_binary_classes(examples: Sequence[BuildExample], minimum: int, label: str) -> None:
    require_binary_class_floor(
        examples,
        lambda item: item.build_failed,
        minimum,
        f"{label} needs at least {minimum} positive and negative examples",
    )


def _require_check_classes(examples: Sequence[BuildExample], check: str, minimum: int) -> None:
    positives = sum(check in item.failed_optional for item in examples)
    negatives = len(examples) - positives
    if positives < minimum or negatives < minimum:
        raise TrainingError(
            "class-imbalance",
            f"optional check {check} needs at least {minimum} failed and passed examples",
        )


def _fit_output(
    examples: Sequence[BuildExample],
    label_of,
    means: tuple[float, ...],
    scales: tuple[float, ...],
) -> tuple[tuple[float, ...], float]:
    return fit_balanced_logistic(
        examples,
        label_of,
        lambda item: item.features,
        means,
        scales,
        iterations=240,
        normalize=_normalized_features,
        sigmoid=_sigmoid,
    )


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
    family_fields = {
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
                {"check": check, "index": index} for index, check in enumerate(optional, start=1)
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
    }
    return numeric_training_report(
        family_fields,
        model_path,
        len(BUILD_FEATURES),
        digest=file_digest,
    )
