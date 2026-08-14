"""Bounded Backblaze SMART intake and device-neutral failure-risk training.

The source is deliberately a local directory of official daily CSV files. The
service never downloads training data, serial numbers never enter the emitted
model, and a row is labelled only from a later failure observation. This makes
the portable ONNX graph reusable across accelerator targets without claiming
that a Backblaze fleet model is already suitable for an operator's devices.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Protocol

from omnitensor.preparation import file_digest

from .contracts import TrainingError, write_training_report
from .tabular import (
    checked_model,
    fit_balanced_logistic,
    load_onnx_dependency,
    numeric_training_report,
    population_normalization,
    publish_numeric_training,
    ranked_auc,
    require_binary_class_floor,
    save_onnx_atomic,
)
from .tabular import stable_sigmoid as _sigmoid
from .tabular.fit import unchecked_normalized_features

BACKBLAZE_TERMS = "Backblaze-Drive-Stats"
BACKBLAZE_CITATION = "Backblaze Drive Stats"
BACKBLAZE_SOURCE_URI = "https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data"
STORAGE_RECIPE = "backblaze-smart-risk-v1"
FEATURE_COLUMNS = (
    "smart_5_raw",
    "smart_9_raw",
    "smart_187_raw",
    "smart_188_raw",
    "smart_197_raw",
    "smart_198_raw",
)
REQUIRED_COLUMNS = frozenset({"date", "serial_number", "failure", *FEATURE_COLUMNS})
DEFAULT_HORIZON_DAYS = 7
DEFAULT_MAX_ROWS = 50_000_000
DEFAULT_MAX_SAMPLES = 500_000
DEFAULT_MAX_DEVICES = 500_000
DEFAULT_NEGATIVE_KEEP_MODULUS = 64
MAX_CSV_FILES = 400
MAX_CSV_BYTES = 2 * 1024 * 1024 * 1024
MAX_SMART_VALUE = 2**53
MAX_SERIAL_LENGTH = 200


@dataclass(frozen=True, slots=True)
class StorageExample:
    observed_on: date
    features: tuple[float, ...]
    failed_within_horizon: int


@dataclass(frozen=True, slots=True)
class StorageDataset:
    examples: tuple[StorageExample, ...]
    rows_read: int
    rows_missing_features: int
    schema_sha256: str
    corpus_sha256: str
    first_date: date
    last_date: date
    horizon_days: int


@dataclass(frozen=True, slots=True)
class StorageRiskModel:
    weights: tuple[float, ...]
    intercept: float
    means: tuple[float, ...]
    scales: tuple[float, ...]

    def predict(self, features: Sequence[float]) -> float:
        if len(features) != len(self.weights):
            raise TrainingError("features-invalid", "storage feature width disagrees")
        standardized = []
        for value, mean, scale in zip(features, self.means, self.scales, strict=True):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TrainingError("features-invalid", "storage features must be numbers")
            if not math.isfinite(value):
                raise TrainingError("features-invalid", "storage features must be finite")
            standardized.append((float(value) - mean) / scale)
        score = self.intercept + sum(
            weight * value for weight, value in zip(self.weights, standardized, strict=True)
        )
        return _sigmoid(score)


class StorageModelExporter(Protocol):
    def export(self, model: StorageRiskModel, destination: Path) -> None: ...


class BackblazeDatasetBuilder:
    """Stream official daily CSVs into bounded, future-labelled examples."""

    def __init__(
        self,
        *,
        accept_terms: str,
        horizon_days: int = DEFAULT_HORIZON_DAYS,
        max_rows: int = DEFAULT_MAX_ROWS,
        max_samples: int = DEFAULT_MAX_SAMPLES,
        max_devices: int = DEFAULT_MAX_DEVICES,
        negative_keep_modulus: int = DEFAULT_NEGATIVE_KEEP_MODULUS,
    ) -> None:
        if accept_terms != BACKBLAZE_TERMS:
            raise TrainingError(
                "dataset-terms-not-accepted",
                f"review the Backblaze data terms and pass {BACKBLAZE_TERMS}",
            )
        values = (horizon_days, max_rows, max_samples, max_devices, negative_keep_modulus)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("storage intake bounds must be integers")
        if not 1 <= horizon_days <= 90:
            raise ValueError("storage failure horizon must be between 1 and 90 days")
        if min(max_rows, max_samples, max_devices) < 1:
            raise ValueError("storage intake bounds must be positive")
        if not 1 <= negative_keep_modulus <= 1024:
            raise ValueError("negative sampling modulus must be between 1 and 1024")
        self._horizon_days = horizon_days
        self._max_rows = max_rows
        self._max_samples = max_samples
        self._max_devices = max_devices
        self._negative_keep_modulus = negative_keep_modulus

    def build(self, root: Path | str) -> StorageDataset:
        files = _csv_files(Path(root))
        pending: dict[str, deque[tuple[date, tuple[float, ...]]]] = {}
        examples: list[StorageExample] = []
        schema_digest = hashlib.sha256()
        rows_read = 0
        missing = 0
        first_date = _file_date(files[0])
        last_date = first_date
        for path in files:
            file_date = _file_date(path)
            last_date = file_date
            read, absent, header = self._read_daily(path, file_date, pending, examples, rows_read)
            rows_read += read
            missing += absent
            schema_digest.update(json.dumps(header, separators=(",", ":")).encode())
            schema_digest.update(b"\n")
        if not examples:
            raise TrainingError("insufficient-history", "dataset produced no labelled examples")
        corpus = hashlib.sha256()
        for example in examples:
            corpus.update(
                json.dumps(
                    {
                        "date": example.observed_on.isoformat(),
                        "features": example.features,
                        "failure": example.failed_within_horizon,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            )
            corpus.update(b"\n")
        return StorageDataset(
            tuple(examples),
            rows_read,
            missing,
            schema_digest.hexdigest(),
            corpus.hexdigest(),
            first_date,
            last_date,
            self._horizon_days,
        )

    def _read_daily(
        self,
        path: Path,
        file_date: date,
        pending: dict[str, deque[tuple[date, tuple[float, ...]]]],
        examples: list[StorageExample],
        prior_rows: int,
    ) -> tuple[int, int, tuple[str, ...]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = _validated_header(reader.fieldnames)
            seen_serials: set[str] = set()
            missing = 0
            for count, row in enumerate(reader, start=1):
                if prior_rows + count > self._max_rows:
                    raise TrainingError("dataset-too-large", "CSV row limit exceeded")
                missing += self._consume_row(
                    row, path.name, file_date, seen_serials, pending, examples
                )
        return len(seen_serials), missing, header

    def _consume_row(
        self,
        row: dict[str, str],
        filename: str,
        file_date: date,
        seen_serials: set[str],
        pending: dict[str, deque[tuple[date, tuple[float, ...]]]],
        examples: list[StorageExample],
    ) -> int:
        serial = _serial(row.get("serial_number"))
        if serial in seen_serials:
            raise TrainingError("row-duplicate", f"duplicate serial in {filename}: {serial}")
        seen_serials.add(serial)
        if row.get("date") != file_date.isoformat():
            raise TrainingError("date-mismatch", f"row date disagrees with {filename}")
        features = _features(row)
        failed = _failure(row.get("failure"))
        history = pending.get(serial)
        if history is None:
            if len(pending) >= self._max_devices:
                raise TrainingError("dataset-too-large", "active device limit exceeded")
            history = pending[serial] = deque()
        self._release_negatives(serial, file_date, history, examples)
        if failed:
            for observed_on, previous in history:
                _append_example(
                    examples,
                    StorageExample(observed_on, previous, 1),
                    self._max_samples,
                )
            pending.pop(serial, None)
            return 0
        if features is None:
            return 1
        history.append((file_date, features))
        return 0

    def _release_negatives(
        self,
        serial: str,
        current: date,
        history: deque[tuple[date, tuple[float, ...]]],
        examples: list[StorageExample],
    ) -> None:
        while history and (current - history[0][0]).days > self._horizon_days:
            observed_on, features = history.popleft()
            if _keep_negative(serial, observed_on, self._negative_keep_modulus):
                _append_example(
                    examples,
                    StorageExample(observed_on, features, 0),
                    self._max_samples,
                )


class StorageTrainer:
    """Fit and gate one portable risk source without compiling any target lane."""

    def __init__(
        self,
        exporter: StorageModelExporter | None = None,
        *,
        minimum_positive_examples: int = 25,
        minimum_auc: float = 0.6,
    ) -> None:
        if (
            isinstance(minimum_positive_examples, bool)
            or not isinstance(minimum_positive_examples, int)
            or minimum_positive_examples < 1
        ):
            raise ValueError("minimum positive examples must be a positive integer")
        if not math.isfinite(minimum_auc) or not 0.5 < minimum_auc <= 1:
            raise ValueError("minimum AUC must be in (0.5, 1]")
        self._exporter = exporter or OnnxStorageExporter()
        self._minimum_positive_examples = minimum_positive_examples
        self._minimum_auc = minimum_auc

    def train(self, dataset: StorageDataset, output_dir: Path | str) -> dict:
        train, holdout, split_date = _time_split(dataset.examples, dataset.horizon_days)
        _require_classes(train, self._minimum_positive_examples, "training")
        _require_classes(holdout, self._minimum_positive_examples, "holdout")
        model = _fit_logistic(train)
        quality = _quality(model, holdout)
        if quality["auc"] < self._minimum_auc:
            raise TrainingError(
                "model-not-useful", "held-out storage risk AUC does not beat the required gate"
            )
        return publish_numeric_training(
            output_dir,
            model,
            self._exporter,
            report_name="storage-training-report.json",
            report_prefix=".storage-training-report-",
            report=lambda model_path: _training_report(
                dataset,
                train,
                holdout,
                split_date,
                quality,
                model_path,
            ),
            writer=write_training_report,
        )


class OnnxStorageExporter:
    """Export normalization-folded MatMul/Add/Sigmoid portable inference."""

    def export(self, model: StorageRiskModel, destination: Path) -> None:
        onnx, tensor_proto, helper = load_onnx_dependency()
        raw_weights = [
            weight / scale for weight, scale in zip(model.weights, model.scales, strict=True)
        ]
        raw_bias = model.intercept - sum(
            weight * mean for weight, mean in zip(raw_weights, model.means, strict=True)
        )
        width = len(raw_weights)
        weights = helper.make_tensor("weights", tensor_proto.FLOAT, [width, 1], raw_weights)
        bias = helper.make_tensor("bias", tensor_proto.FLOAT, [1], [raw_bias])
        graph = helper.make_graph(
            [
                helper.make_node("MatMul", ["input", "weights"], ["linear"]),
                helper.make_node("Add", ["linear", "bias"], ["logit"]),
                helper.make_node("Sigmoid", ["logit"], ["risk"]),
            ],
            "omnitensor-storage-risk",
            [helper.make_tensor_value_info("input", tensor_proto.FLOAT, [1, width])],
            [helper.make_tensor_value_info("risk", tensor_proto.FLOAT, [1, 1])],
            [weights, bias],
        )
        portable = checked_model(graph, onnx, helper)
        save_onnx_atomic(portable, destination, onnx, prefix=".storage-model-")


def _training_report(
    dataset: StorageDataset,
    training: Sequence[StorageExample],
    holdout: Sequence[StorageExample],
    split_date: date,
    quality: dict,
    model_path: Path,
) -> dict:
    family_fields = {
        "version": 1,
        "recipe": STORAGE_RECIPE,
        "dataset": {
            "source": BACKBLAZE_SOURCE_URI,
            "citation": BACKBLAZE_CITATION,
            "terms": BACKBLAZE_TERMS,
            "termsAccepted": True,
            "rowsRead": dataset.rows_read,
            "rowsMissingFeatures": dataset.rows_missing_features,
            "schemaSha256": dataset.schema_sha256,
            "corpusSha256": dataset.corpus_sha256,
            "firstDate": dataset.first_date.isoformat(),
            "lastDate": dataset.last_date.isoformat(),
        },
        "split": {
            "kind": "time-purged",
            "holdoutFrom": split_date.isoformat(),
            "purgeDays": dataset.horizon_days,
        },
        "features": list(FEATURE_COLUMNS),
        "label": {"kind": "future-failure", "horizonDays": dataset.horizon_days},
        "samples": {"training": len(training), "holdout": len(holdout)},
        "quality": quality,
    }
    return numeric_training_report(
        family_fields,
        model_path,
        len(FEATURE_COLUMNS),
        digest=file_digest,
    )


def _csv_files(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        raise TrainingError("dataset-invalid", "Backblaze CSV root is not a directory")
    files = tuple(sorted(root.glob("*.csv")))
    if not files:
        raise TrainingError("dataset-invalid", "Backblaze CSV root contains no daily files")
    if len(files) > MAX_CSV_FILES:
        raise TrainingError("dataset-too-large", "too many daily CSV files")
    for path in files:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CSV_BYTES:
            raise TrainingError("dataset-invalid", f"unsafe daily CSV: {path.name}")
    return files


def _file_date(path: Path) -> date:
    try:
        parsed = date.fromisoformat(path.stem)
    except ValueError as error:
        raise TrainingError("dataset-invalid", f"daily CSV name is invalid: {path.name}") from error
    if path.name != f"{parsed.isoformat()}.csv":
        raise TrainingError("dataset-invalid", f"daily CSV name is invalid: {path.name}")
    return parsed


def _validated_header(fieldnames: list[str] | None) -> tuple[str, ...]:
    if fieldnames is None or len(fieldnames) != len(set(fieldnames)):
        raise TrainingError("schema-drift", "CSV header is absent or duplicated")
    missing = sorted(REQUIRED_COLUMNS - set(fieldnames))
    if missing:
        raise TrainingError("schema-drift", f"CSV header has no {missing[0]}")
    return tuple(fieldnames)


def _serial(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_SERIAL_LENGTH:
        raise TrainingError("row-invalid", "serial number is invalid")
    return value


def _failure(value: object) -> bool:
    if value not in {"0", "1"}:
        raise TrainingError("row-invalid", "failure must be 0 or 1")
    return value == "1"


def _features(row: dict[str, str]) -> tuple[float, ...] | None:
    if any(row.get(name, "").strip() == "" for name in FEATURE_COLUMNS):
        return None
    values = []
    for name in FEATURE_COLUMNS:
        try:
            exact = Decimal(row[name])
        except InvalidOperation as error:
            raise TrainingError("row-invalid", f"{name} must be numeric") from error
        if not exact.is_finite() or not 0 <= exact <= MAX_SMART_VALUE:
            raise TrainingError("row-invalid", f"{name} is outside the supported range")
        values.append(float(exact))
    return tuple(values)


def _keep_negative(serial: str, observed_on: date, modulus: int) -> bool:
    digest = hashlib.sha256(f"{serial}\0{observed_on.isoformat()}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % modulus == 0


def _append_example(examples: list[StorageExample], example: StorageExample, maximum: int) -> None:
    if len(examples) >= maximum:
        raise TrainingError("dataset-too-large", "labelled sample limit exceeded")
    examples.append(example)


def _time_split(
    examples: Sequence[StorageExample], horizon_days: int
) -> tuple[tuple[StorageExample, ...], tuple[StorageExample, ...], date]:
    dates = sorted({example.observed_on for example in examples})
    if len(dates) < 3:
        raise TrainingError("insufficient-history", "at least three labelled dates are required")
    split_index = min(len(dates) - 1, max(1, int(len(dates) * 0.8)))
    split_date = dates[split_index]
    train_before = split_date - timedelta(days=horizon_days)
    train = tuple(example for example in examples if example.observed_on < train_before)
    holdout = tuple(example for example in examples if example.observed_on >= split_date)
    if not train or not holdout:
        raise TrainingError("insufficient-history", "time-purged split is empty")
    return train, holdout, split_date


def _require_classes(examples: Sequence[StorageExample], minimum_positive: int, label: str) -> None:
    require_binary_class_floor(
        examples,
        lambda example: example.failed_within_horizon,
        minimum_positive,
        f"{label} split needs at least {minimum_positive} positive and negative examples",
    )


def _fit_logistic(examples: Sequence[StorageExample]) -> StorageRiskModel:
    means, scales = population_normalization(
        examples,
        lambda example: example.features,
        scale_floor=1.0,
        width=len(FEATURE_COLUMNS),
    )
    weights, intercept = fit_balanced_logistic(
        examples,
        lambda example: example.failed_within_horizon,
        lambda example: example.features,
        means,
        scales,
        iterations=200,
        normalize=unchecked_normalized_features,
        sigmoid=_sigmoid,
    )
    return StorageRiskModel(weights, intercept, means, scales)


def _quality(model: StorageRiskModel, examples: Sequence[StorageExample]) -> dict:
    scored = [
        (model.predict(example.features), example.failed_within_horizon) for example in examples
    ]
    positives = sum(label for _score, label in scored)
    negatives = len(scored) - positives
    predicted = [(score >= 0.5, label) for score, label in scored]
    true_positive = sum(is_positive and label == 1 for is_positive, label in predicted)
    false_positive = sum(is_positive and label == 0 for is_positive, label in predicted)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / positives
    return {
        "auc": _auc(scored),
        "precisionAtHalf": round(precision, 6),
        "recallAtHalf": round(recall, 6),
        "positiveExamples": positives,
        "negativeExamples": negatives,
        "identityFeatures": False,
        "futureLabelOnly": True,
    }


def _auc(scored: Iterable[tuple[float, int]]) -> float:
    return ranked_auc(scored, empty_class_detail=None)
