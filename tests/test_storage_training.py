from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import date, timedelta
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.training.cli import storage_train_main
from omnitensor.training.contracts import TrainingError
from omnitensor.training.storage import (
    BACKBLAZE_CITATION,
    BACKBLAZE_SOURCE_URI,
    BACKBLAZE_TERMS,
    DEFAULT_HORIZON_DAYS,
    FEATURE_COLUMNS,
    MAX_SERIAL_LENGTH,
    MAX_SMART_VALUE,
    BackblazeDatasetBuilder,
    OnnxStorageExporter,
    StorageDataset,
    StorageExample,
    StorageRiskModel,
    StorageTrainer,
    _append_example,
    _auc,
    _csv_files,
    _failure,
    _features,
    _file_date,
    _fit_logistic,
    _keep_negative,
    _quality,
    _require_classes,
    _serial,
    _sigmoid,
    _time_split,
    _validated_header,
)

HEADER = ("date", "serial_number", "model", "capacity_bytes", "failure", *FEATURE_COLUMNS)


class FakeExporter:
    def __init__(self, content: bytes = b"portable storage risk") -> None:
        self.content = content
        self.model = None

    def export(self, model, destination) -> None:
        self.model = model
        Path(destination).write_bytes(self.content)


def write_day(
    root: Path,
    observed_on: date,
    rows: list[dict[str, str]],
    *,
    header: tuple[str, ...] = HEADER,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{observed_on.isoformat()}.csv"
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return destination


def row(
    observed_on: date,
    serial: str,
    *,
    failed: bool = False,
    value: float = 0,
) -> dict[str, str]:
    document = {
        "date": observed_on.isoformat(),
        "serial_number": serial,
        "model": "not-a-feature",
        "capacity_bytes": "1000000000000",
        "failure": "1" if failed else "0",
    }
    document.update({name: str(value + index) for index, name in enumerate(FEATURE_COLUMNS)})
    return document


def build_fleet(root: Path) -> None:
    start = date(2026, 1, 1)
    failure_days = {
        "failed-early-a": 12,
        "failed-early-b": 12,
        "failed-middle-a": 22,
        "failed-middle-b": 22,
        "failed-late-a": 40,
        "failed-late-b": 40,
        "failed-last-a": 48,
        "failed-last-b": 48,
    }
    for offset in range(50):
        observed_on = start + timedelta(days=offset)
        rows = [row(observed_on, serial) for serial in ("healthy-a", "healthy-b", "healthy-c")]
        for serial, failure_day in failure_days.items():
            if offset <= failure_day:
                imminent = 0 < failure_day - offset <= DEFAULT_HORIZON_DAYS
                rows.append(
                    row(
                        observed_on,
                        serial,
                        failed=offset == failure_day,
                        value=100 if imminent else 0,
                    )
                )
        write_day(root, observed_on, rows)


def builder(**changes) -> BackblazeDatasetBuilder:
    return BackblazeDatasetBuilder(
        accept_terms=BACKBLAZE_TERMS,
        negative_keep_modulus=1,
        **changes,
    )


def example(day: int, failed: int, value: float = 0) -> StorageExample:
    return StorageExample(
        date(2026, 1, 1) + timedelta(days=day),
        tuple(value + index for index in range(len(FEATURE_COLUMNS))),
        failed,
    )


def dataset(examples: tuple[StorageExample, ...], *, horizon: int = 7) -> StorageDataset:
    return StorageDataset(
        examples,
        len(examples),
        0,
        "a" * 64,
        "b" * 64,
        min(item.observed_on for item in examples),
        max(item.observed_on for item in examples),
        horizon,
    )


def error_code(call) -> str:
    with pytest.raises(TrainingError) as raised:
        call()
    return raised.value.code


def test_backblaze_fleet_build_and_portable_train_are_identity_free(tmp_path):
    build_fleet(tmp_path / "csv")
    source = builder().build(tmp_path / "csv")
    report = StorageTrainer(minimum_positive_examples=2, minimum_auc=0.55).train(
        source, tmp_path / "fit"
    )

    assert source.rows_read == 402
    assert len(source.examples) == 370
    assert source.horizon_days == 7
    expected_schema = hashlib.sha256()
    for _index in range(50):
        expected_schema.update(json.dumps(HEADER, separators=(",", ":")).encode())
        expected_schema.update(b"\n")
    expected_corpus = hashlib.sha256()
    for item in source.examples:
        expected_corpus.update(
            json.dumps(
                {
                    "date": item.observed_on.isoformat(),
                    "features": item.features,
                    "failure": item.failed_within_horizon,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        expected_corpus.update(b"\n")
    assert source.schema_sha256 == expected_schema.hexdigest()
    assert source.corpus_sha256 == expected_corpus.hexdigest()
    assert report["dataset"] == {
        "source": BACKBLAZE_SOURCE_URI,
        "citation": BACKBLAZE_CITATION,
        "terms": BACKBLAZE_TERMS,
        "termsAccepted": True,
        "rowsRead": source.rows_read,
        "rowsMissingFeatures": 0,
        "schemaSha256": source.schema_sha256,
        "corpusSha256": source.corpus_sha256,
        "firstDate": "2026-01-01",
        "lastDate": "2026-02-19",
    }
    assert report["quality"]["auc"] >= 0.55
    assert report["quality"]["identityFeatures"] is False
    assert report["quality"]["futureLabelOnly"] is True
    assert set(report) == {
        "version",
        "recipe",
        "dataset",
        "split",
        "features",
        "label",
        "samples",
        "quality",
        "model",
        "tensorContract",
        "outputContract",
        "targets",
    }
    assert report["version"] == 1
    assert report["recipe"] == "backblaze-smart-risk-v1"
    assert report["split"] == {
        "kind": "time-purged",
        "holdoutFrom": "2026-02-08",
        "purgeDays": 7,
    }
    assert report["features"] == list(FEATURE_COLUMNS)
    assert report["label"] == {"kind": "future-failure", "horizonDays": 7}
    assert set(report["model"]) == {"format", "filename", "sha256"}
    assert report["model"]["format"] == "onnx"
    assert report["model"]["filename"] == "model.onnx"
    assert report["targets"] == {"tpu": "uncompiled", "npu": "uncompiled", "gpu": "uncompiled"}
    assert report["tensorContract"]["inputs"] == [
        {"shape": [1, 6], "dtype": "float32", "layout": "NC"}
    ]
    assert report["outputContract"] == {"kind": "raw"}
    serialized = (tmp_path / "fit/storage-training-report.json").read_text()
    assert json.loads(serialized) == report
    assert "failed-early" not in serialized
    assert "not-a-feature" not in serialized

    onnx = pytest.importorskip("onnx")
    graph = onnx.load(tmp_path / "fit/model.onnx").graph
    assert [node.op_type for node in graph.node] == ["MatMul", "Add", "Sigmoid"]
    input_shape = [dimension.dim_value for dimension in graph.input[0].type.tensor_type.shape.dim]
    output_shape = [dimension.dim_value for dimension in graph.output[0].type.tensor_type.shape.dim]
    assert input_shape == [1, 6]
    assert output_shape == [1, 1]


def test_future_failure_labels_only_prior_horizon_and_excludes_failure_row(tmp_path):
    start = date(2026, 3, 1)
    for offset in range(12):
        observed = start + timedelta(days=offset)
        write_day(
            tmp_path,
            observed,
            [
                row(observed, "failing", failed=offset == 10, value=offset),
                row(observed, "healthy", value=offset),
            ],
        )
    result = builder().build(tmp_path)
    positives = [item for item in result.examples if item.failed_within_horizon]
    negatives = [item for item in result.examples if not item.failed_within_horizon]

    assert [item.observed_on for item in positives] == [
        start + timedelta(days=offset) for offset in range(3, 10)
    ]
    assert all(item.observed_on < start + timedelta(days=10) for item in positives)
    assert start in [item.observed_on for item in negatives]


def test_missing_features_are_counted_but_never_zero_filled(tmp_path):
    start = date(2026, 4, 1)
    for offset in range(10):
        observed = start + timedelta(days=offset)
        healthy = row(observed, "healthy")
        if offset == 1:
            healthy[FEATURE_COLUMNS[2]] = ""
        write_day(tmp_path, observed, [healthy, row(observed, "failing", failed=offset == 9)])
    result = builder().build(tmp_path)
    assert result.rows_missing_features == 1
    assert all(len(item.features) == len(FEATURE_COLUMNS) for item in result.examples)


@pytest.mark.parametrize(
    ("value", "detail"),
    [
        (None, "serial number is invalid"),
        ("", "serial number is invalid"),
        ("x" * (MAX_SERIAL_LENGTH + 1), "serial number is invalid"),
    ],
)
def test_serial_contract(value, detail):
    with pytest.raises(TrainingError, match=detail) as raised:
        _serial(value)
    assert raised.value.code == "row-invalid"
    assert _serial("drive-1") == "drive-1"


@pytest.mark.parametrize("value", [None, "", "false", 0, "2"])
def test_failure_flag_is_exact(value):
    with pytest.raises(TrainingError) as raised:
        _failure(value)
    assert raised.value.code == "row-invalid"
    assert raised.value.detail == "failure must be 0 or 1"
    assert _failure("0") is False
    assert _failure("1") is True


@pytest.mark.parametrize("value", ["nan", "inf", "-1", str(MAX_SMART_VALUE + 1)])
def test_smart_features_refuse_nonfinite_or_out_of_range(value):
    document = row(date(2026, 1, 1), "drive")
    document[FEATURE_COLUMNS[0]] = value
    assert error_code(lambda: _features(document)) == "row-invalid"


def test_smart_features_preserve_exact_order_and_missing_state():
    document = row(date(2026, 1, 1), "drive", value=10)
    assert _features(document) == (10, 11, 12, 13, 14, 15)
    document[FEATURE_COLUMNS[-1]] = " "
    assert _features(document) is None
    document[FEATURE_COLUMNS[-1]] = "wrong"
    assert error_code(lambda: _features(document)) == "row-invalid"


@pytest.mark.parametrize(
    "fieldnames",
    [None, [*HEADER, "date"], [name for name in HEADER if name != FEATURE_COLUMNS[0]]],
)
def test_schema_drift_fails_closed(fieldnames):
    with pytest.raises(TrainingError) as raised:
        _validated_header(fieldnames)
    assert raised.value.code == "schema-drift"
    expected = (
        "CSV header is absent or duplicated"
        if fieldnames is None or len(fieldnames) != len(set(fieldnames))
        else f"CSV header has no {FEATURE_COLUMNS[0]}"
    )
    assert raised.value.detail == expected
    assert _validated_header(list(HEADER)) == HEADER


def test_csv_root_and_daily_name_are_bounded_and_canonical(tmp_path, monkeypatch):
    with pytest.raises(TrainingError) as missing:
        _csv_files(tmp_path / "missing")
    assert missing.value.detail == "Backblaze CSV root is not a directory"
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(TrainingError) as empty:
        _csv_files(tmp_path)
    assert empty.value.detail == "Backblaze CSV root contains no daily files"
    invalid = tmp_path / "drive.csv"
    invalid.write_text("x")
    assert error_code(lambda: _file_date(invalid)) == "dataset-invalid"
    compact = tmp_path / "20260101.csv"
    invalid.rename(compact)
    assert error_code(lambda: _file_date(compact)) == "dataset-invalid"
    valid = tmp_path / "2026-01-01.csv"
    compact.rename(valid)
    assert _file_date(valid) == date(2026, 1, 1)
    monkeypatch.setattr("omnitensor.training.storage.MAX_CSV_BYTES", 1)
    assert _csv_files(tmp_path) == (valid,)
    monkeypatch.setattr("omnitensor.training.storage.MAX_CSV_BYTES", 0)
    with pytest.raises(TrainingError) as unsafe:
        _csv_files(tmp_path)
    assert unsafe.value.detail == "unsafe daily CSV: 2026-01-01.csv"


def test_csv_root_refuses_symlinks_and_too_many_files(tmp_path, monkeypatch):
    target = tmp_path / "2026-01-01.csv.real"
    target.write_text("x")
    (tmp_path / "2026-01-01.csv").symlink_to(target)
    assert error_code(lambda: _csv_files(tmp_path)) == "dataset-invalid"
    (tmp_path / "2026-01-01.csv").unlink()
    for offset in range(3):
        (tmp_path / f"2026-01-0{offset + 1}.csv").write_text("x")
    monkeypatch.setattr("omnitensor.training.storage.MAX_CSV_FILES", 2)
    third = tmp_path / "2026-01-03.csv"
    third.unlink()
    assert len(_csv_files(tmp_path)) == 2
    third.write_text("x")
    with pytest.raises(TrainingError) as excess:
        _csv_files(tmp_path)
    assert excess.value.code == "dataset-too-large"
    assert excess.value.detail == "too many daily CSV files"


def test_utf8_bom_header_and_exact_row_limit_are_supported(tmp_path):
    observed = date(2026, 1, 1)
    destination = tmp_path / f"{observed.isoformat()}.csv"
    tmp_path.mkdir(exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerow(row(observed, "drive"))
    with pytest.raises(TrainingError) as no_labels:
        builder(max_rows=1).build(tmp_path)
    assert no_labels.value.code == "insufficient-history"
    assert no_labels.value.detail == "dataset produced no labelled examples"

    with destination.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writerow(row(observed, "second"))
    with pytest.raises(TrainingError) as limit:
        builder(max_rows=1).build(tmp_path)
    assert limit.value.code == "dataset-too-large"
    assert limit.value.detail == "CSV row limit exceeded"


def test_builder_requires_reviewed_terms_and_valid_bounds():
    with pytest.raises(TrainingError) as terms:
        BackblazeDatasetBuilder(accept_terms="")
    assert terms.value.code == "dataset-terms-not-accepted"
    assert terms.value.detail == (
        "review the Backblaze data terms and pass Backblaze-Drive-Stats"
    )
    invalid = (
        ({"horizon_days": True}, "storage intake bounds must be integers"),
        ({"max_rows": 0}, "storage intake bounds must be positive"),
        ({"max_samples": 0}, "storage intake bounds must be positive"),
        ({"max_devices": 0}, "storage intake bounds must be positive"),
        ({"negative_keep_modulus": 0}, "negative sampling modulus must be between 1 and 1024"),
        ({"negative_keep_modulus": 1025}, "negative sampling modulus must be between 1 and 1024"),
        ({"horizon_days": 91}, "storage failure horizon must be between 1 and 90 days"),
    )
    for changes, message in invalid:
        with pytest.raises(ValueError) as raised:
            BackblazeDatasetBuilder(accept_terms=BACKBLAZE_TERMS, **changes)
        assert str(raised.value) == message
    for changes in (
        {"horizon_days": 1},
        {"horizon_days": 90},
        {"negative_keep_modulus": 1},
        {"negative_keep_modulus": 1024},
    ):
        BackblazeDatasetBuilder(accept_terms=BACKBLAZE_TERMS, **changes)


def test_builder_refuses_row_identity_date_and_resource_boundaries(tmp_path):
    observed = date(2026, 1, 1)
    duplicate = row(observed, "same")
    write_day(tmp_path / "duplicate", observed, [duplicate, duplicate])
    assert error_code(lambda: builder().build(tmp_path / "duplicate")) == "row-duplicate"

    wrong_date = row(observed, "drive")
    wrong_date["date"] = "2025-12-31"
    write_day(tmp_path / "date", observed, [wrong_date])
    assert error_code(lambda: builder().build(tmp_path / "date")) == "date-mismatch"

    write_day(tmp_path / "rows", observed, [row(observed, "a"), row(observed, "b")])
    assert error_code(lambda: builder(max_rows=1).build(tmp_path / "rows")) == "dataset-too-large"
    assert error_code(lambda: builder(max_devices=1).build(tmp_path / "rows")) == (
        "dataset-too-large"
    )


def test_sample_limit_and_empty_labelled_dataset_refuse(tmp_path):
    start = date(2026, 1, 1)
    for offset in range(10):
        observed = start + timedelta(days=offset)
        write_day(tmp_path, observed, [row(observed, "healthy")])
    assert error_code(lambda: builder(max_samples=1).build(tmp_path)) == "dataset-too-large"

    one = tmp_path / "one"
    write_day(one, start, [row(start, "healthy")])
    assert error_code(lambda: builder().build(one)) == "insufficient-history"


def test_custom_horizon_flows_into_leakage_purge_and_report(tmp_path):
    samples = tuple(example(day, day % 2, 10 if day % 2 else 0) for day in range(20))
    source = dataset(samples, horizon=3)
    exporter = FakeExporter()
    report = StorageTrainer(exporter, minimum_positive_examples=1, minimum_auc=1).train(
        source, tmp_path / "nested/fit"
    )
    assert report["split"]["purgeDays"] == 3
    assert report["label"]["horizonDays"] == 3
    assert report["split"]["holdoutFrom"] == "2026-01-17"
    assert report["samples"] == {"training": 13, "holdout": 4}


def test_time_split_requires_dates_and_nonempty_purged_sides():
    assert error_code(lambda: _time_split((example(0, 0), example(1, 1)), 1)) == (
        "insufficient-history"
    )
    compressed = (example(0, 0), example(1, 1), example(2, 0))
    assert error_code(lambda: _time_split(compressed, 90)) == "insufficient-history"
    train, holdout, split = _time_split(compressed, 1)
    assert train == (compressed[0],)
    assert holdout == (compressed[2],)
    assert split == compressed[2].observed_on


def test_class_balance_and_quality_gates_refuse_unusable_fit(tmp_path):
    balanced = tuple(example(day, day % 2, 0) for day in range(30))
    assert error_code(lambda: _require_classes(balanced[:2], 2, "training")) == (
        "class-imbalance"
    )
    assert error_code(
        lambda: StorageTrainer(
            FakeExporter(), minimum_positive_examples=1, minimum_auc=1
        ).train(dataset(balanced), tmp_path)
    ) == "model-not-useful"

    positive_short = (example(0, 0), example(1, 0), example(2, 1))
    negative_short = (example(0, 1), example(1, 1), example(2, 0))
    for values in (positive_short, negative_short):
        with pytest.raises(TrainingError) as raised:
            _require_classes(values, 2, "exact")
        assert raised.value.detail == (
            "exact split needs at least 2 positive and negative examples"
        )
    _require_classes((example(0, 0), example(1, 0), example(2, 1), example(3, 1)), 2, "ok")


def test_trainer_validates_configuration_and_export_result(tmp_path):
    for positive in (True, 0):
        with pytest.raises(ValueError) as raised:
            StorageTrainer(minimum_positive_examples=positive)
        assert str(raised.value) == "minimum positive examples must be a positive integer"
    for auc in (0.5, 1.1, math.nan):
        with pytest.raises(ValueError) as raised:
            StorageTrainer(minimum_auc=auc)
        assert str(raised.value) == "minimum AUC must be in (0.5, 1]"

    defaults = StorageTrainer(FakeExporter())
    assert defaults._minimum_positive_examples == 25
    assert defaults._minimum_auc == 0.6

    samples = tuple(example(day, day % 2, 10 if day % 2 else 0) for day in range(20))
    with pytest.raises(TrainingError) as raised:
        StorageTrainer(
            FakeExporter(b""), minimum_positive_examples=1, minimum_auc=0.5 + 1e-9
        ).train(dataset(samples, horizon=3), tmp_path)
    assert raised.value.code == "export-failed"
    assert raised.value.detail == "exporter produced no portable model"


def test_logistic_fit_is_deterministic_and_regression_locked():
    samples = []
    for index in range(9):
        failed = 1 if index in {1, 5, 8} else 0
        base = float(index * index + (7 if failed else 0))
        samples.append(
            StorageExample(
                date(2026, 1, 1) + timedelta(days=index),
                tuple(base + feature * (index + 1) for feature in range(6)),
                failed,
            )
        )
    model = _fit_logistic(tuple(samples))
    assert model.weights == pytest.approx(
        (
            0.21856718568277828,
            0.16751006613882513,
            0.12549154753050024,
            0.09038293877001741,
            0.06065418081996534,
            0.03518427816371408,
        ),
        rel=1e-12,
        abs=1e-12,
    )
    assert model.intercept == pytest.approx(-0.0450410881983949, rel=1e-12, abs=1e-12)
    assert model.means == (25, 30, 35, 40, 45, 50)
    assert model.scales == pytest.approx(
        (
            22.494443758403985,
            24.94883653488564,
            27.42666990763228,
            29.922121136933683,
            32.43112359721411,
            34.95075901258163,
        ),
        rel=1e-12,
        abs=1e-12,
    )


def test_storage_risk_model_validates_width_type_and_finiteness():
    model = StorageRiskModel((1, 2), 0, (0, 0), (1, 2))
    assert 0 < model.predict((1, 2)) < 1
    for values in ((1,), (True, 2), (math.inf, 2), ("1", 2)):
        assert error_code(lambda values=values: model.predict(values)) == "features-invalid"


def test_sigmoid_is_stable_at_extremes():
    assert 0 < _sigmoid(-1000) < 1e-300
    assert _sigmoid(0) == 0.5
    assert _sigmoid(1000) == 1.0
    assert _sigmoid(-1) == pytest.approx(0.2689414213699951)
    assert _sigmoid(1) == pytest.approx(0.7310585786300049)
    assert _sigmoid(-701) == pytest.approx(9.85967654375977e-305)


def test_auc_handles_ties_and_quality_reports_exact_counts():
    assert _auc([(0.1, 0), (0.5, 0), (0.5, 1), (0.9, 1)]) == 0.875
    model = StorageRiskModel((1,) * 6, 0, (0,) * 6, (1,) * 6)
    report = _quality(model, (example(0, 0, -10), example(1, 1, 2)))
    assert report == {
        "auc": 1.0,
        "precisionAtHalf": 1.0,
        "recallAtHalf": 1.0,
        "positiveExamples": 1,
        "negativeExamples": 1,
        "identityFeatures": False,
        "futureLabelOnly": True,
    }
    threshold = StorageRiskModel((0,) * 6, 0, (0,) * 6, (1,) * 6)
    threshold_report = _quality(threshold, (example(0, 0), example(1, 1)))
    assert threshold_report["precisionAtHalf"] == 0.5
    assert threshold_report["recallAtHalf"] == 1.0


def test_append_and_negative_sampling_are_bounded_and_deterministic():
    samples = []
    item = example(0, 0)
    _append_example(samples, item, 1)
    assert samples == [item]
    with pytest.raises(TrainingError) as raised:
        _append_example(samples, item, 1)
    assert raised.value.code == "dataset-too-large"
    assert raised.value.detail == "labelled sample limit exceeded"
    first = _keep_negative("drive", date(2026, 1, 1), 7)
    assert _keep_negative("drive", date(2026, 1, 1), 7) is first
    assert _keep_negative("drive", date(2026, 1, 1), 1) is True


@given(
    st.lists(
        st.floats(min_value=0, max_value=1_000_000, allow_nan=False, allow_infinity=False),
        min_size=len(FEATURE_COLUMNS),
        max_size=len(FEATURE_COLUMNS),
    )
)
def test_feature_parser_property_preserves_finite_device_neutral_values(values):
    document = row(date(2026, 1, 1), "identity-never-emitted")
    for name, value in zip(FEATURE_COLUMNS, values, strict=True):
        document[name] = repr(value)
    parsed = _features(document)
    assert parsed == tuple(values)
    assert all(math.isfinite(value) for value in parsed)


def test_onnx_exporter_reports_missing_or_invalid_producer(tmp_path, monkeypatch):
    model = StorageRiskModel((1,) * 6, 0, (0,) * 6, (1,) * 6)
    import onnx

    def invalid(_model):
        raise ValueError("bad")

    monkeypatch.setattr(onnx.checker, "check_model", invalid)
    assert error_code(lambda: OnnxStorageExporter().export(model, tmp_path / "bad.onnx")) == (
        "export-failed"
    )


def test_onnx_exporter_reports_missing_dependency(tmp_path, monkeypatch):
    model = StorageRiskModel((1,) * 6, 0, (0,) * 6, (1,) * 6)
    original_import = __import__

    def without_onnx(name, *args, **kwargs):
        if name == "onnx":
            raise ImportError("absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", without_onnx)
    assert error_code(lambda: OnnxStorageExporter().export(model, tmp_path / "model.onnx")) == (
        "exporter-missing"
    )


def test_onnx_exporter_cleans_staging_file_when_save_fails(tmp_path, monkeypatch):
    model = StorageRiskModel((1,) * 6, 0, (0,) * 6, (1,) * 6)
    import onnx

    monkeypatch.setattr(onnx, "save_model", lambda *_args: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError, match="disk"):
        OnnxStorageExporter().export(model, tmp_path / "model.onnx")
    assert list(tmp_path.iterdir()) == []


def test_storage_cli_requires_terms_and_reports_portable_next_step(tmp_path, capsys):
    build_fleet(tmp_path / "csv")
    bad = storage_train_main(
        ["--csv-root", str(tmp_path / "csv"), "--accept-dataset-terms", "wrong"]
    )
    assert bad == 1
    assert "dataset-terms-not-accepted" in capsys.readouterr().err

    code = storage_train_main(
        [
            "--csv-root",
            str(tmp_path / "csv"),
            "--output-dir",
            str(tmp_path / "fit"),
            "--accept-dataset-terms",
            BACKBLAZE_TERMS,
            "--negative-keep-modulus",
            "1",
            "--minimum-positive-examples",
            "2",
            "--minimum-auc",
            "0.55",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert code == 0
    assert output["portableModel"] == str(tmp_path / "fit/model.onnx")
    assert output["next"] == "compile and qualify this source separately for each target lane"


def test_storage_cli_catches_invalid_numeric_bounds(tmp_path, capsys):
    code = storage_train_main(
        [
            "--csv-root",
            str(tmp_path),
            "--accept-dataset-terms",
            BACKBLAZE_TERMS,
            "--negative-keep-modulus",
            "0",
        ]
    )
    assert code == 1
    assert "storage training failed" in capsys.readouterr().err


def test_storage_cli_help_and_required_arguments_are_stable(capsys):
    with pytest.raises(SystemExit) as help_exit:
        storage_train_main(["--help"])
    help_text = capsys.readouterr().out
    normalized_help = " ".join(help_text.split())
    assert help_exit.value.code == 0
    assert "usage: omnitensor-train-storage-model" in help_text
    assert "Fit a portable storage risk source from official Backblaze daily CSVs" in help_text
    assert "--csv-root CSV_ROOT" in help_text
    assert "--accept-dataset-terms ACCEPT_DATASET_TERMS" in help_text
    assert "after reviewing the source terms, pass exactly Backblaze-Drive-Stats" in normalized_help

    with pytest.raises(SystemExit) as missing:
        storage_train_main([])
    error = capsys.readouterr().err
    assert missing.value.code == 2
    assert "the following arguments are required: --csv-root, --accept-dataset-terms" in error


def test_storage_cli_forwards_exact_defaults_and_prints_exact_envelope(
    tmp_path, capsys, monkeypatch
):
    captured = {}

    class CapturingBuilder:
        def __init__(self, **kwargs):
            captured["builder"] = kwargs

        def build(self, root):
            captured["csv_root"] = root
            return "dataset"

    class CapturingTrainer:
        def __init__(self, **kwargs):
            captured["trainer"] = kwargs

        def train(self, source, output):
            captured["train"] = (source, output)
            return {"samples": {"training": 30, "holdout": 10}, "quality": {"auc": 0.75}}

    monkeypatch.setattr("omnitensor.training.cli.BackblazeDatasetBuilder", CapturingBuilder)
    monkeypatch.setattr("omnitensor.training.cli.StorageTrainer", CapturingTrainer)
    code = storage_train_main(
        [
            "--csv-root",
            str(tmp_path / "csv"),
            "--output-dir",
            str(tmp_path / "fit"),
            "--accept-dataset-terms",
            BACKBLAZE_TERMS,
        ]
    )
    rendered = capsys.readouterr().out
    assert code == 0
    assert captured == {
        "builder": {"accept_terms": BACKBLAZE_TERMS, "negative_keep_modulus": 64},
        "csv_root": tmp_path / "csv",
        "trainer": {"minimum_positive_examples": 25, "minimum_auc": 0.6},
        "train": ("dataset", tmp_path / "fit"),
    }
    assert rendered.startswith("{\n  \"report\":")
    assert json.loads(rendered) == {
        "report": str(tmp_path / "fit/storage-training-report.json"),
        "portableModel": str(tmp_path / "fit/model.onnx"),
        "samples": {"training": 30, "holdout": 10},
        "quality": {"auc": 0.75},
        "next": "compile and qualify this source separately for each target lane",
    }
