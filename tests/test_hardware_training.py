from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins.hardware_collection import SensorKind
from omnitensor.training.cli import hardware_train_main
from omnitensor.training.contracts import TrainingError
from omnitensor.training.hardware import (
    DEFAULT_MAX_FALSE_POSITIVE_RATE,
    DEFAULT_MIN_AUC,
    DEFAULT_MIN_CLASS_EXAMPLES,
    DEFAULT_MIN_RECALL,
    DEFAULT_WINDOW,
    HARDWARE_LABEL_CONFIRMATION,
    HARDWARE_RECIPE,
    MAX_ROLE_LENGTH,
    MAX_TRAINING_SENSORS,
    MAX_WINDOW,
    HardwareDataset,
    HardwareHealthTrainer,
    HardwareObservation,
    OnnxHardwareExporter,
    SensorBinding,
    _binding_from_text,
    _hardware_item,
    _history_line,
    _observation_from_document,
    _quality,
    _require_classes,
    _sensor_features,
    _strict_observation_times,
    _time_split,
    _unit_metric,
    _validate_snapshot_header,
    _validated_bindings,
    _window_examples,
    load_hardware_history,
    parse_sensor_bindings,
)

BINDINGS = (SensorBinding("cpu-temperature", "cpu-package-0"), SensorBinding("ecc", "dimm-0"))


class FakeExporter:
    def __init__(self, content: bytes = b"portable hardware risk") -> None:
        self.content = content
        self.model = None

    def export(self, model, destination) -> None:
        self.model = model
        Path(destination).write_bytes(self.content)


def item(
    stable_id: str,
    observed_at_ms: int,
    *,
    kind: str,
    health: str = "ok",
    value: int | None = 1,
    unit: str,
) -> dict:
    return {
        "id": stable_id,
        "kind": kind,
        "health": health,
        "value": value,
        "unit": unit,
        "label": "private host label",
        "observedAtMs": observed_at_ms,
    }


def document(index: int, *, fault: bool | None = None, injected: bool = False) -> dict:
    observed = 1_000_000 + index * 1_000
    failed = index % 2 == 1 if fault is None else fault
    return {
        "label": "fault" if failed else "baseline",
        "labelSource": "injected" if injected else "observed",
        "snapshot": {
            "schemaVersion": 1,
            "source": "hwmon-edac-power-service",
            "sourceHealth": "ready",
            "observedAtMs": observed,
            "items": [
                item(
                    "cpu-package-0",
                    observed,
                    kind="thermal",
                    health="critical" if failed else "ok",
                    value=95_000 if failed else 40_000,
                    unit="millidegC",
                ),
                item(
                    "dimm-0",
                    observed,
                    kind="memory-errors",
                    health="degraded" if failed else "ok",
                    value=100 if failed else 0,
                    unit="count",
                ),
            ],
            "churn": {"added": [], "removed": [], "changed": []},
            "truncatedItems": 0,
        },
    }


def write_history(path: Path, count: int = 100) -> bytes:
    raw = b"".join(
        json.dumps(
            document(index, injected=index % 4 == 3),
            separators=(",", ":"),
        ).encode()
        + b"\n"
        for index in range(count)
    )
    path.write_bytes(raw)
    return raw


def training_error(call) -> TrainingError:
    with pytest.raises(TrainingError) as raised:
        call()
    return raised.value


def error_code(call) -> str:
    return training_error(call).code


def trainer(exporter=None, **changes) -> HardwareHealthTrainer:
    options = {
        "exporter": exporter,
        "window": 1,
        "minimum_class_examples": 2,
        "minimum_auc": 0.5,
        "minimum_recall": 0.5,
        "maximum_false_positive_rate": 0.5,
    }
    options.update(changes)
    return HardwareHealthTrainer(**options)


def test_labelled_history_trains_private_portable_hardware_source(tmp_path):
    history = tmp_path / "hardware.jsonl"
    raw = write_history(history)
    dataset = load_hardware_history(
        history,
        bindings=BINDINGS,
        confirm_labels=HARDWARE_LABEL_CONFIRMATION,
    )
    report = trainer().train(dataset, tmp_path / "fit")

    assert dataset.history_sha256 == hashlib.sha256(raw).hexdigest()
    assert len(dataset.observations) == 100
    assert dataset.roles == ("cpu-temperature", "ecc")
    assert dataset.kinds == ("thermal", "memory-errors")
    assert dataset.units == ("millidegC", "count")
    assert report["version"] == 1
    assert report["recipe"] == HARDWARE_RECIPE
    assert report["dataset"] == {
        "historySha256": dataset.history_sha256,
        "observations": 100,
        "observedFaultLabels": 25,
        "injectedFaultLabels": 25,
        "stableIdsPersisted": False,
        "labelsConfirmed": True,
    }
    assert report["split"] == {
        "kind": "chronological",
        "training": 80,
        "holdout": 20,
        "holdoutFromMs": 1_080_000,
    }
    assert report["featureContract"] == {
        "version": 1,
        "roles": [
            {"role": "cpu-temperature", "kind": "thermal", "unit": "millidegC"},
            {"role": "ecc", "kind": "memory-errors", "unit": "count"},
        ],
        "perRoleFeatures": ["logValue", "degraded", "critical", "missing"],
        "window": 1,
        "observationOrder": "oldest-first",
        "flattenOrder": "observations-then-roles-then-features",
    }
    assert report["resultContract"] == {
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
    }
    assert report["quality"] == {
        "auc": 1.0,
        "recallAtHalf": 1.0,
        "falsePositiveRateAtHalf": 0.0,
        "faultExamples": 10,
        "baselineExamples": 10,
    }
    assert report["tensorContract"] == {
        "inputs": [{"shape": [1, 8], "dtype": "float32", "layout": "NC"}]
    }
    assert report["targets"] == {"tpu": "uncompiled", "npu": "uncompiled", "gpu": "uncompiled"}
    rendered = (tmp_path / "fit/hardware-training-report.json").read_text()
    assert json.loads(rendered) == report
    assert "cpu-package-0" not in rendered
    assert "dimm-0" not in rendered
    assert "private host label" not in rendered

    onnx = pytest.importorskip("onnx")
    model = onnx.load(tmp_path / "fit/model.onnx")
    onnx.checker.check_model(model)
    assert [node.op_type for node in model.graph.node] == ["Sub", "Div", "MatMul", "Add", "Sigmoid"]
    input_shape = [value.dim_value for value in model.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [value.dim_value for value in model.graph.output[0].type.tensor_type.shape.dim]
    assert input_shape == [1, 8]
    assert output_shape == [1, 1]
    assert model.opset_import[0].version == 13
    assert model.ir_version <= 8


@pytest.mark.parametrize(
    "values",
    ["role=id", (), ("bad",), ("role=",), ("=id",), (f"{'x' * (MAX_ROLE_LENGTH + 1)}=id",)],
)
def test_sensor_binding_parser_rejects_invalid_shapes(values):
    with pytest.raises(ValueError):
        parse_sensor_bindings(values)


def test_sensor_binding_parser_errors_are_stable():
    for value, detail in (
        (None, "sensor binding must be ROLE=STABLE_ID"),
        ("bad role=id", "sensor role is invalid"),
        ("role=bad identity", "sensor stable identity is invalid"),
    ):
        with pytest.raises(ValueError) as raised:
            _binding_from_text(value)
        assert str(raised.value) == detail
    boundary_role = "x" * MAX_ROLE_LENGTH
    assert _binding_from_text(f"{boundary_role}=id").role == boundary_role


def test_sensor_binding_catalog_bounds_and_uniqueness():
    assert parse_sensor_bindings(("single=id",)) == (SensorBinding("single", "id"),)
    maximum = tuple(
        f"role-{index}=id-{index}" for index in range(MAX_TRAINING_SENSORS)
    )
    assert len(parse_sensor_bindings(maximum)) == MAX_TRAINING_SENSORS
    assert parse_sensor_bindings(("cpu-temperature=cpu-package-0", "ecc=dimm-0")) == BINDINGS
    with pytest.raises(ValueError) as sequence_error:
        parse_sensor_bindings(b"role=id")
    assert str(sequence_error.value) == "sensor bindings must be a sequence"
    with pytest.raises(ValueError, match="roles must be unique"):
        parse_sensor_bindings(("same=a", "same=b"))
    with pytest.raises(ValueError, match="identities must be unique"):
        parse_sensor_bindings(("a=same", "b=same"))
    with pytest.raises(ValueError, match="1 to"):
        parse_sensor_bindings(
            tuple(
                f"role-{index}=id-{index}"
                for index in range(MAX_TRAINING_SENSORS + 1)
            )
        )
    assert _binding_from_text("role=id") == SensorBinding("role", "id")


def test_binding_validation_wraps_untrusted_api_values():
    assert _validated_bindings(BINDINGS) == BINDINGS
    sequence_error = training_error(lambda: _validated_bindings(b"bad"))
    assert (sequence_error.code, sequence_error.detail) == (
        "bindings-invalid",
        "sensor bindings must be a sequence",
    )
    assert error_code(lambda: _validated_bindings((object(),))) == "bindings-invalid"
    assert error_code(lambda: _validated_bindings((SensorBinding("bad role", "id"),))) == (
        "bindings-invalid"
    )


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("label", "unknown", "label-invalid"),
        ("labelSource", "unknown", "label-invalid"),
        ("snapshot", [], "snapshot-invalid"),
    ],
)
def test_labelled_wrapper_is_closed_and_typed(field, value, code):
    value_document = document(0)
    value_document[field] = value
    assert error_code(lambda: _observation_from_document(value_document, BINDINGS)) == code
    extra = document(0)
    extra["note"] = "private"
    assert error_code(lambda: _observation_from_document(extra, BINDINGS)) == "snapshot-invalid"


def test_only_fault_labels_may_be_injected():
    value = document(0, fault=False, injected=True)
    error = training_error(lambda: _observation_from_document(value, BINDINGS))
    assert (error.code, error.detail) == (
        "label-invalid",
        "only a fault may have an injected label",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schemaVersion", True),
        ("schemaVersion", 2),
        ("source", "wrong"),
        ("sourceHealth", "degraded"),
        ("observedAtMs", True),
        ("observedAtMs", -1),
        ("truncatedItems", True),
        ("truncatedItems", 1),
        ("churn", []),
        ("churn", {"added": [], "removed": []}),
    ],
)
def test_snapshot_header_is_exact(field, value):
    snapshot = document(0)["snapshot"]
    snapshot[field] = value
    assert error_code(lambda: _validate_snapshot_header(snapshot)) == "snapshot-invalid"


def test_churn_values_and_items_are_arrays():
    value = document(0)
    value["snapshot"]["churn"]["added"] = "bad"
    assert error_code(lambda: _observation_from_document(value, BINDINGS)) == "snapshot-invalid"
    value = document(0)
    value["snapshot"]["items"] = "bad"
    assert error_code(lambda: _observation_from_document(value, BINDINGS)) == "snapshot-invalid"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kind", "bad"),
        ("health", "bad"),
        ("value", -1),
        ("unit", 1),
        ("observedAtMs", True),
        ("observedAtMs", -1),
        ("observedAtMs", 2_000_000),
    ],
)
def test_hardware_item_values_are_typed_and_bounded(field, value):
    item_document = document(0)["snapshot"]["items"][0]
    item_document[field] = value
    assert error_code(lambda: _hardware_item(item_document, 1_000_000)) == "snapshot-invalid"


def test_hardware_item_document_is_closed():
    item_document = document(0)["snapshot"]["items"][0]
    item_document["serial"] = "private"
    assert error_code(lambda: _hardware_item(item_document, 1_000_000)) == "snapshot-invalid"


def test_sensor_set_duplicates_missing_extra_and_semantic_drift_are_refused(tmp_path):
    missing = document(0)
    missing["snapshot"]["items"].pop()
    assert error_code(lambda: _observation_from_document(missing, BINDINGS)) == (
        "sensor-set-mismatch"
    )
    duplicate = document(0)
    duplicate["snapshot"]["items"][1]["id"] = "cpu-package-0"
    assert error_code(lambda: _observation_from_document(duplicate, BINDINGS)) == (
        "sensor-set-mismatch"
    )
    extra = document(0)
    extra["snapshot"]["items"].append(
        item("fan-0", 1_000_000, kind="fan", unit="rpm")
    )
    assert error_code(lambda: _observation_from_document(extra, BINDINGS)) == (
        "sensor-set-mismatch"
    )

    path = tmp_path / "history.jsonl"
    first = document(0)
    second = document(1)
    second["snapshot"]["items"][0]["unit"] = "celsius"
    path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    assert error_code(
        lambda: load_hardware_history(
            path, bindings=BINDINGS, confirm_labels=HARDWARE_LABEL_CONFIRMATION
        )
    ) == "sensor-drift"


def test_sensor_features_distinguish_health_and_missing_without_fake_zero():
    observed = 1
    base = document(0)["snapshot"]["items"][0]
    assert _sensor_features(_hardware_item(base, 1_000_000)) == (
        math.log1p(40_000),
        0.0,
        0.0,
        0.0,
    )
    missing = item(
        "cpu-package-0",
        observed,
        kind="thermal",
        health="missing",
        value=None,
        unit="millidegC",
    )
    assert _sensor_features(_hardware_item(missing, observed)) == (0.0, 0.0, 0.0, 1.0)


def test_history_file_safety_labels_bounds_and_order(tmp_path, monkeypatch):
    path = tmp_path / "history.jsonl"
    label_error = training_error(
        lambda: load_hardware_history(path, bindings=BINDINGS, confirm_labels="wrong")
    )
    assert (label_error.code, label_error.detail) == (
        "labels-not-confirmed",
        f"review the labels and pass exactly {HARDWARE_LABEL_CONFIRMATION}",
    )
    missing_error = training_error(
        lambda: load_hardware_history(
            path, bindings=BINDINGS, confirm_labels=HARDWARE_LABEL_CONFIRMATION
        )
    )
    assert (missing_error.code, missing_error.detail) == (
        "history-invalid",
        "hardware history is not a safe regular file",
    )
    path.mkdir()
    assert error_code(
        lambda: load_hardware_history(
            path, bindings=BINDINGS, confirm_labels=HARDWARE_LABEL_CONFIRMATION
        )
    ) == "history-invalid"
    path.rmdir()
    write_history(path, 2)
    alias = tmp_path / "alias"
    alias.symlink_to(path)
    assert error_code(
        lambda: load_hardware_history(
            alias, bindings=BINDINGS, confirm_labels=HARDWARE_LABEL_CONFIRMATION
        )
    ) == "history-invalid"
    monkeypatch.setattr("omnitensor.training.hardware.MAX_HISTORY_BYTES", path.stat().st_size - 1)
    assert error_code(
        lambda: load_hardware_history(
            path, bindings=BINDINGS, confirm_labels=HARDWARE_LABEL_CONFIRMATION
        )
    ) == "history-too-large"


@pytest.mark.parametrize("raw", [b"\n", b"not-json\n", b"\xff\n"])
def test_history_line_rejects_invalid_json(raw):
    assert error_code(lambda: _history_line(raw, 7, BINDINGS)) == "history-invalid"


def test_history_line_errors_preserve_line_and_cause():
    empty = training_error(lambda: _history_line(b"\n", 7, BINDINGS))
    assert (empty.code, empty.detail) == (
        "history-invalid",
        "hardware history line 7 is invalid",
    )
    malformed = training_error(lambda: _history_line(b"not-json\n", 9, BINDINGS))
    assert (malformed.code, malformed.detail) == (
        "history-invalid",
        "hardware history line 9: invalid JSON",
    )


def test_history_reader_numbers_the_first_line_from_one(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_bytes(b"not-json\n")
    error = training_error(
        lambda: load_hardware_history(
            path,
            bindings=BINDINGS,
            confirm_labels=HARDWARE_LABEL_CONFIRMATION,
        )
    )
    assert error.detail == "hardware history line 1: invalid JSON"


def test_history_line_exact_limit_and_count(tmp_path, monkeypatch):
    raw = json.dumps(document(0)).encode() + b"\n"
    monkeypatch.setattr("omnitensor.training.hardware.MAX_HISTORY_LINE_BYTES", len(raw))
    observation, semantics = _history_line(raw, 1, BINDINGS)
    assert observation.observed_at_ms == 1_000_000
    assert semantics == ((SensorKind.THERMAL, "millidegC"), (SensorKind.MEMORY_ERRORS, "count"))

    path = tmp_path / "history.jsonl"
    write_history(path, 2)
    monkeypatch.setattr("omnitensor.training.hardware.MAX_HISTORY_SNAPSHOTS", 1)
    assert error_code(
        lambda: load_hardware_history(
            path, bindings=BINDINGS, confirm_labels=HARDWARE_LABEL_CONFIRMATION
        )
    ) == "history-too-large"


def test_empty_and_unordered_history_are_refused(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_bytes(b"")
    assert error_code(
        lambda: load_hardware_history(
            path, bindings=BINDINGS, confirm_labels=HARDWARE_LABEL_CONFIRMATION
        )
    ) == "insufficient-history"
    observations = (
        HardwareObservation(2, 0, "observed", (0.0,) * 8),
        HardwareObservation(2, 1, "observed", (1.0,) * 8),
    )
    order_error = training_error(lambda: _strict_observation_times(observations))
    assert (order_error.code, order_error.detail) == (
        "observations-unordered",
        "hardware observation times must increase",
    )


def test_windows_preserve_oldest_first_role_feature_order():
    observations = (
        HardwareObservation(1, 0, "observed", (1.0, 2.0)),
        HardwareObservation(2, 1, "observed", (3.0, 4.0)),
        HardwareObservation(3, 0, "observed", (5.0, 6.0)),
    )
    windows = _window_examples(observations, 2)
    assert [(item.started_at_ms, item.features, item.build_failed) for item in windows] == [
        (2, (1.0, 2.0, 3.0, 4.0), 1),
        (3, (3.0, 4.0, 5.0, 6.0), 0),
    ]
    assert error_code(lambda: _window_examples(observations, 4)) == "insufficient-history"


def test_split_class_and_metric_boundaries():
    examples = tuple(
        _window_examples(
            (HardwareObservation(index, index % 2, "observed", (float(index),)),),
            1,
        )[0]
        for index in range(10)
    )
    training, holdout = _time_split(examples)
    assert len(training) == 8
    assert len(holdout) == 2
    assert tuple(map(len, _time_split(examples[:2]))) == (1, 1)
    split_error = training_error(lambda: _time_split(examples[:1]))
    assert (split_error.code, split_error.detail) == (
        "insufficient-history",
        "hardware windows cannot be split",
    )
    _require_classes(examples, 5, "balanced")
    assert error_code(lambda: _require_classes(examples, 6, "short")) == "class-imbalance"
    all_faults = tuple(
        HardwareObservation(index, 1, "observed", (1.0,)) for index in range(4)
    )
    all_fault_examples = _window_examples(all_faults, 1)
    class_error = training_error(
        lambda: _require_classes(all_fault_examples, 1, "training")
    )
    assert (class_error.code, class_error.detail) == (
        "class-imbalance",
        "training split needs at least 1 baseline and fault examples",
    )
    for value in (True, -1, 2, math.inf, "bad"):
        with pytest.raises(ValueError):
            _unit_metric(value, "metric")
    assert _unit_metric(0, "metric") == 0.0
    assert _unit_metric(1, "metric") == 1.0


def test_trainer_configuration_and_width_are_bounded(tmp_path):
    for value in (True, 0, MAX_WINDOW + 1, 1.5):
        with pytest.raises(ValueError, match="window"):
            HardwareHealthTrainer(window=value)
    for value in (True, 0, 1.5):
        with pytest.raises(ValueError, match="minimum_class_examples"):
            HardwareHealthTrainer(minimum_class_examples=value)
    for name in ("minimum_auc", "minimum_recall", "maximum_false_positive_rate"):
        with pytest.raises(ValueError, match=name):
            HardwareHealthTrainer(**{name: -1})

    observations = tuple(
        HardwareObservation(index, index % 2, "observed", (0.0,) * (MAX_TRAINING_SENSORS * 4))
        for index in range(20)
    )
    dataset = HardwareDataset(
        observations,
        "a" * 64,
        tuple(f"role-{index}" for index in range(MAX_TRAINING_SENSORS)),
        ("thermal",) * MAX_TRAINING_SENSORS,
        ("unit",) * MAX_TRAINING_SENSORS,
    )
    assert error_code(lambda: HardwareHealthTrainer(window=5).train(dataset, tmp_path)) == (
        "features-too-wide"
    )


@pytest.mark.parametrize(
    ("quality", "changes", "expected"),
    [
        (
            {"auc": 0.4, "recallAtHalf": 1.0, "falsePositiveRateAtHalf": 0.0},
            {"minimum_auc": 0.5},
            ("model-not-useful", "held-out hardware AUC is below the gate"),
        ),
        (
            {"auc": 1.0, "recallAtHalf": 0.4, "falsePositiveRateAtHalf": 0.0},
            {"minimum_recall": 0.5},
            ("model-not-useful", "held-out hardware recall is below the gate"),
        ),
        (
            {"auc": 1.0, "recallAtHalf": 1.0, "falsePositiveRateAtHalf": 0.6},
            {"maximum_false_positive_rate": 0.5},
            (
                "model-not-stable",
                "held-out hardware false-positive rate exceeds the gate",
            ),
        ),
    ],
)
def test_quality_gates(tmp_path, monkeypatch, quality, changes, expected):
    history = tmp_path / "history.jsonl"
    write_history(history)
    dataset = load_hardware_history(
        history, bindings=BINDINGS, confirm_labels=HARDWARE_LABEL_CONFIRMATION
    )
    complete_quality = {
        **quality,
        "faultExamples": 10,
        "baselineExamples": 10,
    }
    monkeypatch.setattr("omnitensor.training.hardware._quality", lambda *_args: complete_quality)
    error = training_error(
        lambda: trainer(FakeExporter(), **changes).train(dataset, tmp_path / expected[0])
    )
    assert (error.code, error.detail) == expected


def test_quality_gate_boundaries_are_accepted(tmp_path, monkeypatch):
    history = tmp_path / "history.jsonl"
    write_history(history)
    dataset = load_hardware_history(
        history, bindings=BINDINGS, confirm_labels=HARDWARE_LABEL_CONFIRMATION
    )
    quality = {
        "auc": 0.5,
        "recallAtHalf": 0.5,
        "falsePositiveRateAtHalf": 0.5,
        "faultExamples": 10,
        "baselineExamples": 10,
    }
    monkeypatch.setattr("omnitensor.training.hardware._quality", lambda *_args: quality)
    report = trainer(
        FakeExporter(),
        minimum_auc=0.5,
        minimum_recall=0.5,
        maximum_false_positive_rate=0.5,
    ).train(dataset, tmp_path / "boundaries")
    assert report["quality"] == quality


def test_empty_export_is_refused(tmp_path):
    history = tmp_path / "history.jsonl"
    write_history(history)
    dataset = load_hardware_history(
        history, bindings=BINDINGS, confirm_labels=HARDWARE_LABEL_CONFIRMATION
    )
    error = training_error(
        lambda: trainer(FakeExporter(b"")).train(dataset, tmp_path / "empty")
    )
    assert (error.code, error.detail) == (
        "export-failed",
        "exporter produced no portable model",
    )


def test_quality_helper_and_report_are_exact(tmp_path):
    class Model:
        def predict(self, features):
            return (features[0],)

    examples = tuple(
        _window_examples(
            (
                HardwareObservation(
                    index,
                    index % 2,
                    "observed",
                    (float(index % 2),),
                ),
            ),
            1,
        )[0]
        for index in range(10)
    )
    assert _quality(Model(), examples) == {
        "auc": 1.0,
        "recallAtHalf": 1.0,
        "falsePositiveRateAtHalf": 0.0,
        "faultExamples": 5,
        "baselineExamples": 5,
    }


def test_onnx_exporter_missing_validation_and_cleanup(tmp_path, monkeypatch):
    from omnitensor.training.build import BuildAdvisorModel

    model = BuildAdvisorModel((0.0,) * 8, (1.0,) * 8, tuple((1.0,) for _ in range(8)), (0.0,))
    original_import = __import__

    def without_onnx(name, *args, **kwargs):
        if name == "onnx":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", without_onnx)
    assert error_code(lambda: OnnxHardwareExporter().export(model, tmp_path / "missing")) == (
        "exporter-missing"
    )
    monkeypatch.setattr("builtins.__import__", original_import)
    import onnx

    monkeypatch.setattr(
        onnx.checker,
        "check_model",
        lambda _model: (_item for _item in ()).throw(ValueError("bad")),
    )
    assert error_code(lambda: OnnxHardwareExporter().export(model, tmp_path / "bad")) == (
        "export-failed"
    )
    monkeypatch.undo()
    monkeypatch.setattr(
        onnx,
        "save_model",
        lambda *_args: (_item for _item in ()).throw(OSError("disk")),
    )
    with pytest.raises(OSError, match="disk"):
        OnnxHardwareExporter().export(model, tmp_path / "failed")
    assert list(tmp_path.glob(".hardware-model-*")) == []


def test_hardware_cli_help_defaults_and_errors(tmp_path, capsys, monkeypatch):
    with pytest.raises(SystemExit) as help_exit:
        hardware_train_main(["--help"])
    help_text = capsys.readouterr().out
    assert help_exit.value.code == 0
    expected_help = (
        "usage: omnitensor-train-hardware-model [-h] --history HISTORY "
        "--sensor SENSOR\n"
        "                                       [--output-dir OUTPUT_DIR]\n"
        "                                       --confirm-labels CONFIRM_LABELS\n"
        "                                       [--window WINDOW]\n"
        "                                       "
        "[--minimum-class-examples MINIMUM_CLASS_EXAMPLES]\n"
        "                                       [--minimum-auc MINIMUM_AUC]\n"
        "                                       [--minimum-recall MINIMUM_RECALL]\n"
        "                                       "
        "[--maximum-false-positive-rate MAXIMUM_FALSE_POSITIVE_RATE]\n\n"
        "Fit a portable hardware-health risk source from reviewed labels\n\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
        "  --history HISTORY\n"
        "  --sensor SENSOR       ROLE=STABLE_ID\n"
        "  --output-dir OUTPUT_DIR\n"
        "  --confirm-labels CONFIRM_LABELS\n"
        "                        after reviewing baseline/fault labels, pass exactly\n"
        "                        I-confirm-hardware-labels-are-reviewed\n"
        "  --window WINDOW\n"
        "  --minimum-class-examples MINIMUM_CLASS_EXAMPLES\n"
        "  --minimum-auc MINIMUM_AUC\n"
        "  --minimum-recall MINIMUM_RECALL\n"
        "  --maximum-false-positive-rate MAXIMUM_FALSE_POSITIVE_RATE\n"
    )
    assert help_text == expected_help

    captured = {}

    def fake_parse(values):
        captured["parse"] = values
        return BINDINGS

    def fake_load(path, **kwargs):
        captured["load"] = (path, kwargs)
        return "dataset"

    class CapturingTrainer:
        def __init__(self, **kwargs):
            captured["trainer"] = kwargs

        def train(self, dataset, output):
            captured["train"] = (dataset, output)
            return {"split": {"training": 80, "holdout": 20}, "quality": {"auc": 1}}

    monkeypatch.setattr("omnitensor.training.cli.parse_sensor_bindings", fake_parse)
    monkeypatch.setattr("omnitensor.training.cli.load_hardware_history", fake_load)
    monkeypatch.setattr("omnitensor.training.cli.HardwareHealthTrainer", CapturingTrainer)
    code = hardware_train_main(
        [
            "--history",
            str(tmp_path / "history"),
            "--sensor",
            "cpu-temperature=cpu-package-0",
            "--sensor",
            "ecc=dimm-0",
            "--output-dir",
            str(tmp_path / "fit"),
            "--confirm-labels",
            HARDWARE_LABEL_CONFIRMATION,
            "--window",
            str(DEFAULT_WINDOW),
            "--minimum-class-examples",
            str(DEFAULT_MIN_CLASS_EXAMPLES),
            "--minimum-auc",
            str(DEFAULT_MIN_AUC),
            "--minimum-recall",
            str(DEFAULT_MIN_RECALL),
            "--maximum-false-positive-rate",
            str(DEFAULT_MAX_FALSE_POSITIVE_RATE),
        ]
    )
    assert code == 0
    assert captured["parse"] == [
        "cpu-temperature=cpu-package-0",
        "ecc=dimm-0",
    ]
    assert captured["load"] == (
        tmp_path / "history",
        {"bindings": BINDINGS, "confirm_labels": HARDWARE_LABEL_CONFIRMATION},
    )
    assert captured["trainer"] == {
        "window": DEFAULT_WINDOW,
        "minimum_class_examples": DEFAULT_MIN_CLASS_EXAMPLES,
        "minimum_auc": DEFAULT_MIN_AUC,
        "minimum_recall": DEFAULT_MIN_RECALL,
        "maximum_false_positive_rate": DEFAULT_MAX_FALSE_POSITIVE_RATE,
    }
    assert captured["train"] == ("dataset", tmp_path / "fit")
    assert capsys.readouterr().out == json.dumps(
        {
            "report": str(tmp_path / "fit/hardware-training-report.json"),
            "portableModel": str(tmp_path / "fit/model.onnx"),
            "samples": {"training": 80, "holdout": 20},
            "quality": {"auc": 1},
            "next": "compile and qualify this source separately for each target lane",
        },
        indent=2,
    ) + "\n"

    monkeypatch.setattr(
        "omnitensor.training.cli.parse_sensor_bindings",
        lambda _values: (_item for _item in ()).throw(ValueError("bad binding")),
    )
    assert hardware_train_main(
        [
            "--history",
            "missing",
            "--sensor",
            "bad",
            "--confirm-labels",
            HARDWARE_LABEL_CONFIRMATION,
        ]
    ) == 1
    assert capsys.readouterr().err == "hardware training failed: bad binding\n"


@given(
    value=st.integers(min_value=0, max_value=200_000),
    health=st.sampled_from(["ok", "degraded", "critical"]),
)
def test_sensor_feature_property_is_finite_and_bounded(value, health):
    sample_document = item(
        "cpu-package-0",
        1,
        kind="thermal",
        health=health,
        value=value,
        unit="millidegC",
    )
    features = _sensor_features(_hardware_item(sample_document, 1))
    assert len(features) == 4
    assert all(math.isfinite(feature) and feature >= 0 for feature in features)
    assert sum(features[1:]) <= 1
