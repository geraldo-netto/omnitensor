from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins.build_ingestion import BuildOutcome, BuildRecord
from omnitensor.training.build import (
    BUILD_FEATURES,
    BUILD_PROVENANCE_CONFIRMATION,
    BUILD_RECIPE,
    DEFAULT_MIN_CHECK_CLASS_EXAMPLES,
    DEFAULT_MIN_CLASS_EXAMPLES,
    DEFAULT_MIN_RANKING_MRR,
    DEFAULT_MIN_RISK_AUC,
    MAX_CHANGED_PATHS,
    MAX_CHECK_NAME,
    MAX_CHECKS,
    MAX_PATH_LENGTH,
    BuildAdvisorModel,
    BuildAdvisorTrainer,
    BuildDataset,
    BuildExample,
    BuildTrainingRecord,
    OnnxBuildExporter,
    _auc,
    _build_examples,
    _history_line,
    _normalization,
    _path_features,
    _positive_integer,
    _ranking_mrr,
    _record_from_document,
    _require_binary_classes,
    _require_check_classes,
    _risk_auc,
    _time_split,
    _unit_interval,
    _validate_checks,
    _validate_paths,
    load_build_history,
)
from omnitensor.training.cli import build_train_main
from omnitensor.training.contracts import TrainingError
from omnitensor.training.tabular import fit_output
from omnitensor.training.tabular import normalized_features as _normalized_features
from omnitensor.training.tabular import stable_sigmoid as _sigmoid

MANDATORY = ("security",)
OPTIONAL = ("lint", "integration")


class FakeExporter:
    def __init__(self, content: bytes = b"portable build advisor") -> None:
        self.content = content
        self.model = None

    def export(self, model, destination) -> None:
        self.model = model
        Path(destination).write_bytes(self.content)


def document(
    index: int,
    *,
    outcome: str | None = None,
    changed_paths: list[str] | None = None,
    executed_checks: list[str] | None = None,
    failed_checks: list[str] | None = None,
) -> dict:
    mode = index % 4
    default_paths = {
        0: [f"docs/guide-{index}.md"],
        1: [f"src/module-{index}.py", f"src/helper-{index}.py"],
        2: [f"tests/test_feature_{index}.py", f"src/feature-{index}.py"],
        3: ["pyproject.toml", f"config/settings-{index}.json"],
    }
    failures = {0: [], 1: ["lint"], 2: ["integration"], 3: []}
    selected_failures = failures[mode] if failed_checks is None else failed_checks
    return {
        "schemaVersion": 1,
        "buildId": f"private-build-{index}",
        "outcome": outcome or ("failed" if selected_failures else "succeeded"),
        "durationMs": 1_000 + index,
        "startedAtMs": 1_000_000 + index * 1_000,
        "changedPaths": default_paths[mode] if changed_paths is None else changed_paths,
        "executedChecks": list(MANDATORY + OPTIONAL)
        if executed_checks is None
        else executed_checks,
        "failedChecks": selected_failures,
    }


def write_history(path: Path, count: int = 100) -> bytes:
    raw = b"".join(
        json.dumps(document(index), separators=(",", ":")).encode() + b"\n"
        for index in range(count)
    )
    path.write_bytes(raw)
    return raw


def load_fixture(path: Path, count: int = 100) -> BuildDataset:
    write_history(path, count)
    return load_build_history(path, accept_provenance=BUILD_PROVENANCE_CONFIRMATION)


def training_error(call) -> TrainingError:
    with pytest.raises(TrainingError) as raised:
        call()
    return raised.value


def error_code(call) -> str:
    return training_error(call).code


def trainer(exporter=None, **changes) -> BuildAdvisorTrainer:
    options = {
        "mandatory_checks": MANDATORY,
        "optional_checks": OPTIONAL,
        "exporter": exporter,
        "minimum_class_examples": 2,
        "minimum_check_class_examples": 2,
        "minimum_risk_auc": 0.5,
        "minimum_ranking_mrr": 0.5,
    }
    options.update(changes)
    return BuildAdvisorTrainer(**options)


def test_history_load_and_portable_training_are_provenance_safe(tmp_path):
    history = tmp_path / "history.jsonl"
    raw = write_history(history)

    dataset = load_build_history(
        history,
        accept_provenance=BUILD_PROVENANCE_CONFIRMATION,
    )
    report = trainer().train(dataset, tmp_path / "fit")

    assert dataset.history_sha256 == hashlib.sha256(raw).hexdigest()
    assert len(dataset.records) == 100
    assert report["version"] == 1
    assert report["recipe"] == BUILD_RECIPE
    assert report["dataset"] == {
        "historySha256": dataset.history_sha256,
        "recordsRead": 100,
        "ignoredCancelledOrUnknown": 0,
        "pathsPersisted": False,
        "buildIdsPersisted": False,
        "logsOrSourceContentIngested": False,
        "provenanceConfirmed": True,
    }
    assert report["split"] == {
        "kind": "chronological",
        "training": 80,
        "holdout": 20,
        "holdoutFromMs": 1_080_000,
    }
    assert report["features"] == list(BUILD_FEATURES)
    assert report["outputs"] == {
        "buildFailureRiskIndex": 0,
        "optionalCheckRisk": [
            {"check": "lint", "index": 1},
            {"check": "integration", "index": 2},
        ],
        "mandatoryChecks": ["security"],
        "mandatoryChecksAlwaysIncluded": True,
        "modelMayOnlyReorderOptionalChecks": True,
    }
    assert report["featureContract"] == {
        "version": 1,
        "recipe": BUILD_RECIPE,
        "featureNames": list(BUILD_FEATURES),
        "pathContentRead": False,
    }
    assert report["resultContract"] == {
        "version": 1,
        "kind": "build-advice-scores",
        "shape": [1, 3],
        "mandatorySource": "deterministic-report-catalog",
        "mandatoryOmissionAllowed": False,
    }
    assert report["quality"]["holdoutBuildRiskAuc"] >= 0.5
    assert report["quality"]["holdoutOptionalCheckMrr"] >= 0.5
    assert report["quality"]["rankedHoldoutRecords"] == 10
    assert report["tensorContract"] == {
        "inputs": [{"shape": [1, 12], "dtype": "float32", "layout": "NC"}]
    }
    assert report["outputContract"] == {"kind": "raw"}
    assert report["targets"] == {"tpu": "uncompiled", "npu": "uncompiled", "gpu": "uncompiled"}
    report_path = tmp_path / "fit/build-training-report.json"
    rendered = report_path.read_text()
    assert json.loads(rendered) == report
    assert "private-build" not in rendered
    assert "guide-" not in rendered
    assert "module-" not in rendered

    onnx = pytest.importorskip("onnx")
    assert onnx.__version__ == "1.22.0", "review serialized goldens with producer upgrades"
    model_path = tmp_path / "fit/model.onnx"
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    assert [node.op_type for node in model.graph.node] == [
        "Sub",
        "Div",
        "MatMul",
        "Add",
        "Sigmoid",
    ]
    input_shape = [item.dim_value for item in model.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [item.dim_value for item in model.graph.output[0].type.tensor_type.shape.dim]
    assert input_shape == [1, 12]
    assert output_shape == [1, 3]
    assert model.opset_import[0].version == 13
    assert model.ir_version <= 8
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == (
        "ddd161eef9f993053e2707babbcac19a49a452be6a7e9d42fcb616de6c271fa1"
    )
    assert hashlib.sha256(model.graph.SerializeToString()).hexdigest() == (
        "ba3849d0d93a23f9675a8b8e104620ff28cf0245ffd9a9d4f2c41e95b4cd36a5"
    )
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == (
        "bbaed2ee17daa2c83b5f7a12105b6442fc860b1ba7781d07a9f0dc60191c2560"
    )
    assert [
        (item.name, hashlib.sha256(item.SerializeToString()).hexdigest())
        for item in model.graph.initializer
    ] == [
        ("means", "fd06a72b1a69d98334e31ccbb20d44e3a0048ae0bcf977885018675c07d2295e"),
        ("scales", "c6a8961b3f2b1abdfbec9c257a724506bc9c0aeab5840dd3072feceac9383a30"),
        ("weights", "a10490e212d3339c1dbe8e07f46f645020fda2504d622df3f6eb36398448d90e"),
        (
            "intercepts",
            "6400b51d17abe5259318584fe4273ab773cbaf3d33ac14907dca0f823fc48f6a",
        ),
    ]


def test_record_parser_preserves_only_declared_metadata():
    parsed = _record_from_document(document(1))
    assert parsed.record == BuildRecord(
        "private-build-1",
        BuildOutcome.FAILED,
        1_001,
        1_001_000,
        ("src/module-1.py", "src/helper-1.py"),
        ("lint",),
    )
    assert parsed.executed_checks == MANDATORY + OPTIONAL


@pytest.mark.parametrize(
    ("field", "value", "detail"),
    [
        ("schemaVersion", True, "version"),
        ("schemaVersion", 2, "version"),
        ("buildId", "", "identity"),
        ("outcome", "bad", "values"),
        ("durationMs", True, "duration"),
        ("startedAtMs", -1, "start"),
        ("changedPaths", "src/a.py", "must be an array"),
        ("executedChecks", "lint", "must be an array"),
        ("failedChecks", "lint", "must be an array"),
    ],
)
def test_record_field_types_are_exact(field, value, detail):
    item = document(0)
    item[field] = value
    error = training_error(lambda: _record_from_document(item))
    assert error.code == "record-invalid"
    assert detail in error.detail


def test_record_schema_is_closed_and_complete():
    item = document(0)
    item["log"] = "secret"
    assert error_code(lambda: _record_from_document(item)) == "record-invalid"
    del item["log"]
    del item["buildId"]
    assert error_code(lambda: _record_from_document(item)) == "record-invalid"


@pytest.mark.parametrize(
    "paths",
    [
        [],
        ["same.py", "same.py"],
        ["x.py"] * (MAX_CHANGED_PATHS + 1),
        [""],
        ["x" * (MAX_PATH_LENGTH + 1)],
        ["/absolute.py"],
        ["../escape.py"],
        ["src/../escape.py"],
        ["src//file.py"],
        ["src/file.py/"],
        ["src\\file.py"],
        ["src/control\n.py"],
        [1],
    ],
)
def test_changed_paths_are_bounded_safe_relative_metadata(paths):
    assert error_code(lambda: _validate_paths(tuple(paths))) == "record-invalid"


def test_changed_path_exact_boundaries_are_accepted():
    _validate_paths(("x",))
    _validate_paths(("file name.py",))
    _validate_paths(("x" * MAX_PATH_LENGTH,))
    _validate_paths(tuple(f"src/{index}.py" for index in range(MAX_CHANGED_PATHS)))


@pytest.mark.parametrize(
    ("paths", "detail"),
    [
        ((), "changed paths are empty, duplicated, or excessive"),
        (("",), "changed path is invalid"),
        (("../escape.py",), "changed path is not safe relative metadata"),
        (("bad\nname.py",), "changed path contains unsupported characters"),
    ],
)
def test_changed_path_rejections_have_stable_details(paths, detail):
    error = training_error(lambda: _validate_paths(paths))
    assert (error.code, error.detail) == ("record-invalid", detail)


@pytest.mark.parametrize(
    "checks",
    [
        ["same", "same"],
        [f"check-{index}" for index in range(MAX_CHECKS + 1)],
        [""],
        ["x" * (MAX_CHECK_NAME + 1)],
        ["bad\nname"],
        [1],
    ],
)
def test_check_names_are_unique_bounded_identifiers(checks):
    assert error_code(lambda: _validate_checks(tuple(checks), "executed")) == ("record-invalid")


def test_check_name_and_count_boundaries_are_accepted():
    _validate_checks(("x" * MAX_CHECK_NAME,), "optional")
    _validate_checks(tuple(f"check-{index}" for index in range(MAX_CHECKS)), "optional")


def test_failed_checks_must_have_executed():
    item = document(1, executed_checks=["security", "integration"])
    error = training_error(lambda: _record_from_document(item))
    assert (error.code, error.detail) == (
        "record-invalid",
        "failed checks were not executed",
    )


def test_provenance_file_and_line_boundaries(tmp_path, monkeypatch):
    path = tmp_path / "history.jsonl"
    provenance_error = training_error(lambda: load_build_history(path, accept_provenance="wrong"))
    assert (provenance_error.code, provenance_error.detail) == (
        "provenance-not-confirmed",
        f"review the export and pass exactly {BUILD_PROVENANCE_CONFIRMATION}",
    )
    assert (
        error_code(
            lambda: load_build_history(path, accept_provenance=BUILD_PROVENANCE_CONFIRMATION)
        )
        == "history-invalid"
    )
    path.mkdir()
    assert (
        error_code(
            lambda: load_build_history(path, accept_provenance=BUILD_PROVENANCE_CONFIRMATION)
        )
        == "history-invalid"
    )
    path.rmdir()
    write_history(path, 2)
    alias = tmp_path / "alias"
    alias.symlink_to(path)
    assert (
        error_code(
            lambda: load_build_history(alias, accept_provenance=BUILD_PROVENANCE_CONFIRMATION)
        )
        == "history-invalid"
    )

    monkeypatch.setattr("omnitensor.training.build.MAX_HISTORY_BYTES", path.stat().st_size - 1)
    assert (
        error_code(
            lambda: load_build_history(path, accept_provenance=BUILD_PROVENANCE_CONFIRMATION)
        )
        == "history-too-large"
    )


@pytest.mark.parametrize("raw", [b"\n", b"not-json\n", b"\xff\n"])
def test_history_line_rejects_blank_or_invalid_json(raw):
    assert error_code(lambda: _history_line(raw, 7)) == "history-invalid"


def test_history_line_exact_limit_and_context(monkeypatch):
    raw = json.dumps(document(0)).encode() + b"\n"
    monkeypatch.setattr("omnitensor.training.build.MAX_HISTORY_LINE_BYTES", len(raw))
    assert _history_line(raw, 1).record.build_id == "private-build-0"

    invalid = training_error(lambda: _history_line(b"not-json\n", 9))
    assert invalid.detail == "build history line 9: invalid JSON"
    invalid = training_error(lambda: _history_line(b"{}\n", 3))
    assert invalid.detail == "build history line 3: build record fields are invalid"


def test_empty_and_record_count_limits(tmp_path, monkeypatch):
    path = tmp_path / "history.jsonl"
    path.write_bytes(b"")
    assert (
        error_code(
            lambda: load_build_history(path, accept_provenance=BUILD_PROVENANCE_CONFIRMATION)
        )
        == "insufficient-history"
    )

    write_history(path, 2)
    monkeypatch.setattr("omnitensor.training.build.DEFAULT_MAX_BUILD_RECORDS", 1)
    error = training_error(
        lambda: load_build_history(path, accept_provenance=BUILD_PROVENANCE_CONFIRMATION)
    )
    assert (error.code, error.detail) == (
        "history-too-large",
        "build record limit exceeded",
    )


@pytest.mark.parametrize(
    ("mandatory", "optional", "message"),
    [
        ("security", OPTIONAL, "sequence"),
        ((), OPTIONAL, "must not be empty"),
        (MANDATORY, "lint", "sequence"),
        (MANDATORY, (), "must not be empty"),
        (("same",), ("same",), "overlap"),
    ],
)
def test_check_catalog_configuration_is_explicit(mandatory, optional, message):
    with pytest.raises(ValueError, match=message):
        BuildAdvisorTrainer(mandatory_checks=mandatory, optional_checks=optional)


def test_check_catalog_wraps_identifier_errors_as_configuration_errors():
    with pytest.raises(ValueError, match="check name is invalid"):
        BuildAdvisorTrainer(mandatory_checks=("bad\nname",), optional_checks=OPTIONAL)


def test_trainer_configuration_error_messages_name_the_exact_field():
    cases = (
        (
            {"mandatory_checks": (), "optional_checks": OPTIONAL},
            "mandatory checks must not be empty",
        ),
        (
            {"mandatory_checks": MANDATORY, "optional_checks": ()},
            "optional checks must not be empty",
        ),
        (
            {"mandatory_checks": ("same",), "optional_checks": ("same",)},
            "mandatory and optional check catalogs overlap",
        ),
        (
            {
                "mandatory_checks": MANDATORY,
                "optional_checks": OPTIONAL,
                "minimum_class_examples": 0,
            },
            "minimum_class_examples must be a positive integer",
        ),
        (
            {
                "mandatory_checks": MANDATORY,
                "optional_checks": OPTIONAL,
                "minimum_check_class_examples": 0,
            },
            "minimum_check_class_examples must be a positive integer",
        ),
        (
            {
                "mandatory_checks": MANDATORY,
                "optional_checks": OPTIONAL,
                "minimum_risk_auc": -1,
            },
            "minimum_risk_auc must be in [0, 1]",
        ),
        (
            {
                "mandatory_checks": MANDATORY,
                "optional_checks": OPTIONAL,
                "minimum_ranking_mrr": -1,
            },
            "minimum_ranking_mrr must be in [0, 1]",
        ),
    )
    for options, message in cases:
        with pytest.raises(ValueError) as raised:
            BuildAdvisorTrainer(**options)
        assert str(raised.value) == message


def test_numeric_trainer_configuration_is_bounded():
    for name in ("minimum_class_examples", "minimum_check_class_examples"):
        for value in (True, 0, 1.5):
            with pytest.raises(ValueError, match="positive integer"):
                BuildAdvisorTrainer(
                    mandatory_checks=MANDATORY,
                    optional_checks=OPTIONAL,
                    **{name: value},
                )
    for name in ("minimum_risk_auc", "minimum_ranking_mrr"):
        for value in (True, -0.1, 1.1, math.inf, "bad"):
            with pytest.raises(ValueError, match=r"\[0, 1\]"):
                BuildAdvisorTrainer(
                    mandatory_checks=MANDATORY,
                    optional_checks=OPTIONAL,
                    **{name: value},
                )


def test_build_examples_refuse_catalog_and_order_drift_and_skip_ambiguous_outcomes(tmp_path):
    dataset = load_fixture(tmp_path / "history.jsonl", 8)
    records = list(dataset.records)
    unknown = BuildTrainingRecord(records[0].record, (*records[0].executed_checks, "unknown"))
    assert (
        error_code(lambda: _build_examples((unknown,), MANDATORY, OPTIONAL))
        == "check-catalog-mismatch"
    )
    missing = BuildTrainingRecord(records[0].record, OPTIONAL)
    assert (
        error_code(lambda: _build_examples((missing,), MANDATORY, OPTIONAL))
        == "mandatory-check-missing"
    )
    assert (
        error_code(lambda: _build_examples((records[1], records[0]), MANDATORY, OPTIONAL))
        == "observations-unordered"
    )
    duplicate_time = BuildRecord(
        records[1].record.build_id,
        records[1].record.outcome,
        records[1].record.duration_ms,
        records[0].record.started_at_ms,
        records[1].record.changed_paths,
        records[1].record.failed_checks,
    )
    duplicate_error = training_error(
        lambda: _build_examples(
            (records[0], BuildTrainingRecord(duplicate_time, records[1].executed_checks)),
            MANDATORY,
            OPTIONAL,
        )
    )
    assert (duplicate_error.code, duplicate_error.detail) == (
        "observations-unordered",
        "build start times must increase",
    )

    cancelled_record = BuildRecord(
        "cancelled",
        BuildOutcome.CANCELLED,
        1,
        records[-1].record.started_at_ms + 1,
        ("src/a.py",),
        (),
    )
    examples, ignored = _build_examples(
        (*records, BuildTrainingRecord(cancelled_record, MANDATORY + OPTIONAL)),
        MANDATORY,
        OPTIONAL,
    )
    assert len(examples) == 8
    assert ignored == 1


def test_path_features_are_fixed_identity_free_and_order_independent():
    paths = (
        "src/main.py",
        "tests/test_main.py",
        "docs/guide.md",
        ".github/workflows/ci.yaml",
        "package-lock.json",
        "assets/logo.png",
    )
    features = _path_features(paths)
    assert len(features) == len(BUILD_FEATURES)
    assert all(math.isfinite(value) for value in features)
    assert features == _path_features(tuple(reversed(paths)))
    assert features[0] == math.log1p(6)
    assert features[1:4] == (5.0, 6.0, 3.0)
    assert features[4:] == pytest.approx((2 / 6, 1 / 6, 1 / 6, 2 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6))
    assert "main.py" not in repr(features)


def test_path_feature_categories_cover_singular_and_automation_directories():
    paths = (
        "test/example.bin",
        "spec/example.bin",
        "specs/example.bin",
        "docs/example.bin",
        ".gitlab/pipeline.bin",
        "ci/build.bin",
    )
    features = _path_features(paths)
    assert features[5] == 3 / 6
    assert features[6] == 1 / 6
    assert features[9] == 2 / 6


def test_time_split_and_class_gates_are_exact():
    examples = tuple(
        BuildExample(index, (0.0,) * 12, index % 2, OPTIONAL, (OPTIONAL[index % 2],))
        for index in range(10)
    )
    training, holdout = _time_split(examples)
    assert [item.started_at_ms for item in training] == list(range(8))
    assert [item.started_at_ms for item in holdout] == [8, 9]
    one_train, one_holdout = _time_split(examples[:2])
    assert len(one_train) == len(one_holdout) == 1
    assert error_code(lambda: _time_split(examples[:1])) == "insufficient-history"
    _require_binary_classes(examples, 5, "balanced")
    assert error_code(lambda: _require_binary_classes(examples, 6, "unbalanced")) == (
        "class-imbalance"
    )
    _require_check_classes(examples, "lint", 5)
    assert error_code(lambda: _require_check_classes(examples, "lint", 6)) == ("class-imbalance")
    positive_short = examples[:1] + tuple(item for item in examples if not item.build_failed)
    negative_short = examples[1:2] + tuple(item for item in examples if item.build_failed)
    assert (
        error_code(lambda: _require_binary_classes(positive_short, 2, "positive-short"))
        == "class-imbalance"
    )
    assert (
        error_code(lambda: _require_binary_classes(negative_short, 2, "negative-short"))
        == "class-imbalance"
    )
    lint_short = tuple(
        BuildExample(item.started_at_ms, item.features, item.build_failed, OPTIONAL, ())
        for item in examples
    )
    lint_negative_short = tuple(
        BuildExample(item.started_at_ms, item.features, item.build_failed, OPTIONAL, ("lint",))
        for item in examples
    )
    assert error_code(lambda: _require_check_classes(lint_short, "lint", 1)) == ("class-imbalance")
    assert (
        error_code(lambda: _require_check_classes(lint_negative_short, "lint", 1))
        == "class-imbalance"
    )


def test_fit_prediction_auc_and_ranking_contract():
    examples = tuple(
        BuildExample(
            index,
            (float(index % 2),) + (0.0,) * 11,
            index % 2,
            OPTIONAL,
            (("lint",) if index % 2 else ("integration",)),
        )
        for index in range(20)
    )
    means, scales = _normalization(examples)
    weights = []
    intercepts = []
    for label_of in (
        lambda item: item.build_failed,
        lambda item: int("lint" in item.failed_optional),
        lambda item: int("integration" in item.failed_optional),
    ):
        weight, intercept = fit_output(examples, label_of, means, scales)
        weights.append(weight)
        intercepts.append(intercept)
    model = BuildAdvisorModel(
        means,
        scales,
        tuple(tuple(output[row] for output in weights) for row in range(12)),
        tuple(intercepts),
    )
    assert _risk_auc(model, examples) == 1.0
    mrr, count = _ranking_mrr(model, examples, OPTIONAL)
    assert mrr == 1.0
    assert count == 20
    assert all(0 <= value <= 1 for value in model.predict(examples[0].features))
    assert _ranking_mrr(model, (), OPTIONAL) == (0.0, 0)
    assert _auc(((0.5, 0), (0.5, 1))) == 0.5
    assert error_code(lambda: _auc(((0.5, 1),))) == "class-imbalance"

    class RiskIndexModel:
        def predict(self, features):
            return (features[0], 1 - features[0])

    assert _risk_auc(RiskIndexModel(), examples) == 1.0


def test_fit_output_numeric_regression_is_deterministic():
    examples = tuple(
        BuildExample(
            index,
            (float(index % 2),) + tuple(float((index + column) % 3) for column in range(11)),
            index % 2,
            OPTIONAL,
            (("lint",) if index % 2 else ("integration",)),
        )
        for index in range(10)
    )
    means, scales = _normalization(examples)
    weights, intercept = fit_output(
        examples,
        lambda item: item.build_failed,
        means,
        scales,
    )
    assert weights == pytest.approx(
        (
            1.6209815288320895,
            -0.024443942919380846,
            0.06160624488482913,
            -0.03300417665335261,
            -0.024443942919380846,
            0.06160624488482913,
            -0.03300417665335261,
            -0.024443942919380846,
            0.06160624488482913,
            -0.03300417665335261,
            -0.024443942919380846,
            0.06160624488482913,
        ),
        abs=1e-12,
    )
    assert intercept == pytest.approx(-6.954801200567604e-05, abs=1e-14)


def test_feature_and_numeric_helpers_reject_invalid_values():
    means = (0.0,) * 12
    scales = (1.0,) * 12
    assert _normalized_features((1.0,) * 12, means, scales) == (1.0,) * 12
    width_error = training_error(lambda: _normalized_features((1.0,) * 11, means, scales))
    assert (width_error.code, width_error.detail) == (
        "features-invalid",
        "build feature width disagrees",
    )
    number_error = training_error(lambda: _normalized_features((True,) * 12, means, scales))
    assert (number_error.code, number_error.detail) == (
        "features-invalid",
        "build features must be numbers",
    )
    finite_error = training_error(lambda: _normalized_features((math.inf,) * 12, means, scales))
    assert (finite_error.code, finite_error.detail) == (
        "features-invalid",
        "build features must be finite",
    )
    assert _normalized_features((3.0,) * 12, (1.0,) * 12, (4.0,) * 12) == (0.5,) * 12
    for value in (True, 0, 1.5):
        with pytest.raises(ValueError):
            _positive_integer(value, "bound")
    assert _positive_integer(1, "bound") is None
    for value in (True, -1, 2, math.nan, "bad"):
        with pytest.raises(ValueError):
            _unit_interval(value, "metric")
    assert _unit_interval(0, "metric") == 0.0
    assert _unit_interval(1, "metric") == 1.0
    assert _sigmoid(1_000) == 1.0
    assert _sigmoid(-1_000) >= 0
    assert _sigmoid(2) == pytest.approx(0.8807970779778823)
    assert _sigmoid(-2) == pytest.approx(0.11920292202211755)


def test_quality_gates_ranking_measurement_and_empty_export(tmp_path):
    dataset = load_fixture(tmp_path / "history.jsonl")
    inverted_holdout = []
    for index, item in enumerate(dataset.records):
        if index >= 80:
            outcome = (
                BuildOutcome.SUCCEEDED
                if item.record.outcome is BuildOutcome.FAILED
                else BuildOutcome.FAILED
            )
            changed = BuildRecord(
                item.record.build_id,
                outcome,
                item.record.duration_ms,
                item.record.started_at_ms,
                item.record.changed_paths,
                item.record.failed_checks,
            )
            item = BuildTrainingRecord(changed, item.executed_checks)
        inverted_holdout.append(item)
    bad_risk = BuildDataset(tuple(inverted_holdout), dataset.history_sha256)
    assert (
        error_code(
            lambda: trainer(FakeExporter(), minimum_risk_auc=0.5).train(bad_risk, tmp_path / "auc")
        )
        == "model-not-useful"
    )

    inverted_ranking = []
    for index, item in enumerate(dataset.records):
        if index >= 80 and item.record.failed_checks:
            failed = ("integration",) if item.record.failed_checks == ("lint",) else ("lint",)
            changed = BuildRecord(
                item.record.build_id,
                item.record.outcome,
                item.record.duration_ms,
                item.record.started_at_ms,
                item.record.changed_paths,
                failed,
            )
            item = BuildTrainingRecord(changed, item.executed_checks)
        inverted_ranking.append(item)
    bad_ranking = BuildDataset(tuple(inverted_ranking), dataset.history_sha256)
    assert (
        error_code(
            lambda: trainer(FakeExporter(), minimum_ranking_mrr=0.75).train(
                bad_ranking, tmp_path / "mrr"
            )
        )
        == "model-not-useful"
    )

    records = tuple(
        BuildTrainingRecord(
            item.record,
            item.executed_checks,
        )
        for item in dataset.records
    )
    no_holdout_optional_failure = []
    for index, item in enumerate(records):
        if index >= 80:
            no_fail = BuildRecord(
                item.record.build_id,
                item.record.outcome,
                item.record.duration_ms,
                item.record.started_at_ms,
                item.record.changed_paths,
                (),
            )
            item = BuildTrainingRecord(no_fail, item.executed_checks)
        no_holdout_optional_failure.append(item)
    unranked = BuildDataset(tuple(no_holdout_optional_failure), "a" * 64)
    assert (
        error_code(lambda: trainer(FakeExporter()).train(unranked, tmp_path / "unranked"))
        == "ranking-unmeasured"
    )

    empty = trainer(FakeExporter(b""), minimum_risk_auc=0, minimum_ranking_mrr=0)
    assert error_code(lambda: empty.train(dataset, tmp_path / "empty")) == "export-failed"


def test_onnx_exporter_missing_dependency_validation_and_cleanup(tmp_path, monkeypatch):
    model = BuildAdvisorModel(
        (0.0,) * 12,
        (1.0,) * 12,
        tuple((1.0, 0.0, 0.0) for _index in range(12)),
        (0.0, 0.0, 0.0),
    )
    original_import = __import__

    def without_onnx(name, *args, **kwargs):
        if name == "onnx":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", without_onnx)
    assert error_code(lambda: OnnxBuildExporter().export(model, tmp_path / "missing.onnx")) == (
        "exporter-missing"
    )
    monkeypatch.setattr("builtins.__import__", original_import)

    import onnx

    monkeypatch.setattr(
        onnx.checker,
        "check_model",
        lambda _model: (_item for _item in ()).throw(ValueError("bad graph")),
    )
    assert error_code(lambda: OnnxBuildExporter().export(model, tmp_path / "bad.onnx")) == (
        "export-failed"
    )
    monkeypatch.undo()
    monkeypatch.setattr(
        onnx,
        "save_model",
        lambda *_args: (_item for _item in ()).throw(OSError("disk")),
    )
    with pytest.raises(OSError, match="disk"):
        OnnxBuildExporter().export(model, tmp_path / "failed.onnx")
    assert list(tmp_path.glob(".build-model-*")) == []


def test_build_cli_help_required_arguments_and_exact_envelope(tmp_path, capsys, monkeypatch):
    with pytest.raises(SystemExit) as help_exit:
        build_train_main(["--help"])
    help_text = capsys.readouterr().out
    assert help_exit.value.code == 0
    assert "usage: omnitensor-train-build-model" in help_text
    assert "--mandatory-check MANDATORY_CHECK" in help_text
    assert "--optional-check OPTIONAL_CHECK" in help_text
    assert "I-confirm-build-metadata-is-approved" in "".join(help_text.split())

    with pytest.raises(SystemExit) as missing:
        build_train_main([])
    assert missing.value.code == 2
    assert "--history" in capsys.readouterr().err

    captured = {}

    def fake_load(path, **kwargs):
        captured["load"] = (path, kwargs)
        return "dataset"

    class CapturingTrainer:
        def __init__(self, **kwargs):
            captured["trainer"] = kwargs

        def train(self, dataset, output):
            captured["train"] = (dataset, output)
            return {"split": {"training": 80, "holdout": 20}, "quality": {"auc": 1}}

    monkeypatch.setattr("omnitensor.training.cli.load_build_history", fake_load)
    monkeypatch.setattr("omnitensor.training.cli.BuildAdvisorTrainer", CapturingTrainer)
    code = build_train_main(
        [
            "--history",
            str(tmp_path / "history.jsonl"),
            "--output-dir",
            str(tmp_path / "fit"),
            "--mandatory-check",
            "security",
            "--optional-check",
            "lint",
            "--accept-provenance",
            BUILD_PROVENANCE_CONFIRMATION,
        ]
    )
    rendered = capsys.readouterr().out
    assert code == 0
    assert captured == {
        "load": (
            tmp_path / "history.jsonl",
            {"accept_provenance": BUILD_PROVENANCE_CONFIRMATION},
        ),
        "trainer": {
            "mandatory_checks": ["security"],
            "optional_checks": ["lint"],
            "minimum_class_examples": DEFAULT_MIN_CLASS_EXAMPLES,
            "minimum_check_class_examples": DEFAULT_MIN_CHECK_CLASS_EXAMPLES,
            "minimum_risk_auc": DEFAULT_MIN_RISK_AUC,
            "minimum_ranking_mrr": DEFAULT_MIN_RANKING_MRR,
        },
        "train": ("dataset", tmp_path / "fit"),
    }
    assert json.loads(rendered) == {
        "report": str(tmp_path / "fit/build-training-report.json"),
        "portableModel": str(tmp_path / "fit/model.onnx"),
        "samples": {"training": 80, "holdout": 20},
        "quality": {"auc": 1},
        "next": "compile and qualify this source separately for each target lane",
    }


def test_build_cli_catches_provenance_and_path_errors(tmp_path, capsys):
    code = build_train_main(
        [
            "--history",
            str(tmp_path / "missing"),
            "--mandatory-check",
            "security",
            "--optional-check",
            "lint",
            "--accept-provenance",
            "wrong",
        ]
    )
    assert code == 1
    assert "build training failed: provenance-not-confirmed" in capsys.readouterr().err


@given(
    suffix=st.sampled_from([".py", ".md", ".json", ".png", ".rs"]),
    depth=st.integers(min_value=1, max_value=8),
    count=st.integers(min_value=1, max_value=32),
)
def test_path_feature_property_is_finite_bounded_and_content_free(suffix, depth, count):
    paths = tuple(
        "/".join([*(f"d{level}" for level in range(depth - 1)), f"file-{index}{suffix}"])
        for index in range(count)
    )
    _validate_paths(paths)
    features = _path_features(paths)
    assert len(features) == len(BUILD_FEATURES)
    assert all(math.isfinite(value) for value in features)
    assert all(value >= 0 for value in features)
    assert not any(f"file-{index}" in repr(features) for index in range(count))
