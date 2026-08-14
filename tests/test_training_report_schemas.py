from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.acceptance import REQUIRED_SCHEMAS, check_schemas
from omnitensor.preparation import file_digest
from omnitensor.registry import load_schema, validate_document
from omnitensor.training.contracts import (
    NUMERIC_TRAINING_REPORT_SCHEMA,
    TRAINING_REPORT_SCHEMA,
    TrainingError,
    TrainingReport,
    TrainingSpec,
    validate_training_report_document,
    write_training_report,
)
from omnitensor.training.numeric_promotion import NUMERIC_RECIPES, NumericTrainingReport


def base_document(model_digest: str = "b" * 64) -> dict:
    spec = TrainingSpec(
        "resource-scheduler",
        "local-forecast",
        "1.0.0",
        ("load", "queue"),
        "load",
        1,
    )
    return TrainingReport(spec, 4, "a" * 64, model_digest, {"mae": 0.25}).document()


def numeric_document(recipe: str = "backblaze-smart-risk-v1", shape=None) -> dict:
    return {
        "version": 1,
        "recipe": recipe,
        "quality": {"accepted": True, "score": 0.5},
        "model": {
            "format": "onnx",
            "filename": "model.onnx",
            "sha256": "a" * 64,
        },
        "tensorContract": {
            "inputs": [
                {
                    "shape": shape or [1, 2],
                    "dtype": "float32",
                    "layout": "NC",
                }
            ]
        },
        "outputContract": {"kind": "raw"},
        "targets": {"tpu": "uncompiled", "npu": "uncompiled", "gpu": "uncompiled"},
    }


def _refs(value: object) -> tuple[str, ...]:
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "$ref":
                found.append(item)
            found.extend(_refs(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_refs(item))
    return tuple(found)


def test_training_report_schemas_are_installed_and_use_only_local_definitions():
    names = (TRAINING_REPORT_SCHEMA, NUMERIC_TRAINING_REPORT_SCHEMA)
    assert set(names) <= set(REQUIRED_SCHEMAS)
    assert check_schemas(names).ok
    for name in names:
        schema = load_schema(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert _refs(schema)
        assert all(reference.startswith("#/$defs/") for reference in _refs(schema))


@pytest.mark.parametrize("recipe", NUMERIC_RECIPES)
def test_numeric_schema_accepts_each_recipe_family(recipe):
    document = numeric_document(recipe)
    document["familyPayload"] = {"recipeSpecific": [1, 2, 3]}
    assert validate_document(NUMERIC_TRAINING_REPORT_SCHEMA, document) == []


@pytest.mark.parametrize(
    "field",
    ["version", "recipe", "quality", "model", "tensorContract", "outputContract", "targets"],
)
def test_numeric_schema_requires_the_entire_common_core(field):
    document = numeric_document()
    document.pop(field)
    assert validate_document(NUMERIC_TRAINING_REPORT_SCHEMA, document)


@pytest.mark.parametrize("shape", [[1, 1], [1, 65536], [1, 1, 2, 3, 4, 5]])
def test_numeric_schema_accepts_shape_boundaries(shape):
    assert validate_document(NUMERIC_TRAINING_REPORT_SCHEMA, numeric_document(shape=shape)) == []


@pytest.mark.parametrize("shape", [[1], [1] * 7, [2, 1], [1, 0], [1, 65537], [1, True]])
def test_numeric_schema_refuses_shape_overflow_and_ambiguous_integers(shape):
    assert validate_document(NUMERIC_TRAINING_REPORT_SCHEMA, numeric_document(shape=shape))


@given(st.lists(st.integers(min_value=1, max_value=65536), min_size=1, max_size=5))
def test_numeric_schema_accepts_every_bounded_rank_and_dimension(dimensions):
    document = numeric_document(shape=[1, *dimensions])
    assert validate_document(NUMERIC_TRAINING_REPORT_SCHEMA, document) == []


@pytest.mark.parametrize(
    ("container", "extra"),
    [
        ("model", "source"),
        ("tensorContract", "outputs"),
        ("outputContract", "labels"),
        ("targets", "cpu"),
    ],
)
def test_numeric_common_core_objects_are_closed(container, extra):
    document = numeric_document()
    document[container][extra] = True
    assert validate_document(NUMERIC_TRAINING_REPORT_SCHEMA, document)


def test_base_schema_is_closed_at_every_contract_boundary():
    mutations = []
    top = base_document()
    top["unexpected"] = True
    mutations.append(top)
    for container, extra in (
        ("spec", "unexpected"),
        ("model", "source"),
        ("tensorContract", "outputs"),
        ("outputContract", "labels"),
    ):
        document = base_document()
        document[container][extra] = True
        mutations.append(document)
    item = base_document()
    item["tensorContract"]["inputs"][0]["name"] = "input"
    mutations.append(item)
    assert all(validate_document(TRAINING_REPORT_SCHEMA, document) for document in mutations)


def _write_base(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "training-report.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_base_reader_retains_cross_field_semantics_beyond_the_schema(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    document = base_document(file_digest(model))
    document["tensorContract"]["inputs"][0]["shape"] = [1, 1]
    assert validate_document(TRAINING_REPORT_SCHEMA, document) == []
    with pytest.raises(TrainingError) as refusal:
        TrainingReport.load(_write_base(tmp_path, document))
    assert refusal.value.detail == "tensor contract disagrees with training spec"


def test_base_reader_retains_feature_and_finite_quality_semantics(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")

    target_drift = base_document(file_digest(model))
    target_drift["spec"]["targetFeature"] = "queue"
    assert validate_document(TRAINING_REPORT_SCHEMA, target_drift) == []
    with pytest.raises(TrainingError, match="target feature must be the first feature"):
        TrainingReport.load(_write_base(tmp_path, target_drift))

    non_finite = base_document(file_digest(model))
    non_finite["quality"] = {"mae": float("nan")}
    assert validate_document(TRAINING_REPORT_SCHEMA, non_finite) == []
    with pytest.raises(TrainingError, match="quality metrics are invalid"):
        TrainingReport.load(_write_base(tmp_path, non_finite))


def test_registry_failures_keep_the_stable_training_error_mapping(tmp_path):
    invalid_base = base_document()
    invalid_base["unexpected"] = True
    with pytest.raises(TrainingError) as base_error:
        validate_training_report_document(invalid_base)
    assert base_error.value.code == "report-invalid"
    assert base_error.value.detail.startswith("training report violates schema: /:")

    model = tmp_path / "model.onnx"
    model.write_bytes(b"portable")
    invalid_numeric = numeric_document()
    invalid_numeric["model"]["sha256"] = file_digest(model)
    invalid_numeric["model"]["unexpected"] = True
    report_path = tmp_path / "numeric.json"
    report_path.write_text(json.dumps(invalid_numeric), encoding="utf-8")
    with pytest.raises(TrainingError) as numeric_error:
        NumericTrainingReport.load(report_path)
    assert numeric_error.value.code == "report-invalid"
    assert numeric_error.value.detail.startswith("numeric training report violates schema: model:")


def test_validated_writer_preserves_compact_bytes_and_fails_before_publication(tmp_path):
    document = base_document()
    target = tmp_path / "training-report.json"
    write_training_report(target, document, prefix=".training-report-")
    assert target.read_bytes() == json.dumps(document, separators=(",", ":")).encode("utf-8")

    invalid = copy.deepcopy(document)
    invalid["unexpected"] = True
    refused = tmp_path / "refused.json"
    with pytest.raises(TrainingError, match="violates schema"):
        write_training_report(refused, invalid, prefix=".training-report-")
    assert not refused.exists()


def test_every_training_report_producer_uses_the_validated_writer():
    training_root = Path(__file__).parents[1] / "src/omnitensor/training"
    forecast = (training_root / "forecast.py").read_text(encoding="utf-8")
    assert "write_training_report(" in forecast
    assert "write_json_atomic(" not in forecast

    for name in ("storage", "network", "build", "hardware", "desktop"):
        source = (training_root / f"{name}.py").read_text(encoding="utf-8")
        assert "publish_numeric_training(" in source
        assert "writer=write_training_report" in source
        assert "write_json_atomic(" not in source

    shared = (training_root / "tabular/report.py").read_text(encoding="utf-8")
    assert "writer(" in shared
    assert "numeric=True" in shared
    assert "write_json_atomic(" not in shared
