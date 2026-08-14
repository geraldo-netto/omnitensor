from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import omnitensor.training.document_model as document_model
from omnitensor.acceptance import REQUIRED_SCHEMAS, check_schemas
from omnitensor.registry import load_schema, validate_document
from omnitensor.training.document_model import (
    DOCUMENT_MODEL_REPORT_SCHEMA,
    RECIPE_ID,
    DocumentModelError,
    DocumentModelEvidence,
    _report_document,
    _write_document_model_report,
    native_tensor_contract,
)
from omnitensor.training.recipes import load_model_recipe

ROOT = Path(__file__).parents[1]


def report_document(tmp_path: Path) -> dict:
    receipt = tmp_path / "source-receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    source = SimpleNamespace(
        recipe=SimpleNamespace(
            id="bge-small-en-v1-5",
            version="1.0.0",
            document_sha256="c" * 64,
            tensor_contract={
                "inputs": [
                    {"shape": [1, 128], "dtype": "int64", "layout": "NC"},
                    {"shape": [1, 128], "dtype": "int64", "layout": "NC"},
                    {"shape": [1, 128], "dtype": "int64", "layout": "NC"},
                ]
            },
            producer={"inputNames": ["input_ids", "attention_mask", "token_type_ids"]},
        ),
        receipt_path=receipt,
    )
    holdout = SimpleNamespace(
        license_id="CC0-1.0",
        corpus_sha256="d" * 64,
        documents=("document",) * 12,
    )
    evidence = DocumentModelEvidence(0, "GPU", 28, 0.999, 0.9, 0.001, 2)
    return _report_document(source, "a" * 64, "b" * 64, holdout, evidence)


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


def test_document_model_schema_is_installed_strict_and_self_contained():
    assert DOCUMENT_MODEL_REPORT_SCHEMA in REQUIRED_SCHEMAS
    assert check_schemas((DOCUMENT_MODEL_REPORT_SCHEMA,)).ok
    schema = load_schema(DOCUMENT_MODEL_REPORT_SCHEMA)
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert _refs(schema)
    assert all(reference.startswith("#/$defs/") for reference in _refs(schema))


def test_archived_measured_report_satisfies_the_canonical_schema():
    report_path = ROOT / "acceptance-evidence/bge-rx6600xt-report.json"
    evidence_path = ROOT / "acceptance-evidence/document-question-rx6600xt-evidence.json"
    accepted_path = ROOT / "acceptance-evidence/document-question-rx6600xt-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    assert report["cpuFallback"] is False
    assert "cpuFallback" not in report["limitations"]
    recipe = load_model_recipe(ROOT / f"model-recipes/{RECIPE_ID}.json")
    assert report["tensorContract"] == native_tensor_contract(recipe)
    assert validate_document(DOCUMENT_MODEL_REPORT_SCHEMA, report) == []
    report_digest = document_model.file_digest(report_path)
    assert evidence["embedding"]["reportSha256"] == report_digest
    assert accepted["models"]["embedding"]["reportSha256"] == report_digest
    assert accepted["evidenceSha256"] == document_model.file_digest(evidence_path)


def test_cpu_fallback_is_required_only_at_the_top_level(tmp_path):
    missing = report_document(tmp_path)
    missing.pop("cpuFallback")
    legacy_only = report_document(tmp_path)
    legacy_only.pop("cpuFallback")
    legacy_only["limitations"]["cpuFallback"] = "forbidden"

    assert validate_document(DOCUMENT_MODEL_REPORT_SCHEMA, missing)
    assert validate_document(DOCUMENT_MODEL_REPORT_SCHEMA, legacy_only)


@given(st.one_of(st.none(), st.just(True), st.text(), st.integers()))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_cpu_fallback_refuses_every_non_false_json_scalar(tmp_path, value):
    report = report_document(tmp_path)
    report["cpuFallback"] = value
    assert validate_document(DOCUMENT_MODEL_REPORT_SCHEMA, report)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["tensorContract"].update(outputs=[]),
        lambda value: value["tensorContract"]["inputs"].pop(),
        lambda value: value["tensorContract"]["inputs"][0].update(dtype="float32"),
        lambda value: value["tensorContract"]["inputs"][1].update(dtype="int32"),
        lambda value: value["tensorContract"]["inputs"][2].update(shape=[1, 127]),
        lambda value: value["outputContract"].update(kind="raw"),
    ],
)
def test_tensor_and_output_contracts_are_exact(tmp_path, mutate):
    report = report_document(tmp_path)
    mutate(report)
    assert validate_document(DOCUMENT_MODEL_REPORT_SCHEMA, report)


def test_gate_accepts_exact_quality_boundaries(tmp_path):
    report = report_document(tmp_path)
    report["nativeGate"].update(
        samples=1,
        minimumCosineSimilarity=0.999,
        minimumTop10Overlap=0.9,
        maximumAbsoluteError=0.001,
        expectedTopHits=1,
    )
    assert validate_document(DOCUMENT_MODEL_REPORT_SCHEMA, report) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("samples", 0),
        ("minimumCosineSimilarity", math.nextafter(0.999, 0.0)),
        ("minimumTop10Overlap", math.nextafter(0.9, 0.0)),
        ("maximumAbsoluteError", math.nextafter(0.001, 1.0)),
        ("expectedTopHits", 0),
        ("accepted", False),
    ],
)
def test_gate_refuses_values_beyond_the_measured_boundaries(tmp_path, field, value):
    report = report_document(tmp_path)
    report["nativeGate"][field] = value
    assert validate_document(DOCUMENT_MODEL_REPORT_SCHEMA, report)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("index", -1),
        ("index", 65536),
        ("name", ""),
        ("name", " "),
        ("name", "x" * 201),
        ("name", "cpu"),
        ("name", " Cpu "),
        ("name", "LlVmPiPe"),
    ],
)
def test_device_identity_is_named_and_bounded(tmp_path, field, value):
    report = report_document(tmp_path)
    report["device"][field] = value
    assert validate_document(DOCUMENT_MODEL_REPORT_SCHEMA, report)


@pytest.mark.parametrize(
    ("index", "name"),
    [(0, "G"), (65535, "x" * 200)],
)
def test_device_identity_accepts_exact_boundaries(tmp_path, index, name):
    report = report_document(tmp_path)
    report["device"] = {"index": index, "name": name}
    assert validate_document(DOCUMENT_MODEL_REPORT_SCHEMA, report) == []


@pytest.mark.parametrize(
    ("container", "extra"),
    [
        (None, "unexpected"),
        ("tensorContract", "outputs"),
        ("outputContract", "shape"),
        ("holdout", "source"),
        ("device", "backend"),
        ("nativeGate", "compiler"),
        ("limitations", "cpuFallback"),
    ],
)
def test_report_contract_is_closed_at_every_object_boundary(tmp_path, container, extra):
    report = report_document(tmp_path)
    target = report if container is None else report[container]
    target[extra] = True
    assert validate_document(DOCUMENT_MODEL_REPORT_SCHEMA, report)


def test_report_writer_preserves_bytes_and_refuses_before_publication(tmp_path):
    report = report_document(tmp_path)
    path = tmp_path / "document-model-report.json"
    _write_document_model_report(path, report)
    assert path.read_bytes() == json.dumps(report, separators=(",", ":")).encode("utf-8")

    invalid = copy.deepcopy(report)
    invalid["cpuFallback"] = True
    refused = tmp_path / "refused.json"
    with pytest.raises(DocumentModelError) as caught:
        _write_document_model_report(refused, invalid)
    assert caught.value.code == "report-invalid"
    assert caught.value.detail.startswith("document model report violates schema: cpuFallback:")
    assert not refused.exists()


def test_registry_failure_has_a_stable_document_model_error(monkeypatch, tmp_path):
    monkeypatch.setattr(
        document_model,
        "validate_document",
        lambda _schema, _report: ["/: schema unavailable"],
    )
    with pytest.raises(DocumentModelError) as caught:
        _write_document_model_report(tmp_path / "report.json", report_document(tmp_path))
    assert (caught.value.code, caught.value.detail) == (
        "report-invalid",
        "document model report violates schema: /: schema unavailable",
    )

    monkeypatch.setattr(
        document_model,
        "validate_document",
        lambda _schema, _report: (_ for _ in ()).throw(OSError("missing schema")),
    )
    with pytest.raises(DocumentModelError) as unavailable:
        _write_document_model_report(tmp_path / "unavailable.json", report_document(tmp_path))
    assert (unavailable.value.code, unavailable.value.detail) == (
        "report-invalid",
        "cannot validate document model report: missing schema",
    )
