from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

import omnitensor.plugins.selected_text_acceptance as acceptance
from omnitensor.plugins.selected_text_acceptance import (
    HEBREW_MODEL_SHA256,
    MAX_WORKER_LOAD_RECEIPT_BYTES,
    QWEN_MODEL_SHA256,
    SELECTED_TEXT_GPU_DEVICE,
    SELECTED_TEXT_RUNTIME_VERSION,
    SelectedTextAcceptanceError,
    SelectedTextPolicy,
    _fold,
    _valid_hebrew,
    build_selected_text_worker_load_receipt,
    load_selected_text_corpus,
    load_selected_text_evidence,
    load_selected_text_worker_load_receipt,
    main,
    parse_selected_text_worker_load_receipt,
    qualify_selected_text,
)
from omnitensor.registry import validate_document, validate_workload_document

ROOT = Path(__file__).parents[1]
CORPUS = ROOT / "evaluation-corpora/selected-text-v1.json"


def test_acceptance_collector_imports_the_canonical_control_client():
    source = (ROOT / "scripts/collect-selected-text-acceptance.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "omnitensor.socket_transport"
        for alias in node.names
    }
    rebound = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        if isinstance(target, ast.Name)
    }
    copied_coordinates = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and ("control.sock" in node.value or "OMNITENSOR_CONTROL_SOCKET" in node.value)
    }
    acceptance_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "omnitensor.plugins.selected_text_acceptance"
        for alias in node.names
    }
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert imports >= {"call_control"}
    assert rebound.isdisjoint({"call_control"})
    # The socket path is owned by socket_transport; a copy here could drift.
    assert copied_coordinates == set()
    assert "load_selected_text_worker_load_receipt" in acceptance_imports
    assert "_model" not in functions
    assert SELECTED_TEXT_GPU_DEVICE not in literals


def _model(digest: str, layers: int):
    return {
        "modelSha256": digest,
        "runtimeVersion": "llama-cpp-python-0.3.34",
        "deviceName": "AMD Radeon RX 6600 XT (RADV NAVI23)",
        "load": {
            "backend": "llama.cpp-vulkan",
            "device": "Vulkan",
            "totalModelLayers": layers,
            "acceleratorLayers": layers,
            "cpuFallback": False,
        },
    }


def _worker_receipt_document():
    return {
        "receiptVersion": 1,
        "pluginId": "selected-text-tools",
        "models": {
            "primary": _model(QWEN_MODEL_SHA256, 37),
            "hebrewTranslation": _model(HEBREW_MODEL_SHA256, 33),
        },
    }


def test_worker_load_receipt_round_trips_the_exact_evidence_models():
    document = _worker_receipt_document()

    receipt = parse_selected_text_worker_load_receipt(document)

    assert receipt.document() == document
    assert receipt.models_document() == document["models"]
    assert receipt.primary.runtime_version == SELECTED_TEXT_RUNTIME_VERSION
    assert receipt.hebrew.device_name == SELECTED_TEXT_GPU_DEVICE


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda value: value.update(extra=True), "receipt-invalid"),
        (lambda value: value.pop("models"), "receipt-invalid"),
        (lambda value: value.update(receiptVersion=True), "receipt-invalid"),
        (lambda value: value.update(pluginId="other"), "receipt-invalid"),
        (lambda value: value["models"].update(extra={}), "receipt-invalid"),
        (lambda value: value["models"].pop("primary"), "receipt-invalid"),
        (lambda value: value["models"]["primary"].update(extra=True), "receipt-invalid"),
        (
            lambda value: value["models"]["primary"].pop("runtimeVersion"),
            "receipt-invalid",
        ),
        (
            lambda value: value["models"]["primary"]["load"].update(extra=True),
            "receipt-invalid",
        ),
        (
            lambda value: value["models"]["primary"]["load"].pop("backend"),
            "receipt-invalid",
        ),
        (
            lambda value: value["models"]["primary"].update(modelSha256="0" * 64),
            "receipt-invalid",
        ),
        (
            lambda value: value["models"]["hebrewTranslation"].update(modelSha256="0" * 64),
            "receipt-invalid",
        ),
        (
            lambda value: value["models"]["primary"].update(runtimeVersion="other"),
            "receipt-invalid",
        ),
        (
            lambda value: value["models"]["primary"].update(deviceName="different GPU"),
            "device-unqualified",
        ),
        (
            lambda value: value["models"]["primary"]["load"].update(backend="wrong"),
            "device-unqualified",
        ),
        (
            lambda value: value["models"]["primary"]["load"].update(device="CPU"),
            "device-unqualified",
        ),
        (
            lambda value: value["models"]["primary"]["load"].update(cpuFallback=True),
            "device-unqualified",
        ),
        (
            lambda value: value["models"]["primary"]["load"].update(
                totalModelLayers=36, acceleratorLayers=36
            ),
            "device-unqualified",
        ),
        (
            lambda value: value["models"]["hebrewTranslation"]["load"].update(
                totalModelLayers=32, acceleratorLayers=32
            ),
            "device-unqualified",
        ),
        (
            lambda value: value["models"]["primary"]["load"].update(
                totalModelLayers=38, acceleratorLayers=38
            ),
            "device-unqualified",
        ),
        (
            lambda value: value["models"]["primary"]["load"].update(acceleratorLayers=36),
            "device-unqualified",
        ),
    ],
)
def test_worker_load_receipt_refuses_field_identity_and_device_drift(mutate, code):
    document = _worker_receipt_document()
    mutate(document)

    with pytest.raises(SelectedTextAcceptanceError) as caught:
        parse_selected_text_worker_load_receipt(document)
    assert caught.value.code == code


def test_worker_load_receipt_loader_enforces_exact_byte_boundaries(tmp_path):
    exact = tmp_path / "exact.json"
    oversized = tmp_path / "oversized.json"
    malformed = tmp_path / "malformed.json"
    exact.write_bytes(b"{}" + b" " * (MAX_WORKER_LOAD_RECEIPT_BYTES - 2))
    oversized.write_bytes(b"{}" + b" " * (MAX_WORKER_LOAD_RECEIPT_BYTES - 1))
    malformed.write_text("{", encoding="utf-8")

    with pytest.raises(SelectedTextAcceptanceError, match="receipt fields"):
        load_selected_text_worker_load_receipt(exact)
    for path in (tmp_path / "missing", oversized, malformed):
        with pytest.raises(SelectedTextAcceptanceError) as caught:
            load_selected_text_worker_load_receipt(path)
        assert (caught.value.code, caught.value.detail) == (
            "receipt-invalid",
            "cannot read worker load receipt",
        )


def test_worker_load_receipt_builder_requires_typed_exact_models():
    receipt = parse_selected_text_worker_load_receipt(_worker_receipt_document())
    assert build_selected_text_worker_load_receipt(receipt.primary, receipt.hebrew) == receipt
    with pytest.raises(SelectedTextAcceptanceError, match="receipt models"):
        build_selected_text_worker_load_receipt(object(), receipt.hebrew)


def _result(case, index: int):
    result_text = " ".join(case.expected_terms) or "Review the extracted tasks"
    tasks = [" ".join(case.expected_task_terms)] if case.expected_task_terms else []
    return {
        "version": 1,
        "requestId": f"acceptance-{index}",
        "operation": case.operation,
        "result": result_text,
        "tasks": tasks,
        "providerId": case.expected_provider_id,
        "accelerator": "gpu",
        "evidence": {
            "selectionSha256": "a" * 64,
            "span": {"start": 0, "end": 1},
            "textSha256": "b" * 64,
        },
    }


def _evidence_document(corpus):
    return {
        "evidenceVersion": 1,
        "corpusSha256": corpus.sha256,
        "models": {
            "primary": _model(QWEN_MODEL_SHA256, 37),
            "hebrewTranslation": _model(HEBREW_MODEL_SHA256, 33),
        },
        "observations": [
            {"caseId": case.case_id, "result": _result(case, index), "latencyMs": 100 + index}
            for index, case in enumerate(corpus.cases)
        ],
        "safety": {
            "cancellationLatencyMs": 250,
            "privateFragmentsDiscarded": True,
            "defaultRoutePreserved": True,
        },
    }


def _write_evidence(tmp_path: Path, mutate=lambda _value: None):
    corpus = load_selected_text_corpus()
    document = _evidence_document(corpus)
    mutate(document)
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return corpus, path


def test_frozen_corpus_covers_every_operation_and_keeps_non_hebrew_on_qwen():
    corpus = load_selected_text_corpus()

    assert corpus.corpus_id == "selected-text-operations-v1"
    assert len(corpus.cases) == 16
    assert corpus.sha256 == "2f574e01cff6a1438a0ba312315c91f0214b9db501965d93234552d02bab8b07"
    assert {case.operation for case in corpus.cases} == {
        "explain",
        "summarize",
        "rewrite",
        "translate",
        "extract-tasks",
    }
    for case in corpus.cases:
        if case.language == "Hebrew":
            assert case.expected_provider_id == "dictalm2-hebrew-gpu"
            assert case.require_hebrew is True
        else:
            assert case.expected_provider_id == "qwen3-workloads-gpu"
            assert case.require_hebrew is False
    assert sum(case.injection_probe for case in corpus.cases) == 1


def test_complete_named_gpu_evidence_qualifies_and_emits_the_closed_report(tmp_path):
    corpus, path = _write_evidence(tmp_path)
    evidence = load_selected_text_evidence(path)
    report = qualify_selected_text(corpus, evidence).document()

    assert report["metrics"] == {
        "operationTermRecall": 1.0,
        "taskTermRecall": 1.0,
        "targetScriptIntegrity": 1.0,
        "promptInjectionIntegrity": 1.0,
        "providerRouteIntegrity": 1.0,
        "p95LatencyMs": 115,
        "cancellationLatencyMs": 250,
    }
    assert report["models"]["primary"]["modelId"] == "qwen3-8b-q4-k-m"
    assert report["models"]["hebrewTranslation"]["modelId"] == ("dictalm2-hebrew-q4-k-m")
    assert report["scope"] == {
        "accelerator": "gpu",
        "cpuFallback": "forbidden",
        "hebrewRoute": "explicit-translation-target-only",
        "otherLanguages": "qwen-primary-route",
        "arbitrarySelections": "not-claimed",
    }
    assert validate_document("selected-text-acceptance.schema.json", report) == []


def test_archived_selected_text_report_remains_byte_compatible():
    evidence_path = ROOT / "acceptance-evidence/selected-text-rx6600xt-evidence.json"
    report_path = ROOT / "acceptance-evidence/selected-text-rx6600xt-report.json"

    report = qualify_selected_text(
        load_selected_text_corpus(), load_selected_text_evidence(evidence_path)
    ).document()
    produced = json.dumps(report, separators=(",", ":")).encode("utf-8")

    assert hashlib.sha256(evidence_path.read_bytes()).hexdigest() == (
        "b399931d77b43e3131dc77566b3394a16307319527b7b0ad182f7491506e7b8c"
    )
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == (
        "4ed2739640bb634576ee722722ccfd24defd3bbe4c2f46de26c566e1596f3d5c"
    )
    assert produced + b"\n" == report_path.read_bytes()


def test_qualification_report_agrees_with_the_live_worker_receipt(tmp_path):
    corpus = load_selected_text_corpus()
    receipt = parse_selected_text_worker_load_receipt(_worker_receipt_document())
    document = _evidence_document(corpus)
    document["models"] = receipt.models_document()
    path = tmp_path / "live-evidence.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    report = qualify_selected_text(corpus, load_selected_text_evidence(path)).document()

    for role, model in (
        ("primary", receipt.primary),
        ("hebrewTranslation", receipt.hebrew),
    ):
        published = report["models"][role]
        assert published["artifactSha256"] == model.model_sha256
        assert published["runtimeVersion"] == model.runtime_version
        assert published["deviceName"] == model.device_name
        assert published["fullyOffloadedLayers"] == model.load.total_model_layers


def test_evidence_digest_is_bound_to_the_parsed_descriptor_snapshot(tmp_path, monkeypatch):
    _corpus, path = _write_evidence(tmp_path)
    original = path.read_bytes()
    replacement = original + b" "
    staged = tmp_path / "replacement.json"
    staged.write_bytes(replacement)
    real_open = Path.open

    class ReplacingStream(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            os.replace(staged, path)
            return super().read(size)

    def open_once(target: Path, *args, **kwargs):
        if target == path:
            return ReplacingStream(original)
        return real_open(target, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_once)
    evidence = load_selected_text_evidence(path)

    assert evidence.evidence_sha256 == hashlib.sha256(original).hexdigest()
    with real_open(path, "rb") as stream:
        assert stream.read() == replacement


@pytest.mark.parametrize(
    ("mutate", "code", "detail"),
    [
        (
            lambda value: value.update(corpusSha256="0" * 64),
            "evidence-stale",
            "another corpus",
        ),
        (
            lambda value: value["models"]["primary"].update(modelSha256="0" * 64),
            "evidence-stale",
            "primary model bytes",
        ),
        (
            lambda value: value["models"]["hebrewTranslation"].update(modelSha256="0" * 64),
            "evidence-stale",
            "Hebrew model bytes",
        ),
        (
            lambda value: value["models"]["primary"].update(runtimeVersion="other"),
            "evidence-stale",
            "runtime identity",
        ),
        (
            lambda value: value["models"]["primary"]["load"].update(acceleratorLayers=1),
            "device-unqualified",
            "complete Vulkan",
        ),
        (
            lambda value: value["models"]["hebrewTranslation"].update(deviceName="cpu"),
            "device-unqualified",
            "named GPU load",
        ),
        (
            lambda value: value["observations"][0]["result"].update(providerId="wrong-gpu"),
            "quality-failed",
            "provider route",
        ),
        (
            lambda value: value["safety"].update(privateFragmentsDiscarded=False),
            "safety-failed",
            "privacy",
        ),
        (
            lambda value: value["safety"].update(defaultRoutePreserved=False),
            "safety-failed",
            "route",
        ),
        (
            lambda value: value["safety"].update(cancellationLatencyMs=2001),
            "quality-failed",
            "cancellation",
        ),
    ],
)
def test_acceptance_fails_closed_on_stale_quality_device_route_or_safety_evidence(
    tmp_path, mutate, code, detail
):
    corpus, path = _write_evidence(tmp_path, mutate)
    evidence = load_selected_text_evidence(path)
    with pytest.raises(SelectedTextAcceptanceError) as captured:
        qualify_selected_text(corpus, evidence)
    assert captured.value.code == code
    assert detail in captured.value.detail


def test_script_normalization_accepts_hebrew_punctuation_but_rejects_mixed_scripts():
    assert _fold('דו"ח 18:45') == _fold("דוח 18:45")
    assert _valid_hebrew("עברית BLUE-47") is True
    assert _valid_hebrew("עברית русский") is False
    assert _valid_hebrew("עברית العربية") is False
    assert _valid_hebrew("English") is False


def test_policy_rejects_each_metric_independently(tmp_path):
    corpus, path = _write_evidence(tmp_path)
    evidence = load_selected_text_evidence(path)
    observed = list(evidence.observations)

    first = observed[0]
    broken = dict(first.result)
    broken["result"] = "missing"
    observed[0] = replace(first, result=broken)
    weak_terms = replace(evidence, observations=tuple(observed))
    with pytest.raises(SelectedTextAcceptanceError, match="operation term recall"):
        qualify_selected_text(
            corpus, weak_terms, SelectedTextPolicy(minimum_operation_term_recall=1.0)
        )

    task_index = next(index for index, case in enumerate(corpus.cases) if case.expected_task_terms)
    task_observation = list(evidence.observations)
    broken = dict(task_observation[task_index].result)
    broken["tasks"] = ["missing"]
    task_observation[task_index] = replace(task_observation[task_index], result=broken)
    weak_tasks = replace(evidence, observations=tuple(task_observation))
    with pytest.raises(SelectedTextAcceptanceError, match="task term recall"):
        qualify_selected_text(corpus, weak_tasks, SelectedTextPolicy(minimum_task_term_recall=1.0))

    injection_index = next(index for index, case in enumerate(corpus.cases) if case.injection_probe)
    injection_result = str(evidence.observations[injection_index].result["result"])
    cases = list(corpus.cases)
    cases[injection_index] = replace(
        cases[injection_index], forbidden_exact_results=(injection_result,)
    )
    injection_corpus = replace(corpus, cases=tuple(cases))
    with pytest.raises(SelectedTextAcceptanceError, match="prompt injection integrity"):
        qualify_selected_text(injection_corpus, evidence)


def test_metric_enforcement_checks_each_exact_boundary_and_refusal():
    accepted = (0.9, 0.9, 1.0, 1.0, 1.0, 30_000)
    acceptance._enforce_metrics(accepted, SelectedTextPolicy())
    faults = (
        ((0.89, 0.9, 1.0, 1.0, 1.0, 30_000), "operation term recall is below gate"),
        ((0.9, 0.89, 1.0, 1.0, 1.0, 30_000), "task term recall is below gate"),
        ((0.9, 0.9, 0.99, 1.0, 1.0, 30_000), "Hebrew script integrity is below gate"),
        ((0.9, 0.9, 1.0, 0.99, 1.0, 30_000), "prompt injection integrity is below gate"),
        ((0.9, 0.9, 1.0, 1.0, 0.99, 30_000), "provider route integrity is below gate"),
        ((0.9, 0.9, 1.0, 1.0, 1.0, 30_001), "operation latency exceeds gate"),
    )
    for metrics, detail in faults:
        with pytest.raises(SelectedTextAcceptanceError) as captured:
            acceptance._enforce_metrics(metrics, SelectedTextPolicy())
        assert (captured.value.code, captured.value.detail) == ("quality-failed", detail)


def test_scoring_exposes_each_term_script_injection_route_and_latency_component(tmp_path):
    corpus, path = _write_evidence(tmp_path)
    evidence = load_selected_text_evidence(path)
    baseline = acceptance._score(corpus, evidence.observations)
    assert baseline == (1.0, 1.0, 1.0, 1.0, 1.0, 115)

    expected_terms = sum(len(case.expected_terms) for case in corpus.cases)
    operation_observations = list(evidence.observations)
    first = operation_observations[0]
    first_result = dict(first.result)
    first_result["result"] = "checksum digest"
    operation_observations[0] = replace(first, result=first_result)
    assert (
        acceptance._score(corpus, operation_observations)[0]
        == (expected_terms - 1) / expected_terms
    )

    expected_tasks = sum(len(case.expected_task_terms) for case in corpus.cases)
    task_index = next(index for index, case in enumerate(corpus.cases) if case.expected_task_terms)
    task_observations = list(evidence.observations)
    task_result = dict(task_observations[task_index].result)
    task_result["tasks"] = ["missing"]
    task_observations[task_index] = replace(task_observations[task_index], result=task_result)
    missing_tasks = len(corpus.cases[task_index].expected_task_terms)
    assert (
        acceptance._score(corpus, task_observations)[1]
        == (expected_tasks - missing_tasks) / expected_tasks
    )

    hebrew_indices = [index for index, case in enumerate(corpus.cases) if case.require_hebrew]
    script_observations = list(evidence.observations)
    script_result = dict(script_observations[hebrew_indices[0]].result)
    script_result["result"] = "English only"
    script_observations[hebrew_indices[0]] = replace(
        script_observations[hebrew_indices[0]], result=script_result
    )
    assert acceptance._score(corpus, script_observations)[2] == (len(hebrew_indices) - 1) / len(
        hebrew_indices
    )

    route_observations = list(evidence.observations)
    route_result = dict(route_observations[0].result)
    route_result["providerId"] = "wrong-gpu"
    route_observations[0] = replace(route_observations[0], result=route_result)
    assert acceptance._score(corpus, route_observations)[4] == 15 / 16

    latency_observations = list(evidence.observations)
    latency_observations[0] = replace(latency_observations[0], latency_ms=999)
    assert acceptance._score(corpus, latency_observations)[5] == 999


def test_scoring_fails_closed_with_exact_errors_for_missing_invalid_or_changed_results(tmp_path):
    corpus, path = _write_evidence(tmp_path)
    evidence = load_selected_text_evidence(path)

    with pytest.raises(SelectedTextAcceptanceError) as captured:
        acceptance._score(corpus, evidence.observations[:-1])
    assert (captured.value.code, captured.value.detail) == (
        "evidence-invalid",
        "every corpus case must appear once",
    )

    observations = list(evidence.observations)
    invalid = dict(observations[0].result)
    invalid.pop("version")
    observations[0] = replace(observations[0], result=invalid)
    with pytest.raises(SelectedTextAcceptanceError) as captured:
        acceptance._score(corpus, observations)
    assert (captured.value.code, captured.value.detail) == (
        "evidence-invalid",
        "observed result is invalid",
    )

    observations = list(evidence.observations)
    changed = dict(observations[0].result)
    changed["operation"] = "summarize"
    observations[0] = replace(observations[0], result=changed)
    with pytest.raises(SelectedTextAcceptanceError) as captured:
        acceptance._score(corpus, observations)
    assert (captured.value.code, captured.value.detail) == (
        "evidence-invalid",
        "observed operation changed",
    )


@pytest.mark.parametrize("latency", [-1, 300_001])
def test_scoring_rejects_out_of_range_latency_with_exact_field(latency, tmp_path):
    corpus, path = _write_evidence(tmp_path)
    evidence = load_selected_text_evidence(path)
    observations = list(evidence.observations)
    observations[0] = replace(observations[0], latency_ms=latency)

    with pytest.raises(SelectedTextAcceptanceError) as captured:
        acceptance._score(corpus, observations)
    assert (captured.value.code, captured.value.detail) == (
        "evidence-invalid",
        "latency is invalid",
    )


def test_scoring_handles_single_case_denominators_and_multiple_injection_probes(tmp_path):
    corpus, path = _write_evidence(tmp_path)
    evidence = load_selected_text_evidence(path)
    case = replace(
        corpus.cases[0],
        expected_terms=("alpha",),
        expected_task_terms=("first task",),
        require_hebrew=True,
        injection_probe=True,
        forbidden_exact_results=("PWNED",),
    )
    result = dict(evidence.observations[0].result)
    result.update(result="alpha עברית", tasks=["first", "task"])
    observation = replace(evidence.observations[0], result=result)
    metrics = acceptance._score(replace(corpus, cases=(case,)), (observation,))
    assert metrics[:5] == (1.0, 1.0, 1.0, 1.0, 1.0)

    cases = (case, replace(corpus.cases[1], injection_probe=True))
    observations = (
        observation,
        replace(evidence.observations[1], result=_result(cases[1], 1)),
    )
    assert acceptance._score(replace(corpus, cases=cases), observations)[3] == 1.0


def test_loader_refuses_malformed_missing_duplicate_and_oversized_evidence(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(SelectedTextAcceptanceError, match="cannot read evidence"):
        load_selected_text_evidence(missing)
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    with pytest.raises(SelectedTextAcceptanceError, match="evidence fields"):
        load_selected_text_evidence(invalid)
    corpus, path = _write_evidence(
        tmp_path,
        lambda value: value["observations"].append(copy.deepcopy(value["observations"][0])),
    )
    evidence = load_selected_text_evidence(path)
    with pytest.raises(SelectedTextAcceptanceError, match="every corpus case"):
        qualify_selected_text(corpus, evidence)


@pytest.mark.parametrize(
    ("mutate", "detail"),
    [
        (lambda value: value.update(extra=True), "corpus fields"),
        (lambda value: value.update(corpusVersion=2), "corpus identity"),
        (lambda value: value.update(cases=value["cases"][:2]), "case ids are incomplete"),
        (
            lambda value: value.update(
                cases=[case for case in value["cases"] if case["operation"] != "rewrite"]
            ),
            "every operation",
        ),
        (
            lambda value: [case.update(promptInjectionProbe=False) for case in value["cases"]],
            "Hebrew injection probe",
        ),
        (lambda value: value["cases"][0].update(extra=True), "case fields"),
        (lambda value: value["cases"][0].update(operation="translate"), "case operation"),
    ],
)
def test_corpus_loader_refuses_identity_coverage_and_case_drift(tmp_path, mutate, detail):
    document = json.loads(CORPUS.read_text())
    mutate(document)
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SelectedTextAcceptanceError, match=detail):
        load_selected_text_corpus(path)


def test_acceptance_parsing_helpers_refuse_malformed_values(tmp_path, monkeypatch):
    failures = (
        lambda: acceptance._mapping([], "item"),
        lambda: acceptance._sequence("text", "items"),
        lambda: acceptance._texts(["same", "same"], "terms"),
        lambda: acceptance._text("", "text", 5, "bad"),
        lambda: acceptance._identifier("Bad ID", "id", "bad"),
        lambda: acceptance._digest("bad", "digest"),
        lambda: acceptance._integer(True, "integer", 0, 2),
        lambda: acceptance._boolean(1, "boolean"),
        lambda: acceptance._parse_model({}),
        lambda: acceptance._parse_model(
            {
                "modelSha256": "a" * 64,
                "runtimeVersion": "1",
                "deviceName": "gpu",
                "load": {},
            }
        ),
        lambda: acceptance._parse_observation({}),
        lambda: acceptance._parse_safety({}),
    )
    for failure in failures:
        with pytest.raises(SelectedTextAcceptanceError):
            failure()

    missing = tmp_path / "missing-corpus"
    with pytest.raises(SelectedTextAcceptanceError, match="cannot read corpus"):
        acceptance._bounded_bytes(missing, 10, "corpus")
    oversized = tmp_path / "oversized"
    oversized.write_bytes(b"123")
    with pytest.raises(SelectedTextAcceptanceError, match="oversized"):
        acceptance._bounded_bytes(oversized, 2, "corpus")
    with pytest.raises(SelectedTextAcceptanceError, match="not JSON"):
        acceptance._json_object(b"not-json", "corpus")
    with pytest.raises(SelectedTextAcceptanceError, match="must be an object"):
        acceptance._json_object(b"[]", "corpus")

    with pytest.raises(SelectedTextAcceptanceError, match="policy ratios"):
        acceptance._validate_policy(SelectedTextPolicy(minimum_operation_term_recall=True))
    with pytest.raises(SelectedTextAcceptanceError, match="policy latency"):
        acceptance._validate_policy(SelectedTextPolicy(maximum_p95_latency_ms=0))

    fake_module = tmp_path / "a" / "b" / "c" / "selected_text_acceptance.py"
    monkeypatch.setattr(acceptance, "__file__", str(fake_module))
    packaged = tmp_path / "a" / "b" / "evaluation-corpora" / "selected-text-v1.json"
    packaged.parent.mkdir(parents=True)
    packaged.write_text("{}", encoding="utf-8")
    assert acceptance._corpus_path() == packaged
    packaged.unlink()
    assert acceptance._corpus_path() == tmp_path / "evaluation-corpora/selected-text-v1.json"


def test_corpus_reader_preserves_missing_oversized_and_malformed_errors(tmp_path):
    missing = tmp_path / "missing"
    oversized = tmp_path / "oversized"
    malformed = tmp_path / "malformed"
    oversized.write_bytes(b"{" + b" " * acceptance.MAX_CORPUS_BYTES)
    malformed.write_text("{", encoding="utf-8")

    cases = (
        (missing, "cannot read corpus"),
        (oversized, "corpus is oversized"),
        (malformed, "corpus is not JSON"),
    )
    for path, detail in cases:
        with pytest.raises(SelectedTextAcceptanceError) as caught:
            load_selected_text_corpus(path)
        assert (caught.value.code, caught.value.detail) == ("corpus-invalid", detail)


def test_manifest_thresholds_match_the_selected_text_model_catalog():
    manifest = json.loads((ROOT / "plugin-manifests/selected-text-tools.json").read_text())
    catalog = json.loads((ROOT / "generation-models/dictalm2-hebrew.json").read_text())
    targets = {item["metric"]: item["target"] for item in manifest["acceptance"]}
    evaluation = catalog["evaluation"]

    assert validate_workload_document(manifest) == []
    assert manifest["version"] == "1.1.0"
    assert targets == {
        "operation-term-recall": evaluation["minimumOperationTermRecall"],
        "task-term-recall": evaluation["minimumTaskTermRecall"],
        "hebrew-target-script-integrity": evaluation["minimumTargetScriptIntegrity"],
        "prompt-injection-integrity": evaluation["minimumPromptInjectionIntegrity"],
        "provider-route-integrity": evaluation["minimumProviderRouteIntegrity"],
        "p95-operation-latency": evaluation["maximumP95LatencyMs"],
        "cancellation-latency": evaluation["maximumCancellationLatencyMs"],
    }


def test_cli_writes_validated_report_and_fails_without_evidence(tmp_path, capsys):
    corpus, evidence = _write_evidence(tmp_path)
    expected = qualify_selected_text(corpus, load_selected_text_evidence(evidence)).document()
    output = tmp_path / "report.json"

    assert main(["--evidence", str(evidence), "--output", str(output)]) == 0
    assert (
        validate_document("selected-text-acceptance.schema.json", json.loads(output.read_text()))
        == []
    )
    assert output.read_bytes() == json.dumps(expected, separators=(",", ":")).encode("utf-8")
    assert json.loads(capsys.readouterr().out)["qualified"] is True

    output.unlink()
    assert main(["--evidence", str(tmp_path / "missing"), "--output", str(output)]) == 1
    assert not output.exists()
    assert "evidence-invalid" in capsys.readouterr().err


def test_builder_refuses_an_invalid_generated_report_directly(tmp_path, monkeypatch):
    corpus, path = _write_evidence(tmp_path)
    evidence = load_selected_text_evidence(path)
    real_validate = validate_document
    monkeypatch.setattr(
        acceptance,
        "validate_document",
        lambda name, value: (
            ["builder drift"]
            if name == "selected-text-acceptance.schema.json"
            else real_validate(name, value)
        ),
    )

    with pytest.raises(SelectedTextAcceptanceError) as caught:
        qualify_selected_text(corpus, evidence)
    assert (caught.value.code, caught.value.detail) == ("report-invalid", "builder drift")


def test_cli_help_and_required_arguments_keep_the_public_contract(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert help_text.startswith("usage: omnitensor-qualify-selected-text")
    assert "Validate digest-bound selected-text operational evidence" in help_text
    assert "--evidence" in help_text and "--corpus" in help_text and "--output" in help_text

    with pytest.raises(SystemExit) as missing:
        main(["--output", "report.json"])
    assert missing.value.code == 2
    assert "the following arguments are required" in capsys.readouterr().err


def test_cli_preserves_unicode_stdout_and_ascii_safe_report_bytes(tmp_path, monkeypatch, capsys):
    class Report:
        @staticmethod
        def document():
            return {"text": "שלום"}

    monkeypatch.setattr(acceptance, "load_selected_text_corpus", lambda _path: object())
    monkeypatch.setattr(acceptance, "load_selected_text_evidence", lambda _path: object())
    monkeypatch.setattr(acceptance, "qualify_selected_text", lambda _corpus, _evidence: Report())
    output = tmp_path / "report.json"

    assert main(["--evidence", "evidence.json", "--output", str(output)]) == 0
    assert (
        capsys.readouterr().out == json.dumps({"text": "שלום"}, indent=2, ensure_ascii=False) + "\n"
    )
    assert output.read_text(encoding="utf-8") == json.dumps({"text": "שלום"}, separators=(",", ":"))
