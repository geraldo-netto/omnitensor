from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

import omnitensor.plugins.document_acceptance as acceptance_module
from omnitensor.plugins.document_acceptance import (
    MAX_CORPUS_BYTES,
    MAX_EVIDENCE_BYTES,
    PRIMARY_MODEL_SHA256,
    DocumentAcceptanceError,
    DocumentAcceptancePolicy,
    _boolean,
    _bounded_bytes,
    _bounded_text,
    _digest,
    _identifier,
    _integer,
    _json_object,
    _mapping,
    _optional_sequence,
    _parse_source,
    _score_observation,
    _sequence,
    load_document_acceptance_corpus,
    load_document_acceptance_evidence,
    main,
    qualify_document_questions,
)
from omnitensor.registry import validate_document, validate_workload_document

ROOT = Path(__file__).parents[1]
CORPUS = ROOT / "evaluation-corpora/document-question-v1.json"
EVIDENCE = ROOT / "acceptance-evidence/document-question-rx6600xt-evidence.json"
BGE_REPORT = ROOT / "acceptance-evidence/bge-rx6600xt-report.json"
REPORT = ROOT / "acceptance-evidence/document-question-rx6600xt-report.json"


def _assert_error(call, code: str, detail: str | None = None):
    with pytest.raises(DocumentAcceptanceError) as caught:
        call()
    assert caught.value.code == code
    if detail is not None:
        assert caught.value.detail == detail
    return caught.value


def _write_corpus(tmp_path: Path, mutate) -> Path:
    value = json.loads(CORPUS.read_text(encoding="utf-8"))
    mutate(value)
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _write_evidence(tmp_path: Path, mutate=lambda _value: None) -> Path:
    value = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    mutate(value)
    (tmp_path / "bge-rx6600xt-report.json").write_bytes(BGE_REPORT.read_bytes())
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _accepted(tmp_path: Path | None = None):
    corpus = load_document_acceptance_corpus()
    evidence = load_document_acceptance_evidence(
        EVIDENCE if tmp_path is None else _write_evidence(tmp_path)
    )
    return corpus, evidence, qualify_document_questions(corpus, evidence)


def test_evidence_digest_is_bound_to_the_parsed_descriptor_snapshot(tmp_path, monkeypatch):
    path = _write_evidence(tmp_path)
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
    evidence = load_document_acceptance_evidence(path)

    assert evidence.evidence_sha256 == hashlib.sha256(original).hexdigest()
    with real_open(path, "rb") as stream:
        assert stream.read() == replacement


def test_archived_named_gpu_evidence_reproduces_exact_qualified_report():
    corpus, evidence, report = _accepted()
    expected = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report.document() == expected
    assert validate_document("document-question-acceptance.schema.json", expected) == []
    assert corpus.corpus_id == "grounded-selected-files-v1"
    assert len(corpus.cases) == 6
    assert corpus.sha256 == "da58477462f26065d23813727e96349ed104171210685051f6c3ca851bfd05d4"
    assert evidence.evidence_sha256 == expected["evidenceSha256"]
    assert report.embedding_device_name == "AMD Radeon RX 6600 XT (RADV NAVI23)"
    assert report.generator_device_name == report.embedding_device_name
    assert report.retrieval_recall == report.grounded_term_recall == 1.0
    assert report.citation_integrity == 1.0
    assert report.p95_latency_ms == 21907
    assert report.peak_memory_bytes == 7637553152
    assert report.cancellation_latency_ms == 269
    assert report.revocation_latency_ms == 83


def test_public_corpus_and_report_contain_no_private_path_or_unbounded_claim():
    corpus = CORPUS.read_text(encoding="utf-8")
    report = REPORT.read_text(encoding="utf-8")
    assert "/home/" not in corpus
    assert "private:" not in report
    assert '"arbitraryPrivateCorpora": "not-claimed"' in report
    assert '"npu": "not-qualified"' in report
    assert '"tpu": "not-qualified"' in report


def test_qwen3_catalog_is_official_revision_pinned_and_gpu_default():
    catalog = json.loads((ROOT / "generation-models/qwen3-8b.json").read_text())
    source = catalog["source"]
    revision = catalog["upstream"]["revision"]

    assert catalog["id"] == "qwen3-8b-q4-k-m"
    assert catalog["license"]["spdx"] == "Apache-2.0"
    assert len(revision) == 40 and revision in source["uri"]
    assert source == {
        "uri": (
            f"https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/{revision}/Qwen3-8B-Q4_K_M.gguf"
        ),
        "filename": "Qwen3-8B-Q4_K_M.gguf",
        "sha256": PRIMARY_MODEL_SHA256,
        "sizeBytes": 5027783488,
    }
    assert [(item["accelerator"], item["default"]) for item in catalog["providers"]] == [
        ("gpu", True),
        ("npu", False),
    ]


def test_manifest_declares_the_same_operational_thresholds_and_answer_bound():
    manifest = json.loads((ROOT / "plugin-manifests/ask-selected-files.json").read_text())
    catalog = json.loads((ROOT / "generation-models/qwen3-8b.json").read_text())
    targets = {item["metric"]: item["target"] for item in manifest["acceptance"]}
    policy = catalog["evaluation"]

    assert validate_workload_document(manifest) == []
    assert targets == {
        "retrieval-recall": policy["minimumRetrievalRecall"],
        "grounded-term-recall": policy["minimumGroundedTermRecall"],
        "citation-integrity": policy["minimumCitationIntegrity"],
        "p95-question-latency": policy["maximumP95LatencyMs"],
        "cancellation-latency": policy["maximumCancellationLatencyMs"],
        "grant-revocation-latency": policy["maximumRevocationLatencyMs"],
    }
    private_schema = json.loads((ROOT / "schemas/document-question-answer.schema.json").read_text())
    public_schema = json.loads((ROOT / "schemas/document-question-result.schema.json").read_text())
    assert private_schema["properties"]["answer"]["maxLength"] == 1024
    assert public_schema["properties"]["answer"]["maxLength"] == 16384


def test_cli_writes_the_exact_validated_report(tmp_path, capsys):
    output = tmp_path / "report.json"
    assert (
        main(["--evidence", str(EVIDENCE), "--corpus", str(CORPUS), "--output", str(output)]) == 0
    )
    expected = json.loads(REPORT.read_text())
    assert json.loads(output.read_text()) == expected
    assert output.read_bytes() == json.dumps(expected, separators=(",", ":")).encode("utf-8")
    assert capsys.readouterr().out == json.dumps(expected, indent=2) + "\n"


def test_cli_fails_closed_without_writing_a_report(tmp_path, capsys):
    output = tmp_path / "report.json"
    assert main(["--evidence", str(tmp_path / "missing"), "--output", str(output)]) == 1
    assert not output.exists()
    assert capsys.readouterr().err.startswith("document acceptance failed: evidence-invalid:")


def test_cli_refuses_an_invalid_generated_report(tmp_path, monkeypatch, capsys):
    output = tmp_path / "report.json"
    real_validate = validate_document
    monkeypatch.setattr(
        acceptance_module,
        "validate_document",
        lambda name, value: (
            ["drift"]
            if name == "document-question-acceptance.schema.json"
            else real_validate(name, value)
        ),
    )
    assert main(["--evidence", str(EVIDENCE), "--output", str(output)]) == 1
    assert not output.exists()
    assert "report-invalid: drift" in capsys.readouterr().err


def test_builder_refuses_an_invalid_generated_report_directly(tmp_path, monkeypatch):
    corpus = load_document_acceptance_corpus()
    evidence = load_document_acceptance_evidence(_write_evidence(tmp_path))
    real_validate = validate_document
    monkeypatch.setattr(
        acceptance_module,
        "validate_document",
        lambda name, value: (
            ["builder drift"]
            if name == "document-question-acceptance.schema.json"
            else real_validate(name, value)
        ),
    )

    _assert_error(
        lambda: qualify_document_questions(corpus, evidence),
        "report-invalid",
        "builder drift",
    )


def test_cli_help_names_inputs_and_output(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert help_text.startswith("usage: omnitensor-qualify-document-questions")
    assert "Validate digest-bound Document Intelligence operational evidence" in help_text
    assert "--evidence" in help_text and "--corpus" in help_text and "--output" in help_text


@pytest.mark.parametrize(
    "arguments",
    [
        ["--output", "report.json"],
        ["--evidence", str(EVIDENCE)],
    ],
)
def test_cli_requires_evidence_and_output(arguments, capsys):
    with pytest.raises(SystemExit) as caught:
        main(arguments)
    assert caught.value.code == 2
    assert "the following arguments are required" in capsys.readouterr().err


def test_cli_honors_an_explicit_corpus_path(tmp_path, capsys):
    invalid_corpus = tmp_path / "corpus.json"
    output = tmp_path / "report.json"
    invalid_corpus.write_text("{}", encoding="utf-8")

    assert (
        main(
            [
                "--evidence",
                str(EVIDENCE),
                "--corpus",
                str(invalid_corpus),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert not output.exists()
    assert "corpus-invalid: corpus fields or version are invalid" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("mutate", "code", "detail"),
    [
        (
            lambda value: value.update(corpusVersion=2),
            "corpus-invalid",
            "corpus fields or version are invalid",
        ),
        (
            lambda value: value.update(extra=True),
            "corpus-invalid",
            "corpus fields or version are invalid",
        ),
        (
            lambda value: value.update(license="private"),
            "corpus-invalid",
            "corpus license must be CC0-1.0",
        ),
        (
            lambda value: value.update(provenance=""),
            "corpus-invalid",
            "corpus provenance must be bounded text",
        ),
        (
            lambda value: value.update(id="Bad ID"),
            "corpus-invalid",
            "corpus id must be a kebab-case identifier",
        ),
        (
            lambda value: value.update(cases=[]),
            "corpus-invalid",
            "corpus cases must be a non-empty sequence",
        ),
        (
            lambda value: value.update(cases=value["cases"][:4]),
            "corpus-invalid",
            "corpus needs at least five uniquely named cases",
        ),
        (
            lambda value: value["cases"].__setitem__(1, copy.deepcopy(value["cases"][0])),
            "corpus-invalid",
            "corpus needs at least five uniquely named cases",
        ),
        (
            lambda value: [item.update(promptInjectionProbe=False) for item in value["cases"]],
            "corpus-invalid",
            "corpus needs a prompt-injection probe",
        ),
        (
            lambda value: value["cases"][0].update(extra=True),
            "corpus-invalid",
            "corpus case fields are invalid",
        ),
        (
            lambda value: value["cases"][0].update(question=""),
            "corpus-invalid",
            "question must be bounded text",
        ),
        (
            lambda value: value["cases"][0].update(caseId="Bad"),
            "corpus-invalid",
            "case id must be a kebab-case identifier",
        ),
        (
            lambda value: value["cases"][0].update(sources=[]),
            "corpus-invalid",
            "sources must be a non-empty sequence",
        ),
        (
            lambda value: value["cases"][0].update(sources=value["cases"][0]["sources"] * 9),
            "corpus-invalid",
            "case sources must be unique and bounded",
        ),
        (
            lambda value: value["cases"][0].update(expectedTerms=[]),
            "corpus-invalid",
            "expected terms must be a non-empty sequence",
        ),
        (
            lambda value: value["cases"][0].update(expectedFileIds=["selected-file-9"]),
            "corpus-invalid",
            "expected file ids are invalid",
        ),
        (
            lambda value: value["cases"][0].update(promptInjectionProbe="yes"),
            "corpus-invalid",
            "prompt-injection probe must be boolean",
        ),
        (
            lambda value: value["cases"][0].update(promptInjectionProbe=True, forbiddenTerms=[]),
            "corpus-invalid",
            "injection probes need forbidden terms",
        ),
        (
            lambda value: value["cases"][0]["sources"][0].update(extra=True),
            "corpus-invalid",
            "case source fields are invalid",
        ),
        (
            lambda value: value["cases"][0]["sources"][0].update(fileName="../secret"),
            "corpus-invalid",
            "source file name must be a basename",
        ),
        (
            lambda value: value["cases"][0]["sources"][0].update(content=""),
            "corpus-invalid",
            "source content must be bounded text",
        ),
    ],
)
def test_corpus_rejects_schema_identity_privacy_and_boundary_drift(tmp_path, mutate, code, detail):
    path = _write_corpus(tmp_path, mutate)
    _assert_error(lambda: load_document_acceptance_corpus(path), code, detail)


def test_corpus_reads_are_bounded_and_json_only(tmp_path):
    missing = tmp_path / "missing.json"
    oversized = tmp_path / "oversized.json"
    malformed = tmp_path / "malformed.json"
    non_object = tmp_path / "array.json"
    oversized.write_bytes(b"{" + b" " * MAX_CORPUS_BYTES)
    malformed.write_text("{", encoding="utf-8")
    non_object.write_text("[]", encoding="utf-8")

    _assert_error(
        lambda: load_document_acceptance_corpus(missing), "corpus-invalid", "cannot read corpus"
    )
    _assert_error(
        lambda: load_document_acceptance_corpus(oversized),
        "corpus-invalid",
        "corpus exceeds its byte limit",
    )
    _assert_error(
        lambda: load_document_acceptance_corpus(malformed), "corpus-invalid", "corpus is not JSON"
    )
    _assert_error(
        lambda: load_document_acceptance_corpus(non_object),
        "corpus-invalid",
        "corpus must be an object",
    )


def test_source_contract_rejects_non_objects_with_stable_context():
    _assert_error(
        lambda: _parse_source([]),
        "corpus-invalid",
        "case source must be an object",
    )


def test_source_contract_enforces_exact_name_and_content_bounds():
    name = "a" * 251 + ".txt"
    source = _parse_source({"fileName": name, "content": "x" * 2048})
    assert source.file_name == name
    assert source.content == "x" * 2048

    _assert_error(
        lambda: _parse_source({"fileName": "a" * 256, "content": "x"}),
        "corpus-invalid",
        "source file name must be bounded text",
    )
    _assert_error(
        lambda: _parse_source({"fileName": "source.txt", "content": "x" * 2049}),
        "corpus-invalid",
        "source content must be bounded text",
    )
    for unsafe_name in ("folder/source.txt", "folder\\source.txt"):
        _assert_error(
            lambda unsafe_name=unsafe_name: _parse_source(
                {"fileName": unsafe_name, "content": "x"}
            ),
            "corpus-invalid",
            "source file name must be a basename",
        )


@pytest.mark.parametrize(
    ("mutate", "detail"),
    [
        (lambda value: value.update(evidenceVersion=2), "evidence fields or version are invalid"),
        (lambda value: value.update(extra=True), "evidence fields or version are invalid"),
        (lambda value: value["embedding"].update(extra=True), "model evidence fields are invalid"),
        (lambda value: value["generation"].update(extra=True), "model evidence fields are invalid"),
        (
            lambda value: value["generation"]["load"].update(extra=True),
            "generation load fields are invalid",
        ),
        (
            lambda value: value.update(corpusSha256="A" * 64),
            "corpus digest must be lowercase SHA-256",
        ),
        (
            lambda value: value["embedding"].update(reportPath=""),
            "BGE report path must be bounded text",
        ),
        (
            lambda value: value["embedding"].update(reportSha256="0"),
            "BGE report digest must be lowercase SHA-256",
        ),
        (
            lambda value: value["generation"].update(modelSha256="0"),
            "Qwen model digest must be lowercase SHA-256",
        ),
        (
            lambda value: value["generation"].update(runtimeVersion=""),
            "runtime version must be bounded text",
        ),
        (
            lambda value: value["generation"]["load"].update(totalModelLayers=0),
            "total model layers is outside its bound",
        ),
        (
            lambda value: value["generation"]["load"].update(acceleratorLayers=True),
            "accelerator layers is outside its bound",
        ),
        (
            lambda value: value["generation"]["load"].update(cpuFallback=0),
            "CPU fallback must be boolean",
        ),
        (lambda value: value.update(observations=[]), "observations must be a non-empty sequence"),
        (
            lambda value: value["observations"][0].update(extra=True),
            "observation fields are invalid",
        ),
        (
            lambda value: value["observations"][0].update(caseId="Bad"),
            "case id must be a kebab-case identifier",
        ),
        (
            lambda value: value["observations"][0].update(retrievedFileIds=[]),
            "retrieved file ids must be a non-empty sequence",
        ),
        (
            lambda value: value["observations"][0].update(latencyMs=0),
            "observation latency is outside its bound",
        ),
        (
            lambda value: value["observations"][0].update(peakMemoryBytes=False),
            "observation memory is outside its bound",
        ),
        (lambda value: value["safety"].update(extra=True), "safety evidence fields are invalid"),
        (lambda value: value["safety"].update(indexRecovered=1), "indexRecovered must be boolean"),
        (
            lambda value: value["safety"].update(cancellationLatencyMs=0),
            "cancellation latency is outside its bound",
        ),
    ],
)
def test_evidence_parser_rejects_closed_contract_drift(tmp_path, mutate, detail):
    path = _write_evidence(tmp_path, mutate)
    _assert_error(lambda: load_document_acceptance_evidence(path), "evidence-invalid", detail)


def test_evidence_read_is_bounded_and_object_only(tmp_path):
    malformed = tmp_path / "malformed"
    oversized = tmp_path / "oversized"
    array = tmp_path / "array"
    malformed.write_text("{", encoding="utf-8")
    oversized.write_bytes(b"{" + b" " * MAX_EVIDENCE_BYTES)
    array.write_text("[]", encoding="utf-8")
    for path in (tmp_path / "missing", malformed, oversized, array):
        error = _assert_error(
            lambda path=path: load_document_acceptance_evidence(path), "evidence-invalid"
        )
        assert (
            error.detail.startswith("cannot read evidence")
            or error.detail == "evidence fields or version are invalid"
        )


@pytest.mark.parametrize(
    ("change", "code", "detail"),
    [
        ({"corpus_sha256": "0" * 64}, "evidence-stale", "evidence names another corpus version"),
        ({"primary_model_sha256": "0" * 64}, "evidence-stale", "evidence names another primary model"),
        ({"generator_device_name": "Vulkan"}, "device-unqualified", "generator GPU must be named"),
        (
            {"generator_device_name": "llvmpipe"},
            "device-unqualified",
            "generator GPU must be named",
        ),
    ],
)
def test_qualification_rejects_stale_identity_or_unnamed_device(tmp_path, change, code, detail):
    corpus, evidence, _report = _accepted(tmp_path)
    _assert_error(
        lambda: qualify_document_questions(corpus, replace(evidence, **change)), code, detail
    )


def test_qualification_rejects_partial_or_cpu_generation_load(tmp_path):
    corpus, evidence, _report = _accepted(tmp_path)
    for load in (
        replace(evidence.generator_load, accelerator_layers=28),
        replace(evidence.generator_load, cpu_fallback=True),
        replace(evidence.generator_load, backend="cpu"),
    ):
        _assert_error(
            lambda load=load: qualify_document_questions(
                corpus, replace(evidence, generator_load=load)
            ),
            "device-unqualified",
            "llama.cpp did not prove complete Vulkan model-layer offload",
        )


@pytest.mark.parametrize(
    ("mutate_report", "change", "code"),
    [
        (lambda value: value.update(kind="other"), {}, "device-unqualified"),
        (lambda value: value.pop("cpuFallback"), {}, "device-unqualified"),
        (lambda value: value.update(cpuFallback=True), {}, "device-unqualified"),
        (lambda value: value.update(cpuFallback="forbidden"), {}, "device-unqualified"),
        (
            lambda value: (
                value.pop("cpuFallback"),
                value["limitations"].update(cpuFallback="forbidden"),
            ),
            {},
            "device-unqualified",
        ),
        (lambda value: value.update(nativeArtifactSha256="0" * 64), {}, "device-unqualified"),
        (lambda value: value["nativeGate"].update(accepted=False), {}, "device-unqualified"),
        (lambda value: value["device"].update(name="llvmpipe"), {}, "device-unqualified"),
        (lambda _value: None, {"bge_artifact_sha256": "0" * 64}, "device-unqualified"),
    ],
)
def test_qualification_rejects_changed_or_software_bge_evidence(
    tmp_path, mutate_report, change, code
):
    corpus, evidence, _report = _accepted(tmp_path)
    report = json.loads(evidence.bge_report_path.read_text())
    mutate_report(report)
    evidence.bge_report_path.write_text(json.dumps(report))
    changed = replace(
        evidence,
        bge_report_sha256=hashlib.sha256(evidence.bge_report_path.read_bytes()).hexdigest(),
        **change,
    )
    _assert_error(lambda: qualify_document_questions(corpus, changed), code)


def test_qualification_rejects_missing_changed_or_nonobject_bge_report(tmp_path):
    corpus, evidence, _report = _accepted(tmp_path)
    missing = replace(evidence, bge_report_path=tmp_path / "missing")
    _assert_error(lambda: qualify_document_questions(corpus, missing), "evidence-stale")
    evidence.bge_report_path.write_text("[]", encoding="utf-8")
    changed = replace(
        evidence,
        bge_report_sha256=hashlib.sha256(evidence.bge_report_path.read_bytes()).hexdigest(),
    )
    _assert_error(
        lambda: qualify_document_questions(corpus, changed),
        "evidence-invalid",
        "BGE report must be an object",
    )
    evidence.bge_report_path.write_text("{", encoding="utf-8")
    malformed = replace(
        evidence,
        bge_report_sha256=hashlib.sha256(evidence.bge_report_path.read_bytes()).hexdigest(),
    )
    _assert_error(
        lambda: qualify_document_questions(corpus, malformed),
        "evidence-invalid",
        "BGE report cannot be read",
    )


def test_qualification_requires_every_case_exactly_once(tmp_path):
    corpus, evidence, _report = _accepted(tmp_path)
    for observations in (
        evidence.observations[:-1],
        evidence.observations[:-1] + (evidence.observations[0],),
    ):
        _assert_error(
            lambda observations=observations: qualify_document_questions(
                corpus, replace(evidence, observations=observations)
            ),
            "evidence-invalid",
            "every corpus case must appear once",
        )


def test_score_refuses_an_observation_from_another_case():
    corpus, evidence, _report = _accepted()
    _assert_error(
        lambda: _score_observation(corpus.cases[0], evidence.observations[1]),
        "evidence-invalid",
        "observation identity disagrees",
    )


@pytest.mark.parametrize(
    ("field", "value", "code", "detail"),
    [
        ("case_id", "other", "evidence-invalid", "every corpus case must appear once"),
        ("latency_ms", 0, "evidence-invalid", "observation latency is outside its bound"),
        ("peak_memory_bytes", 0, "evidence-invalid", "observation memory is outside its bound"),
        (
            "retrieved_file_ids",
            ("selected-file-1", "selected-file-1"),
            "evidence-invalid",
            "retrieval identities are invalid",
        ),
        (
            "retrieved_file_ids",
            ("selected-file-9",),
            "evidence-invalid",
            "retrieval identities are invalid",
        ),
    ],
)
def test_observation_identity_and_bounds_fail_closed(tmp_path, field, value, code, detail):
    corpus, evidence, _report = _accepted(tmp_path)
    changed = replace(evidence.observations[0], **{field: value})
    observations = (changed, *evidence.observations[1:])
    _assert_error(
        lambda: qualify_document_questions(corpus, replace(evidence, observations=observations)),
        code,
        detail,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("providerId", "other-provider"),
        ("accelerator", "npu"),
    ],
)
def test_public_answer_identity_is_bound_to_shared_gpu_provider(tmp_path, field, value):
    corpus, evidence, _report = _accepted(tmp_path)
    result = copy.deepcopy(evidence.observations[0].result)
    result[field] = value
    changed = replace(evidence.observations[0], result=result)
    _assert_error(
        lambda: qualify_document_questions(
            corpus, replace(evidence, observations=(changed, *evidence.observations[1:]))
        ),
        "evidence-invalid",
        "answer identity disagrees",
    )


@pytest.mark.parametrize("leak", ["private:job:span", "/home/alice/report", "\\home\\alice"])
def test_public_answer_cannot_leak_private_references_or_paths(tmp_path, leak):
    corpus, evidence, _report = _accepted(tmp_path)
    result = copy.deepcopy(evidence.observations[0].result)
    result["answer"] = f"answer {leak}"
    changed = replace(evidence.observations[0], result=result)
    _assert_error(
        lambda: qualify_document_questions(
            corpus, replace(evidence, observations=(changed, *evidence.observations[1:]))
        ),
        "privacy-failed",
        "answer leaked a private reference or path",
    )


def test_prompt_injection_and_stale_fact_are_rejected(tmp_path):
    corpus, evidence, _report = _accepted(tmp_path)
    for case_id, forbidden in (("prompt-injection", "PWNED"), ("changed-version", "8443")):
        index = next(i for i, item in enumerate(evidence.observations) if item.case_id == case_id)
        result = copy.deepcopy(evidence.observations[index].result)
        result["answer"] = forbidden
        observations = list(evidence.observations)
        observations[index] = replace(observations[index], result=result)
        _assert_error(
            lambda observations=tuple(observations): qualify_document_questions(
                corpus, replace(evidence, observations=observations)
            ),
            "quality-failed",
            "a forbidden term appeared in the answer",
        )


@given(field=st.sampled_from(["fileName", "sourceSha256", "page", "span", "textSha256"]))
def test_any_citation_identity_or_address_drift_fails_closed(field):
    corpus, evidence, _report = _accepted()
    result = copy.deepcopy(evidence.observations[0].result)
    citation = result["citations"][0]
    replacements = {
        "fileName": "other.txt",
        "sourceSha256": "0" * 64,
        "page": 2,
        "span": {"start": 1, "end": citation["span"]["end"]},
        "textSha256": "0" * 64,
    }
    citation[field] = replacements[field]
    changed = replace(evidence.observations[0], result=result)
    _assert_error(
        lambda: qualify_document_questions(
            corpus, replace(evidence, observations=(changed, *evidence.observations[1:]))
        ),
        "quality-failed",
        "citation address or digest drifted",
    )


def test_unsupported_or_repeated_citation_fails_closed(tmp_path):
    corpus, evidence, _report = _accepted(tmp_path)
    multi = evidence.observations[3]
    for citations, detail in (
        ([copy.deepcopy(multi.result["citations"][2 - 1])], "answer cited an unsupported file"),
        (multi.result["citations"] * 2, "citation source is absent or repeated"),
    ):
        result = copy.deepcopy(multi.result)
        if detail == "answer cited an unsupported file":
            result["citations"][0]["fileId"] = "selected-file-3"
        else:
            result["citations"] = copy.deepcopy(citations)
        changed = replace(multi, result=result)
        observations = (*evidence.observations[:3], changed, *evidence.observations[4:])
        _assert_error(
            lambda observations=observations: qualify_document_questions(
                corpus, replace(evidence, observations=observations)
            ),
            "quality-failed",
            detail,
        )


def test_invalid_public_answer_contract_fails_before_scoring(tmp_path):
    corpus, evidence, _report = _accepted(tmp_path)
    result = copy.deepcopy(evidence.observations[0].result)
    result["extra"] = True
    changed = replace(evidence.observations[0], result=result)
    _assert_error(
        lambda: qualify_document_questions(
            corpus, replace(evidence, observations=(changed, *evidence.observations[1:]))
        ),
        "quality-failed",
        "answer violates its public contract",
    )


@pytest.mark.parametrize(
    ("policy", "detail"),
    [
        (
            DocumentAcceptancePolicy(minimum_retrieval_recall=1.1),
            "minimum_retrieval_recall must be in [0, 1]",
        ),
        (
            DocumentAcceptancePolicy(minimum_grounded_term_recall=True),
            "minimum_grounded_term_recall must be in [0, 1]",
        ),
        (
            DocumentAcceptancePolicy(minimum_citation_integrity=-0.1),
            "minimum_citation_integrity must be in [0, 1]",
        ),
        (
            DocumentAcceptancePolicy(maximum_p95_latency_ms=0),
            "maximum_p95_latency_ms is outside its bound",
        ),
        (
            DocumentAcceptancePolicy(maximum_cancellation_latency_ms=0),
            "maximum_cancellation_latency_ms is outside its bound",
        ),
        (
            DocumentAcceptancePolicy(maximum_revocation_latency_ms=0),
            "maximum_revocation_latency_ms is outside its bound",
        ),
    ],
)
def test_policy_bounds_are_closed(tmp_path, policy, detail):
    corpus, evidence, _report = _accepted(tmp_path)
    _assert_error(
        lambda: qualify_document_questions(corpus, evidence, policy), "policy-invalid", detail
    )
    _assert_error(lambda: qualify_document_questions(corpus, evidence, object()), "policy-invalid")


@pytest.mark.parametrize(
    ("change", "policy", "detail"),
    [
        (
            "retrieval",
            DocumentAcceptancePolicy(minimum_retrieval_recall=1.0),
            "retrieval recall is below the gate",
        ),
        (
            "answer",
            DocumentAcceptancePolicy(minimum_grounded_term_recall=1.0),
            "grounded answer recall is below the gate",
        ),
        (
            "citation",
            DocumentAcceptancePolicy(minimum_citation_integrity=1.0),
            "citation integrity is below the gate",
        ),
        (
            "latency",
            DocumentAcceptancePolicy(maximum_p95_latency_ms=1),
            "question latency exceeds the gate",
        ),
    ],
)
def test_each_quality_threshold_is_enforced(tmp_path, change, policy, detail):
    corpus, evidence, _report = _accepted(tmp_path)
    observation = evidence.observations[0]
    if change == "retrieval":
        observation = replace(observation, retrieved_file_ids=("selected-file-2",))
    elif change == "answer":
        result = copy.deepcopy(observation.result)
        result["answer"] = "A supported but incomplete answer."
        observation = replace(observation, result=result)
    elif change == "citation":
        result = copy.deepcopy(evidence.observations[3].result)
        result["citations"] = result["citations"][:1]
        observation = replace(evidence.observations[3], result=result)
        evidence = replace(
            evidence,
            observations=(*evidence.observations[:3], observation, *evidence.observations[4:]),
        )
        _assert_error(
            lambda: qualify_document_questions(corpus, evidence, policy), "quality-failed", detail
        )
        return
    elif change == "latency":
        observation = replace(observation, latency_ms=2)
    else:
        raise AssertionError(f"unknown threshold: {change}")
    changed = replace(evidence, observations=(observation, *evidence.observations[1:]))
    _assert_error(
        lambda: qualify_document_questions(corpus, changed, policy), "quality-failed", detail
    )


@pytest.mark.parametrize(
    ("field", "value", "detail"),
    [
        ("malicious_document_refused", False, "every document safety probe must pass"),
        ("mutated_document_refused", False, "every document safety probe must pass"),
        ("grant_revocation_refused", False, "every document safety probe must pass"),
        ("stale_version_suppressed", False, "every document safety probe must pass"),
        ("index_recovered", False, "every document safety probe must pass"),
        ("private_fragments_discarded", False, "every document safety probe must pass"),
        ("cancellation_latency_ms", 2001, "cancellation latency exceeds the gate"),
        ("revocation_latency_ms", 501, "grant revocation latency exceeds the gate"),
    ],
)
def test_every_safety_probe_and_latency_is_mandatory(tmp_path, field, value, detail):
    corpus, evidence, _report = _accepted(tmp_path)
    changed = replace(evidence, safety=replace(evidence.safety, **{field: value}))
    _assert_error(lambda: qualify_document_questions(corpus, changed), "safety-failed", detail)


@pytest.mark.parametrize(
    ("call", "code", "detail"),
    [
        (lambda: _mapping([], "thing", "bad"), "bad", "thing must be an object"),
        (lambda: _sequence("x", "things", "bad"), "bad", "things must be a non-empty sequence"),
        (lambda: _sequence([], "things", "bad"), "bad", "things must be a non-empty sequence"),
        (lambda: _optional_sequence("x", "things", "bad"), "bad", "things must be a sequence"),
        (lambda: _bounded_text("", "text", 2, "bad"), "bad", "text must be bounded text"),
        (lambda: _bounded_text("abc", "text", 2, "bad"), "bad", "text must be bounded text"),
        (lambda: _identifier("Bad", "id", "bad"), "bad", "id must be a kebab-case identifier"),
        (
            lambda: _digest("A" * 64, "digest"),
            "evidence-invalid",
            "digest must be lowercase SHA-256",
        ),
        (
            lambda: _integer(True, "integer", 1, 2),
            "evidence-invalid",
            "integer is outside its bound",
        ),
        (lambda: _integer(3, "integer", 1, 2), "evidence-invalid", "integer is outside its bound"),
        (lambda: _boolean(1, "flag", "bad"), "bad", "flag must be boolean"),
        (lambda: _json_object(b"[1]", "value"), "value-invalid", "value must be an object"),
        (lambda: _json_object(b"{", "value"), "value-invalid", "value is not JSON"),
    ],
)
def test_contract_helpers_have_stable_fail_closed_errors(call, code, detail):
    _assert_error(call, code, detail)


def test_contract_helpers_accept_exact_boundaries(tmp_path):
    path = tmp_path / "value"
    path.write_bytes(b"{}")
    assert _bounded_bytes(path, 2, "value") == b"{}"
    assert _json_object(b"{}", "value") == {}
    assert _mapping({}, "value", "bad") == {}
    assert _sequence([1], "values", "bad") == [1]
    assert _optional_sequence([], "values", "bad") == []
    assert _bounded_text("ab", "text", 2, "bad") == "ab"
    assert _identifier("valid-id", "id", "bad") == "valid-id"
    assert _digest("a" * 64, "digest") == "a" * 64
    assert _integer(1, "integer", 1, 2) == 1
    assert _integer(2, "integer", 1, 2) == 2
    assert _boolean(False, "flag") is False


def test_bounded_byte_facade_preserves_missing_and_oversized_errors(tmp_path):
    with pytest.raises(DocumentAcceptanceError) as missing:
        _bounded_bytes(tmp_path / "missing", 2, "value")
    assert (missing.value.code, missing.value.detail) == (
        "value-invalid",
        "cannot read value",
    )

    oversized = tmp_path / "oversized"
    oversized.write_bytes(b"123")
    with pytest.raises(DocumentAcceptanceError) as too_large:
        _bounded_bytes(oversized, 2, "value")
    assert (too_large.value.code, too_large.value.detail) == (
        "value-invalid",
        "value exceeds its byte limit",
    )


def test_corpus_path_prefers_the_packaged_copy(tmp_path, monkeypatch):
    module = tmp_path / "omnitensor/plugins/document_acceptance.py"
    packaged = tmp_path / "omnitensor/evaluation-corpora/document-question-v1.json"
    packaged.parent.mkdir(parents=True)
    packaged.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(acceptance_module, "__file__", str(module))
    assert acceptance_module._corpus_path() == packaged
