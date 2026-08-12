from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

import omnitensor.plugins.qwen as qwen_module
from omnitensor.plugins.cancellation import CancellationReason, JobCancellationToken
from omnitensor.plugins.generation import (
    GenerationRequest,
    GenerationRouter,
    GenerationTask,
    ProviderGenerationError,
    parse_provider_descriptor,
)
from omnitensor.plugins.qwen import (
    MAX_CATALOG_BYTES,
    MAX_CORPUS_BYTES,
    EventProviderEvidence,
    EventProviderObservation,
    EventQualificationPolicy,
    LlamaCppVulkanQwenWorker,
    NativeLoadReport,
    OpenVinoNpuQwenWorker,
    QwenProviderError,
    load_event_corpus,
    load_qwen_catalog,
    qualify_event_provider,
)


class Progress:
    async def report(self, progress):
        return None


class NativeRuntime:
    def __init__(self, report, *, result="ok", load_error=None, generation_error=None):
        self.report = report
        self.result = result
        self.load_error = load_error
        self.generation_error = generation_error
        self.loads = []
        self.generations = []
        self.terminations = []

    async def load(self, artifacts, accelerator):
        self.loads.append((artifacts, accelerator))
        if self.load_error is not None:
            raise self.load_error
        return self.report

    async def generate(self, task, request, cancellation, progress):
        self.generations.append((task, request, cancellation, progress))
        if self.generation_error is not None:
            raise self.generation_error
        return self.result

    async def terminate(self, request_id):
        self.terminations.append(request_id)


def run(coroutine):
    return asyncio.run(coroutine)


def descriptor(accelerator="gpu", *, digest="a" * 64, qualified=True):
    runtime = "llama.cpp-vulkan" if accelerator == "gpu" else "openvino-genai-npu"
    return parse_provider_descriptor(
        {
            "providerId": f"qwen-events-{accelerator}",
            "accelerator": accelerator,
            "runtime": runtime,
            "qualified": qualified,
            "provenance": {
                "modelId": "qwen2-5-vl-7b-instruct",
                "modelVersion": "1.0.0",
                "sha256": digest,
                "sourceUri": "https://example.invalid/qwen.gguf",
                "licenseSpdx": "Apache-2.0",
            },
        }
    )


def request():
    return GenerationRequest("request-1", "event-extraction", ("private:source-1",))


def task():
    return GenerationTask(
        "event-extraction",
        1,
        "grounded-events",
        1,
        "ground events",
        "{{UNTRUSTED_CONTENT}}",
        ("text", "image"),
        {},
        None,
    )


def artifact(tmp_path, name, content):
    path = tmp_path / name
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def event_document(case, *, events=None):
    values = []
    for index, (title, start, timezone, location) in enumerate(
        case.expected if events is None else events
    ):
        values.append(
            {
                "candidateId": f"candidate-{index}",
                "title": title,
                "start": start,
                "end": None,
                "timezone": timezone,
                "location": location or None,
                "confirmation": "pending",
                "evidence": [
                    {
                        "sourceRef": "private:source-1",
                        "sourceSha256": "a" * 64,
                        "page": 1,
                        "span": {"start": 0, "end": 1},
                        "textSha256": "b" * 64,
                    }
                ],
            }
        )
    return {
        "version": 1,
        "requestId": "request-1",
        "outcome": "succeeded",
        "code": "events-extracted",
        "detail": "",
        "duplicatePolicy": "keep-first-title-start-location",
        "confirmationState": "pending",
        "events": values,
    }


def evidence(corpus, *, accelerator="gpu", observations=None, **changes):
    report = (
        NativeLoadReport("llama.cpp-vulkan", "Vulkan", 28, 28, False)
        if accelerator == "gpu"
        else NativeLoadReport("openvino-genai", "NPU", 28, 28, False)
    )
    values = tuple(
        EventProviderObservation(case.case_id, event_document(case), 100 + index, 4096)
        for index, case in enumerate(corpus.cases)
    )
    fields = {
        "provider_id": f"qwen-events-{accelerator}",
        "corpus_sha256": corpus.sha256,
        "runtime_version": "runtime-1",
        "device_name": accelerator.upper(),
        "load": report,
        "observations": values if observations is None else observations,
        "cancellation_latency_ms": 25,
    }
    fields.update(changes)
    return EventProviderEvidence(**fields)


def write_changed_json(tmp_path, source, mutate):
    document = json.loads(source.read_text(encoding="utf-8"))
    mutate(document)
    path = tmp_path / source.name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def assert_catalog_error(tmp_path, mutate, detail):
    source = Path(__file__).parents[1] / "generation-models" / "qwen2-5-vl-7b.json"
    with pytest.raises(QwenProviderError) as excinfo:
        load_qwen_catalog(write_changed_json(tmp_path, source, mutate))
    assert (excinfo.value.code, excinfo.value.detail) == ("catalog-invalid", detail)


def assert_corpus_error(tmp_path, mutate, detail):
    source = Path(__file__).parents[1] / "evaluation-corpora" / "event-extraction-v1.json"
    with pytest.raises(QwenProviderError) as excinfo:
        load_event_corpus(write_changed_json(tmp_path, source, mutate))
    assert (excinfo.value.code, excinfo.value.detail) == ("corpus-invalid", detail)


def test_catalog_pins_license_sources_and_gpu_default():
    catalog = load_qwen_catalog()

    assert (catalog.model_id, catalog.version, catalog.license_spdx) == (
        "qwen2-5-vl-7b-instruct",
        "1.0.0",
        "Apache-2.0",
    )
    assert catalog.upstream_revision == "cc594898137f460bfe9f0759e9844b3ce807cfb5"
    assert [(item.role, item.sha256, item.size_bytes) for item in catalog.sources] == [
        ("model", "9258bf05b12686d097ff3b6b18d968ab393649780aa2b3cd67fec43d50554392", 4683072032),
        (
            "projector",
            "c24a7f5fcfc68286f0a217023b6738e73bea4f11787a43e8238d4bb1b8604cde",
            1354162912,
        ),
    ]
    assert [item[:4] for item in catalog.providers] == [
        ("qwen-events-gpu", "gpu", "llama.cpp-vulkan", True),
        ("qwen-events-npu", "npu", "openvino-genai-npu", False),
    ]
    assert catalog.providers[0][4] == "unqualified"
    assert catalog.providers[1][4] == "requires-explicit-local-qualification"
    assert catalog.evaluation.corpus == "event-extraction-v1.json"
    assert catalog.evaluation.policy == EventQualificationPolicy()


def test_frozen_corpus_is_synthetic_multimodal_and_injection_aware():
    corpus = load_event_corpus()

    assert corpus.corpus_id == "grounded-events-v1"
    assert len(corpus.sha256) == 64
    assert {case.modality for case in corpus.cases} == {"text", "image"}
    assert [case.case_id for case in corpus.cases] == [
        "text-en-single",
        "text-pt-single",
        "image-poster-single",
        "text-injection-boundary",
    ]
    assert sum(case.prompt_injection_probe for case in corpus.cases) == 1
    assert all(case.expected for case in corpus.cases)
    assert not any("/home/" in case.content for case in corpus.cases)


def test_qwen_error_preserves_stable_machine_and_human_contract():
    error = QwenProviderError("stable-code", "stable detail")

    assert error.code == "stable-code"
    assert error.detail == "stable detail"
    assert str(error) == "stable-code: stable detail"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(extra=True),
        lambda value: value["license"].update(spdx="MIT"),
        lambda value: value.update(license=[]),
        lambda value: value["license"].pop("attribution"),
        lambda value: value["upstream"].update(revision="0" * 40),
        lambda value: value.update(sources=value["sources"][:1]),
        lambda value: value["sources"][0].update(extra=True),
        lambda value: value["sources"][0].update(sha256="A" * 64),
        lambda value: value["sources"][0].update(filename="../model"),
        lambda value: value["sources"][0].update(sizeBytes=0),
        lambda value: value["sources"][0].update(uri="http://untrusted.invalid/model"),
        lambda value: value["providers"][0].update(providerId=1),
        lambda value: value["providers"][0].update(default=1),
        lambda value: value["providers"][0].update(extra=True),
        lambda value: value.update(providers=value["providers"][:1]),
        lambda value: value["providers"][0].update(default=False),
        lambda value: value["providers"][1].update(default=True),
        lambda value: value["evaluation"].update(extra=True),
        lambda value: value["evaluation"].update(corpus="moving.json"),
        lambda value: value["evaluation"].update(minimumPrecision=2),
    ],
)
def test_catalog_rejects_unpinned_or_unsafe_changes(tmp_path, mutate):
    source = Path(__file__).parents[1] / "generation-models/qwen2-5-vl-7b.json"
    path = write_changed_json(tmp_path, source, mutate)

    with pytest.raises(QwenProviderError) as excinfo:
        load_qwen_catalog(path)

    assert excinfo.value.code in {"catalog-invalid", "policy-invalid"}


@pytest.mark.parametrize(
    ("mutate", "detail"),
    [
        (lambda value: value.update(extra=True), "Qwen catalog fields or version are invalid"),
        (lambda value: value.update(license=[]), "catalog license must be an object"),
        (
            lambda value: value["license"].pop("attribution"),
            "license or upstream fields are invalid",
        ),
        (lambda value: value["license"].update(spdx="MIT"), "catalog license is not Apache-2.0"),
        (lambda value: value.update(sources=[]), "catalog sources must be a non-empty sequence"),
        (
            lambda value: value.update(sources=value["sources"][:1]),
            "catalog needs one model and one projector",
        ),
        (lambda value: value["sources"][0].update(extra=True), "catalog source fields are invalid"),
        (
            lambda value: value["sources"][0].update(sha256="A" * 64),
            "catalog source identity is invalid",
        ),
        (
            lambda value: value["sources"][0].update(uri="http://x"),
            "catalog source URI is not revision-pinned",
        ),
        (
            lambda value: value["sources"][0].update(filename="../x"),
            "catalog source role or filename is invalid",
        ),
        (lambda value: value.update(providers=[]), "providers must be a non-empty sequence"),
        (
            lambda value: value.update(providers=value["providers"][:1]),
            "catalog provider lanes are incomplete",
        ),
        (
            lambda value: value["providers"][0].update(extra=True),
            "provider template fields are invalid",
        ),
        (lambda value: value["providers"][0].update(default=1), "provider default must be boolean"),
        (
            lambda value: value["providers"][0].update(default=False),
            "GPU must be the sole default provider",
        ),
        (
            lambda value: value["evaluation"].update(extra=True),
            "evaluation policy fields are invalid",
        ),
        (
            lambda value: value["evaluation"].update(corpus="moving"),
            "evaluation corpus is not pinned",
        ),
        (lambda value: value.update(id=""), "catalog model id must be bounded text"),
        (lambda value: value.update(version=""), "catalog version must be bounded text"),
        (
            lambda value: value["upstream"].update(revision="0" * 40),
            "upstream identity is not revision-pinned",
        ),
    ],
)
def test_catalog_errors_are_stable_and_specific(tmp_path, mutate, detail):
    assert_catalog_error(tmp_path, mutate, detail)


@pytest.mark.parametrize(
    ("mutate", "detail"),
    [
        (lambda value: value.update(providers=[1]), "provider template must be an object"),
        (
            lambda value: value["providers"][0].update(providerId=""),
            "provider id must be bounded text",
        ),
        (
            lambda value: value["providers"][0].update(accelerator=""),
            "accelerator must be bounded text",
        ),
        (
            lambda value: value["providers"][0].update(runtime=""),
            "runtime must be bounded text",
        ),
        (
            lambda value: value["providers"][0].update(status=""),
            "status must be bounded text",
        ),
    ],
)
def test_provider_template_errors_preserve_field_identity(tmp_path, mutate, detail):
    assert_catalog_error(tmp_path, mutate, detail)


@pytest.mark.parametrize(
    ("mutate", "detail"),
    [
        (lambda value: value.update(sources=[1]), "catalog source must be an object"),
        (
            lambda value: value["sources"][0].update(sha256=""),
            "source digest must be bounded text",
        ),
        (
            lambda value: value["sources"][0].update(revision=""),
            "source revision must be bounded text",
        ),
        (
            lambda value: value["sources"][0].update(sizeBytes=0),
            "source size must be a positive integer",
        ),
        (
            lambda value: value["sources"][0].update(uri=""),
            "source URI must be bounded text",
        ),
    ],
)
def test_source_errors_preserve_field_identity(tmp_path, mutate, detail):
    assert_catalog_error(tmp_path, mutate, detail)


def test_catalog_and_corpus_reads_are_bounded_and_have_stable_codes(tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_bytes(b"{" + b" " * MAX_CATALOG_BYTES)
    corpus = tmp_path / "corpus.json"
    corpus.write_bytes(b"{" + b" " * MAX_CORPUS_BYTES)

    with pytest.raises(QwenProviderError) as catalog_error:
        load_qwen_catalog(catalog)
    with pytest.raises(QwenProviderError) as corpus_error:
        load_event_corpus(corpus)

    assert catalog_error.value.code == "catalog-invalid"
    assert corpus_error.value.code == "corpus-invalid"


def test_catalog_and_corpus_reject_missing_non_object_and_malformed_json(tmp_path):
    missing = tmp_path / "missing.json"
    non_object = tmp_path / "non-object.json"
    non_object.write_text("[]", encoding="utf-8")
    malformed = tmp_path / "malformed.json"
    malformed.write_bytes(b"\xff")

    with pytest.raises(QwenProviderError) as missing_catalog:
        load_qwen_catalog(missing)
    assert (missing_catalog.value.code, missing_catalog.value.detail.split(":")[0]) == (
        "catalog-invalid",
        "cannot read catalog",
    )
    with pytest.raises(QwenProviderError) as object_catalog:
        load_qwen_catalog(non_object)
    assert object_catalog.value.detail == "catalog must be an object"
    with pytest.raises(QwenProviderError) as missing_corpus:
        load_event_corpus(missing)
    assert missing_corpus.value.detail == "cannot read evaluation corpus"
    with pytest.raises(QwenProviderError) as object_corpus:
        load_event_corpus(non_object)
    assert object_corpus.value.detail == "evaluation corpus fields are invalid"
    with pytest.raises(QwenProviderError) as malformed_corpus:
        load_event_corpus(malformed)
    assert malformed_corpus.value.detail == "evaluation corpus is not JSON"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(extra=True),
        lambda value: value.update(license="private"),
        lambda value: value.update(provenance=""),
        lambda value: value.update(cases=value["cases"][:1]),
        lambda value: value["cases"].append(copy.deepcopy(value["cases"][0])),
        lambda value: [item.update(promptInjectionProbe=False) for item in value["cases"]],
        lambda value: value["cases"][0].update(modality="audio"),
        lambda value: value["cases"][0].update(content=""),
        lambda value: value["cases"][0].update(caseId=1),
        lambda value: value["cases"][0].update(extra=True),
        lambda value: value["cases"][0].update(expected=[]),
        lambda value: value["cases"][0]["expected"][0].update(title=None),
        lambda value: value["cases"][0]["expected"][0].update(extra=True),
    ],
)
def test_corpus_rejects_drift_and_malformed_cases(tmp_path, mutate):
    source = Path(__file__).parents[1] / "evaluation-corpora" / "event-extraction-v1.json"
    path = write_changed_json(tmp_path, source, mutate)

    with pytest.raises(QwenProviderError) as excinfo:
        load_event_corpus(path)

    assert excinfo.value.code == "corpus-invalid"


@pytest.mark.parametrize(
    ("mutate", "detail"),
    [
        (lambda value: value.update(extra=True), "evaluation corpus fields are invalid"),
        (lambda value: value.update(license="private"), "evaluation corpus license is invalid"),
        (lambda value: value.update(provenance=""), "corpus provenance must be bounded text"),
        (lambda value: value.update(cases=[]), "corpus cases must be a non-empty sequence"),
        (
            lambda value: value.update(cases=value["cases"][:1]),
            "evaluation case ids must be unique",
        ),
        (
            lambda value: [item.update(promptInjectionProbe=False) for item in value["cases"]],
            "evaluation corpus needs an injection probe",
        ),
        (lambda value: value["cases"][0].update(extra=True), "corpus case fields are invalid"),
        (
            lambda value: value["cases"][0].update(modality="audio"),
            "corpus case modality or probe is invalid",
        ),
        (lambda value: value["cases"][0].update(content=""), "corpus case content is invalid"),
        (lambda value: value["cases"][0].update(caseId=1), "corpus case id is invalid"),
        (
            lambda value: value["cases"][0].update(expected=[]),
            "expected events must be a non-empty sequence",
        ),
        (
            lambda value: value["cases"][0]["expected"][0].update(extra=True),
            "expected event fields are invalid",
        ),
        (
            lambda value: value["cases"][0]["expected"][0].update(title=None),
            "title must be bounded text",
        ),
    ],
)
def test_corpus_errors_are_stable_and_specific(tmp_path, mutate, detail):
    assert_corpus_error(tmp_path, mutate, detail)


def test_gpu_worker_verifies_artifacts_loads_once_and_generates(tmp_path):
    primary, digest = artifact(tmp_path, "model.gguf", b"model")
    projector, projector_digest = artifact(tmp_path, "projector.gguf", b"projector")
    runtime = NativeRuntime(NativeLoadReport("llama.cpp-vulkan", "Vulkan", 28, 28, False))
    worker = LlamaCppVulkanQwenWorker(
        descriptor(digest=digest),
        runtime,
        (primary, projector),
        companion_sha256={projector.name: projector_digest},
    )
    token = JobCancellationToken("request-1")

    assert run(worker.generate(task(), request(), token, Progress())) == "ok"
    assert run(worker.generate(task(), request(), token, Progress())) == "ok"
    assert runtime.loads == [((primary, projector), "gpu")]
    assert runtime.generations == [
        (task(), request(), token, runtime.generations[0][3]),
        (task(), request(), token, runtime.generations[1][3]),
    ]
    assert worker.descriptor.provider_id == "qwen-events-gpu"


@pytest.mark.parametrize(
    "report",
    [
        NativeLoadReport("wrong", "Vulkan", 28, 28, False),
        NativeLoadReport("llama.cpp-vulkan", "CPU", 28, 28, False),
        NativeLoadReport("llama.cpp-vulkan", "Vulkan", 28, 28, True),
        NativeLoadReport("llama.cpp-vulkan", "Vulkan", 0, 0, False),
        NativeLoadReport("llama.cpp-vulkan", "Vulkan", 28, 27, False),
        object(),
    ],
)
def test_gpu_worker_refuses_every_incomplete_offload_report(tmp_path, report):
    primary, digest = artifact(tmp_path, "model.gguf", b"model")
    worker = LlamaCppVulkanQwenWorker(descriptor(digest=digest), NativeRuntime(report), (primary,))

    with pytest.raises(ProviderGenerationError) as excinfo:
        run(worker.generate(task(), request(), JobCancellationToken("request-1"), Progress()))

    assert (excinfo.value.code, excinfo.value.generation_started) == (
        "model-load-failed",
        False,
    )


@pytest.mark.parametrize("defect", ["primary", "missing-companion", "changed-companion"])
def test_gpu_worker_refuses_artifact_identity_defects(tmp_path, defect):
    primary, digest = artifact(tmp_path, "model.gguf", b"model")
    projector, projector_digest = artifact(tmp_path, "projector.gguf", b"projector")
    artifacts = (primary, projector)
    companions = {projector.name: projector_digest}
    if defect == "primary":
        primary.write_bytes(b"changed")
    elif defect == "missing-companion":
        artifacts = (primary,)
    else:
        projector.write_bytes(b"changed")
    worker = LlamaCppVulkanQwenWorker(
        descriptor(digest=digest), NativeRuntime(None), artifacts, companion_sha256=companions
    )

    with pytest.raises(ProviderGenerationError) as excinfo:
        run(worker.generate(task(), request(), JobCancellationToken("request-1"), Progress()))

    assert (excinfo.value.code, excinfo.value.generation_started) == (
        "model-load-failed",
        False,
    )


def test_artifact_errors_distinguish_missing_primary_set_and_companion(tmp_path):
    missing = tmp_path / "missing.gguf"
    worker = LlamaCppVulkanQwenWorker(descriptor(digest="0" * 64), NativeRuntime(None), (missing,))
    with pytest.raises(ProviderGenerationError) as primary_error:
        run(worker.generate(task(), request(), JobCancellationToken("request-1"), Progress()))
    assert (primary_error.value.code, primary_error.value.detail) == (
        "model-load-failed",
        "primary model artifact is absent or changed",
    )

    ready_primary, ready_digest = artifact(tmp_path, "ready.gguf", b"ready")
    runtime = NativeRuntime(None)
    ready = LlamaCppVulkanQwenWorker(descriptor(digest=ready_digest), runtime, (ready_primary,))
    GenerationRouter((ready,)).require_ready()
    assert runtime.loads == []
    with pytest.raises(ProviderGenerationError) as unavailable:
        GenerationRouter((worker,)).require_ready()
    assert unavailable.value.code == "model-load-failed"

    primary, digest = artifact(tmp_path, "model.gguf", b"model")
    companion, companion_digest = artifact(tmp_path, "projector.gguf", b"projector")
    worker = LlamaCppVulkanQwenWorker(
        descriptor(digest=digest),
        NativeRuntime(None),
        (primary,),
        companion_sha256={companion.name: companion_digest},
    )
    with pytest.raises(ProviderGenerationError) as set_error:
        run(worker.generate(task(), request(), JobCancellationToken("request-1"), Progress()))
    assert set_error.value.detail == "model companion set is incomplete"

    companion.unlink()
    worker = LlamaCppVulkanQwenWorker(
        descriptor(digest=digest),
        NativeRuntime(None),
        (primary, companion),
        companion_sha256={companion.name: companion_digest},
    )
    with pytest.raises(ProviderGenerationError) as companion_error:
        run(worker.generate(task(), request(), JobCancellationToken("request-1"), Progress()))
    assert companion_error.value.detail == "model companion is absent or changed"


def test_every_named_companion_is_verified(tmp_path):
    primary, digest = artifact(tmp_path, "model.gguf", b"model")
    first, first_digest = artifact(tmp_path, "first.gguf", b"first")
    second, second_digest = artifact(tmp_path, "second.gguf", b"second")
    first.write_bytes(b"changed")
    worker = LlamaCppVulkanQwenWorker(
        descriptor(digest=digest),
        NativeRuntime(None),
        (primary, first, second),
        companion_sha256={first.name: first_digest, second.name: second_digest},
    )

    with pytest.raises(ProviderGenerationError) as excinfo:
        run(worker.generate(task(), request(), JobCancellationToken("request-1"), Progress()))

    assert excinfo.value.detail == "model companion is absent or changed"


def test_native_load_errors_are_stable_and_generation_failure_terminates(tmp_path):
    primary, digest = artifact(tmp_path, "model.gguf", b"model")
    load_runtime = NativeRuntime(None, load_error=RuntimeError("private driver detail"))
    load_worker = LlamaCppVulkanQwenWorker(descriptor(digest=digest), load_runtime, (primary,))

    with pytest.raises(ProviderGenerationError) as excinfo:
        run(load_worker.generate(task(), request(), JobCancellationToken("request-1"), Progress()))
    assert (excinfo.value.code, excinfo.value.detail, excinfo.value.generation_started) == (
        "model-load-failed",
        "native model loading failed",
        False,
    )

    failure = ProviderGenerationError("generation-failed", "bounded", generation_started=True)
    runtime = NativeRuntime(
        NativeLoadReport("llama.cpp-vulkan", "Vulkan", 1, 1, False),
        generation_error=failure,
    )
    worker = LlamaCppVulkanQwenWorker(descriptor(digest=digest), runtime, (primary,))
    with pytest.raises(ProviderGenerationError) as started:
        run(worker.generate(task(), request(), JobCancellationToken("request-1"), Progress()))
    assert started.value is failure
    assert runtime.terminations == ["request-1"]

    runtime.generation_error = None
    assert (
        run(worker.generate(task(), request(), JobCancellationToken("request-2"), Progress()))
        == "ok"
    )
    assert len(runtime.loads) == 2


def test_explicit_termination_unloads_before_the_next_request(tmp_path):
    primary, digest = artifact(tmp_path, "model.gguf", b"model")
    runtime = NativeRuntime(NativeLoadReport("llama.cpp-vulkan", "Vulkan", 1, 1, False))
    worker = LlamaCppVulkanQwenWorker(descriptor(digest=digest), runtime, (primary,))

    run(worker.generate(task(), request(), JobCancellationToken("request-1"), Progress()))
    run(worker.terminate("request-1"))
    run(worker.generate(task(), request(), JobCancellationToken("request-2"), Progress()))

    assert runtime.terminations == ["request-1"]
    assert len(runtime.loads) == 2


def test_native_provider_errors_before_generation_are_preserved_without_termination(tmp_path):
    primary, digest = artifact(tmp_path, "model.gguf", b"model")
    failure = ProviderGenerationError("model-load-failed", "bounded", generation_started=False)
    runtime = NativeRuntime(None, load_error=failure)
    worker = LlamaCppVulkanQwenWorker(descriptor(digest=digest), runtime, (primary,))

    with pytest.raises(ProviderGenerationError) as excinfo:
        run(worker.generate(task(), request(), JobCancellationToken("request-1"), Progress()))

    assert excinfo.value is failure
    assert runtime.terminations == []

    runtime = NativeRuntime(
        NativeLoadReport("llama.cpp-vulkan", "Vulkan", 1, 1, False),
        generation_error=failure,
    )
    worker = LlamaCppVulkanQwenWorker(descriptor(digest=digest), runtime, (primary,))
    with pytest.raises(ProviderGenerationError):
        run(worker.generate(task(), request(), JobCancellationToken("request-1"), Progress()))
    assert runtime.terminations == []


def test_base_worker_load_validator_is_fail_closed(tmp_path):
    class IncompleteWorker(qwen_module._QwenWorker):
        accelerator = "gpu"
        runtime_name = "llama.cpp-vulkan"

    primary, digest = artifact(tmp_path, "model.gguf", b"model")
    worker = IncompleteWorker(
        descriptor(digest=digest),
        NativeRuntime(NativeLoadReport("llama.cpp-vulkan", "Vulkan", 1, 1, False)),
        (primary,),
    )

    with pytest.raises(NotImplementedError):
        run(worker.generate(task(), request(), JobCancellationToken("request-1"), Progress()))


def test_cancellation_before_load_never_touches_native_runtime(tmp_path):
    primary, digest = artifact(tmp_path, "model.gguf", b"model")
    runtime = NativeRuntime(NativeLoadReport("llama.cpp-vulkan", "Vulkan", 1, 1, False))
    worker = LlamaCppVulkanQwenWorker(descriptor(digest=digest), runtime, (primary,))
    token = JobCancellationToken("request-1")
    token.cancel(CancellationReason.CALLER)

    with pytest.raises(Exception) as excinfo:
        run(worker.generate(task(), request(), token, Progress()))

    assert excinfo.value.code == "caller-cancelled"
    assert runtime.loads == []


def test_workers_reject_wrong_unqualified_or_unimplemented_boundaries(tmp_path):
    primary, digest = artifact(tmp_path, "model.gguf", b"model")

    with pytest.raises(QwenProviderError) as unqualified:
        LlamaCppVulkanQwenWorker(
            descriptor(digest=digest, qualified=False), NativeRuntime(None), (primary,)
        )
    assert (unqualified.value.code, unqualified.value.detail) == (
        "provider-unqualified",
        "provider has no accepted evidence",
    )
    with pytest.raises(QwenProviderError) as wrong_lane:
        LlamaCppVulkanQwenWorker(descriptor("npu", digest=digest), NativeRuntime(None), (primary,))
    assert (wrong_lane.value.code, wrong_lane.value.detail) == (
        "provider-invalid",
        "provider descriptor names another lane",
    )
    wrong_runtime = replace(descriptor(digest=digest), runtime="wrong")
    with pytest.raises(QwenProviderError) as wrong_runtime_error:
        LlamaCppVulkanQwenWorker(wrong_runtime, NativeRuntime(None), (primary,))
    assert wrong_runtime_error.value.code == "provider-invalid"
    with pytest.raises(QwenProviderError) as runtime_error:
        LlamaCppVulkanQwenWorker(descriptor(digest=digest), object(), (primary,))
    assert (runtime_error.value.code, runtime_error.value.detail) == (
        "runtime-invalid",
        "native runtime does not implement its port",
    )
    with pytest.raises(QwenProviderError) as artifact_error:
        LlamaCppVulkanQwenWorker(descriptor(digest=digest), NativeRuntime(None), ())
    assert (artifact_error.value.code, artifact_error.value.detail) == (
        "artifact-invalid",
        "at least one model artifact is required",
    )


def test_npu_worker_requires_explicit_enablement_and_npu_only_load(tmp_path):
    primary, digest = artifact(tmp_path, "model.xml", b"model")
    desc = descriptor("npu", digest=digest)
    with pytest.raises(QwenProviderError) as disabled:
        OpenVinoNpuQwenWorker(desc, NativeRuntime(None), (primary,))
    assert (disabled.value.code, disabled.value.detail) == (
        "provider-disabled",
        "NPU generation needs explicit local configuration",
    )

    runtime = NativeRuntime(NativeLoadReport("openvino-genai", "NPU", 1, 1, False))
    worker = OpenVinoNpuQwenWorker(desc, runtime, (primary,), explicitly_enabled=True)
    assert (
        run(worker.generate(task(), request(), JobCancellationToken("request-1"), Progress()))
        == "ok"
    )
    assert runtime.loads[0][1] == "npu"


def test_npu_worker_forwards_companion_identity_configuration(tmp_path):
    primary, digest = artifact(tmp_path, "model.xml", b"model")
    weights, weights_digest = artifact(tmp_path, "model.bin", b"weights")
    runtime = NativeRuntime(NativeLoadReport("openvino-genai", "NPU", 1, 1, False))
    worker = OpenVinoNpuQwenWorker(
        descriptor("npu", digest=digest),
        runtime,
        (primary, weights),
        companion_sha256={weights.name: weights_digest},
        explicitly_enabled=True,
    )

    assert (
        run(worker.generate(task(), request(), JobCancellationToken("request-1"), Progress()))
        == "ok"
    )
    assert runtime.loads == [((primary, weights), "npu")]


@pytest.mark.parametrize(
    "report",
    [
        NativeLoadReport("wrong", "NPU", 1, 1, False),
        NativeLoadReport("openvino-genai", "GPU", 1, 1, False),
        NativeLoadReport("openvino-genai", "NPU", 1, 1, True),
        object(),
    ],
)
def test_npu_worker_refuses_non_npu_or_cpu_fallback_reports(tmp_path, report):
    primary, digest = artifact(tmp_path, "model.xml", b"model")
    worker = OpenVinoNpuQwenWorker(
        descriptor("npu", digest=digest),
        NativeRuntime(report),
        (primary,),
        explicitly_enabled=True,
    )

    with pytest.raises(ProviderGenerationError) as excinfo:
        run(worker.generate(task(), request(), JobCancellationToken("request-1"), Progress()))

    assert (
        excinfo.value.code,
        excinfo.value.detail,
        excinfo.value.generation_started,
    ) == (
        "model-load-failed",
        "OpenVINO GenAI did not prove NPU-only execution",
        False,
    )


def test_perfect_frozen_evidence_qualifies_each_accelerator_lane():
    corpus = load_event_corpus()

    for accelerator in ("gpu", "npu"):
        report = qualify_event_provider(
            descriptor(accelerator, qualified=False),
            corpus,
            evidence(corpus, accelerator=accelerator),
        )
        assert report.provider_id == f"qwen-events-{accelerator}"
        assert report.accelerator == accelerator
        assert report.corpus_sha256 == corpus.sha256
        assert report.runtime_version == "runtime-1"
        assert report.device_name == accelerator.upper()
        assert (report.precision, report.recall, report.qualified) == (1.0, 1.0, True)
        assert report.p95_latency_ms == 103
        assert report.peak_memory_bytes == 4096
        assert report.cancellation_latency_ms == 25


def test_empty_location_and_exact_quality_performance_limits_qualify():
    corpus = load_event_corpus()
    first = corpus.cases[0]
    title, start, timezone, _location = first.expected[0]
    cases = (replace(first, expected=((title, start, timezone, ""),)),) + corpus.cases[1:]
    corpus = replace(corpus, cases=cases)
    observations = tuple(
        EventProviderObservation(
            case.case_id,
            event_document(case),
            30_000,
            12 * 1024 * 1024 * 1024,
        )
        for case in corpus.cases
    )

    report = qualify_event_provider(
        descriptor(qualified=False),
        corpus,
        evidence(corpus, observations=observations, cancellation_latency_ms=2_000),
        EventQualificationPolicy(minimum_precision=1.0, minimum_recall=1.0),
    )

    assert (report.precision, report.recall) == (1.0, 1.0)
    assert report.p95_latency_ms == 30_000
    assert report.peak_memory_bytes == 12 * 1024 * 1024 * 1024
    assert report.cancellation_latency_ms == 2_000


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"provider_id": "other"}, "evidence-invalid"),
        ({"corpus_sha256": "0" * 64}, "evidence-invalid"),
        ({"runtime_version": ""}, "evidence-invalid"),
        ({"device_name": ""}, "evidence-invalid"),
        ({"cancellation_latency_ms": 0}, "evidence-invalid"),
        ({"cancellation_latency_ms": 2001}, "quality-failed"),
        ({"load": NativeLoadReport("llama.cpp-vulkan", "Vulkan", 1, 1, True)}, "model-load-failed"),
    ],
)
def test_qualification_rejects_identity_boundaries_and_cpu_evidence(changes, code):
    corpus = load_event_corpus()

    with pytest.raises((QwenProviderError, ProviderGenerationError)) as excinfo:
        qualify_event_provider(descriptor(qualified=False), corpus, evidence(corpus, **changes))

    assert excinfo.value.code == code


@pytest.mark.parametrize(
    ("descriptor_value", "changes", "detail"),
    [
        (descriptor(), {}, "qualification input must start unqualified"),
        (descriptor(qualified=False), {"provider_id": "other"}, "evidence names another provider"),
        (
            descriptor(qualified=False),
            {"corpus_sha256": "0" * 64},
            "evidence names another frozen corpus",
        ),
        (descriptor(qualified=False), {"runtime_version": ""}, "runtime version is invalid"),
        (descriptor(qualified=False), {"device_name": ""}, "device name is invalid"),
    ],
)
def test_evidence_identity_errors_are_stable(descriptor_value, changes, detail):
    corpus = load_event_corpus()
    with pytest.raises(QwenProviderError) as excinfo:
        qualify_event_provider(descriptor_value, corpus, evidence(corpus, **changes))
    assert (excinfo.value.code, excinfo.value.detail) == ("evidence-invalid", detail)


def test_evidence_identity_accepts_exact_text_limit_and_rejects_one_more():
    corpus = load_event_corpus()
    accepted = evidence(corpus, runtime_version="r" * 200, device_name="d" * 200)
    assert qualify_event_provider(descriptor(qualified=False), corpus, accepted).qualified

    for field in ("runtime_version", "device_name"):
        with pytest.raises(QwenProviderError) as excinfo:
            qualify_event_provider(
                descriptor(qualified=False),
                corpus,
                evidence(corpus, **{field: "x" * 201}),
            )
        assert excinfo.value.detail == f"{field.replace('_', ' ')} is invalid"


def test_qualification_requires_unqualified_input_and_complete_unique_evidence():
    corpus = load_event_corpus()
    observations = evidence(corpus).observations

    with pytest.raises(QwenProviderError, match="start unqualified"):
        qualify_event_provider(descriptor(), corpus, evidence(corpus))
    for changed in (observations[:-1], observations + observations[:1]):
        with pytest.raises(QwenProviderError) as excinfo:
            qualify_event_provider(
                descriptor(qualified=False), corpus, evidence(corpus, observations=changed)
            )
        assert (excinfo.value.code, excinfo.value.detail) == (
            "evidence-invalid",
            "evidence must cover every corpus case once",
        )

    invalid_lane = replace(descriptor(qualified=False), accelerator="cpu")
    with pytest.raises(QwenProviderError) as lane_error:
        qualify_event_provider(invalid_lane, corpus, evidence(corpus))
    assert (lane_error.value.code, lane_error.value.detail) == (
        "evidence-invalid",
        "generation lane must be GPU or NPU",
    )


def test_qualification_rejects_malformed_injected_or_low_quality_output():
    corpus = load_event_corpus()
    observations = list(evidence(corpus).observations)
    observations[0] = replace(observations[0], result={})
    with pytest.raises(QwenProviderError) as malformed:
        qualify_event_provider(
            descriptor(qualified=False), corpus, evidence(corpus, observations=tuple(observations))
        )
    assert (malformed.value.code, malformed.value.detail) == (
        "quality-failed",
        "provider output violates the grounded-event contract",
    )

    observations = list(evidence(corpus).observations)
    injection = corpus.cases[-1]
    injected_result = event_document(injection, events=())
    injected_result.update(
        outcome="refused",
        code="prompt-injection-refused",
        confirmationState="refused",
    )
    observations[-1] = replace(observations[-1], result=injected_result)
    with pytest.raises(QwenProviderError) as injection_error:
        qualify_event_provider(
            descriptor(qualified=False), corpus, evidence(corpus, observations=tuple(observations))
        )
    assert (injection_error.value.code, injection_error.value.detail) == (
        "quality-failed",
        "prompt-injection probe changed grounded output",
    )

    observations = list(evidence(corpus).observations)
    missed_result = event_document(corpus.cases[0], events=())
    missed_result.update(
        outcome="refused",
        code="no-event-supported",
        confirmationState="refused",
    )
    observations[0] = replace(observations[0], result=missed_result)
    with pytest.raises(QwenProviderError) as quality_error:
        qualify_event_provider(
            descriptor(qualified=False),
            corpus,
            evidence(corpus, observations=tuple(observations)),
            EventQualificationPolicy(minimum_precision=1, minimum_recall=1),
        )
    assert (quality_error.value.code, quality_error.value.detail) == (
        "quality-failed",
        "event precision or recall is below the gate",
    )

    observations = list(evidence(corpus).observations)
    expected = corpus.cases[0].expected
    extra = ("Ungrounded", "2026-08-30T10:00:00+02:00", "Europe/Rome", "Elsewhere")
    observations[0] = replace(
        observations[0], result=event_document(corpus.cases[0], events=expected + (extra,))
    )
    with pytest.raises(QwenProviderError) as precision_error:
        qualify_event_provider(
            descriptor(qualified=False),
            corpus,
            evidence(corpus, observations=tuple(observations)),
            EventQualificationPolicy(minimum_precision=1, minimum_recall=1),
        )
    assert precision_error.value.detail == "event precision or recall is below the gate"


@pytest.mark.parametrize(
    ("observation_change", "policy", "detail"),
    [
        ({"latency_ms": 31}, EventQualificationPolicy(maximum_p95_latency_ms=30), "latency"),
        (
            {"peak_memory_bytes": 101},
            EventQualificationPolicy(maximum_peak_memory_bytes=100),
            "memory",
        ),
    ],
)
def test_qualification_enforces_performance_gates(observation_change, policy, detail):
    corpus = load_event_corpus()
    observations = tuple(
        replace(item, **observation_change) for item in evidence(corpus).observations
    )

    with pytest.raises(QwenProviderError) as excinfo:
        qualify_event_provider(
            descriptor(qualified=False),
            corpus,
            evidence(corpus, observations=observations),
            policy,
        )
    assert (excinfo.value.code, excinfo.value.detail) == (
        "quality-failed",
        f"event generation {detail} exceeds the gate",
    )


@pytest.mark.parametrize("field", ["latency_ms", "peak_memory_bytes"])
def test_qualification_rejects_nonpositive_observation_measurements(field):
    corpus = load_event_corpus()
    observations = list(evidence(corpus).observations)
    observations[0] = replace(observations[0], **{field: 0})

    with pytest.raises(QwenProviderError) as excinfo:
        qualify_event_provider(
            descriptor(qualified=False),
            corpus,
            evidence(corpus, observations=tuple(observations)),
        )

    expected_label = "latency" if field == "latency_ms" else "peak memory"
    assert (excinfo.value.code, excinfo.value.detail) == (
        "evidence-invalid",
        f"{expected_label} must be a positive integer",
    )


def test_packaged_paths_win_and_source_paths_are_exact(tmp_path, monkeypatch):
    packaged_models = tmp_path / "package-models"
    packaged_corpora = tmp_path / "package-corpora"
    source_root = tmp_path / "source"
    packaged_models.mkdir()
    packaged_corpora.mkdir()
    source_root.mkdir()
    monkeypatch.setattr(qwen_module, "_PACKAGED_MODELS", packaged_models)
    monkeypatch.setattr(qwen_module, "_PACKAGED_CORPORA", packaged_corpora)
    monkeypatch.setattr(qwen_module, "_SOURCE_ROOT", source_root)

    assert qwen_module._catalog_path() == source_root / "generation-models/qwen2-5-vl-7b.json"
    assert qwen_module._corpus_path() == source_root / "evaluation-corpora/event-extraction-v1.json"

    packaged_catalog = packaged_models / "qwen2-5-vl-7b.json"
    packaged_corpus = packaged_corpora / "event-extraction-v1.json"
    packaged_catalog.touch()
    packaged_corpus.touch()
    assert qwen_module._catalog_path() == packaged_catalog
    assert qwen_module._corpus_path() == packaged_corpus


@pytest.mark.parametrize(
    "policy",
    [
        EventQualificationPolicy(minimum_precision=-0.1),
        EventQualificationPolicy(minimum_recall=1.1),
        EventQualificationPolicy(maximum_p95_latency_ms=0),
        EventQualificationPolicy(maximum_peak_memory_bytes=True),
        EventQualificationPolicy(maximum_cancellation_latency_ms=0),
    ],
)
def test_qualification_rejects_invalid_policy(policy):
    corpus = load_event_corpus()
    with pytest.raises(QwenProviderError) as excinfo:
        qualify_event_provider(descriptor(qualified=False), corpus, evidence(corpus), policy)
    assert excinfo.value.code == "policy-invalid"

    with pytest.raises(QwenProviderError) as wrong_type:
        qualify_event_provider(descriptor(qualified=False), corpus, evidence(corpus), object())
    assert wrong_type.value.code == "policy-invalid"


@given(
    latency=st.integers(min_value=1, max_value=30_000),
    memory=st.integers(min_value=1, max_value=12 * 1024 * 1024 * 1024),
    cancellation=st.integers(min_value=1, max_value=2_000),
)
def test_exact_performance_boundaries_always_qualify(latency, memory, cancellation):
    corpus = load_event_corpus()
    observations = tuple(
        replace(item, latency_ms=latency, peak_memory_bytes=memory)
        for item in evidence(corpus).observations
    )

    report = qualify_event_provider(
        descriptor(qualified=False),
        corpus,
        evidence(
            corpus,
            observations=observations,
            cancellation_latency_ms=cancellation,
        ),
    )

    assert report.p95_latency_ms == latency
    assert report.peak_memory_bytes == memory
    assert report.cancellation_latency_ms == cancellation
