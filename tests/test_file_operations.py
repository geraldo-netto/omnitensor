"""The operations core over file sources (OMNI-0617 stage 2).

`ask-selected-files` dispatches on the shared operation enum: ask keeps its
retrieval path (pinned by `test_document_qa.py`), the four transformation
operations run over the whole extracted content, and translate runs the span
pipeline under the one file-operations envelope.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins.document_qa import DocumentQuestionPlugin
from omnitensor.plugins.document_spans import (
    EMBEDDING_DIMENSIONS,
    DocumentQuestionError,
    EmbeddingProvider,
)
from omnitensor.plugins.event_workload import MemoryFragmentStore
from omnitensor.plugins.file_operations import (
    FILE_OPERATIONS,
    file_operations_result,
    grounded_operation_answer,
    operation_control_fragment,
)
from omnitensor.plugins.fragments import SourceFragment
from omnitensor.plugins.generation import (
    ArtifactProvenance,
    GenerationProviderDescriptor,
    GenerationRouter,
)
from omnitensor.plugins.operations import OPERATIONS
from omnitensor.plugins.protocol import PluginContext, PluginRequest, PluginResultStatus
from omnitensor.registry import validate_document
from omnitensor.sdk import CancellationController


class Progress:
    def __init__(self):
        self.items = []

    async def report(self, item):
        self.items.append(item)


class Embedder:
    descriptor = EmbeddingProvider("bge-small-gpu", "gpu", "b" * 64, True)

    async def embed(self, texts, *, query, cancellation):
        cancellation.raise_if_cancelled()
        return [[1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1) for _ in texts]


def _descriptor(provider_id: str) -> GenerationProviderDescriptor:
    return GenerationProviderDescriptor(
        provider_id,
        "gpu",
        "llama.cpp-vulkan",
        True,
        ArtifactProvenance(
            provider_id,
            "1",
            "a" * 64,
            f"https://example.invalid/{provider_id}.gguf",
            "Apache-2.0",
        ),
    )


class OperationWorker:
    """Answers the file-operations contract the way the qualified model must:
    reading the trusted control, transforming the one content fragment, and
    citing it exactly."""

    def __init__(self, store, *, provider_id="qwen3-gpu", mutate=None, transform=None):
        self.store = store
        self.mutate = mutate
        self.transform = transform or (lambda operation, text: f"{operation}: {text[:24]}")
        self.requests = []
        self.descriptor = _descriptor(provider_id)

    async def generate(self, task, request, cancellation, progress):
        cancellation.raise_if_cancelled()
        self.requests.append((task, request))
        first = self.store.resolve(request.request_id, request.content_references[0])
        control = json.loads(first.text)
        fragment = self.store.resolve(request.request_id, request.content_references[1])
        operation = control["operation"]
        document = {
            "version": 1,
            "requestId": request.request_id,
            "operation": operation,
            "answer": self.transform(operation, fragment.text),
            "citations": [
                {
                    "sourceRef": fragment.reference,
                    "sourceSha256": fragment.source_sha256,
                    "page": fragment.page,
                    "span": {"start": 0, "end": len(fragment.text)},
                    "textSha256": fragment.text_sha256,
                }
            ],
        }
        if operation == "extract-tasks":
            document["tasks"] = ["Send the invoice", "File the receipt"]
        if self.mutate is not None:
            self.mutate(document)
        return json.dumps(document)

    async def terminate(self, _request_id):
        return None


def request(source: Path, **payload) -> PluginRequest:
    return PluginRequest(
        "job-1",
        "ask-selected-files",
        "manual",
        {"sources": [str(source)], **payload},
        1,
        None,
    )


async def running_plugin(*, worker_store=None, mutate=None, transform=None, routes=None):
    store = worker_store or MemoryFragmentStore()
    worker = OperationWorker(store, mutate=mutate, transform=transform)
    plugin = DocumentQuestionPlugin(
        Embedder(),
        GenerationRouter((worker,)),
        store,
        translation_routes=routes,
        clock_ms=lambda: 10,
    )
    await plugin.start(
        PluginContext("ask-selected-files", 1, {}, frozenset({"files:read-selected"}))
    )
    return plugin, worker, store


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["explain", "summarize", "rewrite"])
async def test_a_whole_content_operation_answers_in_the_envelope(tmp_path, operation):
    source = tmp_path / "notes.txt"
    source.write_text("The migration moved 1.4 million rows overnight.", encoding="utf-8")
    plugin, worker, _store = await running_plugin()

    result = await plugin.execute(
        request(source, operation=operation), CancellationController(), Progress()
    )

    assert result.status is PluginResultStatus.SUCCEEDED
    assert result.detail == "operation result grounded in selected files"
    output = result.output
    assert validate_document("document-question-result.schema.json", output) == []
    assert output["operation"] == operation
    assert output["answer"].startswith(f"{operation}: ")
    assert "tasks" not in output
    assert "documents" not in output
    # The citations are the real page spans the pass covered, host-built.
    assert output["citations"] == [
        {
            "fileId": "selected-file-1",
            "fileName": "notes.txt",
            "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "page": 1,
            "span": {"start": 0, "end": len(source.read_text())},
            "textSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    ]
    # One generation: the whole selection fits one pass, and the request pairs
    # the closed control with the pass's content fragment.
    ((_task, generated),) = worker.requests
    assert len(generated.content_references) == 2
    assert generated.content_references[0] == "private:job-1:control"


@pytest.mark.asyncio
async def test_extract_tasks_carries_the_tasks_section(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("Send the invoice. File the receipt.", encoding="utf-8")
    plugin, _worker, _store = await running_plugin()

    result = await plugin.execute(
        request(source, operation="extract-tasks"), CancellationController(), Progress()
    )

    assert result.status is PluginResultStatus.SUCCEEDED
    output = result.output
    assert validate_document("document-question-result.schema.json", output) == []
    assert output["operation"] == "extract-tasks"
    assert output["tasks"] == ["Send the invoice", "File the receipt"]
    assert "documents" not in output


@pytest.mark.asyncio
async def test_translate_reassembles_documents_and_cites_every_span(tmp_path):
    source = tmp_path / "report.txt"
    source.write_text("The quarterly report is late.", encoding="utf-8")
    plugin, worker, _store = await running_plugin(
        transform=lambda _operation, text: f"O relatório: {text}"
    )

    result = await plugin.execute(
        request(source, operation="translate", language="Português"),
        CancellationController(),
        Progress(),
    )

    assert result.status is PluginResultStatus.SUCCEEDED
    assert result.detail == "selected documents translated in full"
    output = result.output
    assert validate_document("document-question-result.schema.json", output) == []
    assert output["operation"] == "translate"
    (document,) = output["documents"]
    assert document["fileName"] == "report.txt"
    assert document["spanCount"] == 1
    assert document["text"] == "O relatório: The quarterly report is late."
    assert (
        document["translationSha256"]
        == hashlib.sha256(document["text"].encode("utf-8")).hexdigest()
    )
    assert output["answer"] == document["text"]
    (citation,) = output["citations"]
    assert citation["fileId"] == "selected-file-1"
    assert citation["span"] == {"start": 0, "end": len(source.read_text())}
    # The span fragment rode the request beside the shared control shape.
    (_task, generation_request_sent) = worker.requests[0]
    assert generation_request_sent.content_references[0] == "private:job-1:control"
    assert generation_request_sent.content_references[1].endswith("document-1-span-1")


@pytest.mark.asyncio
async def test_an_explicit_hebrew_translate_rides_the_dedicated_route(tmp_path):
    source = tmp_path / "report.txt"
    source.write_text("The quarterly report is late.", encoding="utf-8")
    store = MemoryFragmentStore()
    hebrew_worker = OperationWorker(
        store,
        provider_id="dictalm2-hebrew-gpu",
        transform=lambda _operation, _text: "הדוח הרבעוני מאחר.",
    )
    plugin, general_worker, _store = await running_plugin(
        worker_store=store, routes={"Hebrew": GenerationRouter((hebrew_worker,))}
    )

    result = await plugin.execute(
        request(source, operation="translate", language="hebrew"),
        CancellationController(),
        Progress(),
    )

    assert result.status is PluginResultStatus.SUCCEEDED
    assert result.output["providerId"] == "dictalm2-hebrew-gpu"
    assert general_worker.requests == []

    other = await plugin.execute(
        PluginRequest(
            "job-2",
            "ask-selected-files",
            "manual",
            {"sources": [str(source)], "operation": "translate", "language": "Italian"},
            1,
            None,
        ),
        CancellationController(),
        Progress(),
    )
    assert other.status is PluginResultStatus.SUCCEEDED
    assert other.output["providerId"] == "qwen3-gpu"


@pytest.mark.asyncio
async def test_an_answer_that_does_not_cite_its_content_is_refused(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("text", encoding="utf-8")
    plugin, _worker, _store = await running_plugin(
        mutate=lambda document: document["citations"][0].update(sourceRef="private:job-1:other")
    )

    result = await plugin.execute(
        request(source, operation="explain"), CancellationController(), Progress()
    )

    assert result.status is PluginResultStatus.FAILED
    assert result.detail.startswith("answer-invalid")


@pytest.mark.asyncio
async def test_tasks_outside_extraction_are_refused(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("text", encoding="utf-8")
    plugin, _worker, _store = await running_plugin(
        mutate=lambda document: document.update(tasks=["Invented"])
    )

    result = await plugin.execute(
        request(source, operation="summarize"), CancellationController(), Progress()
    )

    assert result.status is PluginResultStatus.FAILED
    assert result.detail.startswith("answer-invalid")


@pytest.mark.asyncio
async def test_an_empty_selected_file_is_refused_not_transformed(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("   \n", encoding="utf-8")
    plugin, _worker, _store = await running_plugin()

    for operation, extra in (("summarize", {}), ("translate", {"language": "Italian"})):
        result = await plugin.execute(
            request(source, operation=operation, **extra),
            CancellationController(),
            Progress(),
        )
        assert result.status is PluginResultStatus.FAILED
        assert result.detail.startswith("source-empty")


def test_grounded_operation_answer_enforces_the_contract():
    fragment = SourceFragment(
        "private:job-1:operation-content-1",
        hashlib.sha256(b"content").hexdigest(),
        1,
        "content",
        hashlib.sha256(b"content").hexdigest(),
    )
    citation = {
        "sourceRef": fragment.reference,
        "sourceSha256": fragment.source_sha256,
        "page": fragment.page,
        "span": {"start": 0, "end": len(fragment.text)},
        "textSha256": fragment.text_sha256,
    }
    good = {
        "version": 1,
        "requestId": "job-1",
        "operation": "explain",
        "answer": "An explanation.",
        "citations": [citation],
    }
    assert grounded_operation_answer(good, "job-1", "explain", fragment) == {
        "answer": "An explanation.",
        "tasks": [],
    }
    # An answer with no operation field is still this operation's answer —
    # the host validated the citation, and the field is optional in the
    # contract precisely so the ask path never carries it.
    unnamed = dict(good)
    unnamed.pop("operation")
    assert grounded_operation_answer(unnamed, "job-1", "explain", fragment)["answer"]

    for mutated, code in (
        ({**good, "requestId": "job-2"}, "answer-invalid"),
        ({**good, "operation": "summarize"}, "answer-invalid"),
        ({**good, "tasks": ["Invented"]}, "answer-invalid"),
        ({**good, "citations": [citation, citation]}, "answer-invalid"),
        ({**good, "citations": [{**citation, "page": 2}]}, "answer-invalid"),
        ({**good, "citations": [{**citation, "textSha256": "e" * 64}]}, "answer-invalid"),
        ("not a mapping", "answer-invalid"),
    ):
        with pytest.raises(DocumentQuestionError) as caught:
            grounded_operation_answer(mutated, "job-1", "explain", fragment)
        assert caught.value.code == code


def test_the_envelope_requires_each_section_exactly_when_its_operation_ran():
    citation = {
        "fileId": "selected-file-1",
        "fileName": "notes.txt",
        "sourceSha256": "a" * 64,
        "page": 1,
        "span": {"start": 0, "end": 4},
        "textSha256": "b" * 64,
    }
    document = {
        "documentId": "selected-document-1",
        "fileName": "notes.txt",
        "sourceSha256": "a" * 64,
        "spanCount": 1,
        "text": "texto",
        "translationSha256": "c" * 64,
    }
    base = {
        "version": 1,
        "requestId": "job-1",
        "answer": "answer",
        "providerId": "qwen3-gpu",
        "accelerator": "gpu",
        "citations": [citation],
    }
    # The ask result every client already renders is still a valid instance.
    assert validate_document("document-question-result.schema.json", base) == []
    # Presence and absence follow the operation, both ways.
    accepted = (
        {**base, "operation": "explain"},
        {**base, "operation": "translate", "documents": [document]},
        {**base, "operation": "extract-tasks", "tasks": ["Send the invoice"]},
        {**base, "operation": "extract-tasks", "tasks": []},
    )
    refused = (
        {**base, "documents": [document]},
        {**base, "tasks": ["Send the invoice"]},
        {**base, "operation": "translate"},
        {**base, "operation": "extract-tasks"},
        {**base, "operation": "summarize", "documents": [document]},
        {**base, "operation": "summarize", "tasks": []},
    )
    for payload in accepted:
        assert validate_document("document-question-result.schema.json", payload) == [], payload
    for payload in refused:
        assert validate_document("document-question-result.schema.json", payload) != [], payload


def test_file_operations_result_validates_what_it_publishes():
    with pytest.raises(DocumentQuestionError) as caught:
        file_operations_result(
            "job-1",
            "translate",
            "answer",
            (),
            provider_id="qwen3-gpu",
            accelerator="gpu",
            documents=(),
        )
    assert caught.value.code == "result-invalid"


def test_the_control_fragment_is_the_shared_closed_shape():
    fragment = operation_control_fragment("job-1", "translate", "Hebrew")

    assert fragment.reference == "private:job-1:control"
    assert json.loads(fragment.text) == {"operation": "translate", "language": "Hebrew"}
    assert fragment.source_sha256 == hashlib.sha256(fragment.text.encode()).hexdigest()
    assert fragment.text_sha256 == fragment.source_sha256


# --- request-validation boundary, property-tested -------------------------

_LANGUAGES = st.sampled_from([None, "Hebrew", "Português", "Italian"])
_QUESTIONS = st.sampled_from([None, "What is due?"])
_OPERATIONS = st.sampled_from(sorted(FILE_OPERATIONS) + ["condense", "", None])


@given(operation=_OPERATIONS, question=_QUESTIONS, language=_LANGUAGES)
def test_request_validation_admits_exactly_the_contract(operation, question, language):
    """question iff ask, language iff translate, enum closed — every combination."""
    import asyncio

    plugin = DocumentQuestionPlugin(
        Embedder(),
        GenerationRouter((OperationWorker(MemoryFragmentStore()),)),
        MemoryFragmentStore(),
    )
    asyncio.run(
        plugin.start(PluginContext("ask-selected-files", 1, {}, frozenset({"files:read-selected"})))
    )
    payload: dict = {"sources": ["/tmp/x.txt"]}
    if operation is not None:
        payload["operation"] = operation
    if question is not None:
        payload["question"] = question
    if language is not None:
        payload["language"] = language
    candidate = PluginRequest("job-1", "ask-selected-files", "manual", payload, 1, None)
    effective = "ask" if operation is None else operation
    valid = (
        effective in FILE_OPERATIONS
        and (question is not None) == (effective == "ask")
        and (language is not None) == (effective == "translate")
    )
    if valid:
        assert plugin._validate_request(candidate) == (  # noqa: SLF001
            effective,
            question,
            language,
        )
    else:
        with pytest.raises(DocumentQuestionError) as caught:
            plugin._validate_request(candidate)  # noqa: SLF001
        assert caught.value.code in {
            "operation-invalid",
            "question-invalid",
            "language-invalid",
        }


def test_the_file_enum_is_the_core_enum_plus_ask():
    assert OPERATIONS | {"ask"} == FILE_OPERATIONS
