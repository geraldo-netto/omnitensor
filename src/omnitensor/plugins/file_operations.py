"""The operations core over file sources (OMNI-0617 stage 2).

`ask-selected-files` grew the shared operation enum: `ask` keeps its
retrieval-and-citation pipeline untouched in :mod:`document_qa`, while
explain/summarize/rewrite/extract-tasks run the per-operation instruction
table from :mod:`operations` over the whole extracted content, and
`translate` runs the span pipeline lifted (imported, never moved) from
:mod:`document_translation`. One result envelope serves all of them —
`answer` and `citations` always, `documents` only for translate, `tasks`
only for extract-tasks — so the client keeps a single result schema.

A NEW module on purpose: document_qa/document_spans/document_answer/fragments
are pinned by mutation-selector paths and stay where they are; this is the
part of the workload that did not exist before the dispatch did.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from ..registry import validate_document
from .document_answer import document_question_task
from .document_spans import (
    DocumentQuestionError,
    IndexedSpan,
    answer_passes,
    page_spans,
    select_question_sources,
)
from .event_workload import MemoryFragmentStore, SelectedSource
from .extraction import ExtractionAdapter, SelectedDocumentReader
from .fragments import SourceFragment
from .generation import GenerationRouter, checked_answer, generation_request
from .operations import OPERATIONS, OperationPrompting
from .protocol import (
    CancellationToken,
    PluginRequest,
    ProgressReporter,
    ScaledProgressReporter,
)
from .translation import translate_document

PLUGIN_ID = "ask-selected-files"
# The file-side enum: the five shared operations plus the ask this workload
# has always answered. `ask` is not in the core's OPERATIONS because the
# inline workload does not (yet) take a question; the manifests state each
# side's own surface and the core states what they share.
FILE_OPERATIONS = frozenset({"ask", *OPERATIONS})
# Half the window for the source span, half for its translation — the same
# arithmetic document-translation states, restated here because the constant
# describes this task's window, not that module's.
TRANSLATION_WINDOW_SHARE = 2


class FileOperationsPrompting(OperationPrompting):
    """The file workload's prompting: the operations core, on this task id.

    Active only for operation requests — an ask request's first fragment is
    the question, not a closed control object, so the core contributes
    nothing to it and the reviewed ask prompt is unchanged.
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(PLUGIN_ID, answer_field="answer")


def operation_control_fragment(job_id: str, operation: str, language: str | None) -> SourceFragment:
    """The closed `{operation, language}` control, as the inline side states it.

    One control shape for every workload that reaches the Dicta route: the
    Hebrew runtime parses exactly this object, so the file side publishing it
    is what lets `hebrew.py` keep one contract per task rather than two.
    """
    control_text = json.dumps(
        {"operation": operation, "language": language},
        sort_keys=True,
        separators=(",", ":"),
    )
    control_digest = hashlib.sha256(control_text.encode("utf-8")).hexdigest()
    return SourceFragment(
        f"private:{job_id}:control",
        control_digest,
        1,
        control_text,
        control_digest,
    )


def grounded_operation_answer(
    document: object,
    request_id: str,
    operation: str,
    fragment: SourceFragment,
) -> dict:
    """One generation's answer, checked against the fragment it was given.

    The model must cite exactly the content fragment it transformed — an
    answer built from other material is not this fragment's transformation —
    and may carry tasks only when task extraction asked for them.
    """
    document = checked_answer(
        "document-question-answer.schema.json",
        document,
        error_type=DocumentQuestionError,
        code="answer-invalid",
    )
    if document["requestId"] != request_id:
        raise DocumentQuestionError("answer-invalid", "answer request id does not match")
    stated = document.get("operation")
    if stated is not None and stated != operation:
        raise DocumentQuestionError("answer-invalid", "answer names another operation")
    tasks = list(document.get("tasks", ()))
    if operation != "extract-tasks" and tasks:
        raise DocumentQuestionError("answer-invalid", "tasks are accepted only for task extraction")
    citations = document["citations"]
    if len(citations) != 1:
        raise DocumentQuestionError(
            "answer-invalid", "an operation answer cites its one content fragment"
        )
    citation = citations[0]
    assert isinstance(citation, Mapping)
    if (
        citation["sourceRef"] != fragment.reference
        or citation["sourceSha256"] != fragment.source_sha256
        or citation["page"] != fragment.page
        or citation["span"] != {"start": 0, "end": len(fragment.text)}
        or citation["textSha256"] != fragment.text_sha256
    ):
        raise DocumentQuestionError(
            "answer-invalid", "answer does not cite the content it was given"
        )
    return {"answer": str(document["answer"]).strip(), "tasks": tasks}


def file_operations_result(
    request_id: str,
    operation: str,
    answer: str,
    citations: Sequence[Mapping],
    *,
    provider_id: str,
    accelerator: str,
    tasks: Sequence[str] = (),
    documents: Sequence[Mapping] = (),
) -> dict:
    """The one public envelope every file operation answers in."""
    result: dict = {
        "version": 1,
        "requestId": request_id,
        "operation": operation,
        "answer": answer,
        "providerId": provider_id,
        "accelerator": accelerator,
        "citations": [dict(citation) for citation in citations],
    }
    if operation == "extract-tasks":
        result["tasks"] = list(tasks)
    if operation == "translate":
        result["documents"] = [dict(document) for document in documents]
    violations = validate_document("document-question-result.schema.json", result)
    if violations:
        raise DocumentQuestionError("result-invalid", violations[0])
    return result


async def operate_on_selected_files(
    request: PluginRequest,
    *,
    operation: str,
    language: str | None,
    reader: SelectedDocumentReader,
    adapters: Mapping[str, ExtractionAdapter],
    router: GenerationRouter,
    store: MemoryFragmentStore,
    clock_ms: Callable[[], int],
    cancellation: CancellationToken,
    progress: ProgressReporter,
) -> dict:
    """Run one non-ask operation over the selected files, grounded throughout."""
    task = document_question_task()
    selected = select_question_sources(request.job_id, request.payload.get("sources"))
    control = operation_control_fragment(request.job_id, operation, language)
    await store.publish(request.job_id, (control,))
    extractions = []
    try:
        for index, source in enumerate(selected):
            cancellation.raise_if_cancelled()
            extractions.append((index, source, await reader.read(source.item)))
    finally:
        # An adapter that holds an engine gives it back here, whatever
        # happened: extraction is the only phase that needs it, and
        # generation needs the memory (OMNI-0621).
        unique = {id(adapter): adapter for adapter in adapters.values()}
        for adapter in unique.values():
            closer = getattr(adapter, "close", None)
            if callable(closer):
                closer()
    runner = _Runner(request, task, router, control, store, clock_ms, cancellation, progress)
    if operation == "translate":
        assert language is not None
        return await _translate_documents(runner, language, extractions)
    return await _transform_documents(runner, operation, extractions)


class _Runner:
    """One request's shared generation plumbing, so the two paths state none."""

    __slots__ = (
        "cancellation",
        "clock_ms",
        "control",
        "progress",
        "request",
        "router",
        "store",
        "task",
    )

    def __init__(self, request, task, router, control, store, clock_ms, cancellation, progress):
        self.request = request
        self.task = task
        self.router = router
        self.control = control
        self.store = store
        self.clock_ms = clock_ms
        self.cancellation = cancellation
        self.progress = progress

    async def generate(self, fragment: SourceFragment, *, offset: float, scale: float):
        await self.store.publish(self.request.job_id, (fragment,))
        return await self.router.run(
            self.task,
            generation_request(
                self.request.job_id,
                PLUGIN_ID,
                (self.control.reference, fragment.reference),
            ),
            self.cancellation,
            ScaledProgressReporter(
                self.request,
                self.progress,
                self.clock_ms,
                stage="operate",
                offset=offset,
                scale=scale,
                error_type=DocumentQuestionError,
            ),
        )


async def _transform_documents(
    runner: _Runner,
    operation: str,
    extractions: Sequence[tuple[int, SelectedSource, object]],
) -> dict:
    """Whole-content operation, in as many passes as the window needs.

    Every span of every selected file is in exactly one pass; the answer is
    the passes joined in reading order, because dropping one drops part of
    the person's file. Citations are the real page spans each pass covered,
    host-built from the index — the model proves it processed the pass's
    content fragment, and the host says which file bytes that content was.
    """
    request = runner.request
    spans: list[IndexedSpan] = []
    for index, source, extraction in extractions:
        spans.extend(
            page_spans(
                request.job_id,
                index + 1,
                Path(source.item.path).name,
                source.item.digest,
                extraction.pages,
            )
        )
    if not spans:
        raise DocumentQuestionError("source-empty", "selected files produced no text spans")
    passes = answer_passes(spans, runner.task.limits.context_tokens)
    answers: list[str] = []
    tasks: list[str] = []
    citations: list[dict] = []
    provider_id = ""
    accelerator = ""
    for number, batch in enumerate(passes):
        runner.cancellation.raise_if_cancelled()
        text = "\n\n".join(span.text for span in batch)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        fragment = SourceFragment(
            f"private:{request.job_id}:operation-content-{number + 1}",
            digest,
            number + 1,
            text,
            digest,
        )
        generated = await runner.generate(
            fragment,
            offset=0.1 + 0.85 * number / len(passes),
            scale=0.85 / len(passes),
        )
        partial = grounded_operation_answer(generated.document, request.job_id, operation, fragment)
        provider_id = generated.provider_id
        accelerator = generated.accelerator
        if partial["answer"]:
            answers.append(partial["answer"])
        for item in partial["tasks"]:
            if item not in tasks:
                tasks.append(item)
        citations.extend(_span_citation(span) for span in batch)
    return file_operations_result(
        request.job_id,
        operation,
        "\n\n".join(answers),
        citations,
        provider_id=provider_id,
        accelerator=accelerator,
        tasks=tasks,
    )


async def _translate_documents(
    runner: _Runner,
    language: str,
    extractions: Sequence[tuple[int, SelectedSource, object]],
) -> dict:
    """The document-translation span pipeline, under the file-side envelope.

    Every span is translated or the job fails, progress is spans over spans,
    and cancellation lands between spans — the properties
    :mod:`translation` already guarantees. What is new is only the envelope:
    the translation rides `answer`/`documents`, and each span's citation
    addresses the private fragment it was translated from.
    """
    request = runner.request
    documents: list[dict] = []
    citations: list[dict] = []
    route: dict[str, str] = {}
    count = len(extractions)
    for index, source, extraction in extractions:
        text = extraction.text
        if not text.strip():
            raise DocumentQuestionError(
                "source-empty", "each selected document must contain extractable text"
            )
        offset = 0.1 + 0.85 * index / count
        scale = 0.85 / count

        async def translate_span(
            span, *, _index=index, _source=source, _offset=offset, _scale=scale
        ):
            digest = hashlib.sha256(span.text.encode("utf-8")).hexdigest()
            fragment = SourceFragment(
                f"private:{request.job_id}:document-{_index + 1}-span-{span.index + 1}",
                digest,
                span.index + 1,
                span.text,
                digest,
            )
            generated = await runner.generate(fragment, offset=_offset, scale=_scale)
            partial = grounded_operation_answer(
                generated.document, request.job_id, "translate", fragment
            )
            route["providerId"] = generated.provider_id
            route["accelerator"] = generated.accelerator
            citations.append(
                {
                    "fileId": f"selected-file-{_index + 1}",
                    "fileName": Path(_source.item.path).name,
                    "sourceSha256": _source.item.digest,
                    "page": span.index + 1,
                    "span": {"start": span.start, "end": span.end},
                    "textSha256": digest,
                }
            )
            return partial["answer"]

        translated = await translate_document(
            text,
            output_tokens=_translation_tokens(runner.task),
            translate_span=translate_span,
            raise_if_cancelled=runner.cancellation.raise_if_cancelled,
        )
        documents.append(
            {
                "documentId": f"selected-document-{index + 1}",
                "fileName": Path(source.item.path).name,
                "sourceSha256": source.item.digest,
                "spanCount": translated.span_count,
                "text": translated.text,
                "translationSha256": hashlib.sha256(translated.text.encode("utf-8")).hexdigest(),
            }
        )
    return file_operations_result(
        request.job_id,
        "translate",
        "\n\n".join(document["text"] for document in documents),
        citations,
        provider_id=route.get("providerId", ""),
        accelerator=route.get("accelerator", ""),
        documents=documents,
    )


def _translation_tokens(task) -> int:
    """How much of the window one span's translation may take — arithmetic
    over the context window, honouring a declared budget when a task has one."""
    limits = task.limits
    if limits.output_tokens is not None:
        return limits.output_tokens
    return max(1, limits.context_tokens // TRANSLATION_WINDOW_SHARE)


def _span_citation(span: IndexedSpan) -> dict:
    return {
        "fileId": span.file_id,
        "fileName": span.file_name,
        "sourceSha256": span.source_sha256,
        "page": span.page,
        "span": {"start": span.start, "end": span.end},
        "textSha256": span.text_sha256,
    }


__all__ = [
    "FILE_OPERATIONS",
    "FileOperationsPrompting",
    "PLUGIN_ID",
    "file_operations_result",
    "grounded_operation_answer",
    "operate_on_selected_files",
    "operation_control_fragment",
]
