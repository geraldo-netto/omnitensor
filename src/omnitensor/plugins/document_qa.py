"""Grounded answers over files chosen for one explicit request.

The worker never watches a directory and never turns a selected parent into a
grant.  It extracts only the brokered files in the request, builds a bounded
in-memory BGE span index, sends only the best spans to a qualified generation worker,
and discards every source fragment before returning.  Public results contain
file/page/span addresses and digests, never source text or absolute paths.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from ..registry import validate_document
from ..sdk import (
    ManagedPlugin,
    PluginHealth,
    PluginHealthStatus,
    PluginProgress,
    SDKContractError,
    succeeded_result,
    workload_result,
)
from .document_answer import (
    _answer_failure_detail,
    document_question_task,
    grounded_answer_document,
)
from .document_spans import (
    DIGEST_PATTERN,
    DocumentQuestionError,
    EmbeddingProvider,
    IndexedSpan,
    TextEmbeddingWorker,
    answer_passes,
    page_spans,
    retrieve_spans,
    select_question_sources,
)
from .event_workload import (
    MemoryFragmentStore,
    PlainTextAdapter,
    PyMuPdfAdapter,
    SelectedSource,
)
from .extraction import (
    ExtractionAdapter,
    SelectedDocumentReader,
)
from .file_operations import FILE_OPERATIONS, operate_on_selected_files
from .fragments import FragmentStoreError, SourceFragment
from .generation import (
    GenerationError,
    GenerationRouter,
    generation_request,
)
from .operations import (
    MAX_QUESTION_CHARACTERS,
    translation_route,
    validated_language,
    validated_translation_routes,
)
from .protocol import (
    CancellationToken,
    PluginRequest,
    PluginResult,
    ProgressReporter,
    ScaledProgressReporter,
)
from .translation import TranslationError

PLUGIN_ID = "ask-selected-files"
READ_PERMISSION = "files:read-selected"


class DocumentQuestionPlugin(ManagedPlugin):
    """Manual selected-file retrieval followed by citation-bound generation."""

    plugin_id = PLUGIN_ID

    def __init__(
        self,
        embedder: TextEmbeddingWorker,
        router: GenerationRouter,
        fragment_store: MemoryFragmentStore,
        *,
        translation_routes: Mapping[str, GenerationRouter] | None = None,
        adapters: Mapping[str, ExtractionAdapter] | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(embedder, TextEmbeddingWorker):
            raise DocumentQuestionError("embedder-invalid", "embedding worker is required")
        _validate_embedding_provider(embedder.descriptor)
        if not isinstance(router, GenerationRouter):
            raise DocumentQuestionError("provider-invalid", "generation router is required")
        if not isinstance(fragment_store, MemoryFragmentStore):
            raise DocumentQuestionError("store-invalid", "private fragment store is required")
        if clock_ms is not None and not callable(clock_ms):
            raise DocumentQuestionError("clock-invalid", "clock must be callable")
        self._embedder = embedder
        self._router = router
        self._translation_routes = validated_translation_routes(
            translation_routes, DocumentQuestionError, noun="translation route language"
        )
        self._store = fragment_store
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        defaults: dict[str, ExtractionAdapter] = {
            suffix: PlainTextAdapter() for suffix in (".txt", ".md")
        }
        media = PyMuPdfAdapter()
        defaults.update({suffix: media for suffix in (".pdf", ".png", ".jpg", ".jpeg", ".webp")})
        for suffix, adapter in (adapters or {}).items():
            if suffix not in defaults or not isinstance(adapter, ExtractionAdapter):
                raise DocumentQuestionError("adapter-invalid", "source adapter mapping is invalid")
            defaults[suffix] = adapter
        self._adapters = defaults
        self._reader = SelectedDocumentReader(defaults, DocumentQuestionError)

    async def on_start(self) -> None:
        self.permissions.require(READ_PERMISSION)
        _validate_embedding_provider(self._embedder.descriptor, require_qualified=True)
        self._router.require_ready()
        for route in self._translation_routes.values():
            route.require_ready()

    async def on_health(self) -> PluginHealth:
        descriptor = self._embedder.descriptor
        return PluginHealth(
            PluginHealthStatus.READY,
            f"ready on {descriptor.accelerator}; explicit selected files only",
            self._clock_ms(),
        )

    async def execute(
        self,
        request: PluginRequest,
        cancellation: CancellationToken,
        progress: ProgressReporter,
    ) -> PluginResult:
        return await workload_result(
            request,
            lambda: self._answer(request, cancellation, progress),
            completed_at_ms=self._clock_ms,
            cancelled_detail="selected-file question cancelled",
            failures=(
                DocumentQuestionError,
                # EventWorkloadError subclasses FragmentStoreError; naming the
                # base keeps a store failure inside the plugin result instead
                # of letting it escape and kill the job with no result at all.
                FragmentStoreError,
                GenerationError,
                SDKContractError,
                TranslationError,
            ),
            detail_of=_answer_failure_detail,
            discard=self._store.discard,
        )

    async def _answer(
        self,
        request: PluginRequest,
        cancellation: CancellationToken,
        progress: ProgressReporter,
    ) -> PluginResult:
        """Dispatch on the operation; ask keeps its retrieval path unchanged.

        The ask branch retrieves, generates in as many passes as the selection
        needs, and cites; every other operation of the shared enum runs
        through :mod:`file_operations` (OMNI-0617 stage 2).
        """
        operation, question, language = self._validate_request(request)
        if operation != "ask":
            output = await operate_on_selected_files(
                request,
                operation=operation,
                language=language,
                reader=self._reader,
                adapters=self._adapters,
                router=self._route(operation, language),
                store=self._store,
                clock_ms=self._clock_ms,
                cancellation=cancellation,
                progress=progress,
            )
            await self._progress(request, progress, "terminal", 1.0)
            return succeeded_result(
                request,
                output,
                completed_at_ms=self._clock_ms(),
                detail=(
                    "selected documents translated in full"
                    if operation == "translate"
                    else "operation result grounded in selected files"
                ),
            )
        assert question is not None
        selected = select_question_sources(request.job_id, request.payload.get("sources"))
        await self._progress(request, progress, "extract", 0.05)
        spans = await self._extract_spans(request, selected, cancellation, progress)
        cancellation.raise_if_cancelled()
        await self._progress(request, progress, "retrieve", 0.5)
        retrieved = await retrieve_spans(question, spans, self._embedder, cancellation=cancellation)
        question_digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
        question_fragment = SourceFragment(
            f"private:{request.job_id}:question",
            question_digest,
            1,
            question,
            question_digest,
        )
        task = document_question_task()
        passes = answer_passes(retrieved, task.limits.context_tokens)
        await self._progress(request, progress, "answer", 0.7)
        answers: list[str] = []
        citations: list[dict] = []
        seen_citations: set[tuple] = set()
        provider_id = ""
        accelerator = ""
        # Published once, outside the loop: the store rejects a repeated
        # reference, so republishing the same question on pass 2 refused
        # every multi-pass answer.
        await self._store.publish(request.job_id, (question_fragment,))
        for index, batch in enumerate(passes):
            cancellation.raise_if_cancelled()
            await self._store.publish(
                request.job_id,
                tuple(span.fragment() for span in batch),
            )
            generated = await self._router.run(
                task,
                generation_request(
                    request.job_id,
                    PLUGIN_ID,
                    (question_fragment.reference, *(span.reference for span in batch)),
                ),
                cancellation,
                ScaledProgressReporter(
                    request,
                    progress,
                    self._clock_ms,
                    stage="answer",
                    offset=0.7 + (0.25 * index / len(passes)),
                    scale=0.25 / len(passes),
                    error_type=DocumentQuestionError,
                ),
            )
            partial = grounded_answer_document(
                generated.document,
                request.job_id,
                batch,
                provider_id=generated.provider_id,
                accelerator=generated.accelerator,
            )
            provider_id = partial["providerId"]
            accelerator = partial["accelerator"]
            answers.append(str(partial["answer"]).strip())
            for citation in partial["citations"]:
                span = citation["span"]
                key = (
                    citation["sourceSha256"],
                    citation["page"],
                    span["start"],
                    span["end"],
                )
                if key not in seen_citations:
                    seen_citations.add(key)
                    citations.append(citation)
        output = {
            "version": 1,
            "requestId": request.job_id,
            # Every pass, in the order the document was read. Joined rather
            # than picked between: each pass answered from spans no other
            # pass saw, so dropping one drops part of the person's file.
            "answer": "\n\n".join(answer for answer in answers if answer),
            "providerId": provider_id,
            "accelerator": accelerator,
            "citations": citations,
        }
        violations = validate_document("document-question-result.schema.json", output)
        if violations:
            raise DocumentQuestionError("answer-invalid", violations[0])
        await self._progress(request, progress, "terminal", 1.0)
        return succeeded_result(
            request,
            output,
            completed_at_ms=self._clock_ms(),
            detail="answer grounded in selected files",
        )

    def _validate_request(self, request: PluginRequest) -> tuple[str, str | None, str | None]:
        if not isinstance(request, PluginRequest) or request.plugin_id != PLUGIN_ID:
            raise DocumentQuestionError("request-invalid", "request names another plugin")
        if request.trigger != "manual":
            raise DocumentQuestionError(
                "request-invalid", "selected-file questions are manual only"
            )
        self.permissions.require(READ_PERMISSION)
        # The operation joined the wire contract as optional-with-default
        # (OMNI-0625): an absent field is the ask every payload has always
        # meant. Since stage 2 (OMNI-0617) the whole shared enum dispatches
        # here. Checked as well as schema-validated because a worker defends
        # its own contract, not just the schema's.
        operation = request.payload.get("operation", "ask")
        if operation not in FILE_OPERATIONS:
            raise DocumentQuestionError(
                "operation-invalid", "selected-file operation is unsupported"
            )
        return _validated_parameters(
            str(operation),
            request.payload.get("question"),
            request.payload.get("language"),
        )

    def _route(self, operation: str, language: str | None) -> GenerationRouter:
        """The route this operation rides: language-specific for translate,
        the general one for everything else — the same rule the inline side
        states, so DictaLM answers exactly an explicit Hebrew target."""
        if operation != "translate" or language is None:
            return self._router
        return translation_route(self._router, self._translation_routes, language)

    async def _extract_spans(
        self,
        request: PluginRequest,
        selected: Sequence[SelectedSource],
        cancellation: CancellationToken,
        progress: ProgressReporter,
    ) -> tuple[IndexedSpan, ...]:
        spans: list[IndexedSpan] = []
        try:
            for index, source in enumerate(selected):
                cancellation.raise_if_cancelled()
                extraction = await self._reader.read(source.item)
                spans.extend(
                    page_spans(
                        request.job_id,
                        index + 1,
                        Path(source.item.path).name,
                        source.item.digest,
                        extraction.pages,
                    )
                )
                await self._progress(
                    request,
                    progress,
                    "extract",
                    0.05 + (0.4 * (index + 1) / len(selected)),
                )
        finally:
            # An adapter that holds an engine gives it back here, whatever
            # happened: extraction is the only phase that needs it, and
            # embedding and generation need the memory (OMNI-0621).
            unique = {id(adapter): adapter for adapter in self._adapters.values()}
            for adapter in unique.values():
                closer = getattr(adapter, "close", None)
                if callable(closer):
                    closer()
        if not spans:
            raise DocumentQuestionError("source-empty", "selected files produced no text spans")
        return tuple(spans)

    async def _progress(
        self,
        request: PluginRequest,
        progress: ProgressReporter,
        stage: str,
        fraction: float,
    ) -> None:
        await progress.report(PluginProgress(request.job_id, stage, fraction, "", self._clock_ms()))


def _validated_parameters(
    operation: str, question: object, language: object
) -> tuple[str, str | None, str | None]:
    """question exactly when asking, language exactly when translating."""
    if operation == "ask":
        if language is not None:
            raise DocumentQuestionError(
                "language-invalid", "language is accepted only for translation"
            )
        if (
            not isinstance(question, str)
            or not question.strip()
            or len(question) > MAX_QUESTION_CHARACTERS
        ):
            raise DocumentQuestionError(
                "question-invalid",
                f"question must contain 1-{MAX_QUESTION_CHARACTERS} characters",
            )
        return "ask", question.strip(), None
    if question is not None:
        raise DocumentQuestionError("question-invalid", "a question is accepted only when asking")
    if operation == "translate":
        return (
            operation,
            None,
            validated_language(language, DocumentQuestionError, noun="translation language"),
        )
    if language is not None:
        raise DocumentQuestionError("language-invalid", "language is accepted only for translation")
    return operation, None, None


def _validate_embedding_provider(descriptor: object, *, require_qualified: bool = False) -> None:
    if (
        not isinstance(descriptor, EmbeddingProvider)
        or descriptor.accelerator not in {"gpu", "npu", "tpu"}
        or not descriptor.provider_id
        or DIGEST_PATTERN.fullmatch(descriptor.model_sha256) is None
        or (require_qualified and not descriptor.qualified)
    ):
        raise DocumentQuestionError(
            "embedder-unqualified", "qualified non-CPU BGE provider is required"
        )


__all__ = ["DocumentQuestionPlugin"]
