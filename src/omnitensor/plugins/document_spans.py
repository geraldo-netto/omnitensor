"""The span index every document workload reads a selection through.

A selection becomes an ordered index of page spans, each carrying the exact
offsets it came from, and a question is answered from those spans in as many
passes as the selection needs. It lived inside `document_qa` — the question
workload — so the file organiser, which needs the index and has no questions,
imported the question module to get it, and a change to how an answer is
published touched the module three plugins depend on.

Nothing here bounds a selection: ranking decides the order a document is read
in, never which parts of it are read at all.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..stable_error import StableError
from .event_workload import SelectedSource, select_sources
from .fragments import SourceFragment
from .protocol import CancellationToken
from .search import cosine_similarity

MAX_SOURCES = 16
MAX_SPAN_CHARACTERS = 2_048
EMBEDDING_DIMENSIONS = 384
# Three characters to a token is the ratio these tasks were measured at; it
# decides how many spans fit in one pass, never how long an answer may be.
CHARACTERS_PER_TOKEN = 3
# What the prompt itself costs before a single span is added.
PROMPT_OVERHEAD_TOKENS = 2_048
DIGEST_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_PRIVATE_REFERENCE = re.compile(r"^private:[A-Za-z0-9._:-]{1,220}$")


class DocumentQuestionError(StableError, ValueError):
    """Stable refusal that carries no selected content or path."""


@dataclass(frozen=True, slots=True)
class EmbeddingProvider:
    provider_id: str
    accelerator: str
    model_sha256: str
    qualified: bool


@runtime_checkable
class TextEmbeddingWorker(Protocol):
    """Private BGE execution port implemented by a GPU/NPU/TPU worker."""

    @property
    def descriptor(self) -> EmbeddingProvider: ...

    async def embed(
        self,
        texts: Sequence[str],
        *,
        query: bool,
        cancellation: CancellationToken,
    ) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True, slots=True)
class IndexedSpan:
    reference: str
    file_id: str
    file_name: str
    source_sha256: str
    page: int
    start: int
    end: int
    text: str
    text_sha256: str

    def fragment(self) -> SourceFragment:
        return SourceFragment(
            self.reference,
            self.source_sha256,
            self.page,
            self.text,
            self.text_sha256,
        )


def select_question_sources(request_id: str, value: object) -> tuple[SelectedSource, ...]:
    selected = select_sources(request_id, value)
    if len(selected) > MAX_SOURCES:
        raise DocumentQuestionError("sources-invalid", f"select 1-{MAX_SOURCES} source files")
    if any(source.item.suffix == ".ics" for source in selected):
        raise DocumentQuestionError(
            "source-unsupported", "calendar sources are not document question inputs"
        )
    return selected


def page_spans(
    request_id: str,
    source_number: int,
    file_name: str,
    source_sha256: str,
    pages: Sequence[object],
    *,
    remaining: int | None = None,
) -> tuple[IndexedSpan, ...]:
    """Index every page span, with exact page offsets.

    ``remaining`` is how many spans this call may still add. No caller passes
    it any more — the file organiser did, and classified every document from
    its first two spans — so every path reads the whole file. ``None``, the
    default, is that: the ceiling that used to live here stopped at 512 spans
    and returned what it had without telling anybody the rest existed.
    """
    if (
        not isinstance(request_id, str)
        or not request_id
        or not isinstance(source_number, int)
        or isinstance(source_number, bool)
        or source_number < 1
        or not isinstance(file_name, str)
        or not file_name
        or "/" in file_name
        or "\\" in file_name
        or DIGEST_PATTERN.fullmatch(source_sha256) is None
        or isinstance(remaining, bool)
        or (remaining is not None and (not isinstance(remaining, int) or remaining < 0))
    ):
        raise DocumentQuestionError("span-invalid", "span source identity is invalid")
    indexed: list[IndexedSpan] = []
    for page in pages:
        page_number = getattr(page, "page_number", None)
        text = getattr(page, "text", None)
        if (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number < 1
            or not isinstance(text, str)
        ):
            raise DocumentQuestionError("span-invalid", "extracted page is invalid")
        for start in range(0, len(text), MAX_SPAN_CHARACTERS):
            if remaining is not None and len(indexed) >= remaining:
                return tuple(indexed)
            end = min(start + MAX_SPAN_CHARACTERS, len(text))
            content = text[start:end]
            if not content.strip():
                continue
            reference = (
                f"private:{request_id}:source:{source_number}:page:{page_number}:span:{start}-{end}"
            )
            if _PRIVATE_REFERENCE.fullmatch(reference) is None:
                raise DocumentQuestionError("span-invalid", "private span reference is invalid")
            indexed.append(
                IndexedSpan(
                    reference,
                    f"selected-file-{source_number}",
                    file_name,
                    source_sha256,
                    page_number,
                    start,
                    end,
                    content,
                    hashlib.sha256(content.encode("utf-8")).hexdigest(),
                )
            )
    return tuple(indexed)


def answer_passes(
    spans: Sequence[IndexedSpan],
    context_tokens: int,
) -> tuple[tuple[IndexedSpan, ...], ...]:
    """Split ranked spans into the fewest passes the context window allows.

    The only bound here is arithmetic: what is left of the window once the
    prompt and the question are in it. A document that fits is answered in one
    pass, which is every ordinary selection; one that does not is answered in
    several that accumulate, because the alternative — telling a person about
    the part of their file that happened to fit — is a wrong answer that looks
    like a complete one.
    """
    if not spans:
        return ()
    usable_tokens = context_tokens - PROMPT_OVERHEAD_TOKENS
    # Room for the answer as well as the spans it is drawn from. Half the
    # remaining window, so a long question about a long document can still be
    # answered at length rather than running out of window mid-object.
    budget = max(MAX_SPAN_CHARACTERS, (usable_tokens // 2) * CHARACTERS_PER_TOKEN)
    passes: list[tuple[IndexedSpan, ...]] = []
    current: list[IndexedSpan] = []
    used = 0
    for span in spans:
        cost = len(span.text)
        if current and used + cost > budget:
            passes.append(tuple(current))
            current = []
            used = 0
        current.append(span)
        used += cost
    if current:
        passes.append(tuple(current))
    return tuple(passes)


async def retrieve_spans(
    question: str,
    spans: Sequence[IndexedSpan],
    embedder: TextEmbeddingWorker,
    *,
    cancellation: CancellationToken,
) -> tuple[IndexedSpan, ...]:
    if not isinstance(question, str) or not question or not spans:
        raise DocumentQuestionError("retrieval-invalid", "question or span index is invalid")
    cancellation.raise_if_cancelled()
    query_vectors = await embedder.embed((question,), query=True, cancellation=cancellation)
    span_vectors = await embedder.embed(
        tuple(span.text for span in spans), query=False, cancellation=cancellation
    )
    query = _embedding_vector(query_vectors, 1)[0]
    vectors = _embedding_vector(span_vectors, len(spans))
    ranked = sorted(
        (
            (cosine_similarity(query, vector), span.reference, span)
            for span, vector in zip(spans, vectors, strict=True)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    # Every span, most relevant first. Ranking decides the order the document
    # is read in, never which parts of it are read at all — the slice that
    # used to be here answered from the best eight spans and reported the rest
    # of the file as though it did not exist.
    return tuple(item[2] for item in ranked)


def _embedding_vector(
    vectors: Sequence[Sequence[float]], expected: int
) -> tuple[tuple[float, ...], ...]:
    if isinstance(vectors, (str, bytes)) or not isinstance(vectors, Sequence):
        raise DocumentQuestionError("embedding-invalid", "embedding output is invalid")
    parsed: list[tuple[float, ...]] = []
    for vector in vectors:
        if (
            isinstance(vector, (str, bytes))
            or not isinstance(vector, Sequence)
            or len(vector) != EMBEDDING_DIMENSIONS
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in vector
            )
        ):
            raise DocumentQuestionError("embedding-invalid", "embedding output is invalid")
        parsed.append(tuple(float(value) for value in vector))
    if len(parsed) != expected:
        raise DocumentQuestionError("embedding-invalid", "embedding batch size is invalid")
    return tuple(parsed)
