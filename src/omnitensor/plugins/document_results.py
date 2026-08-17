"""`document-intelligence` reads: classification, similarity, and search.

The redaction guarantee here is inherited rather than reimplemented: the index
refuses to store document text at all, so no projection over it can leak one.
What this layer adds is the classification view — which documents carry which
label, and how many — with the same pagination, provenance, and explicit
stale/indexing reporting as every other read.

Counting is not free of disclosure either.  A per-label count over a small
corpus can identify a single document, so a summary reports the same state a
page does and stays silent when the index is not usable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .document_index import (
    DOCUMENT_INDEX_PERMISSION,
    DOCUMENT_PLUGIN_ID,
    DocumentIntelligenceIndex,
)
from .search import IndexQueryService, ResultPage, ResultState

DOCUMENT_CLASSIFY_PERMISSION = "write:document-intelligence-labels"


@dataclass(frozen=True, slots=True)
class ClassificationSummary:
    """How many documents carry each label, and whether that can be trusted."""

    # Pairs, in the order they are meant to be read — commonest first, ties by
    # label. A dict field made a `frozen=True` value mutable and unhashable,
    # and left the ordering an implicit property of insertion.
    counts: tuple[tuple[str, int], ...]
    state: ResultState
    revision: int
    total: int

    @property
    def usable(self) -> bool:
        return self.state is ResultState.READY

    def document(self) -> dict:
        return {
            "counts": dict(self.counts),
            "state": str(self.state),
            "revision": self.revision,
            "total": self.total,
        }


class DocumentIntelligenceResults(IndexQueryService):
    """Authorized, paginated, redacted access to the document index."""

    read_permission = DOCUMENT_INDEX_PERMISSION
    write_permission = DOCUMENT_CLASSIFY_PERMISSION
    label = "document intelligence results"

    def __init__(self, store: DocumentIntelligenceIndex, permissions, **options) -> None:
        if not isinstance(store, DocumentIntelligenceIndex):
            raise TypeError("store must be a DocumentIntelligenceIndex")
        super().__init__(store, permissions, **options)

    @property
    def plugin_id(self) -> str:
        return DOCUMENT_PLUGIN_ID

    def classified_as(
        self,
        label: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> ResultPage:
        """List the documents carrying one classification label, newest first."""
        return self.browse(tags=(label,), limit=limit, cursor=cursor)

    def classify(self, entry_id: str, labels: Sequence[str]) -> ResultPage:
        """Replace a document's labels, which needs the classification grant."""
        return self.assign_tags(entry_id, labels)

    def classification_summary(self) -> ClassificationSummary:
        """Count documents per label, reporting why when the count is empty."""
        self._permissions.require(self.read_permission)
        state = self._store.load()
        result_state = self._state_of(state)
        if result_state is not ResultState.READY:
            return ClassificationSummary((), result_state, state.revision, 0)
        counts: dict[str, int] = {}
        for entry in state.entries:
            for label in entry.tags:
                counts[label] = counts.get(label, 0) + 1
        # Every label, not the first two hundred. The list was truncated after
        # sorting by descending count, so what a person lost was always the
        # rarest labels — the ones worth seeing — and nothing said so.
        ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
        return ClassificationSummary(
            tuple(ordered),
            result_state,
            state.revision,
            len(state.entries),
        )
