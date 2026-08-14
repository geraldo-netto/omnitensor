"""The `document-intelligence` index: metadata and vectors, never text.

An index is the wrong place to keep document text.  It is a single file that
outlives the documents it describes, it is read by every search, and text in it
would survive the deletion of the source — so a leak here is a leak of
everything the user ever indexed, long after they think they removed it.  The
embedding is the only derived representation kept, and an attribute whose name
suggests it carries text is refused outright rather than truncated, because a
truncated snippet is still the document's contents.

Identity is the content digest, matching :mod:`omnitensor.plugins.visual_index`:
a document that is moved or renamed keeps its entry, and two copies of the same
file are one entry rather than two.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from .index import (
    BoundedIndexStore,
    IndexEntry,
    ModelIdentity,
    ingested_entry,
    integer_attribute_error,
    prefixed_entry_error,
    prefixed_entry_id,
    unsupported_attributes_error,
    validated_tags,
)
from .ingestion import IngestedFile

DOCUMENT_PLUGIN_ID = "document-intelligence"
DOCUMENT_INDEX_NAME = "document-intelligence.index.json"
DOCUMENT_INDEX_PERMISSION = "read:document-intelligence-index"
MAX_PAGES = 100_000
MAX_WORDS = 100_000_000
_LANGUAGE = re.compile(r"^[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})?$")
_ALLOWED_ATTRIBUTES = frozenset(
    {"sizeBytes", "format", "pageCount", "wordCount", "language", "createdAtMs"}
)
_CONTENT_ATTRIBUTES = frozenset(
    {"text", "body", "snippet", "title", "content", "ocrText", "excerpt", "summary"}
)


def document_entry_id(digest: str) -> str:
    """Content identity, so a renamed document keeps its place in the index."""
    return prefixed_entry_id(digest, "doc-", "document")


def document_entry(
    item: IngestedFile,
    embedding: Sequence[float],
    labels: Sequence[str] = (),
    attributes: Mapping[str, object] | None = None,
) -> IndexEntry:
    """Build an index entry from an ingested document and its computed vector."""
    return ingested_entry(
        item,
        embedding,
        labels,
        attributes,
        prefix="doc-",
        family="document",
    )


class DocumentIntelligenceIndex(BoundedIndexStore):
    """A durable, bounded index of document embeddings and classifications."""

    label = "document intelligence index"

    def __init__(self, root, model: ModelIdentity, *, labels: Sequence[str] = (), **bounds):
        super().__init__(root, DOCUMENT_INDEX_NAME, model, **bounds)
        self._labels = frozenset(validated_tags(labels)) if labels else frozenset()

    @property
    def labels(self) -> frozenset[str]:
        return self._labels

    def entry_error(self, entry: IndexEntry) -> str:
        error = prefixed_entry_error(
            entry,
            prefix="doc-",
            family="document",
            tag_name="classification label",
            vocabulary=self._labels,
            vocabulary_name="declared set",
        )
        return error or _attribute_error(entry.attributes)


def _attribute_error(attributes: Mapping[str, object]) -> str:
    carried = sorted(set(attributes) & _CONTENT_ATTRIBUTES)
    if carried:
        # Truncating would not help: a snippet is still the document's text,
        # and it would outlive the document it was taken from.
        return f"the index must not carry document text: {', '.join(carried)}"
    error = unsupported_attributes_error(attributes, _ALLOWED_ATTRIBUTES, "document")
    if error:
        return error
    return _count_error(attributes) or _language_error(attributes)


def _count_error(attributes: Mapping[str, object]) -> str:
    for name, maximum in (("pageCount", MAX_PAGES), ("wordCount", MAX_WORDS)):
        error = integer_attribute_error(attributes, name, 0, maximum)
        if error:
            return error
    return ""


def _language_error(attributes: Mapping[str, object]) -> str:
    language = attributes.get("language")
    if language is None:
        return ""
    if not isinstance(language, str) or not _LANGUAGE.match(language):
        return "language must be a BCP 47 style tag"
    return ""
