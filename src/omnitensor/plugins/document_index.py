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
    IndexStoreError,
    ModelIdentity,
    validated_tags,
)
from .ingestion import IngestedFile

DOCUMENT_PLUGIN_ID = "document-intelligence"
DOCUMENT_INDEX_NAME = "document-intelligence.index.json"
DOCUMENT_INDEX_PERMISSION = "read:document-intelligence-index"
MAX_PAGES = 100_000
MAX_WORDS = 100_000_000
_LABEL = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_LANGUAGE = re.compile(r"^[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})?$")
_ALLOWED_ATTRIBUTES = frozenset(
    {"sizeBytes", "format", "pageCount", "wordCount", "language", "createdAtMs"}
)
_CONTENT_ATTRIBUTES = frozenset(
    {"text", "body", "snippet", "title", "content", "ocrText", "excerpt", "summary"}
)


def document_entry_id(digest: str) -> str:
    """Content identity, so a renamed document keeps its place in the index."""
    if not isinstance(digest, str) or len(digest) != 64:
        raise IndexStoreError("entry-invalid", "document entry identity must be a sha256 digest")
    return f"doc-{digest}"


def document_entry(
    item: IngestedFile,
    embedding: Sequence[float],
    labels: Sequence[str] = (),
    attributes: Mapping[str, object] | None = None,
) -> IndexEntry:
    """Build an index entry from an ingested document and its computed vector."""
    if not isinstance(item, IngestedFile):
        raise IndexStoreError("entry-invalid", "item must be an IngestedFile")
    merged = {
        "sizeBytes": item.size_bytes,
        "format": item.suffix.lstrip(".") or "unknown",
        **dict(attributes or {}),
    }
    return IndexEntry(
        document_entry_id(item.digest),
        item.digest,
        item.path,
        tuple(embedding),
        validated_tags(labels),
        merged,
        item.modified_at_ms,
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
        if not entry.entry_id.startswith("doc-"):
            return "document entry identity is invalid"
        if entry.entry_id != f"doc-{entry.digest}":
            return "document entry identity must be derived from its digest"
        for label in entry.tags:
            if not _LABEL.match(label):
                return f"classification label is not a slug: {label}"
            if self._labels and label not in self._labels:
                return f"classification label is outside the declared set: {label}"
        return _attribute_error(entry.attributes)


def _attribute_error(attributes: Mapping[str, object]) -> str:
    carried = sorted(set(attributes) & _CONTENT_ATTRIBUTES)
    if carried:
        # Truncating would not help: a snippet is still the document's text,
        # and it would outlive the document it was taken from.
        return f"the index must not carry document text: {', '.join(carried)}"
    unknown = set(attributes) - _ALLOWED_ATTRIBUTES
    if unknown:
        return f"unsupported document attributes: {', '.join(sorted(unknown))}"
    return _count_error(attributes) or _language_error(attributes)


def _count_error(attributes: Mapping[str, object]) -> str:
    for name, maximum in (("pageCount", MAX_PAGES), ("wordCount", MAX_WORDS)):
        value = attributes.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{name} must be an integer"
        if not 0 <= value <= maximum:
            return f"{name} is out of range"
    return ""


def _language_error(attributes: Mapping[str, object]) -> str:
    language = attributes.get("language")
    if language is None:
        return ""
    if not isinstance(language, str) or not _LANGUAGE.match(language):
        return "language must be a BCP 47 style tag"
    return ""
