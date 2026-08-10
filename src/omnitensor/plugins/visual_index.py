"""The `visual-library` index: what was seen, keyed by content.

Identity is the content digest rather than the path, so moving a photo between
folders updates one entry instead of deleting one and adding another — which
also means a re-organised library does not silently re-embed from scratch.

The tag vocabulary is closed.  Tags come out of a classifier, and a classifier
that can emit arbitrary strings can emit the contents of whatever it was fed;
constraining them to slugs from a declared vocabulary keeps a label from
becoming a channel for image content.
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

VISUAL_PLUGIN_ID = "visual-library"
VISUAL_INDEX_NAME = "visual-library.index.json"
VISUAL_INDEX_PERMISSION = "read:visual-library-index"
MAX_PIXELS_PER_SIDE = 65_536
_TAG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_ALLOWED_ATTRIBUTES = frozenset({"width", "height", "format", "capturedAtMs", "sizeBytes"})


def visual_entry_id(digest: str) -> str:
    """Content identity, so a move is an update rather than a churn event."""
    if not isinstance(digest, str) or len(digest) != 64:
        raise IndexStoreError("entry-invalid", "visual entry identity must be a sha256 digest")
    return f"vis-{digest}"


def visual_entry(
    item: IngestedFile,
    embedding: Sequence[float],
    tags: Sequence[str] = (),
    attributes: Mapping[str, object] | None = None,
) -> IndexEntry:
    """Build an index entry from an ingested file and its computed vector."""
    if not isinstance(item, IngestedFile):
        raise IndexStoreError("entry-invalid", "item must be an IngestedFile")
    merged = {
        "sizeBytes": item.size_bytes,
        "format": item.suffix.lstrip(".") or "unknown",
        **dict(attributes or {}),
    }
    return IndexEntry(
        visual_entry_id(item.digest),
        item.digest,
        item.path,
        tuple(embedding),
        validated_tags(tags),
        merged,
        item.modified_at_ms,
    )


class VisualLibraryIndex(BoundedIndexStore):
    """A durable, bounded index of visual embeddings and classifier tags."""

    label = "visual library index"

    def __init__(self, root, model: ModelIdentity, *, vocabulary: Sequence[str] = (), **bounds):
        super().__init__(root, VISUAL_INDEX_NAME, model, **bounds)
        self._vocabulary = frozenset(validated_tags(vocabulary)) if vocabulary else frozenset()

    @property
    def vocabulary(self) -> frozenset[str]:
        return self._vocabulary

    def entry_error(self, entry: IndexEntry) -> str:
        if not entry.entry_id.startswith("vis-"):
            return "visual entry identity is invalid"
        if entry.entry_id != f"vis-{entry.digest}":
            # Identity that is not derived from content lets two different
            # images share an entry, or one image occupy two.
            return "visual entry identity must be derived from its digest"
        for tag in entry.tags:
            if not _TAG.match(tag):
                return f"tag is not a slug: {tag}"
            if self._vocabulary and tag not in self._vocabulary:
                return f"tag is outside the declared vocabulary: {tag}"
        return _attribute_error(entry.attributes)


def _attribute_error(attributes: Mapping[str, object]) -> str:
    unknown = set(attributes) - _ALLOWED_ATTRIBUTES
    if unknown:
        return f"unsupported visual attributes: {', '.join(sorted(unknown))}"
    for name in ("width", "height"):
        value = attributes.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{name} must be an integer"
        if not 0 < value <= MAX_PIXELS_PER_SIDE:
            return f"{name} is out of range"
    return ""
