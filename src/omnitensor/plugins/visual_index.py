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
from .search import IndexQueryService

VISUAL_PLUGIN_ID = "visual-library"
VISUAL_INDEX_NAME = "visual-library.index.json"
VISUAL_INDEX_PERMISSION = "read:visual-library-index"
VISUAL_TAG_PERMISSION = "write:visual-library-tags"
MAX_PIXELS_PER_SIDE = 65_536
_ALLOWED_ATTRIBUTES = frozenset({"width", "height", "format", "capturedAtMs", "sizeBytes"})


def visual_entry_id(digest: str) -> str:
    """Content identity, so a move is an update rather than a churn event."""
    return prefixed_entry_id(digest, "vis-", "visual")


def visual_entry(
    item: IngestedFile,
    embedding: Sequence[float],
    tags: Sequence[str] = (),
    attributes: Mapping[str, object] | None = None,
) -> IndexEntry:
    """Build an index entry from an ingested file and its computed vector."""
    return ingested_entry(
        item,
        embedding,
        tags,
        attributes,
        prefix="vis-",
        family="visual",
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
        error = prefixed_entry_error(
            entry,
            prefix="vis-",
            family="visual",
            tag_name="tag",
            vocabulary=self._vocabulary,
            vocabulary_name="declared vocabulary",
        )
        return error or _attribute_error(entry.attributes)


class VisualLibraryResults(IndexQueryService):
    """Authorized, paginated, path-redacted access to the visual library."""

    read_permission = VISUAL_INDEX_PERMISSION
    write_permission = VISUAL_TAG_PERMISSION
    label = "visual library results"

    def __init__(self, store: VisualLibraryIndex, permissions, **options) -> None:
        if not isinstance(store, VisualLibraryIndex):
            raise TypeError("store must be a VisualLibraryIndex")
        super().__init__(store, permissions, **options)

    @property
    def plugin_id(self) -> str:
        return VISUAL_PLUGIN_ID


def _attribute_error(attributes: Mapping[str, object]) -> str:
    error = unsupported_attributes_error(attributes, _ALLOWED_ATTRIBUTES, "visual")
    if error:
        return error
    for name in ("width", "height"):
        error = integer_attribute_error(attributes, name, 1, MAX_PIXELS_PER_SIDE)
        if error:
            return error
    return ""
