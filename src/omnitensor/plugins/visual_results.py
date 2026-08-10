"""`visual-library` reads: search, similarity, and tagging over the index.

Reading and tagging are separate grants.  Being allowed to find a photo is not
being allowed to relabel the library, and a user who wants search without
letting a plugin rewrite their tags should be able to say so.
"""

from __future__ import annotations

from .search import IndexQueryService
from .visual_index import VISUAL_INDEX_PERMISSION, VISUAL_PLUGIN_ID, VisualLibraryIndex

VISUAL_TAG_PERMISSION = "write:visual-library-tags"


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
