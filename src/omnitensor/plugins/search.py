"""Authorized, paginated, redacted reads over a derived index.

Three rules shape this layer.

*Ranking must be total.*  Sorting by score alone leaves ties in whatever order
the dictionary happened to produce, so page 2 can repeat or skip an entry that
page 1 already showed.  Every ordering here breaks ties on the entry id.

*A page must belong to the revision that produced it.*  A cursor carries the
revision and a fingerprint of the query it came from; presenting it against a
changed index is refused rather than silently answered from a different
result set, because a caller paging through search results cannot tell the
difference between "the next page" and "a different page".

*Empty is not the same as unavailable.*  When the index is stale, recovering,
or being rebuilt, the page says so explicitly.  Returning an empty result set
would tell the user their photos contain nothing matching, when the truth is
that nothing has been indexed yet.

Source paths never leave this layer.  Provenance is the opted-in root the
entry came from — the user named that root themselves, so it tells them where
a result lives without disclosing the directory structure inside it.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .index import BoundedIndexStore, IndexEntry, IndexHealth, IndexState, IndexStoreError

DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 200
_CURSOR_VERSION = 1


class ResultState(StrEnum):
    """Why a page holds what it holds — never inferred from emptiness."""

    READY = "ready"
    INDEXING = "indexing"
    STALE = "stale"
    RECOVERING = "recovering"


class SearchError(ValueError):
    """Stable query rejection, safe to return to a caller."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a result came from, at the granularity the user chose themselves."""

    root: str
    updated_at_ms: int

    def document(self) -> dict:
        return {"root": self.root, "updatedAtMs": self.updated_at_ms}


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One ranked entry, projected without its source path."""

    entry_id: str
    score: float
    tags: tuple[str, ...]
    attributes: Mapping[str, object]
    provenance: Provenance

    def document(self) -> dict:
        return {
            "id": self.entry_id,
            "score": self.score,
            "tags": list(self.tags),
            "attributes": dict(self.attributes),
            "provenance": self.provenance.document(),
        }


@dataclass(frozen=True, slots=True)
class ResultPage:
    """One page of results, with the state that explains it."""

    results: tuple[SearchResult, ...]
    state: ResultState
    revision: int
    total: int
    next_cursor: str | None = None

    @property
    def usable(self) -> bool:
        return self.state is ResultState.READY

    def document(self) -> dict:
        return {
            "results": [result.document() for result in self.results],
            "state": str(self.state),
            "revision": self.revision,
            "total": self.total,
            "nextCursor": self.next_cursor,
        }


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, or 0.0 when either side has no direction."""
    # strict: comparing vectors of different lengths silently truncates one,
    # which would rank an entry from another model as merely less similar.
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def query_fingerprint(payload: object) -> str:
    """A stable short digest of a query, so a cursor can be bound to it."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def encode_cursor(revision: int, offset: int, fingerprint: str) -> str:
    document = {"v": _CURSOR_VERSION, "r": revision, "o": offset, "q": fingerprint}
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(encoded).decode("ascii")


def decode_cursor(cursor: object, revision: int, fingerprint: str) -> int:
    """The offset a cursor points at, refusing one that belongs elsewhere."""
    if not isinstance(cursor, str) or not cursor:
        raise SearchError("cursor-invalid", "cursor must be a non-empty string")
    try:
        document = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
    except (ValueError, binascii.Error, UnicodeEncodeError) as error:
        raise SearchError("cursor-invalid", f"cursor is not readable: {error}") from error
    if not isinstance(document, dict) or document.get("v") != _CURSOR_VERSION:
        raise SearchError("cursor-invalid", "cursor version is not recognised")
    offset = document.get("o")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise SearchError("cursor-invalid", "cursor offset is invalid")
    if document.get("q") != fingerprint:
        raise SearchError("cursor-mismatch", "cursor belongs to a different query")
    if document.get("r") != revision:
        # Paging on across a revision would mix two result sets into one list
        # and the caller could not tell.  Restarting is the honest answer.
        raise SearchError("cursor-stale", "the index changed; restart the query")
    return offset


class IndexQueryService:
    """Authorized search, browse, and tagging over one :class:`BoundedIndexStore`."""

    read_permission = "read:index"
    write_permission = "write:index"
    label = "index"

    def __init__(
        self,
        store: BoundedIndexStore,
        permissions,
        *,
        roots: Sequence[Path | str] = (),
        page_size: int = DEFAULT_PAGE_SIZE,
        indexing: bool = False,
    ) -> None:
        if not isinstance(store, BoundedIndexStore):
            raise SearchError("store-invalid", "store must be a BoundedIndexStore")
        if (
            isinstance(page_size, bool)
            or not isinstance(page_size, int)
            or not 1 <= page_size <= MAX_PAGE_SIZE
        ):
            raise SearchError("bounds-invalid", f"page_size must be in [1, {MAX_PAGE_SIZE}]")
        self._store = store
        self._permissions = permissions
        self._roots = tuple(Path(root).resolve() for root in roots)
        self._page_size = page_size
        self._indexing = bool(indexing)

    @property
    def indexing(self) -> bool:
        return self._indexing

    def indexing_started(self) -> None:
        self._indexing = True

    def indexing_finished(self) -> None:
        self._indexing = False

    def search(
        self,
        vector: Sequence[float],
        *,
        tags: Sequence[str] = (),
        limit: int | None = None,
        cursor: str | None = None,
    ) -> ResultPage:
        """Rank entries by similarity to ``vector``, filtered by ``tags``."""
        self._permissions.require(self.read_permission)
        self._require_direction(vector)
        wanted = self._wanted_tags(tags)
        state = self._store.load()
        fingerprint = query_fingerprint(
            {"kind": "search", "vector": list(vector), "tags": sorted(wanted)}
        )
        scored = [
            (cosine_similarity(vector, entry.embedding), entry)
            for entry in state.entries
            if wanted <= set(entry.tags)
        ]
        ranked = sorted(scored, key=lambda pair: (-pair[0], pair[1].entry_id))
        return self._page(state, ranked, fingerprint, limit, cursor)

    def browse(
        self,
        *,
        tags: Sequence[str] = (),
        limit: int | None = None,
        cursor: str | None = None,
    ) -> ResultPage:
        """List entries newest first, without ranking them against a query."""
        self._permissions.require(self.read_permission)
        wanted = self._wanted_tags(tags)
        state = self._store.load()
        fingerprint = query_fingerprint({"kind": "browse", "tags": sorted(wanted)})
        matching = [entry for entry in state.entries if wanted <= set(entry.tags)]
        ranked = [
            (0.0, entry)
            for entry in sorted(matching, key=lambda item: (-item.updated_at_ms, item.entry_id))
        ]
        return self._page(state, ranked, fingerprint, limit, cursor)

    def similar_to(
        self,
        entry_id: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> ResultPage:
        """Rank entries against one already indexed, excluding itself."""
        self._permissions.require(self.read_permission)
        state = self._store.load()
        anchor = state.by_id().get(entry_id)
        if anchor is None:
            # An unknown id and an id belonging to a stale index are the same
            # answer, so neither confirms what the index contains.
            raise SearchError("entry-unknown", "no such entry")
        fingerprint = query_fingerprint({"kind": "similar", "entry": entry_id})
        ranked = sorted(
            (
                (cosine_similarity(anchor.embedding, entry.embedding), entry)
                for entry in state.entries
                if entry.entry_id != entry_id
            ),
            key=lambda pair: (-pair[0], pair[1].entry_id),
        )
        return self._page(state, ranked, fingerprint, limit, cursor)

    def assign_tags(self, entry_id: str, tags: Sequence[str]) -> ResultPage:
        """Replace an entry's tags, which is a write and needs the write grant."""
        self._permissions.require(self.write_permission)
        if not isinstance(entry_id, str) or not entry_id:
            raise SearchError("entry-unknown", "entry id must be a non-empty string")
        state = self._store.load()
        existing = state.by_id().get(entry_id)
        if existing is None:
            raise SearchError("entry-unknown", "no such entry")
        updated = IndexEntry(
            existing.entry_id,
            existing.digest,
            existing.source_path,
            existing.embedding,
            tuple(tags),
            existing.attributes,
            existing.updated_at_ms,
        )
        try:
            self._store.upsert([updated], expected_revision=state.revision)
        except IndexStoreError as error:
            raise SearchError(error.code, error.detail) from error
        return self.browse(limit=1)

    def _require_direction(self, vector: object) -> None:
        if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
            raise SearchError("query-invalid", "query vector must be a sequence of numbers")
        if len(vector) != self._store.model.dimensions:
            raise SearchError(
                "query-invalid",
                f"query vector must have {self._store.model.dimensions} dimensions",
            )
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SearchError("query-invalid", "query vector must contain numbers")
            if not math.isfinite(value):
                raise SearchError("query-invalid", "query vector must be finite")
        if not any(value for value in vector):
            # A zero vector is equally similar to everything, so ranking it
            # returns an arbitrary order dressed up as relevance.
            raise SearchError("query-invalid", "query vector has no direction")

    def _wanted_tags(self, tags: object) -> set[str]:
        if isinstance(tags, (str, bytes)) or not isinstance(tags, Sequence):
            raise SearchError("filter-invalid", "tags must be a sequence of strings")
        if len(tags) > 32:
            raise SearchError("filter-invalid", "at most 32 tag filters are allowed")
        for tag in tags:
            if not isinstance(tag, str) or not tag:
                raise SearchError("filter-invalid", "each tag filter must be a non-empty string")
        return set(tags)

    def _page(
        self,
        state: IndexState,
        ranked: Sequence[tuple[float, IndexEntry]],
        fingerprint: str,
        limit: int | None,
        cursor: str | None,
    ) -> ResultPage:
        size = self._page_size if limit is None else limit
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= MAX_PAGE_SIZE:
            raise SearchError("bounds-invalid", f"limit must be in [1, {MAX_PAGE_SIZE}]")
        result_state = self._state_of(state)
        if result_state is not ResultState.READY:
            # Say why there is nothing rather than letting emptiness imply
            # that nothing matched.
            return ResultPage((), result_state, state.revision, 0)
        offset = 0 if cursor is None else decode_cursor(cursor, state.revision, fingerprint)
        window = ranked[offset : offset + size]
        results = tuple(self._project(score, entry) for score, entry in window)
        next_offset = offset + len(window)
        next_cursor = (
            encode_cursor(state.revision, next_offset, fingerprint)
            if next_offset < len(ranked)
            else None
        )
        return ResultPage(results, result_state, state.revision, len(ranked), next_cursor)

    def _state_of(self, state: IndexState) -> ResultState:
        if state.health is IndexHealth.INCOMPATIBLE:
            return ResultState.STALE
        if state.health is IndexHealth.RECOVERED:
            return ResultState.RECOVERING
        if self._indexing:
            return ResultState.INDEXING
        return ResultState.READY

    def _project(self, score: float, entry: IndexEntry) -> SearchResult:
        return SearchResult(
            entry.entry_id,
            round(float(score), 6),
            entry.tags,
            dict(entry.attributes),
            Provenance(self._root_of(entry.source_path), entry.updated_at_ms),
        )

    def _root_of(self, source_path: str) -> str:
        """Name the opted-in root, never the path inside it."""
        candidate = Path(source_path)
        for root in self._roots:
            if candidate == root or root in candidate.parents:
                return str(root)
        return "unknown"
