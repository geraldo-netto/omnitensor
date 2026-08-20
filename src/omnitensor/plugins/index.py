"""A durable derived index: atomic, versioned, bounded, and self-recovering.

An index over a user's own files is derived data, and that shapes every rule
here.  It can always be rebuilt from the sources, so a corrupt document is
discarded and reported rather than parsed defensively — a half-understood index
serves wrong answers, which is worse than serving none.  It is written by a
read-modify-write cycle, so the whole cycle holds the store lock and commits
through :func:`~omnitensor.atomicio.write_json_atomic`; a reader never sees a
partial document and a crash mid-update loses the update, not the index.

Two rules are less obvious.  Updates are *idempotent*: rescanning an unchanged
tree writes nothing and does not bump the revision, because an index that
churns on every scan burns a disk and destroys the meaning of a revision.  And
embeddings are bound to the model that produced them — comparing vectors from
two different models yields confident nonsense — so the artifact identity is
stored alongside them and a mismatch drops the entries instead of serving them.

``source_path`` is a field of its own rather than an attribute, because the
results layer must never emit it and a dedicated field is something a
projection has to opt into, not something redaction has to remember to strip.
"""

from __future__ import annotations

import base64
import math
import re
import struct
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from ..atomicio import JsonTooLargeError, read_json_bounded, write_json_atomic
from ..stable_error import StableError
from ..storelock import store_lock

INDEX_DOCUMENT_VERSION = 1
DEFAULT_MAX_ENTRIES = 20_000
DEFAULT_MAX_INDEX_BYTES = 64 * 1024 * 1024
MAX_ENTRIES_LIMIT = 200_000
MAX_DIMENSIONS = 4096
MAX_TAGS_PER_ENTRY = 64
MAX_TAG_LENGTH = 64
MAX_ATTRIBUTES = 32
MAX_ATTRIBUTE_LENGTH = 512
MAX_IDENTITY_LENGTH = 200
_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class IndexHealth(StrEnum):
    """What the last load found on disk, in terms a caller can act on."""

    INTACT = "intact"
    RECOVERED = "recovered"
    INCOMPATIBLE = "incompatible"


class IndexStoreError(StableError, ValueError):
    """Stable index contract failure, safe to report to a caller."""


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """The artifact whose embeddings an index holds.

    Vectors are only comparable within one model, so this travels with the
    index and a change to it invalidates every entry.
    """

    artifact_id: str
    artifact_digest: str
    dimensions: int

    def document(self) -> dict:
        return {
            "artifactId": self.artifact_id,
            "artifactDigest": self.artifact_digest,
            "dimensions": self.dimensions,
        }


@dataclass(frozen=True, slots=True)
class IndexEntry:
    """One indexed source: its identity, its vector, and its tags."""

    entry_id: str
    digest: str
    source_path: str
    embedding: tuple[float, ...]
    tags: tuple[str, ...] = ()
    attributes: Mapping[str, object] = field(default_factory=dict)
    updated_at_ms: int = 0

    def comparable(self) -> tuple:
        """Everything that makes this entry different from another.

        ``updated_at_ms`` is deliberately excluded: a rescan that finds nothing
        changed must not count as a change merely because it happened later.
        """
        return (
            self.entry_id,
            self.digest,
            self.source_path,
            self.embedding,
            self.tags,
            tuple(sorted(dict(self.attributes).items())),
        )


@dataclass(frozen=True, slots=True)
class IndexState:
    """The index as it currently stands on disk."""

    revision: int
    model: ModelIdentity
    entries: tuple[IndexEntry, ...]
    health: IndexHealth

    @property
    def stale(self) -> bool:
        """True when the entries cannot be trusted for search."""
        return self.health is not IndexHealth.INTACT

    def by_id(self) -> dict[str, IndexEntry]:
        return {entry.entry_id: entry for entry in self.entries}


@dataclass(frozen=True, slots=True)
class IndexUpdate:
    """What one committed update actually changed."""

    state: IndexState
    added: int = 0
    updated: int = 0
    removed: int = 0
    evicted: int = 0
    unchanged: int = 0

    @property
    def committed(self) -> bool:
        """False when the update was a no-op and nothing was written."""
        return bool(self.added or self.updated or self.removed or self.evicted)


def encode_embedding(values: Sequence[float], dimensions: int) -> str:
    """Pack a vector as base64 float32, exactly and compactly.

    A JSON array of floats is several times larger and rounds on every
    round-trip; the packed form is byte-exact, which matters because a ranking
    that shifts between reads is indistinguishable from a bug.
    """
    if len(values) != dimensions:
        raise IndexStoreError(
            "embedding-invalid", f"expected {dimensions} dimensions, got {len(values)}"
        )
    numbers = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise IndexStoreError("embedding-invalid", "embedding values must be numbers")
        if not math.isfinite(value):
            # A NaN makes every comparison false, so ordering stops being a
            # total order and results reshuffle between identical queries.
            raise IndexStoreError("embedding-invalid", "embedding values must be finite")
        numbers.append(float(value))
    return base64.b64encode(struct.pack(f"<{dimensions}f", *numbers)).decode("ascii")


def decode_embedding(text: object, dimensions: int) -> tuple[float, ...]:
    """Unpack a stored vector, refusing anything that is not exactly one."""
    if not isinstance(text, str):
        raise IndexStoreError("embedding-invalid", "embedding must be a string")
    try:
        raw = base64.b64decode(text, validate=True)
    except (ValueError, TypeError) as error:
        raise IndexStoreError("embedding-invalid", f"embedding is not base64: {error}") from error
    if len(raw) != dimensions * 4:
        raise IndexStoreError(
            "embedding-invalid", f"embedding is {len(raw)} bytes, expected {dimensions * 4}"
        )
    values = struct.unpack(f"<{dimensions}f", raw)
    if any(not math.isfinite(value) for value in values):
        raise IndexStoreError("embedding-invalid", "embedding values must be finite")
    return values


def validated_tags(tags: object) -> tuple[str, ...]:
    """Bound and normalise a tag set, refusing anything unbounded."""
    if isinstance(tags, (str, bytes)) or not isinstance(tags, Sequence):
        raise IndexStoreError("tags-invalid", "tags must be a sequence of strings")
    if len(tags) > MAX_TAGS_PER_ENTRY:
        raise IndexStoreError("tags-invalid", f"at most {MAX_TAGS_PER_ENTRY} tags are allowed")
    cleaned = []
    for tag in tags:
        if not isinstance(tag, str) or not 1 <= len(tag) <= MAX_TAG_LENGTH:
            raise IndexStoreError("tags-invalid", "each tag must be a bounded non-empty string")
        cleaned.append(tag)
    return tuple(sorted(set(cleaned)))


def validated_attributes(attributes: object) -> dict[str, object]:
    """Bound a free-form attribute map to scalars a document can hold."""
    if not isinstance(attributes, Mapping):
        raise IndexStoreError("attributes-invalid", "attributes must be a mapping")
    if len(attributes) > MAX_ATTRIBUTES:
        raise IndexStoreError(
            "attributes-invalid", f"at most {MAX_ATTRIBUTES} attributes are allowed"
        )
    cleaned: dict[str, object] = {}
    for key, value in attributes.items():
        if not isinstance(key, str) or not 1 <= len(key) <= MAX_TAG_LENGTH:
            raise IndexStoreError("attributes-invalid", "attribute keys must be bounded strings")
        error = _attribute_value_error(key, value)
        if error:
            raise IndexStoreError("attributes-invalid", error)
        cleaned[key] = value
    return cleaned


def prefixed_entry_id(digest: object, prefix: str, family: str) -> str:
    """Return a family entry ID while preserving the established digest contract."""
    if not isinstance(digest, str) or len(digest) != 64:
        raise IndexStoreError("entry-invalid", f"{family} entry identity must be a sha256 digest")
    return f"{prefix}{digest}"


def ingested_entry(
    item: object,
    embedding: Sequence[float],
    tags: Sequence[str],
    attributes: Mapping[str, object] | None,
    *,
    prefix: str,
    family: str,
) -> IndexEntry:
    """Build a family-prefixed entry from one accepted file."""
    from .ingestion import IngestedFile

    if not isinstance(item, IngestedFile):
        raise IndexStoreError("entry-invalid", "item must be an IngestedFile")
    merged = {
        "sizeBytes": item.size_bytes,
        "format": item.suffix.lstrip(".") or "unknown",
        **dict(attributes or {}),
    }
    return IndexEntry(
        prefixed_entry_id(item.digest, prefix, family),
        item.digest,
        item.path,
        tuple(embedding),
        validated_tags(tags),
        merged,
        item.modified_at_ms,
    )


def prefixed_entry_error(
    entry: IndexEntry,
    *,
    prefix: str,
    family: str,
    tag_name: str,
    vocabulary: Collection[str],
    vocabulary_name: str,
) -> str:
    """Return the first stable family identity or tag-policy error."""
    if not entry.entry_id.startswith(prefix):
        return f"{family} entry identity is invalid"
    if entry.entry_id != f"{prefix}{entry.digest}":
        return f"{family} entry identity must be derived from its digest"
    for tag in entry.tags:
        if _SLUG.match(tag) is None:
            return f"{tag_name} is not a slug: {tag}"
        if vocabulary and tag not in vocabulary:
            return f"{tag_name} is outside the {vocabulary_name}: {tag}"
    return ""


def unsupported_attributes_error(
    attributes: Mapping[str, object], allowed: Collection[str], family: str
) -> str:
    """Return the stable family error for the first unsupported attribute set."""
    unknown = set(attributes) - set(allowed)
    if unknown:
        return f"unsupported {family} attributes: {', '.join(sorted(unknown))}"
    return ""


def integer_attribute_error(
    attributes: Mapping[str, object], name: str, minimum: int, maximum: int
) -> str:
    """Validate one optional integer attribute against inclusive bounds."""
    value = attributes.get(name)
    if value is None:
        return ""
    if isinstance(value, bool) or not isinstance(value, int):
        return f"{name} must be an integer"
    if not minimum <= value <= maximum:
        return f"{name} is out of range"
    return ""


def _attribute_value_error(key: str, value: object) -> str:
    if isinstance(value, str):
        if len(value) > MAX_ATTRIBUTE_LENGTH:
            return f"attribute {key} exceeds {MAX_ATTRIBUTE_LENGTH} chars"
        return ""
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return "" if math.isfinite(value) else f"attribute {key} is not finite"
    return f"attribute {key} has an unsupported type"


def _document_revision(document: object) -> int | None:
    """The stored revision, or ``None`` when the document is not one of ours."""
    if not isinstance(document, dict) or document.get("version") != INDEX_DOCUMENT_VERSION:
        return None
    revision = document.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        return None
    return revision


def _bounded_identity(value: object, name: str, code: str = "entry-invalid") -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= MAX_IDENTITY_LENGTH:
        raise IndexStoreError(code, f"{name} must be a bounded non-empty string")
    return value


class BoundedIndexStore:
    """A revisioned on-disk index with bounded growth and idempotent updates.

    Subclasses narrow what an entry may contain by overriding
    :meth:`entry_error`; everything about durability, bounding, recovery, and
    compare-and-swap lives here so no profile can implement it slightly wrong.
    """

    label = "index"

    def __init__(
        self,
        root: Path | str,
        name: str,
        model: ModelIdentity,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_index_bytes: int = DEFAULT_MAX_INDEX_BYTES,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if not isinstance(model, ModelIdentity):
            raise IndexStoreError("model-invalid", "model must be a ModelIdentity")
        _bounded_identity(model.artifact_id, "artifact id", "model-invalid")
        _bounded_identity(model.artifact_digest, "artifact digest", "model-invalid")
        if (
            isinstance(model.dimensions, bool)
            or not isinstance(model.dimensions, int)
            or not 1 <= model.dimensions <= MAX_DIMENSIONS
        ):
            raise IndexStoreError(
                "model-invalid", f"dimensions must be an integer in [1, {MAX_DIMENSIONS}]"
            )
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or not 1 <= max_entries <= MAX_ENTRIES_LIMIT
        ):
            raise IndexStoreError(
                "bounds-invalid", f"max_entries must be an integer in [1, {MAX_ENTRIES_LIMIT}]"
            )
        if (
            isinstance(max_index_bytes, bool)
            or not isinstance(max_index_bytes, int)
            or max_index_bytes < 1
        ):
            raise IndexStoreError("bounds-invalid", "max_index_bytes must be a positive integer")
        if not isinstance(name, str) or not name or "/" in name:
            raise IndexStoreError("name-invalid", "index name must be a bare file name")
        self._root = Path(root)
        self._name = name
        self._model = model
        self._max_entries = max_entries
        self._max_index_bytes = max_index_bytes
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))

    @property
    def model(self) -> ModelIdentity:
        return self._model

    @property
    def path(self) -> Path:
        return self._root / self._name

    @property
    def max_entries(self) -> int:
        return self._max_entries

    def entry_error(self, entry: IndexEntry) -> str:
        """Profile-specific rejection reason, or ``""`` when acceptable."""
        return ""

    def load(self) -> IndexState:
        """Read the index, recovering rather than raising on a damaged file."""
        with store_lock(self._root, f"{self._name}.lock"):
            return self._load_locked()

    def upsert(
        self, entries: Sequence[IndexEntry], *, expected_revision: int | None = None
    ) -> IndexUpdate:
        """Merge entries in one atomic revision; unchanged input writes nothing."""
        prepared = [self._prepare(entry) for entry in entries]
        with store_lock(self._root, f"{self._name}.lock"):
            current = self._load_locked()
            self._check_revision(current, expected_revision)
            merged = current.by_id()
            added = updated = unchanged = 0
            for entry in prepared:
                existing = merged.get(entry.entry_id)
                if existing is None:
                    merged[entry.entry_id] = entry
                    added += 1
                elif existing.comparable() == entry.comparable():
                    # An unchanged source must not cost a revision, or every
                    # rescan would look like an edit to anyone watching.
                    unchanged += 1
                else:
                    merged[entry.entry_id] = entry
                    updated += 1
            kept, evicted = self._bound(merged)
            if not (added or updated or evicted):
                return IndexUpdate(current, unchanged=unchanged)
            state = self._commit(current.revision, kept)
            return IndexUpdate(state, added, updated, 0, evicted, unchanged)

    def delete(
        self, entry_ids: Sequence[str], *, expected_revision: int | None = None
    ) -> IndexUpdate:
        """Remove entries in one atomic revision; removing nothing writes nothing."""
        wanted = {entry_id for entry_id in entry_ids if isinstance(entry_id, str)}
        with store_lock(self._root, f"{self._name}.lock"):
            current = self._load_locked()
            self._check_revision(current, expected_revision)
            remaining = [entry for entry in current.entries if entry.entry_id not in wanted]
            removed = len(current.entries) - len(remaining)
            if not removed:
                return IndexUpdate(current)
            state = self._commit(current.revision, tuple(remaining))
            return IndexUpdate(state, removed=removed)

    def reindex(self, entries: Sequence[IndexEntry]) -> IndexUpdate:
        """Replace the whole index, which is how a model change is resolved."""
        prepared = [self._prepare(entry) for entry in entries]
        with store_lock(self._root, f"{self._name}.lock"):
            current = self._load_locked()
            kept, evicted = self._bound({entry.entry_id: entry for entry in prepared})
            state = self._commit(current.revision, kept)
            return IndexUpdate(
                state,
                added=len(kept),
                removed=len(current.entries),
                evicted=evicted,
            )

    def _prepare(self, entry: object) -> IndexEntry:
        if not isinstance(entry, IndexEntry):
            raise IndexStoreError("entry-invalid", "entry must be an IndexEntry")
        _bounded_identity(entry.entry_id, "entry id")
        _bounded_identity(entry.digest, "entry digest")
        _bounded_identity(entry.source_path, "entry source path")
        if (
            isinstance(entry.updated_at_ms, bool)
            or not isinstance(entry.updated_at_ms, int)
            or entry.updated_at_ms < 0
        ):
            raise IndexStoreError("entry-invalid", "entry timestamp is invalid")
        # Round-tripping through the wire form now means an entry that cannot
        # be persisted is refused at the call site rather than at commit time.
        embedding = decode_embedding(
            encode_embedding(entry.embedding, self._model.dimensions), self._model.dimensions
        )
        prepared = IndexEntry(
            entry.entry_id,
            entry.digest,
            entry.source_path,
            embedding,
            validated_tags(entry.tags),
            validated_attributes(entry.attributes),
            entry.updated_at_ms or self._clock_ms(),
        )
        error = self.entry_error(prepared)
        if error:
            raise IndexStoreError("entry-invalid", error)
        return prepared

    def _check_revision(self, current: IndexState, expected: int | None) -> None:
        if expected is None:
            return
        if isinstance(expected, bool) or not isinstance(expected, int):
            raise IndexStoreError("revision-invalid", "expected revision must be an integer")
        if expected != current.revision:
            raise IndexStoreError(
                "revision-conflict",
                f"index is at revision {current.revision}, not {expected}",
            )

    def _bound(self, merged: Mapping[str, IndexEntry]) -> tuple[tuple[IndexEntry, ...], int]:
        """Keep the most recently updated entries; the index is rebuildable."""
        ordered = sorted(merged.values(), key=lambda entry: (-entry.updated_at_ms, entry.entry_id))
        if len(ordered) <= self._max_entries:
            return tuple(sorted(ordered, key=lambda entry: entry.entry_id)), 0
        kept = ordered[: self._max_entries]
        return (
            tuple(sorted(kept, key=lambda entry: entry.entry_id)),
            len(ordered) - self._max_entries,
        )

    def _commit(self, revision: int, entries: tuple[IndexEntry, ...]) -> IndexState:
        document = {
            "version": INDEX_DOCUMENT_VERSION,
            "revision": revision + 1,
            "model": self._model.document(),
            "entries": [self._entry_document(entry) for entry in entries],
        }
        write_json_atomic(self.path, document, f"{self._name}.")
        return IndexState(revision + 1, self._model, entries, IndexHealth.INTACT)

    def _entry_document(self, entry: IndexEntry) -> dict:
        return {
            "id": entry.entry_id,
            "digest": entry.digest,
            "sourcePath": entry.source_path,
            "embedding": encode_embedding(entry.embedding, self._model.dimensions),
            "tags": list(entry.tags),
            "attributes": dict(entry.attributes),
            "updatedAtMs": entry.updated_at_ms,
        }

    def _load_locked(self) -> IndexState:
        try:
            document = read_json_bounded(self.path, self._max_index_bytes)
        except FileNotFoundError:
            return self._empty(IndexHealth.INTACT)
        except (JsonTooLargeError, ValueError, OSError):
            # Derived data: discarding a damaged index costs a rescan, while
            # salvaging half of one costs wrong answers indefinitely.
            return self._empty(IndexHealth.RECOVERED)
        return self._parse(document)

    def _parse(self, document: object) -> IndexState:
        revision = _document_revision(document)
        if revision is None:
            return self._empty(IndexHealth.RECOVERED)
        stored = document.get("model")
        if not isinstance(stored, dict):
            return self._empty(IndexHealth.RECOVERED)
        if stored != self._model.document():
            # The vectors were produced by a different model.  Comparing them
            # with new ones is not degraded search, it is confident nonsense.
            return IndexState(revision, self._model, (), IndexHealth.INCOMPATIBLE)
        raw_entries = document.get("entries")
        if not isinstance(raw_entries, list):
            return self._empty(IndexHealth.RECOVERED)
        entries = []
        for item in raw_entries:
            entry = self._parse_entry(item)
            if entry is None:
                return IndexState(revision, self._model, (), IndexHealth.RECOVERED)
            entries.append(entry)
        return IndexState(revision, self._model, tuple(entries), IndexHealth.INTACT)

    def _parse_entry(self, item: object) -> IndexEntry | None:
        if not isinstance(item, dict):
            return None
        try:
            entry = IndexEntry(
                _bounded_identity(item.get("id"), "entry id"),
                _bounded_identity(item.get("digest"), "entry digest"),
                _bounded_identity(item.get("sourcePath"), "entry source path"),
                decode_embedding(item.get("embedding"), self._model.dimensions),
                validated_tags(item.get("tags", [])),
                validated_attributes(item.get("attributes", {})),
                item.get("updatedAtMs"),
            )
        except IndexStoreError:
            return None
        if (
            isinstance(entry.updated_at_ms, bool)
            or not isinstance(entry.updated_at_ms, int)
            or entry.updated_at_ms < 0
        ):
            return None
        return entry if not self.entry_error(entry) else None

    def _empty(self, health: IndexHealth) -> IndexState:
        return IndexState(0, self._model, (), health)
