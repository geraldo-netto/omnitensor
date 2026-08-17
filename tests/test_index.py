from __future__ import annotations

import base64
import json
import math
import struct

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins.index import (
    MAX_ATTRIBUTE_LENGTH,
    MAX_TAGS_PER_ENTRY,
    BoundedIndexStore,
    IndexEntry,
    IndexHealth,
    IndexStoreError,
    ModelIdentity,
    decode_embedding,
    encode_embedding,
    ingested_entry,
    integer_attribute_error,
    prefixed_entry_error,
    prefixed_entry_id,
    unsupported_attributes_error,
    validated_attributes,
    validated_tags,
)
from omnitensor.plugins.ingestion import IngestedFile

MODEL = ModelIdentity("clip-small", "a" * 64, 4)


def entry(entry_id="e-1", **changes):
    values = {
        "entry_id": entry_id,
        "digest": "d" * 64,
        "source_path": "/home/u/pictures/a.png",
        "embedding": (0.5, -0.25, 0.125, 1.0),
        "tags": ("beach",),
        "attributes": {"width": 100},
        "updated_at_ms": 1_700_000_000_000,
    }
    values.update(changes)
    return IndexEntry(**values)


def store(tmp_path, model=MODEL, **bounds):
    return BoundedIndexStore(tmp_path, "test.index.json", model, **bounds)


@given(digest=st.text(min_size=64, max_size=64))
def test_prefixed_entry_identity_preserves_any_64_character_digest(digest):
    assert prefixed_entry_id(digest, "family-", "sample") == f"family-{digest}"


@pytest.mark.parametrize("digest", [None, 7, "x" * 63, "x" * 65])
def test_prefixed_entry_identity_preserves_exact_error_contract(digest):
    with pytest.raises(IndexStoreError) as refusal:
        prefixed_entry_id(digest, "family-", "sample")
    assert refusal.value.code == "entry-invalid"
    assert refusal.value.detail == "sample entry identity must be a sha256 digest"


@pytest.mark.parametrize(("prefix", "family"), [("doc-", "document"), ("vis-", "visual")])
def test_ingested_entry_is_shared_across_families_and_allows_caller_overrides(prefix, family):
    item = IngestedFile("/owned/a.raw", 4096, 1234, ".raw", "not-hex".ljust(64, "!"))

    result = ingested_entry(
        item,
        (1, 2),
        ("b", "a", "a"),
        {"sizeBytes": 7, "format": "caller"},
        prefix=prefix,
        family=family,
    )

    assert result.entry_id == f"{prefix}{item.digest}"
    assert result.embedding == (1, 2)
    assert result.tags == ("a", "b")
    assert result.attributes == {"sizeBytes": 7, "format": "caller"}
    assert result.updated_at_ms == 1234


def test_ingested_entry_checks_the_source_before_constructing_attributes():
    with pytest.raises(IndexStoreError, match="item must be an IngestedFile"):
        ingested_entry(
            object(),
            (),
            (),
            object(),
            prefix="doc-",
            family="document",
        )


def test_prefixed_entry_policy_preserves_identity_then_slug_then_vocabulary_order():
    candidate = entry("other", digest="d" * 64, tags=("Not Slug",))
    options = {
        "prefix": "doc-",
        "family": "document",
        "tag_name": "classification label",
        "vocabulary": frozenset({"invoice"}),
        "vocabulary_name": "declared set",
    }
    assert prefixed_entry_error(candidate, **options) == "document entry identity is invalid"
    candidate = entry("doc-" + "a" * 64, digest="d" * 64, tags=("Not Slug",))
    assert (
        prefixed_entry_error(candidate, **options)
        == "document entry identity must be derived from its digest"
    )
    candidate = entry("doc-" + "d" * 64, digest="d" * 64, tags=("Not Slug",))
    assert (
        prefixed_entry_error(candidate, **options) == "classification label is not a slug: Not Slug"
    )
    candidate = entry("doc-" + "d" * 64, digest="d" * 64, tags=("contract",))
    assert (
        prefixed_entry_error(candidate, **options)
        == "classification label is outside the declared set: contract"
    )


@given(value=st.integers(min_value=0, max_value=100))
def test_integer_attribute_accepts_every_value_inside_inclusive_bounds(value):
    assert integer_attribute_error({"count": value}, "count", 0, 100) == ""


@pytest.mark.parametrize(
    ("value", "detail"),
    [
        (None, ""),
        (True, "count must be an integer"),
        (1.5, "count must be an integer"),
        (-1, "count is out of range"),
        (101, "count is out of range"),
    ],
)
def test_integer_attribute_preserves_optional_type_and_range_errors(value, detail):
    assert integer_attribute_error({"count": value}, "count", 0, 100) == detail


def test_unsupported_attributes_are_sorted_and_family_specific():
    assert (
        unsupported_attributes_error({"z": 1, "a": 2}, {"kept"}, "sample")
        == "unsupported sample attributes: a, z"
    )
    assert unsupported_attributes_error({"kept": 1}, {"kept"}, "sample") == ""


def test_an_absent_index_loads_empty_and_intact(tmp_path):
    state = store(tmp_path).load()
    assert state.revision == 0
    assert state.entries == ()
    assert state.health is IndexHealth.INTACT
    assert not state.stale


def test_entries_survive_a_write_and_reload(tmp_path):
    subject = store(tmp_path)
    update = subject.upsert([entry()])

    assert update.added == 1
    assert update.state.revision == 1

    reloaded = store(tmp_path).load()
    assert reloaded.revision == 1
    assert reloaded.entries[0].embedding == (0.5, -0.25, 0.125, 1.0)
    assert reloaded.entries[0].source_path == "/home/u/pictures/a.png"


def test_reindexing_unchanged_entries_writes_nothing(tmp_path):
    """A rescan that found no change must not look like an edit."""
    subject = store(tmp_path)
    subject.upsert([entry()])
    before = subject.path.read_bytes()

    update = subject.upsert([entry()])

    assert update.unchanged == 1
    assert not update.committed
    assert update.state.revision == 1
    assert subject.path.read_bytes() == before


def test_a_changed_entry_bumps_the_revision_once(tmp_path):
    subject = store(tmp_path)
    subject.upsert([entry()])
    update = subject.upsert([entry(tags=("beach", "sunset"))])

    assert update.updated == 1
    assert update.state.revision == 2
    assert update.state.entries[0].tags == ("beach", "sunset")


def test_a_later_timestamp_alone_is_not_a_change(tmp_path):
    subject = store(tmp_path)
    subject.upsert([entry()])
    update = subject.upsert([entry(updated_at_ms=1_800_000_000_000)])
    assert update.unchanged == 1
    assert update.state.revision == 1


def test_deletions_remove_entries_and_bump_the_revision(tmp_path):
    subject = store(tmp_path)
    subject.upsert([entry("e-1"), entry("e-2")])

    update = subject.delete(["e-1"])

    assert update.removed == 1
    assert [item.entry_id for item in update.state.entries] == ["e-2"]


def test_deleting_nothing_writes_nothing(tmp_path):
    subject = store(tmp_path)
    subject.upsert([entry()])
    update = subject.delete(["absent"])
    assert not update.committed
    assert update.state.revision == 1


def test_growth_is_bounded_and_the_newest_entries_are_kept(tmp_path):
    subject = store(tmp_path, max_entries=2)
    update = subject.upsert(
        [entry(f"e-{index}", updated_at_ms=1_000 + index) for index in range(5)]
    )

    assert update.evicted == 3
    assert [item.entry_id for item in update.state.entries] == ["e-3", "e-4"]


def test_a_corrupt_document_is_discarded_rather_than_salvaged(tmp_path):
    """Half an index serves wrong answers; no index only costs a rescan."""
    subject = store(tmp_path)
    subject.upsert([entry()])
    subject.path.write_text("{ this is not json")

    state = subject.load()

    assert state.health is IndexHealth.RECOVERED
    assert state.entries == ()
    assert state.stale


def test_an_index_that_outgrew_its_ceiling_is_recovered_not_read(tmp_path):
    subject = store(tmp_path, max_index_bytes=64)
    store(tmp_path).upsert([entry()])
    assert subject.load().health is IndexHealth.RECOVERED


@pytest.mark.parametrize(
    "document",
    [
        [],
        {"version": 99, "revision": 1, "model": MODEL.document(), "entries": []},
        {"version": 1, "revision": -1, "model": MODEL.document(), "entries": []},
        {"version": 1, "revision": 1, "model": "clip", "entries": []},
        {"version": 1, "revision": 1, "model": MODEL.document(), "entries": {}},
        {"version": 1, "revision": 1, "model": MODEL.document(), "entries": [{"id": "x"}]},
        {"version": 1, "revision": 1, "model": MODEL.document(), "entries": ["not-an-entry"]},
    ],
)
def test_a_malformed_document_recovers(tmp_path, document):
    subject = store(tmp_path)
    subject.path.parent.mkdir(parents=True, exist_ok=True)
    subject.path.write_text(json.dumps(document))
    assert subject.load().health is IndexHealth.RECOVERED


def test_entries_from_another_model_are_dropped_not_compared(tmp_path):
    """Vectors from two models are not comparable; ranking them is nonsense."""
    store(tmp_path, MODEL).upsert([entry()])

    other = store(tmp_path, ModelIdentity("clip-large", "b" * 64, 4))
    state = other.load()

    assert state.health is IndexHealth.INCOMPATIBLE
    assert state.entries == ()
    assert state.stale


def test_a_model_change_is_resolved_by_reindexing(tmp_path):
    store(tmp_path, MODEL).upsert([entry()])
    other = store(tmp_path, ModelIdentity("clip-large", "b" * 64, 4))

    update = other.reindex([entry()])

    assert update.state.health is IndexHealth.INTACT
    assert other.load().model.artifact_id == "clip-large"
    assert len(other.load().entries) == 1


def test_a_stale_revision_is_refused(tmp_path):
    subject = store(tmp_path)
    subject.upsert([entry()])
    with pytest.raises(IndexStoreError, match="revision-conflict"):
        subject.upsert([entry("e-2")], expected_revision=0)
    with pytest.raises(IndexStoreError, match="revision-conflict"):
        subject.delete(["e-1"], expected_revision=7)


def test_the_current_revision_is_accepted(tmp_path):
    subject = store(tmp_path)
    subject.upsert([entry()])
    assert subject.upsert([entry("e-2")], expected_revision=1).added == 1


def test_a_non_integer_revision_is_refused(tmp_path):
    with pytest.raises(IndexStoreError, match="revision-invalid"):
        store(tmp_path).upsert([entry()], expected_revision=True)


def test_an_interrupted_write_leaves_the_previous_revision(tmp_path, monkeypatch):
    subject = store(tmp_path)
    subject.upsert([entry()])

    def explode(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr("omnitensor.plugins.index.write_json_atomic", explode)
    with pytest.raises(OSError):
        subject.upsert([entry("e-2")])
    monkeypatch.undo()

    state = subject.load()
    assert state.revision == 1
    assert [item.entry_id for item in state.entries] == ["e-1"]


def test_embeddings_round_trip_exactly():
    encoded = encode_embedding((0.5, -0.25, 0.125, 1.0), 4)
    assert decode_embedding(encoded, 4) == (0.5, -0.25, 0.125, 1.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_non_finite_embedding_is_refused(value):
    """A NaN makes comparison non-transitive, so ranking stops being stable."""
    with pytest.raises(IndexStoreError, match="finite"):
        encode_embedding((value, 0.0, 0.0, 0.0), 4)


def test_a_wrong_length_embedding_is_refused():
    with pytest.raises(IndexStoreError, match="4 dimensions"):
        encode_embedding((1.0, 2.0), 4)
    with pytest.raises(IndexStoreError, match="expected 16"):
        decode_embedding(encode_embedding((1.0, 2.0), 2), 4)


@pytest.mark.parametrize("value", ["not base64!!", 7, None])
def test_an_unreadable_embedding_is_refused(value):
    with pytest.raises(IndexStoreError, match="embedding-invalid"):
        decode_embedding(value, 4)


def test_a_non_numeric_embedding_value_is_refused():
    with pytest.raises(IndexStoreError, match="numbers"):
        encode_embedding(("1.0", 0.0, 0.0, 0.0), 4)
    with pytest.raises(IndexStoreError, match="numbers"):
        encode_embedding((True, 0.0, 0.0, 0.0), 4)


def test_a_stored_non_finite_embedding_is_refused(tmp_path):
    """The bytes can hold a NaN even though the encoder refuses to write one."""
    subject = store(tmp_path)
    subject.upsert([entry()])
    document = json.loads(subject.path.read_text())
    poisoned = base64.b64encode(struct.pack("<4f", math.nan, 0.0, 0.0, 0.0)).decode("ascii")
    document["entries"][0]["embedding"] = poisoned
    subject.path.write_text(json.dumps(document))

    assert subject.load().health is IndexHealth.RECOVERED


def test_tags_are_bounded_and_deduplicated():
    assert validated_tags(["b", "a", "a"]) == ("a", "b")
    with pytest.raises(IndexStoreError, match="tags-invalid"):
        validated_tags("beach")
    with pytest.raises(IndexStoreError, match="tags-invalid"):
        validated_tags([""])
    with pytest.raises(IndexStoreError, match="tags-invalid"):
        validated_tags(["x"] * (MAX_TAGS_PER_ENTRY + 1))


def test_attributes_are_bounded_to_scalars():
    assert validated_attributes({"a": 1, "b": "x", "c": True, "d": 1.5}) == {
        "a": 1,
        "b": "x",
        "c": True,
        "d": 1.5,
    }
    for bad in (
        [],
        {"a": {"nested": 1}},
        {"a": "x" * (MAX_ATTRIBUTE_LENGTH + 1)},
        {"": 1},
        {"a": math.nan},
        {f"k{index}": 1 for index in range(40)},
    ):
        with pytest.raises(IndexStoreError, match="attributes-invalid"):
            validated_attributes(bad)


@pytest.mark.parametrize(
    "changes",
    [
        {"entry_id": ""},
        {"digest": 7},
        {"source_path": ""},
        {"updated_at_ms": -1},
        {"updated_at_ms": True},
    ],
)
def test_a_malformed_entry_is_refused(tmp_path, changes):
    with pytest.raises(IndexStoreError, match="entry-invalid"):
        store(tmp_path).upsert([entry(**changes)])


def test_a_non_entry_is_refused(tmp_path):
    with pytest.raises(IndexStoreError, match="entry-invalid"):
        store(tmp_path).upsert([{"id": "e-1"}])


def test_an_entry_without_a_timestamp_is_stamped(tmp_path):
    subject = BoundedIndexStore(tmp_path, "t.json", MODEL, clock_ms=lambda: 4242)
    update = subject.upsert([entry(updated_at_ms=0)])
    assert update.state.entries[0].updated_at_ms == 4242


def test_a_subclass_can_narrow_what_an_entry_may_contain(tmp_path):
    class OnlyPng(BoundedIndexStore):
        def entry_error(self, item):
            return "" if item.source_path.endswith(".png") else "not a png"

    subject = OnlyPng(tmp_path, "t.json", MODEL)
    subject.upsert([entry()])
    with pytest.raises(IndexStoreError, match="not a png"):
        subject.upsert([entry("e-2", source_path="/home/u/a.txt")])


def test_a_stored_entry_a_subclass_now_refuses_recovers(tmp_path):
    class OnlyPng(BoundedIndexStore):
        def entry_error(self, item):
            return "" if item.source_path.endswith(".png") else "not a png"

    BoundedIndexStore(tmp_path, "t.json", MODEL).upsert([entry(source_path="/home/u/a.txt")])
    assert OnlyPng(tmp_path, "t.json", MODEL).load().health is IndexHealth.RECOVERED


@pytest.mark.parametrize(
    "changes",
    [
        {"model": ModelIdentity("", "a" * 64, 4)},
        {"model": ModelIdentity("m", "", 4)},
        {"model": ModelIdentity("m", "a" * 64, 0)},
        {"model": ModelIdentity("m", "a" * 64, 99_999)},
        {"model": ModelIdentity("m", "a" * 64, True)},
        {"model": "clip"},
    ],
)
def test_the_model_identity_is_validated(tmp_path, changes):
    with pytest.raises(IndexStoreError, match="model-invalid"):
        BoundedIndexStore(tmp_path, "t.json", **changes)


@pytest.mark.parametrize(
    "bounds", [{"max_entries": 0}, {"max_entries": True}, {"max_index_bytes": 0}]
)
def test_the_bounds_are_validated(tmp_path, bounds):
    with pytest.raises(IndexStoreError, match="bounds-invalid"):
        store(tmp_path, **bounds)


@pytest.mark.parametrize("name", ["", "sub/dir.json", 7])
def test_the_index_name_must_be_a_bare_file_name(tmp_path, name):
    with pytest.raises(IndexStoreError, match="name-invalid"):
        BoundedIndexStore(tmp_path, name, MODEL)


def test_a_stored_entry_with_an_impossible_timestamp_recovers(tmp_path):
    subject = store(tmp_path)
    subject.upsert([entry()])
    document = json.loads(subject.path.read_text())
    document["entries"][0]["updatedAtMs"] = -1
    subject.path.write_text(json.dumps(document))

    assert subject.load().health is IndexHealth.RECOVERED


def test_the_store_reports_its_model_and_ceiling(tmp_path):
    subject = store(tmp_path, max_entries=7)
    assert subject.model is MODEL
    assert subject.max_entries == 7


def test_the_lock_file_is_not_mistaken_for_the_index(tmp_path):
    subject = store(tmp_path)
    subject.upsert([entry()])
    assert (tmp_path / "test.index.json.lock").exists()
    assert subject.load().entries
