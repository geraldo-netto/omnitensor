from __future__ import annotations

import pytest

from omnitensor.plugins.index import BoundedIndexStore, IndexHealth, IndexStoreError, ModelIdentity
from omnitensor.plugins.ingestion import IngestedFile
from omnitensor.plugins.visual_index import (
    MAX_PIXELS_PER_SIDE,
    VISUAL_INDEX_NAME,
    VisualLibraryIndex,
    visual_entry,
    visual_entry_id,
)

MODEL = ModelIdentity("mobileclip", "a" * 64, 4)
DIGEST = "d" * 64
VECTOR = (0.1, 0.2, 0.3, 0.4)


def ingested(path="/home/u/pictures/beach.png", digest=DIGEST, **changes):
    values = {
        "path": path,
        "size_bytes": 2048,
        "modified_at_ms": 1_700_000_000_000,
        "suffix": ".png",
        "digest": digest,
    }
    values.update(changes)
    return IngestedFile(**values)


def index(tmp_path, **changes):
    return VisualLibraryIndex(tmp_path, MODEL, **changes)


def test_an_ingested_image_becomes_an_indexable_entry():
    item = visual_entry(ingested(), VECTOR, ("beach",), {"width": 800, "height": 600})

    assert item.entry_id == f"vis-{DIGEST}"
    assert item.attributes["format"] == "png"
    assert item.attributes["sizeBytes"] == 2048
    assert item.updated_at_ms == 1_700_000_000_000


def test_visual_entry_allows_the_caller_to_override_derived_size_and_format():
    item = visual_entry(ingested(), VECTOR, attributes={"sizeBytes": 7, "format": "reviewed"})
    assert item.attributes["sizeBytes"] == 7
    assert item.attributes["format"] == "reviewed"


def test_identity_follows_content_so_a_move_is_an_update(tmp_path):
    """A re-organised library must not re-embed every file from scratch."""
    subject = index(tmp_path)
    subject.upsert([visual_entry(ingested("/home/u/pictures/a.png"), VECTOR)])

    update = subject.upsert([visual_entry(ingested("/home/u/albums/2024/a.png"), VECTOR)])

    assert update.added == 0
    assert update.updated == 1
    assert len(update.state.entries) == 1
    assert update.state.entries[0].source_path == "/home/u/albums/2024/a.png"


def test_two_different_images_are_two_entries(tmp_path):
    subject = index(tmp_path)
    subject.upsert(
        [
            visual_entry(ingested(digest="a" * 64), VECTOR),
            visual_entry(ingested(digest="b" * 64), VECTOR),
        ]
    )
    assert len(subject.load().entries) == 2


def test_an_identity_not_derived_from_content_is_refused(tmp_path):
    item = visual_entry(ingested(), VECTOR)
    forged = type(item)(
        "vis-" + "0" * 64, DIGEST, item.source_path, VECTOR, (), {}, item.updated_at_ms
    )
    with pytest.raises(IndexStoreError, match="derived from its digest"):
        index(tmp_path).upsert([forged])


def test_an_identity_without_the_visual_prefix_is_refused(tmp_path):
    item = visual_entry(ingested(), VECTOR)
    forged = type(item)("img-1", DIGEST, item.source_path, VECTOR, (), {}, 1)
    with pytest.raises(IndexStoreError, match="identity is invalid"):
        index(tmp_path).upsert([forged])


def test_a_digest_that_is_not_a_digest_is_refused():
    with pytest.raises(IndexStoreError, match="sha256"):
        visual_entry_id("short")


def test_visual_identity_keeps_the_legacy_length_only_digest_contract():
    digest = "not-hex".ljust(64, "!")
    assert visual_entry_id(digest) == f"vis-{digest}"


def test_a_tag_outside_the_declared_vocabulary_is_refused(tmp_path):
    """A classifier that can emit any string can emit the image's contents."""
    subject = index(tmp_path, vocabulary=("beach", "sunset"))

    subject.upsert([visual_entry(ingested(), VECTOR, ("beach",))])
    with pytest.raises(IndexStoreError, match="outside the declared vocabulary"):
        subject.upsert([visual_entry(ingested(), VECTOR, ("passport-number",))])


def test_any_slug_is_accepted_when_no_vocabulary_is_declared(tmp_path):
    subject = index(tmp_path)
    assert subject.upsert([visual_entry(ingested(), VECTOR, ("anything-here",))]).added == 1


@pytest.mark.parametrize("tag", ["Beach", "two words", "trailing-", "sym!bol"])
def test_a_tag_that_is_not_a_slug_is_refused(tmp_path, tag):
    with pytest.raises(IndexStoreError, match="not a slug"):
        index(tmp_path).upsert([visual_entry(ingested(), VECTOR, (tag,))])


@pytest.mark.parametrize(
    "attributes",
    [
        {"exifGps": "48.8,2.3"},
        {"width": 0},
        {"height": 100_000},
        {"width": True},
        {"width": 1.5},
    ],
)
def test_an_unsupported_or_impossible_attribute_is_refused(tmp_path, attributes):
    with pytest.raises(IndexStoreError, match="entry-invalid"):
        index(tmp_path).upsert([visual_entry(ingested(), VECTOR, (), attributes)])


@pytest.mark.parametrize(
    ("attributes", "detail"),
    [
        ({"unknown": 1, "width": 0}, "unsupported visual attributes: unknown"),
        ({"width": 0, "height": 0}, "width is out of range"),
    ],
)
def test_visual_attribute_errors_keep_their_established_order(tmp_path, attributes, detail):
    with pytest.raises(IndexStoreError) as refusal:
        index(tmp_path).upsert([visual_entry(ingested(), VECTOR, (), attributes)])
    assert refusal.value.detail == detail


@pytest.mark.parametrize(
    "attributes",
    [
        {"width": 1},
        {"width": MAX_PIXELS_PER_SIDE},
        {"height": 1},
        {"height": MAX_PIXELS_PER_SIDE},
    ],
)
def test_visual_dimensions_accept_their_inclusive_boundaries(tmp_path, attributes):
    assert index(tmp_path).upsert([visual_entry(ingested(), VECTOR, (), attributes)]).added == 1


def test_a_non_ingested_file_is_refused():
    with pytest.raises(IndexStoreError, match="IngestedFile"):
        visual_entry({"path": "/home/u/a.png"}, VECTOR)


def test_a_file_without_a_suffix_records_an_unknown_format():
    item = visual_entry(ingested("/home/u/pictures/raw", suffix=""), VECTOR)
    assert item.attributes["format"] == "unknown"


def test_a_deleted_image_is_removed_from_the_index(tmp_path):
    subject = index(tmp_path)
    subject.upsert([visual_entry(ingested(), VECTOR)])

    update = subject.delete([visual_entry_id(DIGEST)])

    assert update.removed == 1
    assert subject.load().entries == ()


def test_the_index_survives_a_reload(tmp_path):
    index(tmp_path).upsert([visual_entry(ingested(), VECTOR, ("beach",), {"width": 800})])

    state = index(tmp_path).load()

    assert state.health is IndexHealth.INTACT
    assert state.entries[0].tags == ("beach",)
    assert state.entries[0].attributes["width"] == 800


def test_a_stored_entry_that_violates_visual_policy_recovers(tmp_path):
    BoundedIndexStore(tmp_path, VISUAL_INDEX_NAME, MODEL).upsert(
        [visual_entry(ingested(), VECTOR, (), {"width": 0})]
    )
    state = index(tmp_path).load()
    assert state.health is IndexHealth.RECOVERED
    assert state.entries == ()


def test_a_new_model_invalidates_the_library(tmp_path):
    index(tmp_path).upsert([visual_entry(ingested(), VECTOR)])

    other = VisualLibraryIndex(tmp_path, ModelIdentity("mobileclip", "b" * 64, 4))

    assert other.load().health is IndexHealth.INCOMPATIBLE
    assert other.load().entries == ()


def test_the_declared_vocabulary_is_reported(tmp_path):
    assert index(tmp_path, vocabulary=("sunset", "beach")).vocabulary == {"beach", "sunset"}
    assert index(tmp_path).vocabulary == frozenset()


def test_the_vocabulary_itself_is_validated(tmp_path):
    with pytest.raises(IndexStoreError, match="tags-invalid"):
        index(tmp_path, vocabulary="beach")
