from __future__ import annotations

import json

import pytest

from omnitensor.plugins.document_index import (
    DOCUMENT_INDEX_NAME,
    MAX_PAGES,
    MAX_WORDS,
    DocumentIntelligenceIndex,
    document_entry,
    document_entry_id,
)
from omnitensor.plugins.index import BoundedIndexStore, IndexHealth, IndexStoreError, ModelIdentity
from omnitensor.plugins.ingestion import IngestedFile

MODEL = ModelIdentity("minilm", "a" * 64, 4)
DIGEST = "d" * 64
VECTOR = (0.1, 0.2, 0.3, 0.4)


def ingested(path="/home/u/docs/report.pdf", digest=DIGEST, **changes):
    values = {
        "path": path,
        "size_bytes": 40_960,
        "modified_at_ms": 1_700_000_000_000,
        "suffix": ".pdf",
        "digest": digest,
    }
    values.update(changes)
    return IngestedFile(**values)


def index(tmp_path, **changes):
    return DocumentIntelligenceIndex(tmp_path, MODEL, **changes)


def test_an_ingested_document_becomes_an_indexable_entry():
    item = document_entry(ingested(), VECTOR, ("invoice",), {"pageCount": 3, "language": "pt-BR"})

    assert item.entry_id == f"doc-{DIGEST}"
    assert item.attributes["format"] == "pdf"
    assert item.attributes["pageCount"] == 3
    assert item.tags == ("invoice",)


def test_document_entry_allows_the_caller_to_override_derived_size_and_format():
    item = document_entry(ingested(), VECTOR, attributes={"sizeBytes": 7, "format": "reviewed"})
    assert item.attributes["sizeBytes"] == 7
    assert item.attributes["format"] == "reviewed"


@pytest.mark.parametrize(
    "attribute", ["text", "body", "snippet", "title", "content", "ocrText", "excerpt", "summary"]
)
def test_document_text_is_refused_rather_than_truncated(tmp_path, attribute):
    """A snippet is still the document's contents, and it outlives the file."""
    with pytest.raises(IndexStoreError, match="must not carry document text"):
        index(tmp_path).upsert(
            [document_entry(ingested(), VECTOR, (), {attribute: "Bank statement, acct 1234"})]
        )


def test_no_stored_document_holds_anything_but_metadata(tmp_path):
    subject = index(tmp_path)
    subject.upsert([document_entry(ingested(), VECTOR, ("invoice",), {"pageCount": 2})])

    stored = json.loads(subject.path.read_text())

    assert set(stored["entries"][0]["attributes"]) <= {
        "sizeBytes",
        "format",
        "pageCount",
        "wordCount",
        "language",
        "createdAtMs",
    }


def test_an_unsupported_attribute_is_refused(tmp_path):
    with pytest.raises(IndexStoreError, match="unsupported document attributes"):
        index(tmp_path).upsert([document_entry(ingested(), VECTOR, (), {"authorEmail": "a@b"})])


def test_forbidden_content_precedes_unknown_and_typed_attribute_errors(tmp_path):
    item = document_entry(
        ingested(),
        VECTOR,
        (),
        {"text": "private", "authorEmail": "a@b", "pageCount": -1, "language": "english"},
    )
    with pytest.raises(IndexStoreError) as refusal:
        index(tmp_path).upsert([item])
    assert refusal.value.detail == "the index must not carry document text: text"


def test_a_renamed_document_keeps_its_entry(tmp_path):
    subject = index(tmp_path)
    subject.upsert([document_entry(ingested("/home/u/docs/report.pdf"), VECTOR)])

    update = subject.upsert([document_entry(ingested("/home/u/archive/2024-report.pdf"), VECTOR)])

    assert update.updated == 1
    assert len(update.state.entries) == 1


def test_two_copies_of_one_document_are_one_entry(tmp_path):
    subject = index(tmp_path)
    subject.upsert(
        [
            document_entry(ingested("/home/u/docs/a.pdf"), VECTOR),
            document_entry(ingested("/home/u/backup/a.pdf"), VECTOR),
        ]
    )
    assert len(subject.load().entries) == 1


def test_a_digest_that_is_not_a_digest_is_refused():
    with pytest.raises(IndexStoreError, match="sha256"):
        document_entry_id("nope")


def test_document_identity_keeps_the_legacy_length_only_digest_contract():
    digest = "not-hex".ljust(64, "!")
    assert document_entry_id(digest) == f"doc-{digest}"


def test_an_identity_not_derived_from_content_is_refused(tmp_path):
    item = document_entry(ingested(), VECTOR)
    forged = type(item)("doc-" + "0" * 64, DIGEST, item.source_path, VECTOR, (), {}, 1)
    with pytest.raises(IndexStoreError, match="derived from its digest"):
        index(tmp_path).upsert([forged])


def test_an_identity_without_the_document_prefix_is_refused(tmp_path):
    item = document_entry(ingested(), VECTOR)
    forged = type(item)("pdf-1", DIGEST, item.source_path, VECTOR, (), {}, 1)
    with pytest.raises(IndexStoreError, match="identity is invalid"):
        index(tmp_path).upsert([forged])


def test_a_label_outside_the_declared_set_is_refused(tmp_path):
    subject = index(tmp_path, labels=("invoice", "contract"))

    subject.upsert([document_entry(ingested(), VECTOR, ("invoice",))])
    with pytest.raises(IndexStoreError, match="outside the declared set"):
        subject.upsert([document_entry(ingested(), VECTOR, ("account-number-1234",))])


@pytest.mark.parametrize("label", ["Invoice", "two words", "under_score"])
def test_a_label_that_is_not_a_slug_is_refused(tmp_path, label):
    with pytest.raises(IndexStoreError, match="not a slug"):
        index(tmp_path).upsert([document_entry(ingested(), VECTOR, (label,))])


@pytest.mark.parametrize(
    "attributes",
    [
        {"pageCount": -1},
        {"pageCount": 200_000},
        {"pageCount": True},
        {"pageCount": 1.5},
        {"wordCount": -1},
        {"language": "english"},
        {"language": 7},
        {"language": "e"},
    ],
)
def test_an_impossible_metadata_value_is_refused(tmp_path, attributes):
    with pytest.raises(IndexStoreError, match="entry-invalid"):
        index(tmp_path).upsert([document_entry(ingested(), VECTOR, (), attributes)])


@pytest.mark.parametrize(
    ("attributes", "detail"),
    [
        ({"unknown": 1, "pageCount": -1}, "unsupported document attributes: unknown"),
        ({"pageCount": -1, "wordCount": -1}, "pageCount is out of range"),
        ({"wordCount": -1, "language": "english"}, "wordCount is out of range"),
    ],
)
def test_document_metadata_errors_keep_their_established_order(tmp_path, attributes, detail):
    with pytest.raises(IndexStoreError) as refusal:
        index(tmp_path).upsert([document_entry(ingested(), VECTOR, (), attributes)])
    assert refusal.value.detail == detail


@pytest.mark.parametrize("language", ["en", "pt-BR", "zh-Hans"])
def test_a_well_formed_language_tag_is_accepted(tmp_path, language):
    assert (
        index(tmp_path)
        .upsert([document_entry(ingested(), VECTOR, (), {"language": language})])
        .added
        == 1
    )


@pytest.mark.parametrize(
    "attributes",
    [
        {"pageCount": 0},
        {"pageCount": MAX_PAGES},
        {"wordCount": 0},
        {"wordCount": MAX_WORDS},
    ],
)
def test_document_counts_accept_their_inclusive_boundaries(tmp_path, attributes):
    assert index(tmp_path).upsert([document_entry(ingested(), VECTOR, (), attributes)]).added == 1


def test_a_non_ingested_file_is_refused():
    with pytest.raises(IndexStoreError, match="IngestedFile"):
        document_entry({"path": "/home/u/a.pdf"}, VECTOR)


def test_a_document_without_a_suffix_records_an_unknown_format():
    item = document_entry(ingested("/home/u/docs/scan", suffix=""), VECTOR)
    assert item.attributes["format"] == "unknown"


def test_a_deleted_document_is_removed_from_the_index(tmp_path):
    subject = index(tmp_path)
    subject.upsert([document_entry(ingested(), VECTOR)])

    assert subject.delete([document_entry_id(DIGEST)]).removed == 1
    assert subject.load().entries == ()


def test_the_index_survives_a_reload(tmp_path):
    index(tmp_path).upsert([document_entry(ingested(), VECTOR, ("contract",), {"pageCount": 12})])

    state = index(tmp_path).load()

    assert state.health is IndexHealth.INTACT
    assert state.entries[0].tags == ("contract",)
    assert state.entries[0].attributes["pageCount"] == 12


def test_a_stored_entry_that_violates_document_policy_recovers(tmp_path):
    BoundedIndexStore(tmp_path, DOCUMENT_INDEX_NAME, MODEL).upsert(
        [document_entry(ingested(), VECTOR, (), {"text": "must not survive"})]
    )
    state = index(tmp_path).load()
    assert state.health is IndexHealth.RECOVERED
    assert state.entries == ()


def test_a_new_embedding_model_invalidates_the_index(tmp_path):
    index(tmp_path).upsert([document_entry(ingested(), VECTOR)])

    other = DocumentIntelligenceIndex(tmp_path, ModelIdentity("minilm", "b" * 64, 4))

    assert other.load().health is IndexHealth.INCOMPATIBLE
    assert other.reindex([document_entry(ingested(), VECTOR)]).state.health is IndexHealth.INTACT


def test_the_visual_and_document_indexes_do_not_share_a_file(tmp_path):
    from omnitensor.plugins.visual_index import VisualLibraryIndex

    documents = index(tmp_path)
    visuals = VisualLibraryIndex(tmp_path, MODEL)

    documents.upsert([document_entry(ingested(), VECTOR)])

    assert documents.path != visuals.path
    assert visuals.load().entries == ()


def test_the_declared_labels_are_reported(tmp_path):
    assert index(tmp_path, labels=("contract", "invoice")).labels == {"contract", "invoice"}
    assert index(tmp_path).labels == frozenset()


def test_the_label_set_itself_is_validated(tmp_path):
    with pytest.raises(IndexStoreError, match="tags-invalid"):
        index(tmp_path, labels="invoice")
