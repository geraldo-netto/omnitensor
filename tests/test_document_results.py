from __future__ import annotations

import json

import pytest

from omnitensor.plugins.document_index import (
    DOCUMENT_INDEX_PERMISSION,
    DocumentIntelligenceIndex,
    document_entry,
)
from omnitensor.plugins.document_results import (
    DOCUMENT_CLASSIFY_PERMISSION,
    DocumentIntelligenceResults,
)
from omnitensor.plugins.index import ModelIdentity
from omnitensor.plugins.ingestion import IngestedFile
from omnitensor.plugins.search import ResultState, SearchError
from omnitensor.sdk.helpers import PermissionView, SDKContractError

MODEL = ModelIdentity("minilm", "a" * 64, 4)
ROOT = "/home/u/documents"


def permissions(*, read=True, write=True):
    declared = frozenset({DOCUMENT_INDEX_PERMISSION, DOCUMENT_CLASSIFY_PERMISSION})
    granted = frozenset(
        {
            *({DOCUMENT_INDEX_PERMISSION} if read else set()),
            *({DOCUMENT_CLASSIFY_PERMISSION} if write else set()),
        }
    )
    return PermissionView(declared, granted)


def ingested(name, digest, modified=1_000):
    return IngestedFile(f"{ROOT}/finance/{name}", 40_960, modified, ".pdf", digest)


def entry(name, digest, vector, labels=(), modified=1_000):
    return document_entry(ingested(name, digest, modified), vector, labels)


def index(tmp_path, entries=(), **changes):
    store = DocumentIntelligenceIndex(tmp_path, MODEL, **changes)
    if entries:
        store.upsert(list(entries))
    return store


def service(tmp_path, entries=(), *, roots=(ROOT,), **changes):
    return DocumentIntelligenceResults(
        index(tmp_path, entries), permissions(**changes), roots=roots
    )


def test_documents_are_ranked_by_similarity(tmp_path):
    subject = service(
        tmp_path,
        [
            entry("a.pdf", "a" * 64, (1.0, 0.0, 0.0, 0.0)),
            entry("b.pdf", "b" * 64, (0.0, 1.0, 0.0, 0.0)),
        ],
    )

    page = subject.search((1.0, 0.0, 0.0, 0.0))

    assert [result.entry_id[:5] for result in page.results] == ["doc-a", "doc-b"]
    assert page.state is ResultState.READY


def test_no_document_text_or_path_can_reach_a_result(tmp_path):
    """The index refuses to hold text, so no projection over it can leak one."""
    subject = service(tmp_path, [entry("bank-statement.pdf", "a" * 64, (1.0, 0, 0, 0))])

    encoded = json.dumps(subject.search((1.0, 0, 0, 0)).document())

    assert "bank-statement" not in encoded
    assert "finance" not in encoded
    assert ROOT in encoded


def test_a_document_carrying_text_cannot_even_be_indexed(tmp_path):
    from omnitensor.plugins.index import IndexStoreError

    with pytest.raises(IndexStoreError, match="must not carry document text"):
        index(
            tmp_path,
            [
                document_entry(
                    ingested("a.pdf", "a" * 64),
                    (1.0, 0, 0, 0),
                    (),
                    {"snippet": "Account 1234 balance 900"},
                )
            ],
        )


def test_documents_can_be_listed_by_classification(tmp_path):
    subject = service(
        tmp_path,
        [
            entry("a.pdf", "a" * 64, (1.0, 0, 0, 0), ("invoice",), modified=1_000),
            entry("b.pdf", "b" * 64, (1.0, 0, 0, 0), ("invoice",), modified=5_000),
            entry("c.pdf", "c" * 64, (1.0, 0, 0, 0), ("contract",)),
        ],
    )

    page = subject.classified_as("invoice")

    assert [result.entry_id[:5] for result in page.results] == ["doc-b", "doc-a"]
    assert page.total == 2


def test_a_classification_summary_counts_each_label(tmp_path):
    subject = service(
        tmp_path,
        [
            entry("a.pdf", "a" * 64, (1.0, 0, 0, 0), ("invoice",)),
            entry("b.pdf", "b" * 64, (1.0, 0, 0, 0), ("invoice", "scanned")),
            entry("c.pdf", "c" * 64, (1.0, 0, 0, 0), ("contract",)),
        ],
    )

    summary = subject.classification_summary()

    assert summary.counts == (("invoice", 2), ("contract", 1), ("scanned", 1))
    assert summary.total == 3
    assert summary.usable
    assert summary.document()["state"] == "ready"


def test_a_summary_over_a_stale_index_counts_nothing_and_says_why(tmp_path):
    """A count that looks like zero would read as 'you have no invoices'."""
    index(tmp_path, [entry("a.pdf", "a" * 64, (1.0, 0, 0, 0), ("invoice",))])
    other = DocumentIntelligenceIndex(tmp_path, ModelIdentity("minilm", "b" * 64, 4))

    summary = DocumentIntelligenceResults(other, permissions()).classification_summary()

    assert summary.counts == ()
    assert summary.state is ResultState.STALE
    assert not summary.usable


def test_a_summary_while_indexing_says_so(tmp_path):
    subject = service(tmp_path, [entry("a.pdf", "a" * 64, (1.0, 0, 0, 0), ("invoice",))])
    subject.indexing_started()
    assert subject.classification_summary().state is ResultState.INDEXING


def test_a_summary_requires_the_read_grant(tmp_path):
    subject = service(tmp_path, read=False)
    with pytest.raises(SDKContractError, match="permission-denied"):
        subject.classification_summary()


def test_classifying_requires_the_write_grant(tmp_path):
    subject = service(tmp_path, [entry("a.pdf", "a" * 64, (1.0, 0, 0, 0))], write=False)
    with pytest.raises(SDKContractError, match="permission-denied"):
        subject.classify("doc-" + "a" * 64, ("invoice",))


def test_a_document_can_be_classified_and_is_then_listed(tmp_path):
    subject = service(tmp_path, [entry("a.pdf", "a" * 64, (1.0, 0, 0, 0))])

    subject.classify("doc-" + "a" * 64, ("invoice",))

    assert subject.classified_as("invoice").total == 1
    assert subject.classification_summary().counts == (("invoice", 1),)


def test_a_label_the_index_refuses_surfaces_as_a_query_error(tmp_path):
    store = DocumentIntelligenceIndex(tmp_path, MODEL, labels=("invoice",))
    store.upsert([entry("a.pdf", "a" * 64, (1.0, 0, 0, 0))])
    subject = DocumentIntelligenceResults(store, permissions())

    with pytest.raises(SearchError, match="entry-invalid"):
        subject.classify("doc-" + "a" * 64, ("account-1234",))


def test_similarity_finds_related_documents(tmp_path):
    subject = service(
        tmp_path,
        [
            entry("a.pdf", "a" * 64, (1.0, 0, 0, 0)),
            entry("b.pdf", "b" * 64, (0.95, 0.05, 0, 0)),
            entry("c.pdf", "c" * 64, (0.0, 1.0, 0, 0)),
        ],
    )

    page = subject.similar_to("doc-" + "a" * 64)

    assert [result.entry_id[:5] for result in page.results] == ["doc-b", "doc-c"]


def test_classification_listings_paginate(tmp_path):
    entries = [
        entry(f"{i}.pdf", f"{i}" * 64, (1.0, 0, 0, 0), ("invoice",), modified=1_000 + i)
        for i in range(1, 4)
    ]
    subject = service(tmp_path, entries)

    first = subject.classified_as("invoice", limit=2)
    second = subject.classified_as("invoice", limit=2, cursor=first.next_cursor)

    assert len(first.results) == 2
    assert len(second.results) == 1
    assert second.next_cursor is None


def test_the_store_must_be_a_document_index(tmp_path):
    with pytest.raises(TypeError, match="DocumentIntelligenceIndex"):
        DocumentIntelligenceResults(object(), permissions())


def test_the_plugin_is_named(tmp_path):
    assert service(tmp_path).plugin_id == "document-intelligence"


def test_every_label_is_summarised_however_many_there_are(tmp_path):
    """The summary was truncated at two hundred labels, rarest lost first.

    It is sorted by descending count, so the labels a person stopped being told
    about were exactly the uncommon ones, and nothing in the result said
    anything had been dropped.
    """
    subject = service(
        tmp_path,
        [
            entry(f"{index}.pdf", f"{index:064d}", (1.0, 0, 0, 0), (f"label-{index:03d}",))
            for index in range(250)
        ],
    )

    summary = subject.classification_summary()

    assert len(summary.counts) == 250
    assert summary.total == 250
    assert len(summary.document()["counts"]) == 250
