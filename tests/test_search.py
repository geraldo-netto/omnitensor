from __future__ import annotations

import base64
import json

import pytest

from omnitensor.plugins.index import ModelIdentity
from omnitensor.plugins.ingestion import IngestedFile
from omnitensor.plugins.search import (
    MAX_PAGE_SIZE,
    IndexQueryService,
    ResultState,
    SearchError,
    cosine_similarity,
    decode_cursor,
    encode_cursor,
    query_fingerprint,
)
from omnitensor.plugins.visual_index import (
    VISUAL_INDEX_PERMISSION,
    VisualLibraryIndex,
    visual_entry,
)
from omnitensor.plugins.visual_results import VISUAL_TAG_PERMISSION, VisualLibraryResults
from omnitensor.sdk.helpers import PermissionView, SDKContractError

MODEL = ModelIdentity("mobileclip", "a" * 64, 4)
ROOT = "/home/u/pictures"


def permissions(*, read=True, write=True):
    declared = frozenset({VISUAL_INDEX_PERMISSION, VISUAL_TAG_PERMISSION})
    granted = frozenset(
        {
            *({VISUAL_INDEX_PERMISSION} if read else set()),
            *({VISUAL_TAG_PERMISSION} if write else set()),
        }
    )
    return PermissionView(declared, granted)


def ingested(name, digest, modified=1_000):
    return IngestedFile(f"{ROOT}/holidays/{name}", 2048, modified, ".png", digest)


def index(tmp_path, entries=(), **changes):
    store = VisualLibraryIndex(tmp_path, MODEL, **changes)
    if entries:
        store.upsert(list(entries))
    return store


def entry(name, digest, vector, tags=(), modified=1_000):
    return visual_entry(ingested(name, digest, modified), vector, tags)


def service(tmp_path, entries=(), *, roots=(ROOT,), **changes):
    return VisualLibraryResults(
        index(tmp_path, entries), permissions(**changes), roots=roots
    )


def test_results_are_ranked_by_similarity(tmp_path):
    subject = service(
        tmp_path,
        [
            entry("a.png", "a" * 64, (1.0, 0.0, 0.0, 0.0)),
            entry("b.png", "b" * 64, (0.0, 1.0, 0.0, 0.0)),
            entry("c.png", "c" * 64, (0.9, 0.1, 0.0, 0.0)),
        ],
    )

    page = subject.search((1.0, 0.0, 0.0, 0.0))

    assert [result.entry_id[:5] for result in page.results] == ["vis-a", "vis-c", "vis-b"]
    assert page.results[0].score == 1.0
    assert page.state is ResultState.READY


def test_no_source_path_ever_reaches_a_result(tmp_path):
    """The root is the user's own choice; the tree inside it is not."""
    subject = service(tmp_path, [entry("private-folder/tax.png", "a" * 64, (1.0, 0, 0, 0))])

    page = subject.search((1.0, 0.0, 0.0, 0.0))
    encoded = json.dumps(page.document())

    assert "private-folder" not in encoded
    assert "holidays" not in encoded
    assert page.results[0].provenance.root == ROOT


def test_a_result_outside_every_known_root_reports_an_unknown_root(tmp_path):
    subject = service(tmp_path, [entry("a.png", "a" * 64, (1.0, 0, 0, 0))], roots=("/srv/other",))
    assert subject.search((1.0, 0, 0, 0)).results[0].provenance.root == "unknown"


def test_ties_are_broken_so_paging_cannot_repeat_or_skip(tmp_path):
    """Sorting by score alone leaves ties in dictionary order."""
    vector = (1.0, 0.0, 0.0, 0.0)
    entries = [entry(f"{i}.png", f"{i}" * 64, vector) for i in range(1, 6)]
    subject = service(tmp_path, entries)

    first = subject.search(vector, limit=2)
    second = subject.search(vector, limit=2, cursor=first.next_cursor)
    third = subject.search(vector, limit=2, cursor=second.next_cursor)

    seen = [result.entry_id for page in (first, second, third) for result in page.results]
    assert len(seen) == len(set(seen)) == 5
    assert third.next_cursor is None


def test_the_last_page_reports_no_further_cursor(tmp_path):
    subject = service(tmp_path, [entry("a.png", "a" * 64, (1.0, 0, 0, 0))])
    page = subject.search((1.0, 0, 0, 0), limit=5)
    assert page.next_cursor is None
    assert page.total == 1


def test_a_cursor_from_a_changed_index_is_refused(tmp_path):
    """Paging across a revision mixes two result sets into one list."""
    store = index(
        tmp_path,
        [entry("a.png", "a" * 64, (1.0, 0, 0, 0)), entry("b.png", "b" * 64, (0.5, 0.5, 0, 0))],
    )
    subject = VisualLibraryResults(store, permissions(), roots=(ROOT,))
    page = subject.search((1.0, 0, 0, 0), limit=1)
    store.upsert([entry("c.png", "c" * 64, (0.2, 0.8, 0, 0))])

    with pytest.raises(SearchError, match="cursor-stale"):
        subject.search((1.0, 0, 0, 0), limit=1, cursor=page.next_cursor)


def test_a_cursor_from_a_different_query_is_refused(tmp_path):
    subject = service(
        tmp_path,
        [entry("a.png", "a" * 64, (1.0, 0, 0, 0)), entry("b.png", "b" * 64, (0, 1.0, 0, 0))],
    )
    page = subject.search((1.0, 0, 0, 0), limit=1)

    with pytest.raises(SearchError, match="cursor-mismatch"):
        subject.search((0.0, 1.0, 0, 0), limit=1, cursor=page.next_cursor)


@pytest.mark.parametrize("cursor", ["", "not-base64!!", 7, None])
def test_an_unreadable_cursor_is_refused(cursor):
    with pytest.raises(SearchError, match="cursor-invalid"):
        decode_cursor(cursor, 1, "abc")


@pytest.mark.parametrize(
    "document",
    [
        {"v": 99, "r": 1, "o": 0, "q": "abc"},
        {"v": 1, "r": 1, "o": -1, "q": "abc"},
        {"v": 1, "r": 1, "o": True, "q": "abc"},
        ["not", "a", "document"],
    ],
)
def test_a_malformed_cursor_is_refused(document):
    cursor = base64.urlsafe_b64encode(json.dumps(document).encode()).decode()
    with pytest.raises(SearchError, match="cursor-invalid"):
        decode_cursor(cursor, 1, "abc")


def test_a_cursor_round_trips():
    assert decode_cursor(encode_cursor(3, 50, "abc"), 3, "abc") == 50


def test_a_stale_index_says_so_rather_than_returning_nothing(tmp_path):
    """Empty results would say the library contains nothing matching."""
    index(tmp_path, [entry("a.png", "a" * 64, (1.0, 0, 0, 0))])
    other = VisualLibraryIndex(tmp_path, ModelIdentity("mobileclip", "b" * 64, 4))

    page = VisualLibraryResults(other, permissions(), roots=(ROOT,)).search((1.0, 0, 0, 0))

    assert page.state is ResultState.STALE
    assert not page.usable
    assert page.results == ()


def test_a_recovering_index_says_so(tmp_path):
    store = index(tmp_path, [entry("a.png", "a" * 64, (1.0, 0, 0, 0))])
    store.path.write_text("{ broken")

    page = VisualLibraryResults(store, permissions(), roots=(ROOT,)).search((1.0, 0, 0, 0))

    assert page.state is ResultState.RECOVERING


def test_an_indexing_pass_in_flight_is_reported(tmp_path):
    subject = service(tmp_path, [entry("a.png", "a" * 64, (1.0, 0, 0, 0))])
    subject.indexing_started()

    assert subject.search((1.0, 0, 0, 0)).state is ResultState.INDEXING
    assert subject.indexing

    subject.indexing_finished()
    assert subject.search((1.0, 0, 0, 0)).state is ResultState.READY


def test_search_requires_the_read_grant(tmp_path):
    subject = service(tmp_path, [entry("a.png", "a" * 64, (1.0, 0, 0, 0))], read=False)
    with pytest.raises(SDKContractError, match="permission-denied"):
        subject.search((1.0, 0, 0, 0))
    with pytest.raises(SDKContractError, match="permission-denied"):
        subject.browse()
    with pytest.raises(SDKContractError, match="permission-denied"):
        subject.similar_to("vis-" + "a" * 64)


def test_tagging_requires_a_separate_write_grant(tmp_path):
    """Being allowed to find a photo is not being allowed to relabel it."""
    subject = service(tmp_path, [entry("a.png", "a" * 64, (1.0, 0, 0, 0))], write=False)
    with pytest.raises(SDKContractError, match="permission-denied"):
        subject.assign_tags("vis-" + "a" * 64, ("beach",))


def test_tags_can_be_assigned_and_are_then_searchable(tmp_path):
    subject = service(tmp_path, [entry("a.png", "a" * 64, (1.0, 0, 0, 0))])

    subject.assign_tags("vis-" + "a" * 64, ("beach", "sunset"))

    page = subject.search((1.0, 0, 0, 0), tags=("beach",))
    assert page.results[0].tags == ("beach", "sunset")


def test_tagging_an_unknown_entry_is_refused(tmp_path):
    subject = service(tmp_path, [entry("a.png", "a" * 64, (1.0, 0, 0, 0))])
    with pytest.raises(SearchError, match="entry-unknown"):
        subject.assign_tags("vis-" + "9" * 64, ("beach",))
    with pytest.raises(SearchError, match="entry-unknown"):
        subject.assign_tags("", ("beach",))


def test_a_tag_the_index_refuses_surfaces_as_a_query_error(tmp_path):
    store = VisualLibraryIndex(tmp_path, MODEL, vocabulary=("beach",))
    store.upsert([entry("a.png", "a" * 64, (1.0, 0, 0, 0))])
    subject = VisualLibraryResults(store, permissions(), roots=(ROOT,))

    with pytest.raises(SearchError, match="entry-invalid"):
        subject.assign_tags("vis-" + "a" * 64, ("passport-number",))


def test_filtering_by_tag_narrows_the_result_set(tmp_path):
    subject = service(
        tmp_path,
        [
            entry("a.png", "a" * 64, (1.0, 0, 0, 0), ("beach",)),
            entry("b.png", "b" * 64, (1.0, 0, 0, 0), ("indoors",)),
        ],
    )

    page = subject.search((1.0, 0, 0, 0), tags=("beach",))

    assert page.total == 1
    assert page.results[0].tags == ("beach",)


def test_browsing_lists_the_newest_first(tmp_path):
    subject = service(
        tmp_path,
        [
            entry("a.png", "a" * 64, (1.0, 0, 0, 0), modified=1_000),
            entry("b.png", "b" * 64, (1.0, 0, 0, 0), modified=5_000),
        ],
    )

    page = subject.browse()

    assert [result.entry_id[:5] for result in page.results] == ["vis-b", "vis-a"]
    assert page.results[0].score == 0.0


def test_similarity_to_an_indexed_entry_excludes_itself(tmp_path):
    subject = service(
        tmp_path,
        [
            entry("a.png", "a" * 64, (1.0, 0, 0, 0)),
            entry("b.png", "b" * 64, (0.9, 0.1, 0, 0)),
            entry("c.png", "c" * 64, (0.0, 1.0, 0, 0)),
        ],
    )

    page = subject.similar_to("vis-" + "a" * 64)

    assert [result.entry_id[:5] for result in page.results] == ["vis-b", "vis-c"]


def test_similarity_to_an_unknown_entry_is_refused(tmp_path):
    subject = service(tmp_path, [entry("a.png", "a" * 64, (1.0, 0, 0, 0))])
    with pytest.raises(SearchError, match="entry-unknown"):
        subject.similar_to("vis-" + "9" * 64)


@pytest.mark.parametrize(
    "vector",
    [(0.0, 0.0, 0.0, 0.0), (1.0, 0.0), "1,0,0,0", (float("nan"), 0, 0, 0), (True, 0, 0, 0)],
)
def test_an_unusable_query_vector_is_refused(tmp_path, vector):
    with pytest.raises(SearchError, match="query-invalid"):
        service(tmp_path, [entry("a.png", "a" * 64, (1.0, 0, 0, 0))]).search(vector)


@pytest.mark.parametrize("tags", ["beach", [""], [7], ["x"] * 33])
def test_an_unusable_tag_filter_is_refused(tmp_path, tags):
    with pytest.raises(SearchError, match="filter-invalid"):
        service(tmp_path).search((1.0, 0, 0, 0), tags=tags)


@pytest.mark.parametrize("limit", [0, MAX_PAGE_SIZE + 1, True, "10"])
def test_the_page_size_is_bounded(tmp_path, limit):
    with pytest.raises(SearchError, match="bounds-invalid"):
        service(tmp_path).search((1.0, 0, 0, 0), limit=limit)


@pytest.mark.parametrize("page_size", [0, MAX_PAGE_SIZE + 1, True])
def test_the_default_page_size_is_bounded(tmp_path, page_size):
    with pytest.raises(SearchError, match="bounds-invalid"):
        VisualLibraryResults(index(tmp_path), permissions(), page_size=page_size)


def test_the_default_page_size_applies_when_no_limit_is_given(tmp_path):
    entries = [entry(f"{i}.png", f"{i:x}" * 64, (1.0, 0, 0, 0)) for i in range(5)]
    subject = VisualLibraryResults(index(tmp_path, entries), permissions(), page_size=2)
    assert len(subject.search((1.0, 0, 0, 0)).results) == 2


def test_the_store_must_be_an_index(tmp_path):
    with pytest.raises(SearchError, match="store-invalid"):
        IndexQueryService(object(), permissions())
    with pytest.raises(TypeError, match="VisualLibraryIndex"):
        VisualLibraryResults(object(), permissions())


def test_cosine_similarity_of_a_directionless_vector_is_zero():
    assert cosine_similarity((0.0, 0.0), (1.0, 0.0)) == 0.0
    assert cosine_similarity((1.0, 0.0), (0.0, 0.0)) == 0.0


def test_a_query_fingerprint_is_stable_and_query_specific():
    assert query_fingerprint({"a": 1}) == query_fingerprint({"a": 1})
    assert query_fingerprint({"a": 1}) != query_fingerprint({"a": 2})


def test_the_plugin_is_named(tmp_path):
    assert service(tmp_path).plugin_id == "visual-library"
