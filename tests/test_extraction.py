from __future__ import annotations

import asyncio

import pytest

from omnitensor.plugins.document_ingestion import DocumentIngestor
from omnitensor.plugins.extraction import (
    AdapterKind,
    BlockKind,
    DocumentExtractor,
    ExtractionError,
    ExtractionLimits,
    ExtractionOutcome,
    ImageRegion,
    LayoutBlock,
    PageContent,
    normalise_text,
)
from omnitensor.plugins.ingestion import IngestedFile

ITEM = IngestedFile("/home/u/docs/report.pdf", 40_960, 1_000, ".pdf", "d" * 64)


class Adapter:
    """A stand-in for a parser running in its own worker."""

    kind = AdapterKind.PARSER

    def __init__(self, pages=(), *, fail_after=None, error=None, delay=0.0):
        self._pages = list(pages)
        self._fail_after = fail_after
        self._error = error or RuntimeError("segmentation fault in libpoppler")
        self._delay = delay
        self.closed = False
        self.produced = 0

    async def pages(self, item):
        try:
            for index, page in enumerate(self._pages):
                if self._fail_after is not None and index == self._fail_after:
                    raise self._error
                if self._delay:
                    await asyncio.sleep(self._delay)
                self.produced += 1
                yield page
        finally:
            self.closed = True


def page(number=1, text="Hello", blocks=(), images=()):
    return PageContent(number, text, blocks, images)


def extract(adapter, **changes):
    limits = ExtractionLimits(**changes) if changes else None
    return asyncio.run(DocumentExtractor(adapter, limits=limits).extract(ITEM))


def test_a_well_formed_document_extracts_completely():
    result = extract(Adapter([page(1, "First page"), page(2, "Second page")]))

    assert result.outcome is ExtractionOutcome.SUCCEEDED
    assert result.complete
    assert not result.partial
    assert result.text == "First page\nSecond page"
    assert [item.page_number for item in result.pages] == [1, 2]


def test_a_crashing_adapter_keeps_the_pages_it_already_produced():
    """A half-read contract is worth showing, as long as it is labelled."""
    result = extract(Adapter([page(1, "Recital"), page(2, "Never reached")], fail_after=1))

    assert result.outcome is ExtractionOutcome.CRASHED
    assert result.partial
    assert result.text == "Recital"
    assert "segmentation fault" in result.detail


def test_a_crash_before_any_page_is_not_partial():
    result = extract(Adapter([page(1)], fail_after=0))
    assert result.outcome is ExtractionOutcome.CRASHED
    assert result.pages == ()
    assert not result.partial


def test_an_adapter_that_never_finishes_is_bounded_by_a_deadline():
    result = extract(Adapter([page(1, "Slow"), page(2, "Slower")], delay=0.2), timeout_seconds=0.05)

    assert result.outcome is ExtractionOutcome.TIMED_OUT
    assert "deadline" in result.detail


def test_an_adapter_that_raises_before_yielding_anything_is_contained():
    class Broken:
        kind = AdapterKind.OCR

        def pages(self, item):
            raise MemoryError("cannot allocate 8 GiB")

    result = asyncio.run(DocumentExtractor(Broken()).extract(ITEM))

    assert result.outcome is ExtractionOutcome.CRASHED
    assert "MemoryError" in result.detail


def test_cancellation_is_the_callers_decision_not_an_adapter_failure():
    async def scenario():
        extractor = DocumentExtractor(Adapter([page(1)], delay=5), limits=ExtractionLimits())
        task = asyncio.create_task(extractor.extract(ITEM))
        await asyncio.sleep(0)
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scenario())


def test_page_output_is_bounded_and_the_adapter_is_closed():
    adapter = Adapter([page(number) for number in range(1, 11)])

    result = extract(adapter, max_pages=3)

    assert result.outcome is ExtractionOutcome.TRUNCATED
    assert len(result.pages) == 3
    assert result.dropped_pages >= 1
    assert adapter.closed
    assert adapter.produced < 10


def test_character_output_is_bounded():
    result = extract(Adapter([page(1, "x" * 100), page(2, "y" * 100)]), max_characters=120)

    assert result.outcome is ExtractionOutcome.TRUNCATED
    assert len(result.text) <= 121
    assert result.dropped_characters == 80


def test_control_characters_are_stripped():
    result = extract(Adapter([page(1, "Total:\x00 900\x07")]))
    assert result.text == "Total: 900"


def test_bidirectional_overrides_are_stripped():
    """An override makes a line render as something other than what it says."""
    text, _dropped = normalise_text("invoice‮gnp.exe", 100)
    assert "‮" not in text
    assert text == "invoicegnp.exe"


def test_text_is_unicode_normalised():
    combining, _ = normalise_text("café", 100)
    precomposed, _ = normalise_text("café", 100)
    assert combining == precomposed


def test_whitespace_is_collapsed_and_trimmed():
    text, _ = normalise_text("  Heading   spaced  \n   indented line  ", 100)
    assert text == "Heading spaced\nindented line"


def test_a_non_string_page_text_yields_no_text():
    assert normalise_text(None, 10) == ("", 0)


def test_layout_blocks_are_normalised_and_kept():
    block = LayoutBlock(BlockKind.HEADING, 10, 20, 100, 30, "  Terms\x00  ")
    result = extract(Adapter([page(1, "body", blocks=(block,))]))

    [kept] = result.pages[0].blocks
    assert kept.kind is BlockKind.HEADING
    assert kept.text == "Terms"


@pytest.mark.parametrize(
    "block",
    [
        LayoutBlock("heading", 1, 1, 1, 1, "x"),
        LayoutBlock(BlockKind.HEADING, -1, 1, 1, 1, "x"),
        LayoutBlock(BlockKind.HEADING, 1, 1, 2_000_000, 1, "x"),
        LayoutBlock(BlockKind.HEADING, True, 1, 1, 1, "x"),
        "not-a-block",
    ],
)
def test_an_impossible_layout_block_is_dropped(block):
    result = extract(Adapter([page(1, "body", blocks=(block,))]))
    assert result.pages[0].blocks == ()
    assert result.dropped_regions == 1


def test_images_are_described_never_carried():
    image = ImageRegion("img-1", 0, 0, 640, 480)
    result = extract(Adapter([page(1, "body", images=(image,))]))

    [kept] = result.pages[0].images
    assert kept.image_id == "img-1"
    assert not hasattr(kept, "data")
    assert not hasattr(kept, "bytes")


@pytest.mark.parametrize(
    "image",
    [
        ImageRegion("", 0, 0, 10, 10),
        ImageRegion("i" * 200, 0, 0, 10, 10),
        ImageRegion("img", 0, 0, 100_000, 100_000),
        ImageRegion("img", -1, 0, 10, 10),
        7,
    ],
)
def test_an_impossible_image_region_is_dropped(image):
    result = extract(Adapter([page(1, "body", images=(image,))]))
    assert result.pages[0].images == ()
    assert result.dropped_regions == 1


def test_regions_are_bounded_per_page():
    blocks = tuple(LayoutBlock(BlockKind.OTHER, 0, 0, 1, 1, "x") for _ in range(10))
    result = extract(Adapter([page(1, "body", blocks=blocks)]), max_regions_per_page=4)

    assert len(result.pages[0].blocks) == 4
    assert result.dropped_regions == 6
    assert result.outcome is ExtractionOutcome.TRUNCATED


def test_a_region_collection_that_is_not_a_sequence_is_ignored():
    result = extract(Adapter([page(1, "body", blocks="not-a-sequence", images=None)]))
    assert result.pages[0].blocks == ()
    assert result.pages[0].images == ()


def test_a_page_that_is_not_a_page_is_dropped():
    result = extract(Adapter([page(1, "real"), "not-a-page"]))
    assert len(result.pages) == 1
    assert result.dropped_pages == 1


def test_an_absurd_page_number_is_replaced_by_its_position():
    result = extract(Adapter([page(-5, "a"), page(True, "b")]))
    assert [item.page_number for item in result.pages] == [1, 2]


def test_a_declared_expansion_bomb_is_refused_before_the_adapter_runs():
    """Expanding to find out is the attack, so the claim is what is checked."""
    adapter = Adapter([page(1, "never parsed")])
    extractor = DocumentExtractor(
        adapter, ingestor=DocumentIngestor(["/tmp"], max_expansion_ratio=10)
    )
    item = IngestedFile("/tmp/bomb.zip", 1_000, 0, ".zip", "a" * 64)

    result = asyncio.run(extractor.extract(item, declared_uncompressed_bytes=1_000_000))

    assert result.outcome is ExtractionOutcome.REFUSED
    assert "expansion-ratio-exceeded" in result.detail
    assert adapter.produced == 0


def test_a_plausible_expansion_still_extracts():
    extractor = DocumentExtractor(
        Adapter([page(1, "content")]),
        ingestor=DocumentIngestor(["/tmp"], max_expansion_ratio=10),
    )
    item = IngestedFile("/tmp/normal.zip", 1_000, 0, ".zip", "a" * 64)

    result = asyncio.run(extractor.extract(item, declared_uncompressed_bytes=5_000))

    assert result.outcome is ExtractionOutcome.SUCCEEDED


def test_an_expansion_claim_without_an_ingestor_is_not_judged_here():
    result = asyncio.run(
        DocumentExtractor(Adapter([page(1, "x")])).extract(ITEM, declared_uncompressed_bytes=10**9)
    )
    assert result.outcome is ExtractionOutcome.SUCCEEDED


def test_the_adapter_kind_is_declared_and_reported():
    assert DocumentExtractor(Adapter()).kind is AdapterKind.PARSER


@pytest.mark.parametrize(
    "adapter",
    [object(), type("NoKind", (), {"pages": lambda self, item: None})()],
)
def test_an_adapter_that_does_not_meet_the_contract_is_refused(adapter):
    with pytest.raises(ExtractionError, match="adapter-invalid"):
        DocumentExtractor(adapter)


def test_the_ingestor_must_be_an_ingestor():
    with pytest.raises(ExtractionError, match="adapter-invalid"):
        DocumentExtractor(Adapter(), ingestor=object())


@pytest.mark.parametrize(
    "changes",
    [
        {"timeout_seconds": 0},
        {"timeout_seconds": 100_000},
        {"timeout_seconds": True},
        {"max_pages": 0},
        {"max_characters": True},
        {"max_regions_per_page": -1},
        {"max_image_pixels": 0},
    ],
)
def test_the_limits_are_validated(changes):
    with pytest.raises(ExtractionError, match="limits-invalid"):
        DocumentExtractor(Adapter(), limits=ExtractionLimits(**changes))


def test_the_item_must_be_an_ingested_file():
    with pytest.raises(ExtractionError, match="item-invalid"):
        asyncio.run(DocumentExtractor(Adapter()).extract("/home/u/docs/report.pdf"))


def test_the_result_summarises_itself_without_the_text():
    result = extract(Adapter([page(1, "Confidential salary review")]))
    summary = result.document()

    assert "Confidential" not in str(summary)
    assert summary["pageCount"] == 1
    assert summary["characters"] == 26
    assert summary["outcome"] == "succeeded"


def test_the_limits_are_reported():
    assert DocumentExtractor(Adapter()).limits.max_pages == 2_000


def test_a_truncated_extraction_can_say_exactly_what_went_unread():
    """OMNI-0395: what was dropped was counted and then thrown away."""
    from omnitensor.plugins.extraction import (
        TRUNCATED_SOURCE_CODE,
        measured_failure_detail,
        truncation_summary,
    )

    result = extract(Adapter([page(1, "a" * 10), page(2, "b" * 10)]), max_characters=12)

    assert result.outcome is ExtractionOutcome.TRUNCATED
    summary = truncation_summary(result)
    assert "went unread" in summary
    assert "character(s)" in summary
    # A refusal names counts, never the file it read or anything it contained.
    assert "report.pdf" not in summary
    assert "a" * 10 not in summary

    assert truncation_summary(extract(Adapter([page(1, "short")]))) == ""
    assert measured_failure_detail(TRUNCATED_SOURCE_CODE, summary) == (
        f"{TRUNCATED_SOURCE_CODE}: {summary}"
    )
    assert measured_failure_detail(TRUNCATED_SOURCE_CODE, "") == ""
    assert measured_failure_detail("extraction-failed", summary) == ""


def test_the_ceilings_follow_the_file_instead_of_a_policy_number():
    """OMNI-0503: 5,000,000 characters cut a long report for no reason."""
    from omnitensor.plugins.document_ingestion import DEFAULT_MAX_EXPANSION_RATIO
    from omnitensor.plugins.extraction import (
        DEFAULT_MAX_CHARACTERS,
        DEFAULT_MAX_PAGES,
    )

    default = ExtractionLimits()

    # A small file keeps the floor: nothing is tightened by this.
    assert default.for_file(1_000) is default
    assert default.for_file(0) is default
    assert default.for_file(None) is default

    big = default.for_file(80 * 1024 * 1024)
    assert big.max_characters == 80 * 1024 * 1024 * DEFAULT_MAX_EXPANSION_RATIO
    assert big.max_pages == 80 * 1024 * 1024
    assert big.max_characters > DEFAULT_MAX_CHARACTERS
    assert big.max_pages > DEFAULT_MAX_PAGES
    assert big.timeout_seconds == default.timeout_seconds

    # A caller that named a ceiling keeps exactly that ceiling.
    explicit = ExtractionLimits(max_characters=12, max_pages=3)
    assert explicit.for_file(80 * 1024 * 1024) is explicit


def test_a_six_megabyte_selection_is_extracted_whole():
    item = IngestedFile("/home/u/docs/report.pdf", 6 * 1024 * 1024, 1_000, ".pdf", "d" * 64)
    pages = [page(number + 1, "x" * 1_000) for number in range(6_000)]
    result = asyncio.run(DocumentExtractor(Adapter(pages)).extract(item))

    assert result.outcome is ExtractionOutcome.SUCCEEDED
    assert len(result.pages) == 6_000
    assert result.dropped_characters == 0
