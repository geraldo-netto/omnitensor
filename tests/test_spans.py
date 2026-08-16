"""Splitting a document without losing any of it.

This exists because the alternative was a cap, and a cap on a translation is a
truncated document that looks finished. The property everything rests on is
that the spans reassemble into exactly what went in — checked here from every
direction, including on inputs chosen to be awkward, because a splitter that
quietly drops a paragraph is indistinguishable from a working one until
somebody reads the output.
"""

from __future__ import annotations

import pytest

from omnitensor.plugins.spans import (
    CHARACTERS_PER_TOKEN,
    Span,
    covers,
    estimate_tokens,
    reassemble,
    split,
)

PARAGRAPHS = (
    "The supplier must deliver the report by 30 September 2026.\n\n"
    "If the report is late, interest of 3.5 percent per year applies.\n\n"
    "Marta Oliveira signed the agreement in Lisbon on 2026-04-02.\n\n"
) * 20


def limit_characters(budget_tokens: int) -> int:
    return int(budget_tokens * CHARACTERS_PER_TOKEN)


class TestNothingIsLost:
    """The property the rest depends on."""

    @pytest.mark.parametrize("budget", [8, 32, 100, 1_000, 100_000])
    def test_the_spans_reassemble_into_the_original(self, budget):
        spans = split(PARAGRAPHS, budget_tokens=budget)

        assert reassemble([span.text for span in spans]) == PARAGRAPHS
        assert covers(PARAGRAPHS, spans)

    def test_the_offsets_point_back_at_the_original(self):
        """So a failure can name where in the document it happened."""
        spans = split(PARAGRAPHS, budget_tokens=50)

        for span in spans:
            assert PARAGRAPHS[span.start : span.end] == span.text

    def test_the_spans_run_end_to_end_without_gaps_or_overlap(self):
        spans = split(PARAGRAPHS, budget_tokens=50)

        assert spans[0].start == 0
        assert spans[-1].end == len(PARAGRAPHS)
        for earlier, later in zip(spans, spans[1:], strict=False):
            assert earlier.end == later.start

    @pytest.mark.parametrize(
        "text",
        [
            "one line",
            "trailing newline\n",
            "\n\nleading blank lines\n\ntext",
            "no punctuation at all just words running on and on and on",
            "Ünïcödé áccents, שלום, and 漢字 mixed together.\n\nSecond paragraph.",
            "a\n\n\n\n\nb",
            "   ",
        ],
    )
    def test_awkward_documents_are_still_covered_exactly(self, text):
        for budget in (4, 16, 64):
            spans = split(text, budget_tokens=budget)

            assert covers(text, spans), f"{text!r} at budget {budget}"

    def test_an_empty_document_produces_no_spans(self):
        """A generation for nothing is a generation nobody needs."""
        assert split("", budget_tokens=100) == ()


class TestStayingUnderTheBudget:
    def test_every_span_fits_the_budget_when_the_text_can_be_split(self):
        budget = 60
        spans = split(PARAGRAPHS, budget_tokens=budget)

        assert len(spans) > 1
        for span in spans:
            assert len(span.text) <= limit_characters(budget)

    def test_a_short_document_is_one_span(self):
        """Splitting what already fits would cost a generation and gain
        nothing."""
        assert len(split("Short enough.", budget_tokens=1_000)) == 1

    def test_an_unsplittable_run_is_handed_over_whole_rather_than_cut(self):
        """Minified JSON, a base64 blob. It may fail on that span — failing
        loudly on one span beats silently mangling it."""
        blob = "x" * 20_000

        spans = split(blob, budget_tokens=100)

        assert len(spans) == 1
        assert spans[0].text == blob

    def test_a_budget_of_nothing_is_refused(self):
        with pytest.raises(ValueError, match="cannot hold"):
            split(PARAGRAPHS, budget_tokens=0)


class TestWhereTheSplitsGo:
    def test_paragraphs_are_preferred(self):
        """A span that starts mid-sentence costs the model the sentence it was
        in the middle of."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n\n"
        # Room for one paragraph and not two, so the choice of boundary is
        # what the result shows.
        budget = estimate_tokens("First paragraph.\n\n") + 1

        spans = split(text, budget_tokens=budget)

        assert len(spans) == 3
        assert [span.text.strip() for span in spans] == [
            "First paragraph.",
            "Second paragraph.",
            "Third paragraph.",
        ]

    def test_sentences_are_next_when_a_paragraph_is_too_long(self):
        text = "One sentence here. Another sentence here. A third one here."

        spans = split(text, budget_tokens=10)

        assert len(spans) > 1
        assert spans[0].text.strip().endswith(".")

    def test_whitespace_is_the_last_resort_before_giving_up(self):
        text = "word " * 200

        spans = split(text, budget_tokens=20)

        assert len(spans) > 1
        assert all(" " in span.text for span in spans)


class TestTheOutputBudget:
    """The replacement for the fixed cap, and the reason this exists."""

    def test_a_span_carries_a_budget_derived_from_its_own_size(self):
        small = Span(0, 0, 10, "short text")
        large = Span(1, 0, 4_000, "x" * 4_000)

        assert large.output_budget > small.output_budget * 10

    def test_the_budget_leaves_room_for_a_longer_translation(self):
        """Hebrew and German both run longer than English. A budget equal to
        the input is a cap wearing a different hat."""
        span = Span(0, 0, 1_000, "y" * 1_000)

        assert span.output_budget > span.tokens

    def test_even_a_tiny_span_gets_a_usable_budget(self):
        assert Span(0, 0, 2, "hi").output_budget >= 64

    def test_tokens_are_estimated_upward_rather_than_down(self):
        """An estimate that runs low produces spans that do not fit, which is
        the failure this whole module exists to avoid."""
        assert estimate_tokens("") == 1
        assert estimate_tokens("x" * 100) >= 100 / CHARACTERS_PER_TOKEN


class TestPuttingItBackTogether:
    def test_nothing_is_inserted_between_translated_spans(self):
        """The separators travelled with the spans, so the document's shape is
        the input's shape rather than one this invented."""
        assert reassemble(["First.\n\n", "Second."]) == "First.\n\nSecond."

    def test_a_document_translated_span_by_span_keeps_its_paragraphs(self):
        spans = split(PARAGRAPHS, budget_tokens=40)
        translated = [span.text.upper() for span in spans]

        assert reassemble(translated) == PARAGRAPHS.upper()

    def test_covers_notices_a_span_that_went_missing(self):
        """The check that runs before an hour of GPU time is spent."""
        spans = split(PARAGRAPHS, budget_tokens=40)

        assert covers(PARAGRAPHS, spans)
        assert not covers(PARAGRAPHS, spans[:-1])
