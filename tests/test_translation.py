"""Translating a document that does not fit in one generation.

The failure this replaces: a 1,024-token budget silently truncating a long
document, producing something that looks like a finished translation. So the
properties tested here are the ones that make the split honest — every span
translated, nothing dropped, the whole thing failing rather than a paragraph
going missing, and cancellation that lands between spans instead of after all
of them.
"""

from __future__ import annotations

import pytest

from omnitensor.plugins.translation import (
    TranslationError,
    estimated_seconds,
    plan,
    progress_fraction,
    span_budget_tokens,
    translate_document,
)

DOCUMENT = (
    "The supplier must deliver the report by 30 September 2026.\n\n"
    "If the report is late, interest of 3.5 percent per year applies.\n\n"
    "Marta Oliveira signed the agreement in Lisbon on 2026-04-02.\n\n"
) * 30


async def upper(span):
    return span.text.upper()


class TestSizingTheSpans:
    def test_the_span_budget_comes_from_the_tasks_output_budget(self):
        """The task's budget is fixed by its receipt, so the free variable is
        how much source is handed to it at once."""
        assert span_budget_tokens(4_096) < 4_096
        assert span_budget_tokens(8_192) > span_budget_tokens(4_096)

    def test_a_span_is_sized_well_under_the_budget_rather_than_up_to_it(self):
        """A translation can run longer than its source. A span sized to the
        budget exactly is a truncated paragraph waiting to happen."""
        assert span_budget_tokens(4_096) < 4_096 / 1.9

    def test_an_output_budget_too_small_for_anything_is_refused(self):
        with pytest.raises(TranslationError, match="budget-too-small"):
            span_budget_tokens(1)

    def test_the_plan_says_how_many_generations_this_will_be(self):
        """The number a person needs before deciding to wait: this is a coffee
        or an afternoon."""
        spans = plan(DOCUMENT, output_tokens=4_096)

        assert len(spans) >= 1
        assert sum(len(span.text) for span in spans) == len(DOCUMENT)


class TestTranslatingTheWholeThing:
    @pytest.mark.asyncio
    async def test_every_span_appears_in_the_result_in_order(self):
        translation = await translate_document(
            DOCUMENT, output_tokens=4_096, translate_span=upper
        )

        assert translation.text == DOCUMENT.upper()

    @pytest.mark.asyncio
    async def test_a_document_needing_many_generations_still_comes_back_whole(self):
        """The case the old cap could not do at all."""
        translation = await translate_document(
            DOCUMENT, output_tokens=200, translate_span=upper
        )

        assert translation.span_count > 1
        assert translation.text == DOCUMENT.upper()

    @pytest.mark.asyncio
    async def test_an_empty_document_asks_for_no_generations(self):
        asked = []

        async def record(span):
            asked.append(span)
            return span.text

        translation = await translate_document(
            "", output_tokens=4_096, translate_span=record
        )

        assert translation.text == ""
        assert asked == []

    @pytest.mark.asyncio
    async def test_a_span_that_comes_back_empty_fails_the_job(self):
        """A document missing its fourth paragraph, delivered as if complete,
        is worse than an error."""

        async def blank_on_the_second(span):
            return "" if span.index == 1 else span.text

        with pytest.raises(TranslationError, match="span-untranslated"):
            await translate_document(
                DOCUMENT, output_tokens=200, translate_span=blank_on_the_second
            )

    @pytest.mark.asyncio
    async def test_the_failure_says_which_span_it_was(self):
        async def blank_on_the_third(span):
            return "" if span.index == 2 else span.text

        with pytest.raises(TranslationError) as failure:
            await translate_document(
                DOCUMENT, output_tokens=200, translate_span=blank_on_the_third
            )

        assert "span 3 of" in failure.value.detail

    @pytest.mark.asyncio
    async def test_a_span_failing_outright_is_not_swallowed(self):
        async def refuse(_span):
            raise RuntimeError("the worker died")

        with pytest.raises(RuntimeError):
            await translate_document(DOCUMENT, output_tokens=200, translate_span=refuse)


class TestWatchingItHappen:
    @pytest.mark.asyncio
    async def test_progress_is_reported_after_every_span(self):
        seen = []

        await translate_document(
            DOCUMENT,
            output_tokens=200,
            translate_span=upper,
            on_progress=lambda done, total: seen.append((done, total)),
        )

        assert seen[0][0] == 1
        assert seen[-1][0] == seen[-1][1]
        assert [done for done, _total in seen] == list(range(1, len(seen) + 1))

    def test_the_fraction_is_work_finished_rather_than_a_guess_about_time(self):
        assert progress_fraction(0, 10) == 0.0
        assert progress_fraction(5, 10) == 0.5
        assert progress_fraction(10, 10) == 1.0

    def test_the_fraction_survives_a_document_with_no_spans(self):
        assert progress_fraction(0, 0) == 1.0

    def test_the_estimate_comes_from_a_measured_rate(self):
        """Told to somebody before they decide to wait, from what this machine
        actually did rather than from a constant."""
        spans = plan(DOCUMENT, output_tokens=200)

        assert estimated_seconds(spans, 30.0) == pytest.approx(len(spans) * 30.0)
        assert estimated_seconds((), 30.0) == 0.0


class TestChangingYourMind:
    @pytest.mark.asyncio
    async def test_cancellation_lands_between_spans(self):
        """Three spans into two hundred, a person waits for one generation
        rather than for all of them."""
        done = []

        async def count(span):
            done.append(span.index)
            return span.text

        def stop_after_three():
            if len(done) >= 3:
                raise KeyboardInterrupt("cancelled")

        with pytest.raises(KeyboardInterrupt):
            await translate_document(
                DOCUMENT,
                output_tokens=200,
                translate_span=count,
                raise_if_cancelled=stop_after_three,
            )

        assert len(done) == 3
