"""Splitting the work over many sources, without multiplying the answer."""

from __future__ import annotations

from omnitensor.plugins.event_passes import merge, plan_passes
from omnitensor.plugins.events import (
    EventCandidate,
    EventPlace,
    EventWhen,
    GroundedEventResult,
    Stated,
)


def event(candidate_id="e1", label="Project review", date="2026-09-03", time="14:00", place=None):
    return EventCandidate(
        candidate_id=candidate_id,
        label=Stated(label, "stated"),
        when=EventWhen(date=date, time=time, timezone=None),
        confirmation="pending",
        evidence=(),
        where=None if place is None else EventPlace(kind="physical", source="stated", venue=place),
    )


def result(*events, outcome="succeeded"):
    return GroundedEventResult("job-1", outcome, "events-extracted", "", tuple(events))


class TestGroupingSourcesIntoPasses:
    def test_small_sources_share_one_pass(self):
        fragments = [(f"ref-{index}", "a short line") for index in range(5)]

        passes = plan_passes(fragments, context_tokens=32_768)

        assert len(passes) == 1
        assert passes[0] == tuple(f"ref-{index}" for index in range(5))

    def test_a_long_selection_is_split_and_nothing_is_dropped(self):
        fragments = [(f"ref-{index}", "word " * 20_000) for index in range(6)]

        passes = plan_passes(fragments, context_tokens=32_768)

        assert len(passes) > 1
        flattened = [reference for group in passes for reference in group]
        assert flattened == [f"ref-{index}" for index in range(6)]
        assert len(set(flattened)) == len(flattened)

    def test_a_source_too_large_for_any_pass_still_gets_one(self):
        # Trimming it would lose events silently, which is the one outcome
        # worth avoiding; a pass of its own is what it can honestly have.
        fragments = [("huge", "word " * 200_000), ("small", "one line")]

        passes = plan_passes(fragments, context_tokens=32_768)

        assert passes[0] == ("huge",)
        assert "small" in passes[1]

    def test_the_prompt_is_paid_for_out_of_the_same_context(self):
        fragments = [(f"ref-{index}", "word " * 2_000) for index in range(4)]

        generous = plan_passes(fragments, context_tokens=32_768, prompt_tokens=0)
        crowded = plan_passes(fragments, context_tokens=32_768, prompt_tokens=12_000)

        assert len(crowded) >= len(generous)


class TestAccumulatingWhatThePassesFound:
    def test_events_from_several_passes_are_one_answer(self):
        merged = merge([result(event("e1")), result(event("e2", label="Retrospective"))], "job-1")

        assert merged.outcome == "succeeded"
        assert [item.label.text for item in merged.events] == ["Project review", "Retrospective"]

    def test_the_same_event_named_in_two_passes_is_counted_once(self):
        # A recurring meeting mentioned on three pages is one event. Without
        # this, splitting the work would multiply the answer.
        merged = merge([result(event("e1")), result(event("other"))], "job-1")

        assert len(merged.events) == 1
        assert merged.duplicates_dropped == 1

    def test_candidate_ids_stay_unique_when_passes_number_independently(self):
        merged = merge(
            [result(event("e1")), result(event("e1", label="Budget review"))],
            "job-1",
        )

        identities = [item.candidate_id for item in merged.events]
        assert len(set(identities)) == len(identities) == 2

    def test_one_empty_pass_does_not_erase_what_another_found(self):
        merged = merge(
            [result(outcome="refused"), result(event("e1")), result(outcome="refused")],
            "job-1",
        )

        assert merged.outcome == "succeeded"
        assert len(merged.events) == 1

    def test_every_pass_refusing_is_a_refusal(self):
        merged = merge([result(outcome="refused"), result(outcome="refused")], "job-1")

        assert (merged.outcome, merged.code) == ("refused", "no-event-found")
        assert merged.events == ()

    def test_unreadable_material_in_any_pass_makes_the_whole_answer_partial(self):
        merged = merge(
            [result(event("e1")), result(event("e2", date="2026-10-01"), outcome="partial")],
            "job-1",
        )

        assert merged.outcome == "partial"
        assert "could not be read" in merged.detail
        assert len(merged.events) == 2
