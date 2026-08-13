from __future__ import annotations

import copy
from datetime import datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins.events import (
    DUPLICATE_POLICY,
    MAX_EVENTS,
    MAX_EVIDENCE_PER_EVENT,
    EventResultError,
    confirm_event_candidates,
    parse_grounded_event_result,
    render_confirmed_ics,
    source_fragments,
    write_confirmed_events,
)
from omnitensor.plugins.extraction import (
    ExtractionOutcome,
    ExtractionResult,
    NormalisedPage,
)
from omnitensor.registry import validate_document


def evidence(*, page=1, start=0, end=12, source_ref="private:document-1") -> dict:
    return {
        "sourceRef": source_ref,
        "sourceSha256": "a" * 64,
        "page": page,
        "span": {"start": start, "end": end},
        "textSha256": "b" * 64,
    }


def candidate(
    candidate_id="event-1",
    *,
    title="Release planning",
    start="2026-08-12T10:00:00+02:00",
    end="2026-08-12T11:00:00+02:00",
    timezone="Europe/Rome",
    location="Room 2",
    confirmation="pending",
) -> dict:
    return {
        "candidateId": candidate_id,
        "title": title,
        "start": start,
        "end": end,
        "timezone": timezone,
        "location": location,
        "confirmation": confirmation,
        "evidence": [evidence()],
    }


def result_document(*events, outcome="succeeded", state="pending") -> dict:
    return {
        "version": 1,
        "requestId": "request-1",
        "outcome": outcome,
        "code": "ok" if outcome != "refused" else "source-refused",
        "detail": "" if outcome == "succeeded" else "bounded partial result",
        "duplicatePolicy": DUPLICATE_POLICY,
        "confirmationState": state,
        "events": list(events or (candidate(),)),
    }


def test_grounded_result_preserves_source_evidence_and_resolved_times():
    document = result_document(candidate())
    assert validate_document("event-extraction-result.schema.json", document) == []

    parsed = parse_grounded_event_result(document)

    assert parsed.request_id == "request-1"
    assert parsed.outcome == "succeeded"
    assert parsed.confirmation_state == "pending"
    assert parsed.duplicates_dropped == 0
    [event] = parsed.events
    assert event.title == "Release planning"
    assert event.start == datetime.fromisoformat("2026-08-12T10:00:00+02:00")
    assert event.end == datetime.fromisoformat("2026-08-12T11:00:00+02:00")
    assert event.timezone == "Europe/Rome"
    assert event.location == "Room 2"
    [proof] = event.evidence
    assert proof.source_ref == "private:document-1"
    assert proof.page == 1
    assert (proof.span_start, proof.span_end) == (0, 12)


def test_duplicate_policy_keeps_the_first_normalized_title_start_location():
    first = candidate("first")
    duplicate = candidate("second", title="  RELEASE   PLANNING ", location=" room 2 ")
    distinct = candidate("third", location="Room 3")

    parsed = parse_grounded_event_result(result_document(first, duplicate, distinct))

    assert [item.candidate_id for item in parsed.events] == ["first", "third"]
    assert parsed.duplicates_dropped == 1


def test_duplicate_policy_keeps_equal_text_at_different_starts():
    later = candidate("later", start="2026-08-12T12:00:00+02:00", end=None)
    parsed = parse_grounded_event_result(result_document(candidate("first"), later))
    assert [item.candidate_id for item in parsed.events] == ["first", "later"]
    assert parsed.duplicates_dropped == 0


def test_candidate_ids_are_unique_even_when_event_meanings_differ():
    with pytest.raises(EventResultError) as caught:
        parse_grounded_event_result(
            result_document(candidate("same"), candidate("same", location="Elsewhere"))
        )
    assert (caught.value.code, caught.value.detail) == (
        "result-invalid",
        "candidate ids must be unique",
    )


@pytest.mark.parametrize(
    ("changes", "code", "detail"),
    [
        (
            {"start": "xxxxxxxxxxxxxxxx"},
            "date-invalid",
            "event start is not ISO 8601",
        ),
        (
            {"start": "2026-08-12T10:00:00", "timezone": "Europe/Rome"},
            "date-invalid",
            "zoned start must include a UTC offset",
        ),
        (
            {"start": "2026-08-12T10:00:00+00:00", "timezone": "Europe/Rome"},
            "date-invalid",
            "event start offset disagrees with timezone",
        ),
        (
            {"start": "2026-08-12T10:00:00+02:00", "timezone": "Mars/Olympus"},
            "timezone-invalid",
            "unknown timezone: Mars/Olympus",
        ),
        (
            {
                "start": "2026-08-12T10:00:00+02:00",
                "end": "2026-08-12T09:59:59+02:00",
            },
            "date-invalid",
            "event end must be after start",
        ),
        (
            {"start": "2026-08-12T10:00:00+02:00", "timezone": "floating"},
            "date-invalid",
            "floating start must omit a UTC offset",
        ),
    ],
)
def test_date_and_timezone_semantics_fail_closed(changes, code, detail):
    event = candidate()
    event.update(changes)
    with pytest.raises(EventResultError) as caught:
        parse_grounded_event_result(result_document(event))
    assert (caught.value.code, caught.value.detail) == (code, detail)


def test_floating_candidates_are_valid_but_cannot_be_exported():
    event = candidate(
        start="2026-08-12T10:00:00",
        end="2026-08-12T11:00:00",
        timezone="floating",
    )
    parsed = parse_grounded_event_result(result_document(event))
    decided = confirm_event_candidates(parsed, ["event-1"])
    with pytest.raises(EventResultError) as caught:
        render_confirmed_ics(decided)
    assert (caught.value.code, caught.value.detail) == (
        "timezone-required",
        "confirmed events need a named timezone",
    )


def test_evidence_requires_a_forward_span_and_an_opaque_source():
    for changed, expected in (
        ({"span": {"start": 5, "end": 5}}, "evidence span end must be after start"),
        ({"sourceRef": "file:///home/user/private.pdf"}, "does not match"),
    ):
        event = candidate()
        event["evidence"][0].update(changed)
        with pytest.raises(EventResultError) as caught:
            parse_grounded_event_result(result_document(event))
        assert expected in caught.value.detail


def test_closed_schema_rejects_extra_fields_and_unbounded_evidence():
    document = result_document(candidate())
    document["privateText"] = "must never leave the worker"
    with pytest.raises(EventResultError) as caught:
        parse_grounded_event_result(document)
    assert caught.value.code == "result-invalid"
    assert "Additional properties" in caught.value.detail

    document = result_document(candidate())
    document["events"][0]["evidence"] = []
    with pytest.raises(EventResultError, match="non-empty"):
        parse_grounded_event_result(document)


def test_event_and_evidence_bounds_accept_the_exact_maximum():
    events = [
        candidate(f"event-{index}", title=f"Bounded event {index}")
        for index in range(MAX_EVENTS)
    ]
    events[0]["evidence"] = [evidence() for _ in range(MAX_EVIDENCE_PER_EVENT)]

    parsed = parse_grounded_event_result(result_document(*events))

    assert len(parsed.events) == MAX_EVENTS
    assert len(parsed.events[0].evidence) == MAX_EVIDENCE_PER_EVENT


@pytest.mark.parametrize("overflow", ["events", "evidence"])
def test_event_and_evidence_bounds_refuse_max_plus_one(overflow):
    document = result_document(candidate())
    if overflow == "events":
        document["events"] = [
            candidate(f"event-{index}", title=f"Bounded event {index}")
            for index in range(MAX_EVENTS + 1)
        ]
        expected = f"event result exceeds {MAX_EVENTS} events"
    else:
        document["events"][0]["evidence"] = [
            evidence() for _ in range(MAX_EVIDENCE_PER_EVENT + 1)
        ]
        expected = f"event evidence exceeds {MAX_EVIDENCE_PER_EVENT} entries"

    with pytest.raises(EventResultError) as caught:
        parse_grounded_event_result(document)

    assert (caught.value.code, caught.value.detail) == ("result-invalid", expected)


def test_refused_and_partial_outcomes_are_explicit_and_consistent():
    refused = result_document(outcome="refused", state="refused")
    refused["events"] = []
    parsed = parse_grounded_event_result(refused)
    assert parsed.outcome == "refused"
    assert parsed.confirmation_state == "refused"

    refused["events"] = [candidate()]
    with pytest.raises(EventResultError, match="cannot contain events"):
        parse_grounded_event_result(refused)

    refused["events"] = []
    refused["confirmationState"] = "pending"
    with pytest.raises(EventResultError) as caught:
        parse_grounded_event_result(refused)
    assert (caught.value.code, caught.value.detail) == (
        "result-invalid",
        "refused result state must be refused",
    )

    partial = parse_grounded_event_result(
        result_document(candidate(), outcome="partial", state="pending")
    )
    assert partial.outcome == "partial"
    assert partial.detail == "bounded partial result"

    for outcome in ("succeeded", "partial"):
        empty = result_document(outcome=outcome, state="pending")
        empty["events"] = []
        with pytest.raises(EventResultError) as caught:
            parse_grounded_event_result(empty)
        assert (caught.value.code, caught.value.detail) == (
            "result-invalid",
            "succeeded and partial results require an event",
        )


def test_model_output_can_never_confirm_or_reject_itself():
    for model_state in ("confirmed", "rejected"):
        with pytest.raises(EventResultError) as caught:
            parse_grounded_event_result(
                result_document(candidate(confirmation=model_state), state="pending")
            )
        assert caught.value.code == "confirmation-required"
        assert caught.value.detail == "model candidates must start pending"

    with pytest.raises(EventResultError) as caught:
        parse_grounded_event_result(result_document(candidate(), state="confirmed"))
    assert (caught.value.code, caught.value.detail) == (
        "confirmation-required",
        "model candidates must start pending",
    )


class Sink:
    def __init__(self):
        self.events = []

    def write(self, event):
        self.events.append(event)


def test_every_candidate_needs_one_human_decision_before_any_write():
    parsed = parse_grounded_event_result(
        result_document(candidate("keep"), candidate("drop", location="Room 3"))
    )
    sink = Sink()
    with pytest.raises(EventResultError, match="every candidate"):
        write_confirmed_events(parsed, sink)
    assert sink.events == []

    decided = confirm_event_candidates(parsed, ["keep"], rejected_ids=["drop"])
    assert decided.confirmation_state == "mixed"
    assert write_confirmed_events(decided, sink) == 1
    assert [item.candidate_id for item in sink.events] == ["keep"]


@pytest.mark.parametrize(
    ("confirmed", "rejected", "detail"),
    [
        (["event-1"], ["event-1"], "an event cannot be confirmed and rejected"),
        ([], [], "every candidate needs exactly one human decision"),
        (["missing"], [], "every candidate needs exactly one human decision"),
        (["event-1", "event-1"], [], "confirmed ids are invalid"),
        ([1], [], "confirmed ids are invalid"),
        ("event-1", [], "confirmed ids must be a sequence"),
    ],
)
def test_confirmation_decisions_are_closed_complete_and_disjoint(confirmed, rejected, detail):
    parsed = parse_grounded_event_result(result_document(candidate()))
    with pytest.raises(EventResultError) as caught:
        confirm_event_candidates(parsed, confirmed, rejected_ids=rejected)
    assert (caught.value.code, caught.value.detail) == ("confirmation-invalid", detail)


def test_refused_or_non_result_values_cannot_be_confirmed():
    refused_document = result_document(outcome="refused", state="refused")
    refused_document["events"] = []
    refused = parse_grounded_event_result(refused_document)
    for invalid in (refused, object()):
        with pytest.raises(EventResultError) as caught:
            confirm_event_candidates(invalid, [])
        assert (caught.value.code, caught.value.detail) == (
            "confirmation-invalid",
            "result cannot be confirmed",
        )


def test_ics_contains_only_confirmed_candidates_and_escapes_text():
    parsed = parse_grounded_event_result(
        result_document(
            candidate("keep", title="Review, plan; confirm", location="Room\\Two"),
            candidate("drop", location="Room 3"),
        )
    )
    decided = confirm_event_candidates(parsed, ["keep"], rejected_ids=["drop"])

    rendered = render_confirmed_ics(decided)

    assert rendered.startswith("BEGIN:VCALENDAR\r\nVERSION:2.0\r\n")
    assert "DTSTART:20260812T080000Z" in rendered
    assert "DTEND:20260812T090000Z" in rendered
    assert "SUMMARY:Review\\, plan\\; confirm" in rendered
    assert "LOCATION:Room\\\\Two" in rendered
    assert "Room 3" not in rendered
    lines = rendered.splitlines()
    assert lines.count("BEGIN:VEVENT") == 1
    assert lines.count("END:VEVENT") == 1
    [uid] = [line for line in lines if line.startswith("UID:")]
    assert uid.endswith("@omnitensor")
    assert len(uid.removeprefix("UID:").removesuffix("@omnitensor")) == 64
    assert "None" not in uid
    assert rendered.endswith("END:VCALENDAR\r\n")


def test_ics_omits_optional_end_and_location_and_reports_all_confirmed():
    parsed = parse_grounded_event_result(
        result_document(candidate(end=None, location=None))
    )
    decided = confirm_event_candidates(parsed, ["event-1"])
    assert decided.confirmation_state == "confirmed"
    rendered = render_confirmed_ics(decided)
    assert "DTEND:" not in rendered
    assert "LOCATION:" not in rendered


def test_no_rejected_candidate_is_a_valid_export():
    parsed = parse_grounded_event_result(result_document(candidate()))
    decided = confirm_event_candidates(parsed, [], rejected_ids=["event-1"])
    with pytest.raises(EventResultError) as caught:
        render_confirmed_ics(decided)
    assert (caught.value.code, caught.value.detail) == (
        "confirmation-required",
        "no candidate was confirmed",
    )


def test_render_refuses_values_that_are_not_event_results():
    with pytest.raises(EventResultError) as caught:
        render_confirmed_ics(object())
    assert (caught.value.code, caught.value.detail) == (
        "confirmation-required",
        "event result is invalid",
    )


def test_calendar_sink_must_implement_the_port():
    parsed = parse_grounded_event_result(result_document(candidate()))
    decided = confirm_event_candidates(parsed, ["event-1"])
    with pytest.raises(EventResultError) as caught:
        write_confirmed_events(decided, object())
    assert (caught.value.code, caught.value.detail) == (
        "calendar-invalid",
        "sink does not implement the calendar port",
    )


def test_bounded_extraction_pages_become_private_source_fragments():
    extraction = ExtractionResult(
        ExtractionOutcome.TRUNCATED,
        (
            NormalisedPage(1, "Untrusted meeting text", (), ()),
            NormalisedPage(2, "More text", (), ()),
        ),
    )
    fragments = source_fragments(extraction, "private:document-1", "a" * 64)

    assert [item.reference for item in fragments] == [
        "private:document-1:page:1",
        "private:document-1:page:2",
    ]
    assert [item.text for item in fragments] == ["Untrusted meeting text", "More text"]
    assert [item.page for item in fragments] == [1, 2]
    assert [item.source_sha256 for item in fragments] == ["a" * 64, "a" * 64]
    assert all(len(item.text_sha256) == 64 for item in fragments)


@pytest.mark.parametrize(
    ("extraction", "source_ref", "digest", "code", "detail"),
    [
        (
            object(),
            "private:one",
            "a" * 64,
            "source-invalid",
            "extraction must be an ExtractionResult",
        ),
        (
            ExtractionResult(ExtractionOutcome.REFUSED),
            "private:one",
            "a" * 64,
            "source-refused",
            "extraction produced no usable pages",
        ),
        (
            ExtractionResult(ExtractionOutcome.SUCCEEDED),
            "private:one",
            "a" * 64,
            "source-refused",
            "extraction produced no usable pages",
        ),
        (
            ExtractionResult(
                ExtractionOutcome.SUCCEEDED, (NormalisedPage(1, "x", (), ()),)
            ),
            "/home/user/file",
            "a" * 64,
            "source-invalid",
            "source reference must be opaque and private",
        ),
        (
            ExtractionResult(
                ExtractionOutcome.SUCCEEDED, (NormalisedPage(1, "x", (), ()),)
            ),
            "private:one",
            "A" * 64,
            "source-invalid",
            "source digest must be lower-case SHA-256",
        ),
    ],
)
def test_source_fragments_refuse_unusable_or_untrusted_inputs(
    extraction, source_ref, digest, code, detail
):
    with pytest.raises(EventResultError) as caught:
        source_fragments(extraction, source_ref, digest)
    assert (caught.value.code, caught.value.detail) == (code, detail)


@given(
    hour=st.integers(min_value=0, max_value=22),
    duration=st.integers(min_value=1, max_value=59),
    title=st.text(
        alphabet=st.characters(blacklist_categories=("Cc", "Cs")),
        min_size=1,
        max_size=40,
    ),
)
def test_property_valid_utc_candidates_round_trip_and_require_confirmation(
    hour, duration, title
):
    start = f"2026-01-10T{hour:02d}:00:00+00:00"
    end = f"2026-01-10T{hour:02d}:{duration:02d}:00+00:00"
    document = result_document(
        candidate(title=title, start=start, end=end, timezone="UTC", location=None)
    )
    parsed = parse_grounded_event_result(document)
    assert parsed.events[0].title == title
    with pytest.raises(EventResultError, match="human decision"):
        render_confirmed_ics(parsed)


def test_parser_does_not_mutate_provider_documents():
    document = result_document(candidate())
    before = copy.deepcopy(document)
    parse_grounded_event_result(document)
    assert document == before
