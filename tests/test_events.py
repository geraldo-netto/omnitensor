from __future__ import annotations

import copy

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins.event_confirmation import (
    confirm_event_candidates,
    render_confirmed_ics,
    write_confirmed_events,
)
from omnitensor.plugins.events import (
    DUPLICATE_POLICY,
    EventResultError,
    parse_grounded_event_result,
    source_fragments,
)
from omnitensor.plugins.extraction import (
    ExtractionOutcome,
    ExtractionResult,
    NormalisedPage,
)
from omnitensor.registry import validate_document


def evidence(*, page=1, start=0, end=12, source_ref="private:document-1", read_as="text") -> dict:
    return {
        "sourceRef": source_ref,
        "sourceSha256": "a" * 64,
        "page": page,
        "span": {"start": start, "end": end},
        "textSha256": "b" * 64,
        "readAs": read_as,
    }


def when(
    *,
    date="2026-08-12",
    time="10:00",
    end_time="11:00",
    timezone="Europe/Rome",
    all_day=False,
) -> dict:
    """A `when` whose `needs` is derived exactly as the parser derives it."""
    value: dict = {"allDay": all_day}
    if date is not None:
        value["date"] = date
    if time is not None:
        value["time"] = time
    if end_time is not None:
        value["end"] = {"time": end_time}
    if timezone is not None:
        value["timezone"] = {
            "name": timezone,
            "utcOffset": "+02:00" if date is not None else None,
            "source": "stated",
        }
    missing = []
    if date is None:
        missing.append("date")
    if time is None and not all_day:
        missing.append("time")
    if timezone is None and time is not None:
        missing.append("timezone")
    value["needs"] = missing
    return value


def candidate(
    candidate_id="event-1",
    *,
    title="Release planning",
    location="Room 2",
    confirmation="pending",
    **over,
) -> dict:
    event = {
        "candidateId": candidate_id,
        "label": {"text": title, "source": "stated"},
        "when": over.pop("when", when()),
        "confirmation": confirmation,
        "evidence": [evidence()],
    }
    if location is not None:
        event["where"] = {"kind": "physical", "venue": location, "source": "stated"}
    event.update(over)
    return event


def result_document(*events, outcome="succeeded", state="pending") -> dict:
    return {
        "version": 2,
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
    assert event.label.text == "Release planning"
    assert (event.when.date, event.when.time) == ("2026-08-12", "10:00")
    assert event.when.end_time == "11:00"
    assert event.when.timezone.name == "Europe/Rome"
    assert event.when.needs == ()
    assert event.when.placeable is True
    assert event.where.venue == "Room 2"
    [proof] = event.evidence
    assert proof.read_as == "text"
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
    later = candidate("later", when=when(time="12:00", end_time=None))
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
            {"when": {"date": "2026-08-12", "time": "10:00", "needs": []}},
            "date-invalid",
            "event needs ['timezone'] rather than []",
        ),
        (
            {"when": when(timezone="Mars/Olympus")},
            "timezone-invalid",
            "unknown timezone: Mars/Olympus",
        ),
        (
            {"when": {**when(), "end": {"time": "09:00"}}},
            "date-invalid",
            "event end time is not after its start",
        ),
        (
            {"when": {**when(time=None, all_day=True), "time": "10:00"}},
            "date-invalid",
            "an all-day event states no time",
        ),
        (
            {
                "when": {
                    "time": "10:00",
                    "needs": ["date"],
                    "timezone": {"name": "UTC", "utcOffset": "+00:00", "source": "stated"},
                }
            },
            "date-invalid",
            "a UTC offset without a date is not determinate",
        ),
    ],
)
def test_when_semantics_fail_closed(changes, code, detail):
    document = result_document(candidate(**changes))

    with pytest.raises(EventResultError) as caught:
        parse_grounded_event_result(document)

    assert (caught.value.code, caught.value.detail) == (code, detail)


def test_an_event_missing_part_of_its_when_is_kept_and_says_what_it_needs():
    # The whole point of version 2. A banner that names a night and a place but
    # no year is an event somebody wants; version 1 called it a refusal.
    partial = candidate(when=when(date=None, timezone=None))

    [event] = parse_grounded_event_result(result_document(partial)).events

    assert event.when.needs == ("date", "timezone")
    assert event.when.placeable is False
    assert event.label.text == "Release planning"


def test_an_all_day_event_needs_no_time():
    [event] = parse_grounded_event_result(
        result_document(candidate(when=when(time=None, end_time=None, all_day=True)))
    ).events

    assert event.when.needs == ()
    assert event.when.placeable is True


def test_evidence_requires_a_forward_span_and_an_opaque_source():
    for changed, expected in (
        ({"span": {"start": 5, "end": 5}}, "evidence span end must be after start"),
        # v1 refused this in the schema; v2 lets the reference be any opaque
        # string and refuses it where the rule lives, which says why.
        ({"sourceRef": "file:///home/user/private.pdf"}, "opaque and private"),
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


def test_a_long_programme_is_parsed_whole():
    # There was a ceiling here: 64 events and 16 evidence entries, refused as
    # "result-invalid" above that. A conference programme with a hundred talks
    # is not an invalid result, and telling somebody it is loses the talks.
    events = [candidate(f"event-{index}", title=f"Talk {index}") for index in range(250)]
    events[0]["evidence"] = [evidence() for _ in range(64)]

    parsed = parse_grounded_event_result(result_document(*events))

    assert len(parsed.events) == 250
    assert len(parsed.events[0].evidence) == 64


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
    # A zoned event keeps its zone rather than being flattened to UTC: the
    # calendar receiving it knows what Europe/Rome means on that date, and a
    # recurrence resolved in UTC would drift across a daylight-saving change.
    assert "DTSTART;TZID=Europe/Rome:20260812T100000" in rendered
    assert "DTEND;TZID=Europe/Rome:20260812T110000" in rendered
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
        result_document(candidate(when=when(end_time=None), location=None))
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
            ExtractionResult(ExtractionOutcome.SUCCEEDED, (NormalisedPage(1, "x", (), ()),)),
            "/home/user/file",
            "a" * 64,
            "source-invalid",
            "source reference must be opaque and private",
        ),
        (
            ExtractionResult(ExtractionOutcome.SUCCEEDED, (NormalisedPage(1, "x", (), ()),)),
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
def test_property_valid_utc_candidates_round_trip_and_require_confirmation(hour, duration, title):
    document = result_document(
        candidate(
            title=title,
            when=when(
                date="2026-01-10",
                time=f"{hour:02d}:00",
                end_time=f"{hour:02d}:{duration:02d}",
                timezone="UTC",
            ),
            location=None,
        )
    )
    parsed = parse_grounded_event_result(document)
    assert parsed.events[0].label.text == title
    with pytest.raises(EventResultError, match="human decision"):
        render_confirmed_ics(parsed)


def test_parser_does_not_mutate_provider_documents():
    document = result_document(candidate())
    before = copy.deepcopy(document)
    parse_grounded_event_result(document)
    assert document == before
