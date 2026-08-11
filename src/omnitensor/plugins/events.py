"""Grounded event candidates that cannot write before human confirmation.

Generation workers may propose dates, but they do not own calendar effects.
This module validates their closed result, resolves local wall times against a
named timezone and explicit offset, de-duplicates deterministically, and resets
every model-provided candidate to ``pending``.  Only a new result created by
``confirm_event_candidates`` can reach an ICS renderer or calendar sink.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from omnitensor.registry import validate_document

from .extraction import ExtractionOutcome, ExtractionResult

MAX_EVENTS = 64
MAX_EVIDENCE_PER_EVENT = 16
MAX_SOURCE_FRAGMENTS = 2_000
DUPLICATE_POLICY = "keep-first-title-start-location"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
_PRIVATE_REFERENCE = re.compile(r"^private:[A-Za-z0-9._:-]{1,200}$")
_DIGEST = re.compile(r"^[a-f0-9]{64}$")


class EventResultError(ValueError):
    """Stable event-contract or confirmation refusal."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    source_ref: str
    source_sha256: str
    page: int | None
    span_start: int
    span_end: int
    text_sha256: str


@dataclass(frozen=True, slots=True)
class EventCandidate:
    candidate_id: str
    title: str
    start: datetime
    end: datetime | None
    timezone: str
    location: str | None
    confirmation: str
    evidence: tuple[SourceEvidence, ...]


@dataclass(frozen=True, slots=True)
class GroundedEventResult:
    request_id: str
    outcome: str
    code: str
    detail: str
    events: tuple[EventCandidate, ...]
    duplicates_dropped: int = 0

    @property
    def confirmation_state(self) -> str:
        if self.outcome == "refused":
            return "refused"
        states = {event.confirmation for event in self.events}
        if not states or states == {"pending"}:
            return "pending"
        if states == {"confirmed"}:
            return "confirmed"
        return "mixed"


@dataclass(frozen=True, slots=True)
class SourceFragment:
    """Private prompt material retained inside the isolated worker."""

    reference: str
    source_sha256: str
    page: int
    text: str
    text_sha256: str


@runtime_checkable
class CalendarSink(Protocol):
    """A side-effect port implemented by an explicitly invoked client."""

    def write(self, event: EventCandidate) -> None: ...


def parse_grounded_event_result(document: object) -> GroundedEventResult:
    """Validate one model reply and reset all candidates to pending.

    A model cannot confirm its own suggestion: incoming ``confirmed`` or
    ``rejected`` values are refused rather than trusted or silently rewritten.
    """
    violations = validate_document("event-extraction-result.schema.json", document)
    if violations:
        raise EventResultError("result-invalid", violations[0])
    assert isinstance(document, Mapping)
    outcome, raw_events = _result_state(document)

    events: list[EventCandidate] = []
    ids: set[str] = set()
    keys: set[tuple[tuple[str, ...], datetime, tuple[str, ...]]] = set()
    duplicates = 0
    for raw_event in raw_events:
        assert isinstance(raw_event, Mapping)
        if raw_event["confirmation"] != "pending":
            raise EventResultError("confirmation-required", "model candidates must start pending")
        event = _parse_event(raw_event)
        if event.candidate_id in ids:
            raise EventResultError("result-invalid", "candidate ids must be unique")
        ids.add(event.candidate_id)
        key = _duplicate_key(event)
        if key in keys:
            duplicates += 1
            continue
        keys.add(key)
        events.append(event)
    return GroundedEventResult(
        str(document["requestId"]),
        outcome,
        str(document["code"]),
        str(document["detail"]),
        tuple(events),
        duplicates,
    )


def confirm_event_candidates(
    result: GroundedEventResult,
    confirmed_ids: Sequence[str],
    *,
    rejected_ids: Sequence[str] = (),
) -> GroundedEventResult:
    """Apply one explicit human decision to every pending candidate."""
    if not isinstance(result, GroundedEventResult) or result.outcome == "refused":
        raise EventResultError("confirmation-invalid", "result cannot be confirmed")
    confirmed = _decision_ids(confirmed_ids, "confirmed ids")
    rejected = _decision_ids(rejected_ids, "rejected ids")
    if confirmed & rejected:
        raise EventResultError("confirmation-invalid", "an event cannot be confirmed and rejected")
    known = {event.candidate_id for event in result.events}
    if confirmed | rejected != known:
        raise EventResultError(
            "confirmation-invalid", "every candidate needs exactly one human decision"
        )
    events = tuple(
        replace(
            event,
            confirmation="confirmed" if event.candidate_id in confirmed else "rejected",
        )
        for event in result.events
    )
    return replace(result, events=events)


def write_confirmed_events(result: GroundedEventResult, sink: CalendarSink) -> int:
    """Write confirmed candidates only; pending model output never reaches a sink."""
    if not isinstance(sink, CalendarSink):
        raise EventResultError("calendar-invalid", "sink does not implement the calendar port")
    confirmed = _confirmed_events(result)
    for event in confirmed:
        sink.write(event)
    return len(confirmed)


def render_confirmed_ics(result: GroundedEventResult) -> str:
    """Render confirmed candidates as a bounded RFC 5545 calendar document."""
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//OmniTensor//Events 1//EN"]
    for event in _confirmed_events(result):
        lines.extend(_ics_event(event))
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


def source_fragments(
    extraction: ExtractionResult,
    source_ref: str,
    source_sha256: str,
) -> tuple[SourceFragment, ...]:
    """Turn bounded extraction pages into private, evidence-addressable fragments."""
    if not isinstance(extraction, ExtractionResult):
        raise EventResultError("source-invalid", "extraction must be an ExtractionResult")
    _validate_source_identity(source_ref, source_sha256)
    if extraction.outcome is ExtractionOutcome.REFUSED or not extraction.pages:
        raise EventResultError("source-refused", "extraction produced no usable pages")
    fragments = []
    for page in extraction.pages[:MAX_SOURCE_FRAGMENTS]:
        encoded = page.text.encode("utf-8")
        fragments.append(
            SourceFragment(
                f"{source_ref}:page:{page.page_number}",
                source_sha256,
                page.page_number,
                page.text,
                hashlib.sha256(encoded).hexdigest(),
            )
        )
    return tuple(fragments)


def _parse_event(document: Mapping[str, object]) -> EventCandidate:
    start = _event_datetime(document["start"], str(document["timezone"]), "start")
    raw_end = document["end"]
    end = None if raw_end is None else _event_datetime(raw_end, str(document["timezone"]), "end")
    if end is not None and end <= start:
        raise EventResultError("date-invalid", "event end must be after start")
    evidence = tuple(_parse_evidence(item) for item in document["evidence"])
    return EventCandidate(
        str(document["candidateId"]),
        str(document["title"]),
        start,
        end,
        str(document["timezone"]),
        None if document["location"] is None else str(document["location"]),
        "pending",
        evidence,
    )


def _result_state(document: Mapping[str, object]) -> tuple[str, Sequence[object]]:
    outcome = str(document["outcome"])
    raw_events = document["events"]
    assert isinstance(raw_events, Sequence)
    state = document["confirmationState"]
    if outcome == "refused":
        if raw_events:
            raise EventResultError("result-invalid", "refused results cannot contain events")
        if state != "refused":
            raise EventResultError("result-invalid", "refused result state must be refused")
    elif state != "pending":
        raise EventResultError("confirmation-required", "model candidates must start pending")
    return outcome, raw_events


def _event_datetime(value: str, timezone: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise EventResultError("date-invalid", f"event {label} is not ISO 8601") from error
    if timezone == "floating":
        if parsed.tzinfo is not None:
            raise EventResultError("date-invalid", f"floating {label} must omit a UTC offset")
        return parsed
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EventResultError("date-invalid", f"zoned {label} must include a UTC offset")
    try:
        zone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError as error:
        raise EventResultError("timezone-invalid", f"unknown timezone: {timezone}") from error
    local = parsed.astimezone(zone)
    if local.replace(tzinfo=None) != parsed.replace(tzinfo=None):
        raise EventResultError("date-invalid", f"event {label} offset disagrees with timezone")
    return parsed


def _parse_evidence(document: object) -> SourceEvidence:
    assert isinstance(document, Mapping)
    source_ref = str(document["sourceRef"])
    source_sha256 = str(document["sourceSha256"])
    _validate_source_identity(source_ref, source_sha256)
    span = document["span"]
    assert isinstance(span, Mapping)
    start = int(span["start"])
    end = int(span["end"])
    if end <= start:
        raise EventResultError("evidence-invalid", "evidence span end must be after start")
    return SourceEvidence(
        source_ref,
        source_sha256,
        None if document["page"] is None else int(document["page"]),
        start,
        end,
        str(document["textSha256"]),
    )


def _validate_source_identity(source_ref: str, source_sha256: str) -> None:
    if _PRIVATE_REFERENCE.fullmatch(source_ref) is None:
        raise EventResultError("source-invalid", "source reference must be opaque and private")
    if _DIGEST.fullmatch(source_sha256) is None:
        raise EventResultError("source-invalid", "source digest must be lower-case SHA-256")


def _duplicate_key(event: EventCandidate) -> tuple[tuple[str, ...], datetime, tuple[str, ...]]:
    title = tuple(event.title.casefold().split())
    location = () if event.location is None else tuple(event.location.casefold().split())
    return title, event.start, location


def _decision_ids(values: Sequence[str], label: str) -> set[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise EventResultError("confirmation-invalid", f"{label} must be a sequence")
    parsed = set(values)
    if len(parsed) != len(values) or any(
        not isinstance(value, str) or _REQUEST_ID.fullmatch(value) is None for value in parsed
    ):
        raise EventResultError("confirmation-invalid", f"{label} are invalid")
    return parsed


def _confirmed_events(result: GroundedEventResult) -> tuple[EventCandidate, ...]:
    if not isinstance(result, GroundedEventResult):
        raise EventResultError("confirmation-required", "event result is invalid")
    if any(event.confirmation == "pending" for event in result.events):
        raise EventResultError("confirmation-required", "every candidate needs a human decision")
    confirmed = tuple(event for event in result.events if event.confirmation == "confirmed")
    if not confirmed:
        raise EventResultError("confirmation-required", "no candidate was confirmed")
    if any(event.timezone == "floating" for event in confirmed):
        raise EventResultError("timezone-required", "confirmed events need a named timezone")
    return confirmed


def _ics_event(event: EventCandidate) -> tuple[str, ...]:
    assert event.timezone != "floating"
    start = event.start.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    end = event.end.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ") if event.end else None
    identity = hashlib.sha256(
        f"{event.candidate_id}\0{event.start.isoformat()}".encode()
    ).hexdigest()
    lines = ["BEGIN:VEVENT", f"UID:{identity}@omnitensor", f"DTSTART:{start}"]
    if end is not None:
        lines.append(f"DTEND:{end}")
    lines.append(f"SUMMARY:{_ics_text(event.title)}")
    if event.location is not None:
        lines.append(f"LOCATION:{_ics_text(event.location)}")
    lines.append("END:VEVENT")
    return tuple(lines)


def _ics_text(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")
    return escaped.replace("\r", "").replace("\n", "\\n")


__all__ = [
    "CalendarSink",
    "DUPLICATE_POLICY",
    "EventCandidate",
    "EventResultError",
    "GroundedEventResult",
    "SourceEvidence",
    "SourceFragment",
    "confirm_event_candidates",
    "parse_grounded_event_result",
    "render_confirmed_ics",
    "source_fragments",
    "write_confirmed_events",
]
