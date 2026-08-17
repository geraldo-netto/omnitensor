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
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from omnitensor.registry import validate_document

from .extraction import ExtractionOutcome, ExtractionResult

DUPLICATE_POLICY = "keep-first-label-date-time-place"
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
    # `text` or `ocr`. An image never reaches the model as pixels — it goes
    # through PyMuPDF and, when a page has no text layer, Tesseract — so a span
    # from a banner addresses OCR output and is a weaker citation than a span
    # from an email body. Saying which is the difference between a citation a
    # person can trust and one they should read twice.
    read_as: str = "text"


@dataclass(frozen=True, slots=True)
class Stated:
    """A value and where it came from: read, summarised, or worked out."""

    text: str
    source: str


@dataclass(frozen=True, slots=True)
class EventTimezone:
    name: str
    utc_offset: str | None
    source: str


@dataclass(frozen=True, slots=True)
class EventWhen:
    """When an event happens, including the parts the source never said.

    Version 1 demanded a full datetime and a named zone, so *"Jazz night,
    Fridays, Casa da Música"* had nowhere to go and came back as a refusal.
    Here the date and the time are separate and either may be absent, and
    ``needs`` says what a person still has to supply.
    """

    date: str | None = None
    time: str | None = None
    end_date: str | None = None
    end_time: str | None = None
    all_day: bool = False
    timezone: EventTimezone | None = None
    is_past: bool = False

    @property
    def needs(self) -> tuple[str, ...]:
        """What blocks placing this in a calendar — never what merely lacks detail.

        A missing location stops nothing, so it is not listed. A missing zone
        does not stop a calendar either, strictly: it is exported as a floating
        time, which every client reads as local. It is listed because events
        arriving from three countries make "local" the wrong guess often enough
        that a person should be asked.
        """
        missing = []
        if self.date is None:
            missing.append("date")
        if self.time is None and not self.all_day:
            missing.append("time")
        if self.timezone is None and self.time is not None:
            missing.append("timezone")
        return tuple(missing)

    @property
    def placeable(self) -> bool:
        """Whether a calendar can put this somewhere without asking."""
        return self.date is not None


@dataclass(frozen=True, slots=True)
class Recurrence:
    """An RFC 5545 rule, and the words it was read from.

    ``text`` exists so a person can check the rule without reconstructing
    RRULE in their head, and the source is always ``stated``: a recurrence
    inferred from two dates a week apart is how a calendar acquires an event
    every Tuesday for a year.
    """

    freq: str
    text: str
    interval: int = 1
    by_day: tuple[str, ...] = ()
    by_month_day: tuple[int, ...] = ()
    by_month: tuple[int, ...] = ()
    by_set_pos: tuple[int, ...] = ()
    count: int | None = None
    until: str | None = None
    exceptions: tuple[str, ...] = ()
    source: str = "stated"


@dataclass(frozen=True, slots=True)
class PostalAddress:
    full: str
    street: str | None = None
    postal_code: str | None = None
    city: str | None = None
    region: str | None = None
    country: str | None = None


@dataclass(frozen=True, slots=True)
class EventPlace:
    kind: str
    source: str
    venue: str | None = None
    address: PostalAddress | None = None
    url: str | None = None
    joining: str | None = None


@dataclass(frozen=True, slots=True)
class Person:
    name: str
    source: str
    email: str | None = None


@dataclass(frozen=True, slots=True)
class EventPeople:
    organiser: Person | None = None
    participants: tuple[Person, ...] = ()


@dataclass(frozen=True, slots=True)
class Cost:
    amount: str
    currency: str


@dataclass(frozen=True, slots=True)
class Registration:
    source: str
    required: bool = False
    deadline: str | None = None
    url: str | None = None
    cost: Cost | None = None


@dataclass(frozen=True, slots=True)
class Contact:
    source: str
    email: str | None = None
    phone: str | None = None


@dataclass(frozen=True, slots=True)
class EventCandidate:
    candidate_id: str
    label: Stated
    when: EventWhen
    confirmation: str
    evidence: tuple[SourceEvidence, ...]
    what: Stated | None = None
    repeats: Recurrence | None = None
    where: EventPlace | None = None
    people: EventPeople | None = None
    status: str = "scheduled"
    registration: Registration | None = None
    contact: Contact | None = None
    reference: str | None = None
    language: str | None = None


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

    @property
    def unplaceable(self) -> tuple[EventCandidate, ...]:
        """Events a calendar cannot place until somebody says more.

        Kept as a property rather than a separate list so nothing has to
        remember to recompute it: an event is unplaceable exactly while its
        `when` is missing a date.
        """
        return tuple(event for event in self.events if not event.when.placeable)


@dataclass(frozen=True, slots=True)
class SourceFragment:
    """Private prompt material retained inside the isolated worker."""

    reference: str
    source_sha256: str
    page: int
    text: str
    text_sha256: str




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
    # Every page, not the first two thousand: a source truncated here
    # loses events nobody is told about.
    for page in extraction.pages:
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
    when = _parse_when(document["when"])
    evidence = tuple(_parse_evidence(item) for item in document["evidence"])
    return EventCandidate(
        candidate_id=str(document["candidateId"]),
        label=_parse_stated(document["label"], "label"),
        when=when,
        confirmation="pending",
        evidence=evidence,
        what=_optional(document.get("what"), lambda value: _parse_stated(value, "what")),
        repeats=_optional(document.get("repeats"), _parse_recurrence),
        where=_optional(document.get("where"), _parse_place),
        people=_optional(document.get("people"), _parse_people),
        status=str(document.get("status", "scheduled")),
        registration=_optional(document.get("registration"), _parse_registration),
        contact=_optional(document.get("contact"), _parse_contact),
        reference=_optional(document.get("reference"), str),
        language=_optional(document.get("language"), str),
    )


def _optional(value, parse):
    return None if value is None else parse(value)


def _parse_stated(value: object, label: str) -> Stated:
    if not isinstance(value, Mapping):
        raise EventResultError("result-invalid", f"event {label} is not a stated value")
    return Stated(str(value["text"]), str(value["source"]))


def _parse_when(value: object) -> EventWhen:
    if not isinstance(value, Mapping):
        raise EventResultError("result-invalid", "event when is not an object")
    end = value.get("end") or {}
    if not isinstance(end, Mapping):
        raise EventResultError("result-invalid", "event end is not an object")
    stated_date = value.get("date")
    when = EventWhen(
        # `unknown` is how a source that names no date says so. The key is
        # required — a model given the option of omitting it filled exactly one
        # of date and time on 8 of 8 events — so the absence has to be sayable.
        date=None if stated_date in (None, "unknown") else str(stated_date),
        time=_optional(value.get("time"), str),
        end_date=_optional(end.get("date"), str),
        end_time=_optional(end.get("time"), str),
        all_day=bool(value.get("allDay", False)),
        timezone=_optional(value.get("timezone"), _parse_timezone),
        is_past=bool(value.get("isPast", False)),
    )
    _check_when(when, value)
    return when


def _check_when(when: EventWhen, _value: Mapping[str, object]) -> None:
    """Check what the model cannot recompute, and recompute the rest.

    `needs` is a pure function of the date, time, zone and all-day flag, so the
    model's copy carries nothing the parser cannot derive. It used to be
    compared and a disagreement refused the whole job — and the model disagrees
    almost always: measured, it reported `[]` beside a missing date seven times
    and listed three missing fields beside a present one once. Refusing on a
    redundant field turned an answer somebody could use into no answer at all,
    so the derived value simply wins.
    """
    if when.all_day and when.time is not None:
        raise EventResultError("date-invalid", "an all-day event states no time")
    if when.end_date is not None and when.date is not None and when.end_date < when.date:
        raise EventResultError("date-invalid", "event end date precedes its start")
    if (
        when.end_time is not None
        and when.time is not None
        and when.end_date in (None, when.date)
        and when.end_time <= when.time
    ):
        raise EventResultError("date-invalid", "event end time is not after its start")
    if when.timezone is not None and when.timezone.name != "floating":
        try:
            ZoneInfo(when.timezone.name)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise EventResultError(
                "timezone-invalid", f"unknown timezone: {when.timezone.name}"
            ) from error
    if when.timezone is not None and when.timezone.utc_offset is not None and when.date is None:
        # The offset depends on the date — Europe/Lisbon is +00:00 in January
        # and +01:00 in July — so one without a date is a fact with a hole in it.
        raise EventResultError("date-invalid", "a UTC offset without a date is not determinate")


def _parse_timezone(value: object) -> EventTimezone:
    if not isinstance(value, Mapping):
        raise EventResultError("result-invalid", "event timezone is not an object")
    return EventTimezone(
        str(value["name"]),
        _optional(value.get("utcOffset"), str),
        str(value["source"]),
    )


def _parse_recurrence(value: object) -> Recurrence:
    if not isinstance(value, Mapping):
        raise EventResultError("result-invalid", "event recurrence is not an object")
    if value.get("count") is not None and value.get("until") is not None:
        raise EventResultError("date-invalid", "a recurrence states COUNT or UNTIL, never both")
    return Recurrence(
        freq=str(value["freq"]),
        text=str(value["text"]),
        interval=int(value.get("interval", 1)),
        by_day=tuple(str(item) for item in value.get("byDay", ())),
        by_month_day=tuple(int(item) for item in value.get("byMonthDay", ())),
        by_month=tuple(int(item) for item in value.get("byMonth", ())),
        by_set_pos=tuple(int(item) for item in value.get("bySetPos", ())),
        count=_optional(value.get("count"), int),
        until=_optional(value.get("until"), str),
        exceptions=tuple(str(item) for item in value.get("exceptions", ())),
        source=str(value.get("source", "stated")),
    )


def _parse_place(value: object) -> EventPlace:
    if not isinstance(value, Mapping):
        raise EventResultError("result-invalid", "event place is not an object")
    kind = str(value["kind"])
    place = EventPlace(
        kind=kind,
        source=str(value["source"]),
        venue=_optional(value.get("venue"), str),
        address=_optional(value.get("address"), _parse_address),
        url=_optional(value.get("url"), str),
        joining=_optional(value.get("joining"), str),
    )
    if kind in {"online", "hybrid"} and place.url is None:
        raise EventResultError("place-invalid", "an online event states where to join")
    if kind == "physical" and place.address is None and place.venue is None:
        raise EventResultError("place-invalid", "a physical event states where it is")
    if kind == "unknown" and (place.url or place.address or place.venue):
        raise EventResultError("place-invalid", "an unknown place states nothing")
    return place


def _parse_address(value: object) -> PostalAddress:
    if not isinstance(value, Mapping):
        raise EventResultError("result-invalid", "event address is not an object")
    return PostalAddress(
        full=str(value["full"]),
        street=_optional(value.get("street"), str),
        postal_code=_optional(value.get("postalCode"), str),
        city=_optional(value.get("city"), str),
        region=_optional(value.get("region"), str),
        country=_optional(value.get("country"), str),
    )


def _parse_people(value: object) -> EventPeople:
    if not isinstance(value, Mapping):
        raise EventResultError("result-invalid", "event people is not an object")
    return EventPeople(
        organiser=_optional(value.get("organiser"), _parse_person),
        participants=tuple(_parse_person(item) for item in value.get("participants", ())),
    )


def _parse_person(value: object) -> Person:
    if not isinstance(value, Mapping):
        raise EventResultError("result-invalid", "event person is not an object")
    return Person(str(value["name"]), str(value["source"]), _optional(value.get("email"), str))


def _parse_registration(value: object) -> Registration:
    if not isinstance(value, Mapping):
        raise EventResultError("result-invalid", "event registration is not an object")
    cost = value.get("cost")
    return Registration(
        source=str(value["source"]),
        required=bool(value.get("required", False)),
        deadline=_optional(value.get("deadline"), str),
        url=_optional(value.get("url"), str),
        cost=None if cost is None else Cost(str(cost["amount"]), str(cost["currency"])),
    )


def _parse_contact(value: object) -> Contact:
    if not isinstance(value, Mapping):
        raise EventResultError("result-invalid", "event contact is not an object")
    return Contact(
        source=str(value["source"]),
        email=_optional(value.get("email"), str),
        phone=_optional(value.get("phone"), str),
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
    else:
        if not raw_events:
            raise EventResultError(
                "result-invalid", "succeeded and partial results require an event"
            )
        if state != "pending":
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


def _duplicate_key(event: EventCandidate) -> tuple:
    """What makes two extractions the same event.

    Label, date, time and place, with absent compared as absent. A recurring
    meeting named on three pages is one event; "3 September" and "3 September
    at 14:00" are two, and merging them would silently choose one time over
    none.
    """
    label = tuple(event.label.text.casefold().split())
    place = ()
    if event.where is not None:
        stated = event.where.address.full if event.where.address else event.where.venue
        stated = stated or event.where.url
        place = () if stated is None else tuple(stated.casefold().split())
    return label, event.when.date, event.when.time, place
















__all__ = [
    "DUPLICATE_POLICY",
    "EventCandidate",
    "EventResultError",
    "GroundedEventResult",
    "SourceEvidence",
    "SourceFragment",
    "parse_grounded_event_result",
    "source_fragments",
]
