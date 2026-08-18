"""Rendering one confirmed event as RFC 5545 lines.

A serializer for a single destination, kept apart from the contract it reads
and from the gate that decides what may reach it. It is also the part that
grows — alarms, attachments, categories — and none of that is a reason to
touch how a model's reply is parsed.

It decides nothing: `calendar_component` is reachable only through
`event_confirmation`, which is what makes "no model ever put an event in
somebody's calendar" a property rather than a hope.
"""

from __future__ import annotations

import hashlib

from .events import EventCandidate, EventTimezone, Recurrence


def calendar_component(event: EventCandidate) -> tuple[str, ...]:
    """One calendar component, in whichever form the event can honestly take.

    RFC 5545 already has a container for every case, so nothing has to be
    invented and nothing has to be withheld:

    - a date and a time with a zone becomes a zoned ``DTSTART``;
    - a date and a time with no zone becomes a *floating* ``DTSTART``, which
      every client reads as local — exactly "the source did not say, so use
      mine", and the reason an event from a Brazilian invitation does not
      arrive three hours out;
    - a date with no time becomes an all-day ``DTSTART;VALUE=DATE``;
    - no date at all cannot be a ``VEVENT``, because ``DTSTART`` is required
      without a ``METHOD``, so it becomes a ``VTODO`` with no ``DUE`` — a task
      that says decide when, kept off the calendar grid until somebody does.
    """
    identity = hashlib.sha256(
        f"{event.candidate_id}\0{event.when.date or ''}\0{event.when.time or ''}".encode()
    ).hexdigest()
    if not event.when.placeable:
        lines = [
            "BEGIN:VTODO",
            f"UID:{identity}@omnitensor",
            f"SUMMARY:{_ics_text(event.label.text)}",
        ]
        lines.extend(_ics_common(event))
        lines.append("END:VTODO")
        return tuple(lines)

    lines = ["BEGIN:VEVENT", f"UID:{identity}@omnitensor"]
    lines.append(_ics_moment("DTSTART", event.when.date, event.when.time, event.when.timezone))
    end_date = event.when.end_date or (event.when.date if event.when.end_time else None)
    if end_date is not None:
        lines.append(_ics_moment("DTEND", end_date, event.when.end_time, event.when.timezone))
    lines.append(f"SUMMARY:{_ics_text(event.label.text)}")
    if event.repeats is not None:
        lines.append(f"RRULE:{_ics_rrule(event.repeats)}")
        for excluded in event.repeats.exceptions:
            lines.append(f"EXDATE;VALUE=DATE:{excluded.replace('-', '')}")
    lines.extend(_ics_common(event))
    lines.append("END:VEVENT")
    return tuple(lines)


def _ics_common(event: EventCandidate) -> tuple[str, ...]:
    """The parts a task and an event describe the same way."""
    lines = []
    if event.what is not None:
        lines.append(f"DESCRIPTION:{_ics_text(event.what.text)}")
    if event.where is not None:
        place = event.where
        stated = place.address.full if place.address else place.venue
        if stated is not None:
            lines.append(f"LOCATION:{_ics_text(stated)}")
        if place.url is not None:
            lines.append(f"URL:{_ics_text(place.url)}")
    if event.people is not None and event.people.organiser is not None:
        organiser = event.people.organiser
        address = f";MAILTO:{organiser.email}" if organiser.email else ""
        lines.append(f"ORGANIZER;CN={_ics_text(organiser.name)}{address}")
    for participant in () if event.people is None else event.people.participants:
        address = f":MAILTO:{participant.email}" if participant.email else ":"
        lines.append(f"ATTENDEE;CN={_ics_text(participant.name)}{address}")
    if event.status != "scheduled":
        # RFC 5545 has three: a postponed event is tentative until it is
        # rescheduled, which is what the source actually told us.
        mapped = {"cancelled": "CANCELLED", "tentative": "TENTATIVE", "postponed": "TENTATIVE"}
        lines.append(f"STATUS:{mapped[event.status]}")
    return tuple(lines)


def _ics_moment(name: str, date: str, time: str | None, timezone: EventTimezone | None) -> str:
    stamp = date.replace("-", "")
    if time is None:
        return f"{name};VALUE=DATE:{stamp}"
    clock = time.replace(":", "")
    if len(clock) == 4:
        clock = f"{clock}00"
    if timezone is None or timezone.name == "floating":
        # Floating: no zone, no Z. Local to whoever opens it.
        return f"{name}:{stamp}T{clock}"
    if timezone.name == "UTC":
        return f"{name}:{stamp}T{clock}Z"
    return f"{name};TZID={timezone.name}:{stamp}T{clock}"


def _ics_rrule(repeats: Recurrence) -> str:
    parts = [f"FREQ={repeats.freq}"]
    if repeats.interval != 1:
        parts.append(f"INTERVAL={repeats.interval}")
    for name, values in (
        ("BYDAY", repeats.by_day),
        ("BYMONTHDAY", repeats.by_month_day),
        ("BYMONTH", repeats.by_month),
        ("BYSETPOS", repeats.by_set_pos),
    ):
        if values:
            parts.append(f"{name}={','.join(str(value) for value in values)}")
    if repeats.count is not None:
        parts.append(f"COUNT={repeats.count}")
    if repeats.until is not None:
        parts.append(f"UNTIL={repeats.until.replace('-', '')}")
    return ";".join(parts)


def _ics_text(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")
    return escaped.replace("\r", "").replace("\n", "\\n")


__all__ = ["calendar_component"]
