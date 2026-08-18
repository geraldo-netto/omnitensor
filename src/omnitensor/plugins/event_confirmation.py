"""The gate between a model's suggestion and somebody's calendar.

Generation workers may propose dates; they do not own calendar effects. Every
candidate arrives `pending`, a person decides each one explicitly, and only a
result produced here can reach a calendar sink or be rendered. Keeping that in
its own module is the point: it is a safety boundary, not a step in a
pipeline, and it should be visible as one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol, runtime_checkable

from .event_calendar import calendar_component
from .events import _REQUEST_ID, EventCandidate, EventResultError, GroundedEventResult


@runtime_checkable
class CalendarSink(Protocol):
    """A side-effect port implemented by an explicitly invoked client."""

    def write(self, event: EventCandidate) -> None: ...


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
    # A named timezone used to be required here, which is the same rigidity
    # that made the workload refuse every ordinary invitation, one layer down:
    # floating time is what RFC 5545 provides for "the source did not say", and
    # the calendar the person imports into resolves it.
    return confirmed


def render_confirmed_ics(result: GroundedEventResult) -> str:
    """Render confirmed candidates as a bounded RFC 5545 calendar document."""
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//OmniTensor//Events 1//EN"]
    for event in _confirmed_events(result):
        lines.extend(calendar_component(event))
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


__all__ = [
    "CalendarSink",
    "confirm_event_candidates",
    "render_confirmed_ics",
    "write_confirmed_events",
]
