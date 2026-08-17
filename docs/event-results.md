# Grounded event results

Event extraction is a proposal workflow, not a calendar writer. The version 2
contract in `event-extraction-result.schema.json` keeps these boundaries fixed
regardless of the model or runtime that produced a candidate. Version 2
records whatever the source states and says what an event still `needs`;
incomplete is not refused.

## Evidence and outcomes

Every candidate has a bounded `label`, a `when` block whose `date` is always
present (the literal `unknown` when the source states none) with optional
time, end, and timezone, a `needs` list naming what still blocks calendar
placement (`date`, `time`, `timezone`), optional `what`, `where`, `repeats`,
people and status fields, and one or more source evidence records. Evidence
contains an opaque `private:` source reference, source digest, page, character span,
digest of the supporting text, and `readAs` (`text` or `ocr`). It never
carries source text or a filesystem path.

The outcome is explicitly `succeeded`, `partial`, or `refused`; empty output is
not used to guess which one occurred. A refused result cannot carry candidates.
The fixed `keep-first-label-date-time-place` duplicate policy preserves input
order and keeps the first equal candidate, comparing absent fields as absent.
Candidate IDs remain unique even when two candidates describe different
events.

`source_fragments` consumes the existing bounded `ExtractionResult` returned by
parser, renderer, or OCR workers. The fragment text remains private prompt data
inside the isolated workload worker. Public snapshots, logs, and general control
documents receive neither fragments nor event results.

## Dates and timezones

A timezone may be stated by the source or deduced from an unambiguous place,
and each carries its provenance; a UTC offset is accepted only alongside a
known date, because the offset depends on it. An end must be strictly later
than its start. A candidate with no zone exports as a *floating* time —
RFC 5545 local time wherever the calendar is opened — a date without a time
exports as an all-day event, and an event with no date at all exports as a
`VTODO` rather than a mis-placed `VEVENT`.

## Confirmation boundary

A generation worker may emit only `pending` candidates. It cannot mark its own output confirmed or rejected.
`confirm_event_candidates` accepts one explicit
human decision for every candidate and produces a new immutable result. Both
`render_confirmed_ics` and `write_confirmed_events` reject pending output and
ignore rejected candidates; the calendar sink is invoked only for confirmed
candidates. The core domain never chooses a calendar provider or writes a file
on its own.
