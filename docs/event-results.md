# Grounded event results

Event extraction is a proposal workflow, not a calendar writer. The version 1
contract in `event-extraction-result.schema.json` keeps these boundaries fixed
regardless of the model or runtime that produced a candidate.

## Evidence and outcomes

Every candidate has a bounded title, start, optional end, named timezone or
explicitly unresolved `floating` timezone, optional location, and one or more
source evidence records. Evidence contains an opaque `private:` source
reference, source digest, page, character span, and digest of the supporting
text. It never carries source text or a filesystem path.

The outcome is explicitly `succeeded`, `partial`, or `refused`; empty output is
not used to guess which one occurred. A refused result cannot carry candidates.
The fixed `keep-first-title-start-location` duplicate policy normalizes title
and location, preserves input order, and keeps the first equal candidate.
Candidate IDs remain unique even when two candidates describe different
events.

`source_fragments` consumes the existing bounded `ExtractionResult` returned by
parser, renderer, or OCR workers. The fragment text remains private prompt data
inside the isolated workload worker. Public snapshots, logs, and general D-Bus
documents receive neither fragments nor event results.

## Dates and timezones

Named-zone candidates include an explicit UTC offset. The domain resolves the
IANA timezone and refuses an offset that does not describe that local wall time,
making daylight-saving interpretation deterministic. An end must be
strictly later than its start. A `floating` candidate omits its offset and may
be previewed, but cannot be exported until a user resolves it to a named zone.

## Confirmation boundary

A generation worker may emit only `pending` candidates. It cannot mark its own output confirmed or rejected.
`confirm_event_candidates` accepts one explicit
human decision for every candidate and produces a new immutable result. Both
`render_confirmed_ics` and `write_confirmed_events` reject pending output and
ignore rejected candidates; the calendar sink is invoked only for confirmed
candidates. The core domain never chooses a calendar provider or writes a file
on its own.
