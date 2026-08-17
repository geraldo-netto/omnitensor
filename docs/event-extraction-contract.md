# The event contract

Status: implemented. The version 2 result schema
(`event-extraction-result.schema.json`), the `needs`-based partial events,
`readAs` evidence, the floating/all-day/`VTODO` export mapping, and the
removal of the answer ceilings have landed; this document remains the design
record and the measurements that motivated them. Implementing it moved the
task digest, so the qualification receipt was reissued — the receipt now
reports coverage per workload and model rather than gating a person's model
choice.

## Why it changes

Two required fields cannot be satisfied from the evidence the same task insists
on:

```json
"start":    { "type": "string", "minLength": 16 },   // full datetime, no null
"timezone": { "type": "string", "minLength": 1 }     // no null
```

*"Project review with Ana on 2026-09-03 at 14:00 in Lisbon"* states no
timezone. A valid event therefore requires the model to invent one, while the
system prompt says **"Extract only calendar events explicitly supported by
evidence."** Both rules cannot hold at once, and `refused` is the only output
that satisfies both.

That is what a wording theory could not explain: measured on 2026-08-17, seven
answerable documents produced seven byte-identical refusals on `qwen3-4b`,
`qwen3-8b` and `qwen3.5-9b`, on both cards. Models from different generations
agreeing exactly is the signature of a contract contradiction rather than a
phrasing preference.

The second defect is the rule itself. "If no named activity has an explicit
date and time, refuse" throws away a real event because part of it is missing.
A banner that says *Jazz night, Fridays, Casa da Música* is an event somebody
wants to keep; it is not nothing.

A correction to an earlier reading of this file, kept because it cost a
measurement: `events.items` is a `$ref` to `$defs/event`, which specifies all
eight fields and requires them. The schema is not vague about what an event is.
It is precise about it in a way no source can satisfy.

## Measured, not guessed

Five wordings against the seven answerable cases on `qwen3-4b`, 2026-08-17:

| wording | correct | what it changed |
| --- | --- | --- |
| shipped | 0/7 | — |
| timezone rule | 0/7 | told the model what to put in `timezone` when the source is silent |
| fields named | 0/7 | listed every event field the plugin needs |
| plain system | 0/7 | removed the "never confirm events" ambiguity |
| **all three, and 2,048 output tokens** | **2/7** | the above together |

So no single wording moves it, and the combination barely does. Two further
findings explain why, and both are worse than a prompt problem.

**The deterministic fallback matches one sentence nobody writes.** When the
model refuses, the provider retries with a reconsideration prompt, and if that
also refuses it falls back to a regular expression. That expression requires
exactly:

    <title> is in <location> on <day> <Month> <year> from HH:MM to HH:MM <Zone>.

An end time, an explicit IANA zone, exactly one source, and no occurrence of
the words *create, emit, ignore, instruction, make* or *output* anywhere in the
text — a prompt-injection guard that also discards "Team meeting to make the
plan". *"Project review with Ana on 2026-09-03 at 14:00 in Lisbon"* matches
nothing.

**The qualification corpus was written to that shape.** `evaluation-corpora/
event-extraction-v1.json` holds four cases, and the English one reads *"Release
planning is in Room 2 on 12 August 2026 from 10:00 to 11:00 Europe/Rome."* —
the regular expression's own sentence. That is how a workload that cannot read
an ordinary calendar invitation holds a passing receipt.

**And 1,024 output tokens is too few.** The only wording that produced any
event at all was also the only one that raised the budget. One valid event
carries two 64-character digests before anything else is said.

## Principles

1. **Capture everything stated; invent nothing.** Where more detail exists in
   the source, it is recorded. Where it does not, the field is absent — never
   filled with a guess that reads like a fact.
2. **Every derived value says it is derived.** `stated`, `summarised`, or
   `deduced`, per field. A title read off the page and a title the model wrote
   must not look alike to the person deciding.
3. **Incomplete is not refused.** An event that cannot be placed in a calendar
   is still extracted, and says exactly what it needs from the person.
4. **Evidence addresses the material.** Every event carries spans into the
   fragments it came from, and says how that text was read.

## The event

```json
{
  "candidateId": "e1",

  "label":  { "text": "Project review", "source": "stated" },
  "what":   { "text": "Quarterly budget review with the finance team",
              "source": "summarised" },

  "when": {
    "date":     "2026-09-03",
    "time":     "14:00",
    "end":      { "date": "2026-09-05", "time": "15:30" },
    "allDay":   false,
    "timezone": { "name": "Europe/Lisbon", "utcOffset": "+01:00", "source": "deduced" },
    "needs":    [],
    "isPast":   false
  },

  "repeats": {
    "freq":       "WEEKLY",
    "interval":   1,
    "byDay":      ["TU"],
    "byMonthDay": [],
    "byMonth":    [],
    "bySetPos":   [],
    "count":      null,
    "until":      "2026-12-16",
    "exceptions": ["2026-11-04"],
    "text":       "every Tuesday until 16 December, except 4 November",
    "source":     "stated"
  },

  "where": {
    "kind":    "physical",
    "venue":   "Casa da Música",
    "address": { "street": "Av. da Boavista 604", "postalCode": "4149-071",
                 "city": "Porto", "region": null, "country": "PT",
                 "full": "Casa da Música, Av. da Boavista 604, 4149-071 Porto" },
    "url":     null,
    "joining": null,
    "source":  "stated"
  },

  "people": {
    "organiser":    { "name": "Marta Oliveira", "email": null, "source": "stated" },
    "participants": [ { "name": "Ana", "email": null, "source": "stated" } ]
  },

  "status":       "scheduled",
  "registration": { "required": true, "deadline": "2026-09-01", "url": "https://…",
                    "cost": { "amount": "25.00", "currency": "EUR" },
                    "source": "stated" },
  "contact":      { "email": "eventos@example.pt", "phone": null, "source": "stated" },
  "reference":    "https://example.pt/jazz",
  "language":     "pt",

  "confirmation": "pending",

  "evidence": [
    { "sourceRef": "case:…:1", "sourceSha256": "…", "page": 1,
      "span": { "start": 0, "end": 57 }, "textSha256": "…",
      "readAs": "ocr" }
  ]
}
```

Only `candidateId`, `label`, `when`, `confirmation` and `evidence` are required.
Everything else is present when the source supports it and absent when it does
not. An absent field and a null field mean the same thing and both mean *the
source did not say* — never *no*.

### label and what

`label` is the calendar entry's name: the stated one if the source names it,
otherwise **one to four words** the model writes, marked `summarised`.

`what` is the longer description, under the same rule. It exists because a
calendar entry called "Review" three weeks later tells nobody anything, and the
sentence it came from usually does.

### when

`date` and `time` are recorded separately and either may be absent, which is
the change that stops the refusals:

| the source says | date | time | `needs` | what a client does |
| --- | --- | --- | --- | --- |
| `2026-09-03 at 14:00` | ✓ | ✓ | `[]` | places it |
| `3 September` | ✓ | — | `["time"]` | offers all-day, or asks |
| `Tuesday at 14:00` | — | ✓ | `["date"]` | asks which date |
| `soon`, `TBA` | — | — | `["date","time"]` | keeps it, asks for both |
| `14:00`, no place named | ✓ | ✓ | `["timezone"]` | asks which zone |

`needs` lists only what blocks placing the event in a calendar: `date`, `time`,
`timezone`. A missing location does not block anything and is not listed.

`allDay` is true when the source states a day without a time *and means the
whole day* — a public holiday, a deadline date. It is false, not absent, when
a time is simply missing; that case is `needs: ["time"]` and the difference
between the two is real.

`isPast` is computed against the moment of extraction, and the event is still
returned. A file is often selected precisely to find out when something
happened.

### timezone

Three sources of events across Brazil, Italy and Portugal means the zone cannot
be assumed from the machine's locale, and a silent local default would be wrong
about a third of the time.

1. **Stated** — the source names a zone or offset. Use it, `source: stated`.
2. **Deducible** — the source names a place unambiguously enough to determine
   the zone (`Lisboa`, `São Paulo`, `Milano`). Use it, `source: deduced`, and
   the client shows that it was deduced so the person can override.
3. **Neither** — `timezone` is null and `needs` contains `timezone`. The person
   is asked rather than guessed at.

The IANA **name** is the answer; `utcOffset` is filled only when the date is
known, because the offset depends on it. Europe/Lisbon is +00:00 in January and
+01:00 in July, and Brazil abolished daylight saving in 2019 — an offset
without a date is a fact with a hole in it.

A place too ambiguous to resolve is not deduced. There are Springfields in
several zones and a `Milan` in Michigan.

### repeats

Full RFC 5545, because half a recurrence rule in a calendar is worse than none:
`FREQ`, `INTERVAL`, `BYDAY`, `BYMONTHDAY`, `BYMONTH`, `BYSETPOS`, and exactly
one of `COUNT` or `UNTIL` (or neither, meaning it repeats with no stated end).
`exceptions` carries EXDATE.

Two rules that keep this honest. The `text` field holds the phrase the rule was
read from — *"every Tuesday until 16 December"* — so a person can check the
rule against the words without reconstructing RFC 5545 in their head. And a
recurrence is only ever `stated`: a rule nobody wrote down is not deduced from
two dates that happen to be a week apart. Repetition inferred from coincidence
is how a calendar acquires an event every Tuesday for a year.

`repeats` is absent for a single occurrence, which is most events.

### where

`kind` is `physical`, `online`, `hybrid`, or `unknown`, because a client does
different things with each: one opens a map, one opens a link, one offers both.

- `online` and `hybrid` carry `url`, and `joining` for a meeting id or passcode
  when the source states one.
- `physical` and `hybrid` carry `venue` and `address`, broken into parts as far
  as the source allows, plus `full` as the single string a map accepts.
- A place named but not addressed — *"in Lisbon"* — is `venue: null`,
  `address.city: "Lisboa"`, `full: "Lisboa"`. Partial is fine; invented is not.

`source: deduced` covers the case your proposal names: a recurring team call in
an email thread whose venue is stated three messages earlier, or a link that
identifies the platform.

### people

`organiser` and `participants` — because *"Budget review with Marta Oliveira"*
and *"Project review with Ana"* say who, and a calendar entry without the other
person is half the entry. Names as written; emails only when the source shows
them. No inference from a name to an address.

### status, registration, contact, reference, language

`status` is `scheduled`, `tentative`, `postponed`, or `cancelled` — banners and
emails say these words, and an event extracted from a cancellation notice must
not be indistinguishable from the original invitation.

`registration` carries `required`, `deadline`, `url` and `cost`. A deadline to
register is a date a person needs; it stays part of the event rather than
becoming a second one, because it is not something they attend.

`contact`, `reference` and `language` are recorded when stated. `language` is
the source's language, which is what a client needs to decide how to display a
label it did not write.

### evidence

Unchanged in shape, with one addition: `readAs` is `text` or `ocr`.

Images never reach the model as pixels. A `.png`, `.jpg`, `.webp` or a PDF page
with no text layer goes through PyMuPDF and then Tesseract, so what is extracted
is OCR output and the spans address *that* text. This is worth saying in the
result because a span into the OCR of a crooked banner is a much weaker citation
than a span into an email body, and today the two are indistinguishable. It also
predicts the failure mode: a banner whose date reads `2O26` rather than `2026`
produces an event that is wrong in a way no downstream check can catch, and the
person deserves to see that it came from a photograph.

The span stays verifiable either way — the plugin holds the fragment text the
model was given and checks the offsets against it.

## How many events

No ceiling. The shipped schema caps `events` at 64 and `evidence` at 16 per
event, and a cap is a wrong answer waiting for a large enough document: a
conference programme, a year of a venue's listings, a PDF of a school calendar.
Sixty-five events means one of them is discarded, or the whole answer fails
validation — neither of which the person asked for.

Removing the number is the easy half. The real ceiling is arithmetic, and
naming it is the point:

- One fully populated event is roughly 400–600 tokens, most of it the two
  64-character digests each evidence entry carries.
- The task declares 1,024 output tokens and 262,144 output bytes, and the
  provider refuses outright above 4,096 tokens.

None of those three is a fact about the machine; they are policy numbers, and
every one of them is a wrong answer waiting for a long enough document. They
go. The only real bound is the context window: a generation can emit at most
what is left of the context after the prompt, which is arithmetic rather than
a choice. So:

- **`outputBytes` goes**, following `document_qa.py`, which dropped its byte
  ceiling with the note that the token budget bounds an answer and host
  pressure protects the machine.
- **`outputTokens` stops being a budget.** The task states the largest value
  the contract permits below its context, and the runtime asks for what remains
  of the context rather than a constant — generation ends at the model's own
  stop token or at the edge of the context, never at a number somebody picked.
- **`MAX_RUNTIME_OUTPUT_TOKENS` rises to the context.** It is a refusal, not a
  truncation, so a task that asks for more is rejected before it runs.

Passes exist because the context is finite, not because a budget was chosen.
When the remaining sources cannot fit alongside what has already been
extracted, the work splits and continues; the answer never does. That is the
same rule translation reached with spans. The dedup key below is what makes
accumulation safe, since a recurring meeting mentioned on three pages is one
event.

Two ceilings that stay, because they bound *input* rather than an answer:
`MAX_SOURCES` (32 selected files) and `MAX_SOURCE_BYTES` (128 MiB each). Those
describe what a person selected. And a per-event `evidence` list stays bounded
in practice by the passes it was found in, not by a constant — an event
supported by forty spans is a citation list nobody reads, so the passes record
the spans that carried the fields, not every mention.

## Outcome, and what refusal means now

- `succeeded` — one or more events. Some may carry `needs`; that is not a
  failure, it is a question.
- `partial` — one or more events, and some source material could not be
  processed at all (an unreadable page, an adapter that was unavailable).
- `refused` — the material contains no event. Not "no complete event".

`confirmationState` and every event's `confirmation` stay `pending`. A model
never confirms an event; a person does, and that rule is a security property
rather than a wording preference.

## Export, and who resolves `needs`

Decided: the client shows what an event is missing and does **not** ask. The
person decides if and when to import, and their calendar application is where
they fill a gap — it already has the dialogs, the timezone list and the
recurrence editor, and it is where the event ends up anyway.

That makes `needs` an export rule rather than a conversation, and iCalendar
already has a container for every case:

| `needs` | exported as | meaning to the calendar |
| --- | --- | --- |
| `[]` | `DTSTART;TZID=Europe/Lisbon:20260903T140000` | placed exactly |
| `["timezone"]` | `DTSTART:20260903T140000` — floating | local time wherever it is opened |
| `["time"]` | `DTSTART;VALUE=DATE:20260903` | an all-day event |
| `["date"]`, `["date","time"]` | `VTODO` with no `DUE` | a task: decide when |

**Floating time is the answer to the three-country problem.** RFC 5545 defines
a datetime with no zone and no `Z` as local to whoever opens it, which is
precisely "the source did not say, so use mine". It is a standard behaviour of
every calendar client rather than a convention we would be inventing.

An event with no date at all cannot be a `VEVENT`: RFC 5545 requires `DTSTART`
when the object carries no `METHOD`. Writing one anyway produces a file that
some clients reject and others silently place today. `VTODO` has no such
requirement, says what the thing actually is — something to schedule — and
keeps it out of the calendar grid until the person decides.

`plugins/events.py` has to change to allow this. Today `_ics_event` asserts
`event.timezone != "floating"` and `_confirmed_events` raises
`timezone-required` — "confirmed events need a named timezone". Floating was
anticipated and then forbidden, which is the same rigidity that makes the
workload refuse in the first place.

The client's part shrinks to one renderer: `_event_line` in
`workflows/specs/summaries.py` shows the label, what is known, and what is
missing — *"Jazz night · 3 September · needs a time"*. No dialog, no round
trip, no second copy of a calendar UI. Its `limits.events` display cap goes at
the same time, for the same reason the schema's did.

## Where the calendar file is written

Asked, and answered no: the writer stays in the runtime.

It is not presentation. `plugins/events.py` says so in its first line —
*"Generation workers may propose dates, but they do not own calendar effects…
Only a new result created by `confirm_event_candidates` can reach an ICS
renderer or calendar sink."* The single path to a calendar file runs through a
function that resets every model-supplied confirmation to `pending` and demands
an explicit decision per candidate. Move the writer into a client and that
guarantee moves with it, into whichever client happens to implement it; the
runtime could no longer say that no model ever put an event in somebody's
calendar.

RFC 5545 is also a specification rather than a view: line folding at 75 octets,
escaping, stable UIDs, `VEVENT` against `VTODO`, floating time. Written twice,
two clients disagree about the same event.

What is actually missing is a **route**, not a relocation. `confirmed_ics` is
reachable today only through the `omnitensor-import-events` command line, which
is why the client tells a person "nothing is written to a calendar until you
export" and then offers no export. The runtime needs a method that takes the
result document with the confirmed and rejected candidate ids and returns the
calendar text; the client chooses which events and where to save the file.

Three more ceilings live in that module and go with the others: `MAX_EVENTS`
(64), `MAX_EVIDENCE_PER_EVENT` (16) and `MAX_SOURCE_FRAGMENTS` (2,000).

## Deduplication

The current key is title + start + location, which breaks as soon as a start may
be partial. Proposed: `label`, `when.date`, `when.time`, `where.full`, with
absent compared as absent. Two extractions of *"3 September, Lisboa"* are the
same event; *"3 September"* and *"3 September at 14:00"* are not the same, and
merging them would silently choose one time over none.

## What this costs

- **The schema** gains an item contract for `events`, which is what makes the
  grammar able to express a success at all.
- **The prompt** states the success path with the precision the refusal already
  has, and the refusal rule shrinks to "no event at all".
- **Three answer ceilings go**: the task's `outputTokens` and `outputBytes`,
  and the provider's `MAX_RUNTIME_OUTPUT_TOKENS` refusal. What bounds a
  generation afterwards is the context window, and what bounds the answer is
  nothing — the work splits into passes instead.
- **The benchmark cases** are rewritten against the new shape, and gain the ones
  this design exists for: a date without a time, a time without a date, a place
  without a zone, a recurrence, a cancellation, a banner read through OCR.
- **A new qualification receipt**, because all of the above moves the task
  digest.
