# The event contract, proposed

Status: proposal. Nothing here is implemented, and implementing it moves the
task digest, so it lands with a new qualification receipt or not at all — a
worker whose task does not match its receipt refuses to start, which is the
failure OMNI-0352 was.

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
- The task declares 1,024 output tokens. Even raised to 4,096 that is about
  seven events, and a generation that runs out mid-object produces truncated
  JSON — which surfaces as an unexplained invalid output, not as "there were
  more".

So the answer is the same one translation reached: **split the work, never the
answer**. Extraction runs in passes over the source fragments, each pass
bounded by the output budget rather than by a count, and the passes accumulate.
The dedup key below is what makes accumulation safe, since a recurring meeting
mentioned on three pages is one event.

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
- **The output budget** must rise and the event count must lose its ceiling.
  One fully populated event is roughly 400–600 tokens; the task still declares
  1,024 for the whole answer. `ask-selected-files` already had to go from 1,024
  to 2,048 for the same reason — truncation that surfaced as an unexplained
  invalid output. 4,096 per pass, and as many passes as the sources need.
- **The benchmark cases** are rewritten against the new shape, and gain the ones
  this design exists for: a date without a time, a time without a date, a place
  without a zone, a recurrence, a cancellation, a banner read through OCR.
- **A new qualification receipt**, because all of the above moves the task
  digest.
