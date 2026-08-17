# Unattended session, 2026-08-17

What was done while you were away, in order, with where to look. Times are
local. Benchmark numbers land in `benchmark.md` beside this file as each model
finishes; the raw per-case detail is in `benchmark.json`.

## Standing constraints followed

- **Nothing was capped or degraded.** No context was shortened, no cache
  precision lowered, no output bounded to make something fit. Where size was
  the problem, the work was split instead.
- **No model was changed.** Every default, every receipt, every policy is as
  it was. The benchmark reads and measures; deciding is a separate act.

## Committed

| time | item | commit | what |
| --- | --- | --- | --- |
| 01:23 | OMNI-0362 | `489a582` | Fit prober: `python -m omnitensor.probe_cli`. Reads model headers and sysfs, loads nothing. |
| 01:31 | OMNI-0363 | `f9cf3aa` | Benchmark harness through the real generation path, 40 labelled cases. |
| 01:40 | OMNI-0359 | `5a860c6` | A workload runs the model it was told to run — the last link that kept the Hebrew model unreachable. |
| 01:41 | OMNI-0360 (part) | `b94373c` | Span splitter: translation stops being capped and starts being split. |

## What the prober found

At the 32,768-token context the tasks declare, on the discrete RX 6600 XT
(7.88 GiB free):

| artifact | weights | cache | total | fits |
| --- | --- | --- | --- | --- |
| qwen3-8b-q4-k-m | 4.68 GiB | 2.39 GiB | 7.57 GiB | yes, 0.31 GiB spare |
| dictalm2-hebrew-q4-k-m | 4.07 GiB | 2.12 GiB | 6.70 GiB | yes, 1.18 GiB spare |
| qwen3-4b-q4-k-m | 2.33 GiB | 2.39 GiB | 5.22 GiB | yes, 2.67 GiB spare |
| qwen3-0-6b-q8-0 | 0.75 GiB | 1.86 GiB | 3.11 GiB | yes, 4.77 GiB spare |

The ceiling is real: the shipped 8B leaves 0.31 GiB. **The cache is 2.39 GiB of
that** — larger than the 4B model itself.

Correction to what I said earlier: the **integrated 610M has ~45 GiB
available**, not the 4 GiB its VRAM aperture advertises, because its real pool
is mapped system memory. A 14B or a 32B fits there outright. Speed is the open
question — that is OMNI-0364, and it needs the discrete run to finish first so
the two are not competing for the accelerator lease.

## Benchmark run

Started 01:32, `benchmarks/results/run-discrete.log`, models
`qwen3-8b-q4-k-m` then `qwen3-4b-q4-k-m`, all four text workloads, ten cases
each. About 100 seconds per case, so roughly two and a half hours. Results are
rewritten after each model, so an interrupted run still leaves findings.

## Findings so far, while it runs

- **`ask-selected-files` on the 8B: 8/10, and both failures were mine.** The
  two unanswerable cases expected a refusal, and that workload's contract has
  no refusal state — it holds an answer and citations, nothing else. No answer
  could have passed. Fixed in `7eed161`: what those cases check now is
  invention, which is the thing that actually harms somebody — a VAT number the
  invoice never carried, a witness who does not exist. Those two rows are being
  re-measured (see below); everything else in that workload stood.
- **`event-extraction` refuses everything.** Seven answerable documents, seven
  refusals: an ISO date with a named activity and a location, a written month,
  two events in one document, a 24-hour time, and a valid `.ics` with
  `DTSTART:20260903T140000Z`. Its 3/10 is not a partial score — the three it
  "passes" are precisely the three that expect a refusal, so the number is a
  100% refusal rate wearing a score. Every answer was well-formed, so the
  earlier note that the refusal was "not stable either" is wrong: it is
  perfectly stable and perfectly useless. This is OMNI-0354, sharpened, and it
  is a known defect rather than anything this work caused.

- **`file-organizer` scored 0/10, and that too was my cases.** They counted
  `plan`, which is the shape the *plugin* maps an answer into; the model emits
  `suggestions`. Every case asked about a field that is never present, so
  nothing could pass — the 56% rule score is what the model actually did. Fixed,
  re-measurement queued, and there is now a test pairing every case's
  collections against the schema its task declares, so the third instance of
  this mistake fails in a second rather than after 28 minutes of GPU time.

- **`selected-text-tools` scored 10/10 on the 8B, and the Hebrew is still not
  good.** Reading what the rules had passed:
  `בליסון` (Lisbon mistransliterated), `42,000 אירופי` ("European" rather than
  euros), and — the decisive one — `אם הדוח متأخر`, with the **Arabic** word
  for "late" sitting inside a Hebrew sentence. The shipped Hebrew runtime
  already refuses foreign script, so production would have rejected that
  answer; my harness bypassed the check by calling the base runtime. There is
  now a rule mirroring production's exact character range, and the Hebrew
  comparison against DictaLM is queued.

  Worth saying plainly: automated rules are necessary and not sufficient here.
  They cannot see that "אירופי" is the wrong word. Somebody who reads Hebrew
  should look at `benchmark.json` before this decides anything.

- **The 4B matched the 8B on `ask-selected-files`, at 2.3× the speed**: 10/10
  against 8/10, 37.1s per case against 87.1s. Read that carefully — the 4B ran
  *after* the case fix, so it was judged on the invention rules and passed
  them, while the 8B's row was scored under the impossible refusal rules. The
  corrected re-run makes the two comparable; until then the fair statement is
  "the 4B passed everything asked of it, faster", not "the 4B beat the 8B".

- **`event-extraction` fails identically on the 4B**: same 3/10, same seven
  refusals, same 14%. Two independently trained models behaving exactly alike
  is not a model problem — it moves OMNI-0354 into the task itself (prompt,
  grammar projection, evidence rules) and out of "that provider needs tuning".
  That is the most useful thing this run has produced.

- **The 4B also scored 10/10 on `file-organizer`, at 80.8s per case** against
  the 8B's 168.5s. Same caveat as above and for the same reason: the 4B was
  judged on corrected cases, the 8B on broken ones. What can be said now is
  that the 4B satisfied every rule on two of the four workloads while running
  roughly twice as fast; whether the 8B does the same is what the re-run
  answers.

## Queued, in order

1. Discrete card (RX 6600 XT), 8B then 4B, all four workloads — running.
2. Integrated 610M, same models and cases — starts automatically when the
   first finishes, into `benchmarks/results/integrated/`. This is OMNI-0364's
   measurement. Serialised because both take the same accelerator lease.
3. `ask-selected-files` and `file-organizer`, both models, with the corrected
   cases, into `benchmarks/results/corrected/`. Those rows supersede the ones
   in the first table.
4. `selected-text-tools` on DictaLM against the 8B, into
   `benchmarks/results/hebrew/` — the comparison the Hebrew work exists for,
   now that the 8B has been seen to produce Hebrew with Arabic in it.

## Still to do

- OMNI-0360 second half: the `document-translation` workload itself.
- OMNI-0364: the 610M lane, after the discrete run finishes.
- OMNI-0365: the larger candidates. **This one needs your say-so** — it is
  about 15 GB of downloads, and I would rather not pull that unasked.
- OMNI-0361: the qualification runs, which the benchmark now has the harness
  for.
- xpuwlm G2 (Translate route, 10 MB warning) and G3 (model dropdown).

Nothing is deployed. The service and client still run the code they ran
yesterday; a reinstall and reload happens once, at the end, as usual.
