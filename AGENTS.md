# Repository instructions

These rules apply to the entire repository. OmniTensor is a Python service; use the project
venv (`.venv/bin/python`, `.venv/bin/pip`) for every command.

## Finding tracking

- Whenever any finding is discovered (bug, contract risk, race condition, missing validation,
  missing tests, performance problem, packaging issue, technical debt), add or update a row in
  the root `TODO.md` before reporting or acting on it.
- Do not create duplicate rows; update the existing row when a finding changes.
- Move findings that cannot progress without external input, hardware, credentials, or a
  dependency into `Blocked / Deferred`; move them back to `Open` when they become actionable.
- State a dependency in the description, where it can say *why*, rather than as a bare
  list of ids: the ids column carried information nothing read, and a third of the rows
  cited ids that no longer had rows.
- Once a finding is fully resolved and verified, remove its row in the same scoped commit as
  the resolution; no completed row may remain after its resolution is committed.
- A row is removed only when the work is genuinely finished. When an item is implemented
  only in part, keep its row and add to the description exactly what is still missing,
  naming the specific remainder rather than calling it partial. A row deleted after half
  the work silently loses the rest: nothing records it and nobody finds it again. Work
  that the resolution newly reveals is a new row with a new id, not a note appended to
  the old one.
- Findings intentionally rejected or not planned move to `Rejected / Won't fix` with the
  rationale in the description; never mix them into `Open`.
- Use stable sequential IDs in the form `OMNI-0001`.
- Keep root `TODO.md` limited to `# TODO` and exactly three sections in this order: `## Open`,
  `## Blocked / Deferred`, and `## Rejected / Won't fix`.
- Use only `open` or `in_progress` in `Open`, and keep only work implementable now there.
- Use only `blocked` or `deferred` in `Blocked / Deferred`.
- Use only `rejected` or `wont_fix` in `Rejected / Won't fix`.
- Use lowercase status, severity, and effort values. Severities: `critical`, `high`, `medium`,
  `low`. Efforts: `xs`, `s`, `m`, `l`, `xl`. Use `—` when a value is missing.
- Keep descriptions concise, actionable, and specific.
- All three tables use exactly this schema:

  `| id | status | severity | effort | description |`

## Software design and architecture

- New and materially changed code follows SOLID, DDD-lite, and idiomatic Python.
- Keep contracts separate from implementations: consumers depend on the Protocols in
  `omnitensor.ports` and `omnitensor.executors.base`, never on concrete adapters.
  Inject dependencies at constructor boundaries; concrete defaults are wiring, not coupling.
- Domain logic (scheduling, policy, snapshot building) stays free of transport (D-Bus),
  persistence, and process concerns; those live in thin adapters.
- The canonical JSON schemas in `schemas/` are the contract with the Cinnamon applet.
  Never emit a document that has not been validated against them; never weaken a schema
  without coordinating both repositories.
- **Never cap or bound a resource unless explicitly asked to**, and never cap input or
  output tokens. They exist to deliver the expected answer; time and performance are the
  correct price. Every ceiling is a wrong answer waiting for a large enough input — a
  programme with 65 events against `maxItems: 64`, an answer truncated mid-object by a
  token budget and surfaced as "invalid output". Solve size by splitting the work (one
  generation per span, extraction in passes that accumulate), never by bounding the
  answer, and remove a ceiling rather than raising it. The only legitimate bound on a
  generation is the context window itself, which is arithmetic rather than policy.
  Bounds on what a person *selected* (how many files, how large) and protocol frame
  limits are not this; ceilings on what they are told back always are.
- There is deliberately NO CPU *backend* (gpu > npu > tpu only), and no silent CPU
  fallback: a load must reach the accelerator and say how much of it did. A **declared
  partial offload is work, not a fallback** (owner decision 2026-08-21, OMNI-0586) — a
  model larger than VRAM may run with some layers on the GPU and the rest resident in
  RAM, with the real accelerator/total split carried in `NativeLoadReport` and the
  receipts. What stays refused is a load that reaches the GPU not at all, or one the
  runtime itself flags as a CPU fallback. Never refuse work because only part of it
  fits — that is the capping rule wearing a lane costume.
- Keep architecture proportional; record unavoidable compromises in `TODO.md`.

## Quality gates

- Whenever implementing or fixing code, add or update:
  - unit and integration tests (pytest) for the changed behavior;
  - regression tests when fixing a reproducible defect;
  - property/fuzz tests (hypothesis) for changed input boundaries and state transitions;
  - mutation coverage (mutmut) for changed logic.
- Achieve at least 80% coverage for every changed or added module
  (`.venv/bin/python -m pytest --cov=omnitensor --cov-report=term`); aggregate project
  coverage does not replace the per-module requirement.
- Unless the user explicitly requests a long or full suite, run only tests and quality
  gates scoped to the changed behavior; do not run the full pytest suite by default.
- Parallelize independent scoped tests and gates whenever safe, provided their temporary
  files, coverage data, and mutation reports cannot collide.
- If a gate cannot run, record the blocker in `TODO.md`; the work is not complete.

## Git

- Never push automatically; push only on explicit request.
- Commit only green, scoped work — one Conventional Commit per logical unit.
- Do not stage or commit unrelated changes.
- No AI or assistant attribution: no `Co-Authored-By` naming an assistant, no
  generated-by trailers.
