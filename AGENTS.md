# Repository instructions

These rules apply to the entire repository. OmniTensor is a Python service; use the project
venv (`.venv/bin/python`, `.venv/bin/pip`) for every command.

## Finding tracking

- Whenever any finding is discovered (bug, contract risk, race condition, missing validation,
  missing tests, performance problem, packaging issue, technical debt), add or update a row in
  the root `TODO.md` before reporting or acting on it.
- Do not create duplicate rows; update the existing row when a finding changes.
- Once a finding is fully resolved and verified, remove its row in the same scoped commit as
  the resolution. `done` is transitional only; no completed row may remain after its
  resolution is committed.
- Findings intentionally rejected or not planned move to the `Rejected / Won't fix` table with
  the rationale in the description; never mix them into the active `Findings` table.
- Use stable sequential IDs in the form `OMNI-0001`.
- Statuses in `Findings`: `open`, `in_progress`, `blocked`, transitional `done`.
  Statuses in `Rejected / Won't fix`: `rejected`, `wont_fix`.
- Severities: `critical`, `high`, `medium`, `low`. Efforts: `xs`, `s`, `m`, `l`, `xl`.
- `related ids` holds comma-separated IDs, `—` when none.
- Keep descriptions concise, actionable, and specific.
- Both tables use exactly this schema:

  `| id | status | severity | effort | related ids | description |`

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
- There is deliberately NO CPU backend anywhere (tpu > npu > gpu only). Never add CPU
  execution paths or fallbacks, however convenient.
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
- Run `ruff check`, the full pytest suite, coverage, and the applicable property and
  mutation gates before declaring work complete.
- If a gate cannot run, record the blocker in `TODO.md`; the work is not complete.

## Git

- Never push automatically; push only on explicit request.
- Commit only green, scoped work — one Conventional Commit per logical unit.
- Do not stage or commit unrelated changes.
- No AI or assistant attribution: no `Co-Authored-By` naming an assistant, no
  generated-by trailers.
