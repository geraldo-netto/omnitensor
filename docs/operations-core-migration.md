# OMNI-0617 migration map — one operations core, two source kinds

Read-only survey, 2026-08-21. Goal (TODO row: `/backups/disk2/projects/cinnamon/omnitensor/TODO.md:27`):
unify the operation enum (explain, summarize, rewrite, translate, extract-tasks, + **ask** with its
`question` parameter), per-operation prompting, the Dicta-only-for-Hebrew-translate routing rule, and
acceptance metrics into a shared "operations core", instantiated over **inline selection**
(today `selected-text-tools`) and **file sources** (today `ask-selected-files`; `document-translation`
is subsumed as translate-over-files). Staged after OMNI-0615 (OCR fragments for ask, TODO.md:28).

All paths absolute unless prefixed; `OT` = `/backups/disk2/projects/cinnamon/omnitensor`,
`XP` = `/backups/disk2/projects/cinnamon/xpuwlm`, `CX` = `/backups/disk2/projects/cinnamon/cinnamon-xpuwlm`.

---

## 1. omnitensor inventory

### 1.1 The three manifests

**`OT/plugin-manifests/selected-text-tools.json`** (v1.1.0, manifestVersion 2)
- Capabilities: `["explicit-selection", "selected-text-transformation"]` (:5)
- Configuration schema: `answerGuidance` + `answerLength` (brief/standard/thorough) (:14–28) — identical block in all three manifests.
- Input schema (:30–39): required `selection` (1–32768 chars) + `operation` enum
  `["explain","summarize","rewrite","translate","extract-tasks"]` (:36); optional `language`
  (pattern `^\s*[^\W\d_][\w '\-]*\s*$`, max 64) (:37). `additionalProperties: false` — a source
  discriminator cannot be smuggled in without a schema change.
- Output: `$ref selected-text-result.schema.json` (:40)
- Artifacts (:43–69): `qwen3-5-9b-iq4-xs`, `qwen3-8b-q4-k-m` (selectable), `dictalm2-hebrew-q4-k-m`
  (`"selectable": false`, :62).
- Permissions: `["accelerator:gpu", "clipboard:read-once"]` (:70) — **inline side permission**.
- Acceptance (:94–144): `operation-term-recall ≥0.9` (:96), `hebrew-target-script-integrity =1`
  (:103), `task-term-recall ≥0.9` (:110), `prompt-injection-integrity =1` (:117),
  `provider-route-integrity =1` ("Only an explicit Hebrew translation target uses Dicta") (:124),
  `p95-operation-latency ≤30000ms` (:131), `cancellation-latency ≤2000ms` (:138).

**`OT/plugin-manifests/ask-selected-files.json`** (v1.1.0)
- Capabilities: `["explicit-file-selection", "grounded-document-question-answering"]` (:5)
- Input schema (:30–44): required `sources` (array 1–16, uniqueItems, item ≤4096 chars) +
  `question` (1–4096 chars). No operation field today — ask is implicit.
- Output: `$ref document-question-result.schema.json` (:45)
- Artifacts (:48–78): the two Qwens (selectable) + `bge-small-en-v1-5-ask-gpu`
  (`"selectable": false`, ncnn format, companions `model.bin`/`tokenizer.json`) (:66–77). **No Dicta.**
- Permissions: `["accelerator:gpu", "files:read-selected"]` (:79) — **file-broker permission**.
- Host responsibilities (:96–102): broker only selected files, bounded private page-span index,
  exact file/page/span/digest citations.
- Acceptance (:103–146): `retrieval-recall ≥0.9` (:105), `grounded-term-recall ≥0.9` (:112),
  `citation-integrity =1` (:119), `p95-question-latency ≤30000ms` (:126),
  `cancellation-latency ≤2000ms` (:133), `grant-revocation-latency ≤500ms` (:140) — the last one is
  file-side-only and has no inline analogue.

**`OT/plugin-manifests/document-translation.json`** (v1.0.0)
- Capabilities: `["explicit-selection", "document-transformation"]` (:5–8)
- Input schema (:38–61): required `sources` (array, minItems 1, **no maxItems in the manifest** —
  the 16 bound is enforced by `select_question_sources`/`MAX_SOURCES`, see §1.4) + `targetLanguage`
  (same pattern/64 as selected-text `language`) (:54–59).
- Output: `$ref document-translation-result.schema.json` (:62–64)
- Artifacts (:69–95): the two Qwens + `dictalm2-hebrew-q4-k-m` (`selectable:false`, :89) — same
  triple as selected-text-tools, **no BGE** (no retrieval).
- Permissions: `["accelerator:gpu", "files:read-selected"]` (:96–99) — same as ask.
- Acceptance (:128–164): `span-coverage =1` (:130), `translation-term-recall ≥0.9` (:137),
  `target-script-integrity =1` (:144), `prompt-injection-integrity =1` (:151),
  `cancellation-latency ≤2000ms` (:158). **No p95 latency, no route-integrity metric** — and its
  qualification is `"unmeasured"` (§1.5).

### 1.2 Provider wheels (thin shims) and the shim contract

- `OT/providers/selected-text-tools/src/omnitensor_selected_text_tools/__init__.py:3` —
  one re-export: `create = omnitensor_vulkan_runtime.create_selected_text_tools`.
- `OT/providers/document-translation/src/omnitensor_document_translation/__init__.py:3` —
  `create = create_document_translation`.
- `OT/providers/ask-selected-files/src/omnitensor_ask_selected_files/__init__.py` — the only
  non-shim wheel: builds `BgeVulkanEmbedder` from the ncnn distribution, wraps
  `DocumentQuestionPlugin` in `QualifiedWorkload` (:29–49); `BGE_ARTIFACT_ID =
  "bge-small-en-v1-5-ask-gpu"` (:26). Rationale docstring :1–9 (keeping ncnn out of the
  generation-only wheel).
- `OT/providers/shim_identity.py` — shared test helper: shims must be exactly one re-export named
  `create`, and the installed shim object must *be* the runtime factory (`shim_reexport` :28,
  `assert_installed_shim_binds_the_runtime_factory` :50). Consumed by
  `OT/providers/selected-text-tools/tests/test_identity.py` and
  `OT/providers/document-translation/tests/test_identity.py`.
- Entry points (workload id → module:create):
  `OT/providers/selected-text-tools/pyproject.toml:13-14`,
  `OT/providers/ask-selected-files/pyproject.toml:19-20` (dist name
  `omnitensor-qwen-ask-selected-files` :6; force-includes its manifest as
  `omnitensor-plugin.json` :26),
  `OT/providers/document-translation/pyproject.toml:16-17` (manifest force-include :23).

### 1.3 vulkan-runtime: factories, prompting port, routing, qualification, grounding

`OT/providers/vulkan-runtime/src/omnitensor_vulkan_runtime/factories.py`
- `_TASKS` maps workload id → task factory (:292–298): `ask-selected-files → document_question_task`,
  `document-translation → document_translation_task`, `selected-text-tools → selected_text_task`.
  Public copy via `workload_tasks()` (:301–312) — consumed by the benchmark.
- `generation_context(plugin_id, prompting)` (:251–289): bootstrap, chosen-or-default model,
  `load_qualification(plugin_id, model.id, model.sha256, _TASKS[plugin_id]())` (:269–274),
  `MemoryFragmentStore`, `LlamaVulkanRuntime(..., prompting=prompting)` (:276),
  descriptor `qwen3-workloads-gpu` (:277–287), `GenerationRouter`.
- **Dicta route assembly** `_hebrew_route(bootstrap, store)` (:329–351):
  `HEBREW_ARTIFACT_ID = "dictalm2-hebrew-q4-k-m"` (:69), `HebrewTranslationRuntime`,
  descriptor `dictalm2-hebrew-gpu` (:342–349). Shared by exactly the two translating workloads.
- `create_document_translation()` (:354–372): `DocumentTranslationPlugin(router, store,
  translation_routes={"Hebrew": hebrew_router})`, Dicta as `additional_runtimes`.
- `create_selected_text_tools()` (:375–400): `SelectedTextPrompting()` passed into
  `generation_context` (:377); same `translation_routes={"Hebrew": ...}` (:387); publishes the
  selected-text worker load receipt with fixed roles `("primary", ...), ("hebrewTranslation", ...)`
  (:395–399) — role tuple validated in `QualifiedWorkload.__init__` (:101–109).
- `QualifiedWorkload` (:86–200): startup GPU-load proof (`validate_gpu_load`, no-CPU rule comment
  :138–142), embedder preflight for ask (:149–150).

**Prompting port** — `OT/src/omnitensor/plugins/prompting.py`: `TaskPrompting` protocol
(`hint` :25, `reconsideration` :38), `NO_PROMPTING` (:80). Consumed by
`OT/providers/vulkan-runtime/src/omnitensor_vulkan_runtime/runtime.py`:
constructor param (:129, :147); `hint = citable_hint(...) + self._prompting.hint(...)` (:404),
one `reconsideration` re-ask (:425–430). `hints.py:19` (`citable_hint`) supplies the trusted
sourceRef list only.

**Grounding bindings** — `OT/providers/vulkan-runtime/src/omnitensor_vulkan_runtime/grounding.py`:
`TASK_BINDINGS` keyed by task id (:343–384): `selected-text-tools` (:344 — exact 2 refs, span must
match, no page), `ask-selected-files` (:352 — ≥2 refs, control not bindable, citation dedup),
`document-translation` (:364 — 2 refs, host binds the span digest; comment :359–363 explains why the
general route needed it). `bind_grounding_metadata` entry (:19). This table is the third place
(after `_TASKS` and hebrew `_CONTRACTS`) where the workload id triple is hard-coded.

**Dicta routing rule** — three cooperating layers:
1. *Workload layer* (which router is picked): `selected_text.py:208-211` `_route(operation,
   language)` — only `operation == "translate"` with a language looks in `translation_routes`;
   `document_translation.py:242-250` `_route(language)`. Routes installed only for key `"Hebrew"`
   (factories.py:364, :387).
2. *Runtime layer* (Dicta refuses anything else):
   `OT/providers/vulkan-runtime/src/omnitensor_vulkan_runtime/hebrew.py` —
   `HEBREW_TARGET = "hebrew"` (:18), `HebrewTranslationRuntime._generate_sync` (:33),
   `_selected_text_control` requires control `{language, operation}` with
   `operation == "translate"` and `language.casefold() == "hebrew"` (:62–80),
   `_document_translation_control` requires bare control text == "hebrew" (:83–90),
   per-workload answer envelopes `_selected_text_answer` (:93) / `_document_translation_answer`
   (:109), **`_CONTRACTS` keyed by the two workload ids** (:124–127), script gate
   `_validated_hebrew` (Hebrew present, no Arabic/Cyrillic) (:180–190).
3. *Acceptance layer*: `provider-route-integrity` metric (manifest :124–129) and
   `SelectedTextPolicy.minimum_provider_route_integrity` (§1.6).

**Qualification** — `OT/providers/vulkan-runtime/src/omnitensor_vulkan_runtime/qualification.py`:
`task_sha256(task)` = sha256 of the canonical task JSON (:61–62) — **any prompt text change moves
the digest**; `load_qualification(plugin_id, model_id, model_sha256, task)` (:70).
`qualification.json` (same dir): per-workload `default` + per-model `result`/`taskSha256`:
`ask-selected-files` passed, digest `6d1711…` (:38–50); **`document-translation` is `"unmeasured"`
for both models** (:51–61); `selected-text-tools` passed, digest `c6ab03…` (:88–100). Recorded on
the RX 6600 XT (:2) — per memory, only the GPU lane's gates can run on this host.

### 1.4 Domain plugins: enum, prompts, schemas, fragments

**Inline side — `OT/src/omnitensor/plugins/selected_text.py`**
- `PLUGIN_ID = "selected-text-tools"` (:39), `READ_ONCE_PERMISSION = "clipboard:read-once"` (:40).
- **The operation enum**: `OPERATIONS = frozenset({...five...})` (:41);
  `RESTATEMENT_IS_A_NON_ANSWER = {explain, summarize, rewrite}` (:45);
  `MAX_SELECTION_CHARACTERS = 32_768` (:46).
- Control fragment `{operation, language}` published beside the selection (:126–146); route pick
  `_route` (:208); result grounding `grounded_selected_text_result` (:246–291) validates the private
  `selected-text-answer.schema.json` then the public `selected-text-result.schema.json`.
- **Task/prompt**: `selected_text_task()` (:294–326) — prompt id `selected-text-operation` v2,
  taskVersion 2, one prompt serving all five operations, output schema
  `selected-text-answer.schema.json`, contextTokens 32768, deliberately no output ceiling.
- **Per-operation prompting**: `SelectedTextPrompting` (:329–382) — `hint()` reads the trusted
  control and injects `_operation_instruction(operation, language)` (:430–467, the per-operation
  instruction table, incl. translate's language-quoting fix); `reconsideration()` re-asks once when
  explain/summarize/rewrite echo the selection (`ECHOED_SELECTION_REPROMPT` :387).
- Duplicated helpers: `_validated_language` (:214), `_validated_translation_routes` (:223) —
  near-verbatim copies exist in document_translation.py (:313, :323). `target_language.py` already
  centralises the language rule (`OT/src/omnitensor/plugins/target_language.py:110-123`,
  `MAX_LANGUAGE_CHARACTERS`/`TARGET_LANGUAGE_PATTERN`/`valid_target_language`).

**File-translate side — `OT/src/omnitensor/plugins/document_translation.py`**
- `PLUGIN_ID = "document-translation"` (:71), `READ_PERMISSION = "files:read-selected"` (:72),
  `TRANSLATION_WINDOW_SHARE = 2` (:77).
- Module docstring :1–20 states the very split OMNI-0617 undoes ("one shared workload would have to
  weaken both contracts") — grep bait when rewriting.
- Extraction adapters: `.txt/.md` PlainText, `.pdf` PyMuPdf (:113–116) — **narrower than ask's**
  (no images; OMNI-0615 OCR lands on ask's side).
- Sources validated by `select_question_sources` (:231) — the same function ask uses.
- Span pipeline `_translate_source` (:252–301) over `translate_document`
  (`OT/src/omnitensor/plugins/translation.py`) and span sizing `_translation_tokens` (:349–359);
  per-span control = bare targetLanguage fragment (:171–177); span answer check `_span_translation`
  (:362–382); public result `translated_documents_result` (:385–405) →
  `document-translation-result.schema.json`.
- Task/prompt `document_translation_task()` (:418–449): prompt id `document-translation-span` v1,
  taskVersion 1, output schema `document-translation-answer.schema.json`. **No TaskPrompting class**
  — `create_document_translation` passes NO_PROMPTING (factories.py:355).

**File-ask side — `OT/src/omnitensor/plugins/document_qa.py` + `document_answer.py`**
- `document_qa.py`: `PLUGIN_ID = "ask-selected-files"` (:67), `READ_PERMISSION` (:68),
  `MAX_QUESTION_CHARACTERS = 4096` (:69). Adapters incl. images via PyMuPdf
  (`.pdf/.png/.jpg/.jpeg/.webp`, :100–104 — OMNI-0615 replaces the image path with OCR fragments).
  Pipeline `_answer` (:149–248): extract → `page_spans` → `retrieve_spans` (BGE) → question fragment
  → multi-pass `answer_passes` → `grounded_answer_document` per pass → joined answer +
  deduped citations → `document-question-result.schema.json` (:239).
- `document_answer.py`: `grounded_answer_document` citation validation (:24–81), `FAILURE_DETAILS`
  (:92–113), **task/prompt** `document_question_task()` (:129–162): prompt id
  `grounded-selected-file-answer` v1, output schema `document-question-answer.schema.json`,
  contextTokens 32768, no output ceiling (rationale :152–160).
- **Fragment/grounding machinery for ask** — `OT/src/omnitensor/plugins/document_spans.py`:
  `MAX_SOURCES = 16` (:30), `MAX_SPAN_CHARACTERS = 2048` (:31), `select_question_sources` (:92),
  `page_spans` (:103), `answer_passes` (:174), `IndexedSpan` (:71), `TextEmbeddingWorker`/
  `EmbeddingProvider` (:47–55). `OT/src/omnitensor/plugins/fragments.py`: `SourceFragment` (:28),
  `PrivateFragmentStore` protocol (:39), `MemoryFragmentStore` (:55, unique-reference rule :69–70).

### 1.5 Result/answer schemas (per-contract validation targets)

| Contract | Private (model answer) | Public (client result) |
|---|---|---|
| selected-text-tools | `OT/schemas/selected-text-answer.schema.json` (operation enum repeated in it) | `OT/schemas/selected-text-result.schema.json` (operation enum, evidence selectionSha256/span/textSha256; accelerator enum `["gpu","npu"]`) |
| ask-selected-files | `OT/schemas/document-question-answer.schema.json` | `OT/schemas/document-question-result.schema.json` (answer + citations fileId/fileName/sourceSha256/page/span/textSha256) |
| document-translation | `OT/schemas/document-translation-answer.schema.json` (per-span `translation` + sourceRef/textSha256 evidence) | `OT/schemas/document-translation-result.schema.json` (targetLanguage + documents[fileName/spanCount/text/sourceSha256/translationSha256]) |

Acceptance report schemas: `OT/schemas/selected-text-acceptance.schema.json`,
`OT/schemas/document-question-acceptance.schema.json`. (No document-translation acceptance schema —
consistent with it being unmeasured.)

The **operation enum literal appears in 4 places** today: manifest input (:36), both selected-text
schemas, and `selected_text.py:41` — plus xpuwlm's dropdown (§2).

### 1.6 Acceptance / qualification machinery

- `OT/src/omnitensor/plugins/selected_text_acceptance.py`: `SELECTED_TEXT_WORKER_LOAD_RECEIPT`
  (:34), pinned runtime/device (:39–40), `SelectedTextPolicy` minimums (operation-term-recall .9,
  task-term-recall .9, target-script-integrity 1.0, injection 1.0, **provider-route-integrity 1.0**)
  (:122–127), corpus loader requiring every operation represented and a Hebrew injection probe
  (:189–193), `qualify_selected_text` (:317).
- `OT/src/omnitensor/plugins/document_acceptance.py`: `DocumentAcceptancePolicy` (retrieval .9,
  grounded-term .9, citation-integrity 1.0) (:107–110), `qualify_document_questions` (:285).
- Corpora: `OT/evaluation-corpora/selected-text-v1.json`, `document-question-v1.json` (none for
  document-translation). Evidence/reports: `OT/acceptance-evidence/selected-text-rx6600xt-*.json`,
  `document-question-rx6600xt-*.json`.
- Collector: `OT/scripts/collect-selected-text-acceptance.py` (hard-codes
  `"workloadId": "selected-text-tools"` :61, :81, and pins worker version "1.1.0" :113).

### 1.7 Tests pinning these contracts (omnitensor)

- `OT/tests/test_generation_installation.py` — **`DECLARED_ARTIFACTS`** dict pinning each
  manifest's exact artifact id list (:278–299; three workloads at :279, :280–285, :298);
  both-directions manifest↔distribution check (:317–327); entry-point/manifest identity + artifact
  list equality per plugin (:333–342); prose knowledge of ask's two-distribution build (:421, :441,
  :483). Also home of the shim architecture note referenced by `providers/shim_identity.py`.
- `OT/tests/test_selectable_artifacts.py` — **`FIXED_ROLE`** dict (:23–38): ask→{bge…},
  document-translation→{dictalm2…}, selected-text-tools→{dictalm2…}; parametrized checks (:45–80).
- `OT/tests/test_client_workload_contracts.py` — **`CLIENT_PAYLOADS`** transcription of xpuwlm's
  payload builders (:33–54), **`NOT_CLIENT_DRIVEN` currently empty** (:61),
  **`CLIENT_SOURCE_BOUNDS`** (ask 16, :66), operation-enum equality test (:119–126), and the
  both-directions shipped↔client-driven check (:129–144). This file breaks on *any* manifest
  add/remove/rename.
- `OT/tests/test_selected_text.py` (548 lines) — plugin behaviour: enum refusals, language rules,
  control fragment, evidence checks, Dicta route pick.
- `OT/tests/test_document_translation.py` (382) — span pipeline, control, result contract.
- `OT/tests/test_document_qa.py` (1119) — retrieval, passes, citations, embedder gating.
- `OT/tests/test_workload_prompt_contract.py` — every operation reaches the model as an
  instruction (parametrized over the enum, :188–218), Português quoting (:228), echo re-ask
  (:280–304).
- `OT/tests/test_plugin_prompting.py` — the TaskPrompting port contract.
- `OT/tests/test_selected_text_acceptance.py` (831) — metric names ↔ policy mapping (:753–757).
- `OT/tests/test_document_acceptance.py` — metric names (:168–170).
- `OT/tests/test_selected_text_collector.py` — collector script contract.
- `OT/tests/test_target_language.py` — shared language rule (manifests + module in lockstep).
- `OT/tests/test_plugin_configuration_spec.py` — `GENERATION_WORKLOADS` tuple naming all five
  (:220–226); every generation workload must declare the two tunables (:229+).
- `OT/tests/test_documentation.py` — `WORKLOAD_DOCS` map: ask→`docs/document-questions.md`,
  document-translation→`docs/document-translation.md`, selected-text-tools→
  `docs/selected-text-tools.md` (:477–486); content checks of `docs/selected-text-tools.md` (:332).
- `OT/tests/test_benchmark.py` — workload lists (:291–294), answer-shape map
  (`ask-selected-files → document-question-answer`, selected-text entry) (:517–519); reaches tasks
  through `workload_tasks()`. Plus `test_benchmark_reach.py`.
- `OT/tests/test_scripts.py` — collector inventory fixture pins id `selected-text-tools` (:126).
- `OT/tests/test_hebrew_installation.py` — Dicta artifact installation.
- `OT/tests/test_plugin_protocol.py`, `test_workload_result.py`, `test_client_workload_e2e.py`
  (live service), `test_accelerator_routing_regressions.py` — secondary references.
- Provider suites: `OT/providers/selected-text-tools/tests/test_identity.py`,
  `OT/providers/document-translation/tests/test_identity.py`,
  `OT/providers/ask-selected-files/tests/test_workload.py` (default model + plugin id :54, :96),
  `OT/providers/vulkan-runtime/tests/test_runtime.py` — TASK_BINDINGS pins (:168–191: asserts
  selected/ask binding flags and that bindings are *not* a closed list of ids), qualified-task
  digests (:415–416), Dicta contract tests for both workloads (:1294–1477, doc-translation general
  route :1375–1432).

**mutation-selectors scope** (`OT/mutation-selectors.json`): `scope.onlyMutate` includes
`document_answer.py`, `document_qa.py`, `document_spans.py`, `fragments.py` — **not**
`selected_text.py`, `document_translation.py`, or a future operations-core module. Shard
`grounded-answer` = selectors over document_spans/document_answer/document_qa. Moving/renaming
those modules breaks the selector paths. (Per user memory: never write/run mutation tests; but the
selectors file itself must stay path-consistent.) Baselines in `OT/mutation-baselines/`,
launcher pinned by `OT/tests/test_mutation_campaign.py`.

### 1.8 Where each unification concern lives today (summary table)

| Concern | Inline (selected-text-tools) | File-ask | File-translate |
|---|---|---|---|
| Operation enum | `selected_text.py:41`; manifest :36; 2 schemas | implicit "ask" | implicit "translate" |
| Per-op prompting | `SelectedTextPrompting` + `_operation_instruction` (`selected_text.py:329,430`) | system prompt in `document_question_task` (`document_answer.py:129`) | system prompt in `document_translation_task` (`document_translation.py:418`) |
| Dicta rule | `_route` :208 + `hebrew._CONTRACTS`/`_selected_text_control` | n/a | `_route` :242 + `hebrew._document_translation_control` |
| Acceptance | `SelectedTextPolicy` + manifest metrics | `DocumentAcceptancePolicy` + manifest metrics | manifest metrics only; **unmeasured** |
| Fragments | control+selection pair | question + page spans (`document_spans.py`) | targetLanguage control + spans (`translation.py`) |
| Permission | `clipboard:read-once` | `files:read-selected` | `files:read-selected` |

---

## 2. xpuwlm inventory (read-only)

xpuwlm imports omnitensor's schema registry directly (`XP/src/xpuwlm/contracts.py:13` —
`from omnitensor.registry import load_schema, validate_document`), so schema renames propagate
automatically; **workload ids and field names are transcribed**, not derived.

- **Specs (the main transcription)** — `XP/src/xpuwlm/workflows/specs/catalog.py`:
  - `ASK_ACTIONS` / `ASK_ACTION_QUESTIONS` (:48–70): a *client-side proto-enum* over ask
    (summarize/explain/key-points/extract-tasks as prefilled question text — comment :34–46
    explicitly says rewrite/translate were excluded and translate "is a different workload").
    This is the client-side duplicate the operations core absorbs.
  - Payload builders: `_question_payload` (`sources`+`question`, :76), `_translation_payload`
    (`sources`+`targetLanguage`, :84), `_selected_text_payload` (`selection`+`operation`+optional
    `language` only when translate, :91–100).
  - `SPECS["ask-selected-files"]` (:104–144): sources max 16, action choice, question ≤4096,
    `result_schema="document-question-result"`.
  - `SPECS["selected-text-tools"]` (:145–188): selection ≤32768, operation dropdown with the five
    literals (:163–169), language `required_when=("operation","translate")` (:182).
  - `SPECS["document-translation"]` (:229–256): sources max 16 + TRANSLATION_FILTERS,
    targetLanguage ≤64, `result_schema="document-translation-result"`, Stop-between-spans note.
- **Presenters/summaries** — `XP/src/xpuwlm/workflows/specs/summaries.py`:
  `summarize_document_question` (:59, citations fileName/page/span), `summarize_selected_text`
  (:89, result+tasks), `summarize_translation` (:269–300, targetLanguage/documents/spanCount/text).
- **Presentation strings** — `XP/src/xpuwlm/domain/presentation.py`: ask :47, selected-text :52,
  document-translation :67.
- **CLI** — `XP/src/xpuwlm/cli.py`: generic verbs only; `run <workload> --set FIELD=VALUE`
  (:113–121, help text names `selected-text-tools` :114), `ui [workload]` (:108–111),
  `workloads` (:123). No per-workload verbs — CLI resolves through `SPECS`, so re-pointing is a
  catalog change, not a CLI change.
- **Result validation** — `XP/src/xpuwlm/workflows/jobs.py:309,355-356` and
  `workflows/runner.py:110` validate outputs against `spec.result_schema`;
  `XP/src/xpuwlm/ui/workload_panels.py:209-210` displays the contract name.
- **Tests pinning ids/fields**:
  - `XP/tests/test_specs.py`: bounds ask 16 / dt 16+64 / selection 32768 (:46–54); **operation
    tuple equality** (:57–58); language required_when (:61+); payload shapes (:103, :108).
  - `XP/tests/test_command_line.py:72`: `run selected-text-tools --set selection=...`.
  - `XP/tests/test_presenters.py:149, :333`; `XP/tests/test_workload_rows.py:77`;
    `XP/tests/test_domain.py:302-338` (model choice incl. `dictalm2-hebrew` id);
    `XP/tests/test_runtime.py:251,276`; `XP/tests/test_settings_window.py:98-131` (settings rows
    for all three); `XP/tests/test_widget_assembly.py:75-129` and `XP/tests/test_workspace.py:49-111`
    (favourites `ask-selected-files`); `XP/tests/test_activity_log.py:176-177`;
    `XP/tests/conftest.py:241`.
- The omnitensor↔xpuwlm agreement is enforced from the omnitensor side by
  `OT/tests/test_client_workload_contracts.py` (transcribed from `catalog.py`, its header says so).

## 3. cinnamon-xpuwlm applet (read-only)

**No references to any workload id.** `grep` over `CX/files`, `CX/scripts`, `CX/tests` finds none.
The applet is a launcher/status tray: `CX/files/cinnamon-xpuwlm@geraldo-netto/lib/xpuwlm-launcher.js:60-73`
spawns `xpuwlm <verb>` generically; `applet.js:5-15` documents that workload screens moved into
xpuwlm. **No applet re-pointing is needed at any stage** — only xpuwlm.

---

## 4. Proposed staged migration (each stage independently green + committable)

Precondition: OMNI-0615 (OCR fragments for ask) lands first — the TODO row stages 0617 after it,
and the file-side "every operation works on scans" promise rides on OCR fragments carrying
sha256/span provenance like text fragments.

### Stage 0 — extract the operations core, zero contract movement
New module `OT/src/omnitensor/plugins/operations.py` (name free; **do not move**
document_spans/document_qa/document_answer/fragments — they are in `mutation-selectors.json
scope.onlyMutate` and the `grounded-answer` shard):
- `OPERATIONS`, `RESTATEMENT_IS_A_NON_ANSWER`, `_operation_instruction` table, the echo-re-ask
  logic, and a single `validated_translation_routes`/`validated_language` pair (today duplicated at
  `selected_text.py:214,223` and `document_translation.py:313,323`; `target_language.py` is the
  precedent for this kind of extraction — its docstring even records the four-copies problem).
- `selected_text.py` and `document_translation.py` import from it and **re-export every public
  name** (`OPERATIONS`, `SelectedTextPrompting`, etc.) so `tests/test_selected_text.py`,
  `test_workload_prompt_contract.py`, and the vulkan-runtime imports stay green.
- Do **not** touch prompt strings: `task_sha256` (qualification.py:61) would move and invalidate
  `qualification.json` digests for measured workloads.
Green because: no manifest, schema, task, or id changes; pure refactor + re-exports.

### Stage 1 — ask joins the enum (additive, wire-compatible)
- Add `"ask"` to the core enum with its `question` parameter rule (question required iff
  operation==ask, exactly as language is required iff operation==translate — mirror
  `required_when` in xpuwlm and `_validate_request` in the plugin).
- Manifest `ask-selected-files.json`: add **optional** `operation` (enum incl. `"ask"`,
  `default "ask"`) to the input schema; keep `question` required only via a conditional
  (`if operation==ask then require question`); bump manifest version 1.1.0→1.2.0.
- `document_qa.py._validate_request` accepts the absent field as `"ask"`.
- Keep `document_question_task` prompt bytes unchanged in this stage (digest `6d1711…` survives;
  xpuwlm's ASK_ACTION_QUESTIONS comment :57–60 records that payload-only changes don't need
  re-qualification).
- Update: `test_client_workload_contracts.CLIENT_PAYLOADS` (optional field — actually stays green),
  add enum test; xpuwlm may later convert `ASK_ACTIONS` fills into real operations (Stage 4).
Green because the field is optional: old xpuwlm payloads validate unchanged.

### Stage 2 — file-side operations plugin (ask's successor grows the enum internally)
- Introduce a file-source operations plugin (evolve `DocumentQuestionPlugin` or a new
  `FileOperationsPlugin` in a **new** module) that dispatches on operation:
  - `ask` → existing retrieval+citation pipeline (unchanged path).
  - `explain/summarize/rewrite/extract-tasks` → whole-content operation over fragments (reuse
    `page_spans` extraction; per-operation instructions from the core).
  - `translate` → the span pipeline lifted from `document_translation.py`
    (`translate_document`, `_translation_tokens`, `_span_translation`) with `language` as the
    parameter (unifying `targetLanguage` under the core's translate rule).
- Wire the Dicta rule once: give the file side the *selected-text-style* `{operation, language}`
  control so `hebrew.py` needs **one** control contract instead of two; add the new task id to
  `hebrew._CONTRACTS` and `grounding.TASK_BINDINGS` (test_runtime.py:191 deliberately allows
  additional ids). Give the file-side task its own `TaskPrompting` built from the core.
- Manifest: add `dictalm2-hebrew-q4-k-m` (selectable:false) to `ask-selected-files.json`; extend
  acceptance with the translate metrics (span-coverage, target-script-integrity,
  provider-route-integrity).
- **Breaks to update in the same commit**: `DECLARED_ARTIFACTS["ask-selected-files"]`
  (test_generation_installation.py:279), `FIXED_ROLE["ask-selected-files"]`
  (test_selectable_artifacts.py:24), provider wheel gains the Dicta route
  (`providers/ask-selected-files/.../__init__.py` calls `_hebrew_route` — export it or a
  file-side factory from `factories.py`).
- **Re-qualification**: the file-side task prompt changes → new `taskSha256`; on this host only the
  GPU lane can be re-measured (per host-accelerator memory). Until measured, the pair shows
  `unmeasured` — same state document-translation is in today, so no regression.
Green: ask's wire contract still accepts old payloads; new operations are additive.

### Stage 3 — document-translation becomes a compat alias, then retires
- 3a (alias): keep `plugin-manifests/document-translation.json` + wheel, but its `create()` builds
  the Stage-2 file-side plugin pinned to `operation="translate"` and mapping
  `targetLanguage→language` (a ~20-line adapter in `factories.create_document_translation`).
  Old xpuwlm windows and old jobs keep working; `document-translation-result.schema.json` is still
  produced by the adapter (translate the core result back, or keep the result shape as the alias's
  contract). All pins (DECLARED_ARTIFACTS, FIXED_ROLE, CLIENT_PAYLOADS, WORKLOAD_DOCS,
  GENERATION_WORKLOADS, SPECS) stay untouched → green.
- 3b (client re-point): xpuwlm `SPECS["document-translation"]` window is replaced by a translate
  entry over the file-side workload (catalog.py:229–256 folded into the ask/file spec; delete
  `_translation_payload`, `TRANSLATION_FILTERS` merge, keep `summarize_translation` keyed to the
  new result shape). Update `XP/tests/test_specs.py:52-53`, settings-window test :98,
  presentation :67.
- 3c (retire): delete manifest + wheel + `document_translation.py` +
  `document-translation-*` schemas. Update in one commit: `DECLARED_ARTIFACTS` /
  both-direction check (test_generation_installation.py:317–327), `FIXED_ROLE`,
  `CLIENT_PAYLOADS`+`NOT_CLIENT_DRIVEN`+both-direction check
  (test_client_workload_contracts.py:129–144), `GENERATION_WORKLOADS`
  (test_plugin_configuration_spec.py:220–226), `WORKLOAD_DOCS` + `docs/document-translation.md`
  (test_documentation.py:482), `hebrew._CONTRACTS`/`grounding.TASK_BINDINGS` doc-translation
  entries + `test_runtime.py:1375-1432`, `qualification.json` workloads block (:51–61),
  `factories._TASKS`/`create_document_translation`, provider suite, `test_target_language.py`
  references, `test_document_translation.py` (delete/port to file-side tests). Per the
  TODO-completion memory rule, OMNI-0616's fold-in is recorded on the 0617 row.

### Stage 4 — inline side adopts the core enum surface (optional polish)
- xpuwlm: replace `ASK_ACTIONS` question-fills with real `operation` values now that the file side
  understands them (catalog.py:48–70), keeping "Other → ask" as default; one operations dropdown
  component shared by both windows.
- omnitensor: consider `ask` over inline selection too (question about pasted text) — additive to
  `selected-text-tools.json` the same way Stage 1 was for files.

### Acceptance-core unification (parallel to Stages 2–3)
Merge `SelectedTextPolicy`/`DocumentAcceptancePolicy` metric definitions into the core with
per-source-kind extras (retrieval/citation/grant-revocation are file-only; clipboard latency
inline-only). Keep metric *names* in manifests stable until evidence is re-collected — the
collectors (`scripts/collect-selected-text-acceptance.py:61`) and evidence files pin them.

---

## 5. Design point: two manifests sharing a core vs one manifest with a source discriminator

**Recommendation: two manifests sharing one core** (inline = `selected-text-tools`, file =
`ask-selected-files` grown into the file-side operations workload), because:

1. **Permission least-privilege is the decisive one** (the row itself flags it): the inline side
   holds `clipboard:read-once` (selected-text-tools.json:70) and the file side
   `files:read-selected` (ask-selected-files.json:79). One manifest must declare the union, so
   enabling "explain this paragraph" would also grant the file broker — and the host-responsibility
   pipelines differ correspondingly (discard-one-shot-clipboard vs broker-only-selected-files,
   manifest `pipeline` blocks). The grant-revocation acceptance metric (:140) only makes sense on
   the file side.
2. **Artifact sets differ**: file side needs BGE (ask) and will need Dicta; inline needs Dicta but
   not BGE. `FIXED_ROLE`/`DECLARED_ARTIFACTS` and the model-choice dropdown
   (`test_selectable_artifacts.py` header explains exactly the bug a merged list caused) stay clean
   per manifest.
3. **Qualification is keyed per workload id + task digest** (qualification.py:70,
   qualification.json `workloads` block). Two workload ids keep two independently measured task
   digests; one merged id would put retrieval-QA and span-translation under one digest, so any
   change to either re-unmeasures both.
4. **Wire compatibility**: both existing input schemas are `additionalProperties:false` with
   different required sets (`selection`+`operation` vs `sources`+`question`). A discriminator
   manifest is a new contract for *every* caller at once; two manifests let each side move
   additively (Stages 1–2 above never break an old payload).
5. **The distribution split already forces two wheels** (ask embeds ncnn —
   ask-selected-files `__init__.py:11-18`; the others are pure re-exports), and discovery maps one
   entry point per workload id (test_generation_installation.py:333–342).

The cost — the enum and prompting stated once per source kind — is exactly what the shared core
module removes; the manifests then only *reference* the core's enum, and
`test_client_workload_contracts.py:119` / `XP/tests/test_specs.py:57` keep the two surfaces equal.

---

## 6. Break/keep-green matrix per stage

| Stage | What breaks without action | What keeps it green |
|---|---|---|
| 0 | Imports of `OPERATIONS`/`SelectedTextPrompting` etc. from old modules; mutation-selector paths if files were moved | Re-export shims in `selected_text.py`/`document_translation.py`; create a *new* core module instead of moving listed ones; prompt bytes untouched → task digests stable |
| 1 | `ask-selected-files` input schema change → `CLIENT_PAYLOADS` validation (OT:33), xpuwlm `_question_payload` (XP catalog:76), e2e test | Field is **optional with default "ask"**; manifest version bump only; question stays required via conditional; task prompt unchanged → digest `6d1711…` stable |
| 2 | `DECLARED_ARTIFACTS` (:279), `FIXED_ROLE` (:24), ask provider wheel wiring, `hebrew._CONTRACTS`, `TASK_BINDINGS`, qualification digest for the file-side task, acceptance metric lists, `test_document_qa.py` | Same-commit updates of the four pin dicts; new ops additive behind the operation field; new task starts `unmeasured` (allowed state — document-translation lives there today); Dicta control unified on the `{operation,language}` shape hebrew.py already parses |
| 3a | Nothing external — alias preserves id, schema, artifacts, results | Alias `create_document_translation` builds the file-side plugin pinned to translate |
| 3b | xpuwlm SPECS/tests for `document-translation` (catalog:229, test_specs:52, test_settings_window:98, presentation:67); users' favourites/settings referencing the id | Client re-point in one xpuwlm commit; old service alias still answers during rollout |
| 3c | All omnitensor pins listed in Stage 3c; both-direction checks (test_generation_installation:317, test_client_workload_contracts:129); `WORKLOAD_DOCS`; `GENERATION_WORKLOADS`; `qualification.json`; docs page; e2e | One commit removing manifest+wheel+plugin+schemas+doc together with every pin — the both-direction tests are precisely the guard that this commit is complete |
| 4 | `ASK_ACTIONS` semantics change (question-fill → operation) | Keep "other/ask" default; server accepts both a question-only payload (op defaults to ask) and explicit operations |

Cross-repo ordering per stage: omnitensor first (server additive), xpuwlm second (client re-point),
**cinnamon-xpuwlm never** (no workload ids there, §3). Old-workload-id deprecation happens only at
3c, after 3b ships, so no released client ever points at a missing workload.
