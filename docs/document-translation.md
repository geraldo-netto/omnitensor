# Document translation

`document-translation` translates the documents named in one explicit
selection, in full, into one requested language. It is review-only: the
translation is returned to the caller and is never written over — or beside —
a source file. Nothing about the request leaves the machine.

## Runtime and dependencies

Install the base OmniTensor service plus the
`omnitensor-qwen-document-translation` wheel, which depends on
`omnitensor-vulkan-runtime[media]` — the `media` extra carries PyMuPDF,
because a selection that may contain a PDF needs the adapter that reads one.
The declaration prefers GPU and declares no other backend: there is no CPU
fallback, and no NPU or TPU lane is qualified for it.

The workload ships disabled and stays unavailable until all of these are true:

- the `document-translation` provider is installed and its worker reports
  ready;
- the user grants `files:read-selected` and `accelerator:gpu`;
- a qualified GPU is present and the artifacts below are installed.

`docs/qwen-workload-installation.md` covers installing the wheel, importing the
pinned artifacts, and granting the two permissions.

## Models

Three pinned artifacts back the workload, all `Apache-2.0`:

| Artifact | Used for |
| --- | --- |
| `qwen3-5-9b-iq4-xs` | general translation |
| `qwen3-8b-q4-k-m` | general translation |
| `dictalm2-hebrew-q4-k-m` | Hebrew, which is measured on a model trained for it |

A language is routed to the model that was measured for it rather than to
whichever model happens to be loaded, so an answer is never produced by a pair
nobody qualified for that language.

## Qualification

The Qwen pairs are recorded as `unmeasured` in the shipped receipt, so every
answer this workload returns today carries the statement that the pair it ran
on was not measured. That is a statement, not a refusal: the workload loads and
runs. Re-measuring needs the models on the GPU and is tracked as OMNI-0523.

The frozen acceptance metrics the measurement has to clear are in the manifest:
every span of every frozen document translated and reassembled in order; at
least 90% recall of facts, numbers and proper names; every translation written
in the requested script and no other; instructions found *inside* a selected
document translated as data and never followed; and a cancelled translation
reaching a terminal result within two seconds.

## Privacy boundary

One request names its source files explicitly. A selected directory is never
expanded and no folder is scanned. Extracted spans are held in private memory
for the life of that job and discarded on success, refusal, or cancellation.

## Request and result

```json
{"sources":["/absolute/path/report.pdf"],"targetLanguage":"Portuguese"}
```

`targetLanguage` is a language name, at most 64 characters, letters, spaces and
hyphens only.

The closed `document-translation-result.schema.json` reply carries the request
id, the requested language, the provider id and the accelerator that ran it,
and one entry per source document: an opaque `documentId`, the source
*basename* only, the source SHA-256, how many spans were reassembled, the
translated text, and the SHA-256 of that text. Absolute paths never appear in
the result.
