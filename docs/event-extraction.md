# Event extraction workload

`event-extraction` turns files selected by the user into evidence-backed event
candidates. It is a manual, private workflow: OmniTensor never watches or scans
a directory for it, and no source text, path, native token, or event result is
put in the public runtime snapshot.

## Install and model setup

Install base support with `pip install omnitensor`. PDF, image decoding, and
OCR use the optional isolated-worker dependency:

```sh
pip install 'omnitensor[events]'
```

Plain UTF-8 text, Markdown, and iCalendar parsing need no optional package.
PDF, PNG, JPEG, and WebP use PyMuPDF inside the workload process. OCR also
needs Tesseract language data visible to PyMuPDF. A missing parser/OCR runtime
is a bounded `adapter-unavailable` or `ocr-unavailable` refusal; it never causes
the core service to import the dependency.

The model is intentionally not bundled or downloaded. Follow
[Qwen event providers](qwen-event-providers.md), place the two pinned artifacts
in a private directory, then verify them without loading the model:

```sh
omnitensor-import-events check-model /absolute/private/model-directory
```

GPU through llama.cpp/Vulkan is the default and must report every layer on the
GPU. NPU through OpenVINO GenAI is offered only after explicit configuration
and local qualification; it may fall back to GPU only before generation starts.
There is no CPU lane. `plugin-manifests/event-extraction.json`, the provider
catalog, frozen evaluation corpus, and grounded result schema are the
coordinated contracts. Why the result schema has the shape it has — the
`needs`-based partial events, the `readAs` evidence, and the measurements that
forced both — is the design record in
[The event contract](event-extraction-contract.md). A catalog entry is not runtime qualification evidence.

The manifest in this repository is a packaging template, not a bundled live
profile. A provider distribution makes the workload available by shipping that
manifest as its single `omnitensor-plugin.json` resource and an
`omnitensor.workloads` entry point named `event-extraction`. Its factory must
construct `EventExtractionPlugin` with a qualified GPU or explicitly selected
NPU/GPU router, private fragment store, recovery journal, and native provider.
The core service discovers only this metadata; it never imports provider code.

After installation and consent, restart the service and inspect
`DescribePlugins`. The workload is usable only when its entry has
`workerState: "ready"`, advertises `execute`, every declared permission is
granted, and every declared artifact is ready. Empty or unqualified routers and
changed or missing pinned Qwen artifacts fail worker startup before `ready`.
If the provider distribution is absent, the template alone does not appear in
the inventory and `SubmitJob` refuses the workload ID.

## Consent and execution

Grant `files:read-selected` to the plugin. Each request must carry 1–32
absolute paths chosen in that user action. Directories, symlinks, empty files,
files above 128 MiB, unsupported suffixes, and selecting one inode twice are
rejected before parsing. The plugin cannot derive a directory from settings and
has no periodic or event trigger.

The service copies only those validated regular files into a private,
per-request broker directory. The worker sees that directory through one
read-only sandbox mount; it never sees the original path or the rest of its
parent directory. Copies are removed after success, refusal, or cancellation.
The broker opens sources without following symlinks, detects inode reuse and
changes during copying, and never writes, moves, renames, or deletes an
original file.

Extraction, OCR, and the native model run in the killable plugin worker.
Progress exposes only `select`, `extract`, `generate`, `validate`, and
`terminal`; it never includes content or paths. Cancellation is checked between
every source and before generation. The recovery journal stores only a request
ID and stage, then clears abandoned entries on restart. Temporary private
fragments are discarded on success, refusal, cancellation, or crash.
`SubmitJob` routes an installed plugin ID only while that exact worker remains
ready, the `execute` protocol capability was negotiated, its permissions are
active, and the submitted object satisfies the manifest input schema.

## Preview and export

The private job-result client can render a pending result without source text:

```sh
omnitensor-import-events preview /absolute/private/result.json
```

Export requires exactly one explicit decision for every candidate. The output
must be a new absolute path, so an existing calendar cannot be overwritten:

```sh
omnitensor-import-events export /absolute/private/result.json \
  --confirm event-1 --reject event-2 --output /absolute/private/confirmed.ics
```

No model result writes a calendar. An application can instead pass the same
confirmed immutable result to the `CalendarSink` port.

## `lostutils/import_events.py` compatibility

The old command combined recursive discovery, parsers, model setup, inference,
caches, preview, and writes in one process. Those responsibilities now map to
the plugin manifest, explicit source selector, isolated extraction adapters,
provider catalog/router, private fragment store, grounded-result domain, and
confirmation client respectively. `legacy-files` is a transition aid for an
explicitly named directory; it applies the workload's exact type/size/count
checks and never creates or defaults a directory:

```sh
omnitensor-import-events legacy-files /absolute/selected/folder --recursive
```

Legacy options that enable CPU inference, automatic cache downloads, silent
dedup policy changes, or unconfirmed JSON/ICS writes are deliberately not
carried forward. That is a safety change, not an attempt at flag-for-flag
compatibility. The old script can consume the listed paths through the workload
during migration, while its current output stays independent until every event
has passed the new evidence and confirmation contracts.
