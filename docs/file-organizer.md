# File organizer

`file-organizer` suggests tags, names, relative folders, and exact duplicate
groups for files selected in one manual request. It produces a review-only
plan. It has no move, rename, overwrite, delete, or command capability.

## Runtime and dependencies

Install the base OmniTensor service plus a separately installed, qualified Qwen
generation worker. The bundled declaration prefers GPU. An NPU can be used only
when an administrator installs and qualifies a compatible provider; there is no
CPU fallback. PDF and image text extraction use the optional document ingestion
dependencies documented for Ask selected files. Plain `.txt` and `.md` inputs
need only the base runtime.

The worker is disabled until all of these are true:

- an external `file-organizer` worker is installed and reports ready;
- the user grants `files:read-selected`;
- a qualified GPU or explicitly configured NPU Qwen provider is ready.

## Privacy and safety boundary

Each request contains one to sixteen explicit absolute file paths. A selected
directory is never expanded. OmniTensor rejects links, unsupported types,
duplicates, oversized files, and files that are not regular files. It extracts
at most two bounded spans per file, keeps them in private memory for that job,
and discards them on success, refusal, or cancellation.

Qwen proposes descriptive metadata only. OmniTensor validates safe basename and
relative-folder syntax, requires each proposal to cite the selected file's
exact page/span/digests, and derives duplicate groups independently from the
full source SHA-256. Public output contains selected basenames and evidence
addresses, never absolute paths or source text.

The result remains advice. A client may display or export it for review, but
this workload exposes no apply action and cannot change the selected files.

## Request and result

Submit the external plugin with a manual payload:

```json
{"sources":["/absolute/path/notes.md","/absolute/path/copy.md"]}
```

The closed `file-organizer-result.schema.json` response contains one ordered
plan item per source: `tags`, nullable `proposedName`, nullable
`proposedFolder`, nullable host-computed `duplicateGroup`, a bounded `reason`,
and one or more exact evidence addresses. Cancellation and every validation
failure return no plan and never change a file.
