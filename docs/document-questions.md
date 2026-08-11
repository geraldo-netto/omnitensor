# Ask selected files

`ask-selected-files` answers one question over files chosen in the current user
action. It is not a folder indexer, search daemon, or clipboard watcher. The
workflow receives 1–16 brokered regular files plus one bounded question, builds
an in-memory page/span index, returns an answer with exact citations, then
discards every extracted span.

## Models and optional dependencies

The retrieval model is the pinned `BAAI/bge-small-en-v1.5` recipe documented in
[local training](local-training.md#document-intelligence-bge-small-model). Its
upstream portable graph remains ONNX opset 11; the existing MiniLM mean-pooling
producer remains opset 13. Those are graph versions behind the same embedding
contract, not separate service APIs.

The checked-in producer currently qualifies only a named Vulkan GPU/ncnn BGE
variant. The embedding worker port also admits NPU and TPU implementations so a
future qualified conversion does not change the workload or result contract.
It never admits CPU. Planned NPU and TPU variants remain disabled until their
fixed pooling graph, semantic parity, compiler evidence, and named-device runs
pass the recipe gates.

Qwen generation uses the same provider-neutral worker boundary as private event
extraction. GPU through llama.cpp/Vulkan is the default. An NPU through
OpenVINO GenAI is used only after explicit configuration and local
qualification; it may fall back to GPU only before generation starts. There is
no CPU lane. Install producer-only dependencies only where models are built:

```sh
pip install 'omnitensor[document-producers,events]'
```

The base service does not import PyTorch, Transformers, ncnn, llama.cpp,
OpenVINO, PyMuPDF, or OCR libraries. A provider distribution supplies the
`omnitensor.workloads` entry point named `ask-selected-files`, the manifest in
`plugin-manifests/ask-selected-files.json`, a qualified BGE worker, and a
qualified Qwen router. The template alone is not advertised as ready. Startup
fails closed if either provider is absent, changed, CPU-backed, or unqualified.

## Private execution contract

Only a manual trigger is declared. `files:read-selected` grants access to the
files copied by the selected-file broker for this request; it does not grant a
parent folder or a later file. Directories, symlinks, empty/oversized files,
duplicate inodes, unsupported formats, and more than 16 sources are refused.
Text and Markdown use the bounded parser. PDF and images use the optional
PyMuPDF/OCR adapter inside the killable workload process.

Normalised pages are split into at most 512 spans of at most 2,048 characters.
Each entry binds the selected-file ordinal, safe basename, full source SHA-256,
page, character offsets, and span SHA-256. The index exists only in memory.
BGE embeds the explicit question and spans, deterministic cosine ranking keeps
at most eight spans. Qwen receives the explicit question as a separate,
non-citable private fragment followed by only those opaque span references.

Qwen must return the closed `document-question-answer` contract with at least
one citation. The plugin rejects a citation unless its private reference,
source digest, page, exact start/end offsets, and text digest all match a
retrieved span. The public `document-question-result` replaces the private
reference with `selected-file-N` plus the basename and retains the digests and
page/span address. It contains no source text or absolute path. Malformed,
unretrieved, duplicated, or changed citations fail the whole job.

Cancellation is checked before extraction, retrieval, and generation. Progress
contains only `extract`, `retrieve`, `answer`, and `terminal`. Private fragments
are discarded after success, refusal, cancellation, or exception. No result is
written into the public runtime snapshot; the submitting D-Bus owner retrieves
it through the owner-scoped job-result API.

## Application behavior

Cinnamon XPU WLM exposes the workflow only when `DescribePlugins` reports the
exact external distribution ready, executable, granted, and artifact-ready.
The chooser is opened by the user, the question is entered explicitly, and the
answer view renders every citation as file name, page, and character span.
There is no background ingestion, persistent history, or autonomous file
action.

The included deterministic tests exercise the complete private contract and
hostile citation drift. They do not claim production retrieval or answer
quality for arbitrary documents. That broader malicious-document, revocation,
quality, latency, and named-device evidence remains tracked by OMNI-0121.
