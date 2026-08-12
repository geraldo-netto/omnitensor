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
extraction. The pinned text model for this workflow is the official Apache-2.0
`Qwen/Qwen3-0.6B-GGUF` Q8_0 artifact at an immutable upstream revision. GPU through
llama.cpp/Vulkan is the default. An NPU through
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
Brokering places each copy below a private ordinal directory while retaining a
safe original basename, so public citations identify the selected file rather
than an internal staging name.
BGE embeds the explicit question and spans, deterministic cosine ranking keeps
at most eight spans. Qwen receives the explicit question as a separate,
non-citable private fragment followed by only those opaque span references.

Qwen must return the closed `document-question-answer` contract with a concise
answer of at most 1,024 characters and at least one citation. The explicit
bound is accepted by llama.cpp's JSON-schema grammar while the larger job byte
limit continues to bound the complete envelope. The plugin rejects a citation
unless its private reference,
source digest, page, exact start/end offsets, and text digest all match a
retrieved span. The public `document-question-result` replaces the private
reference with `selected-file-N` plus the basename and retains the digests and
page/span address. It contains no raw fragment field or absolute path; the
bounded answer itself may quote facts needed to answer the question. Malformed,
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

## Operational acceptance

`evaluation-corpora/document-question-v1.json` is a CC0 synthetic, immutable
six-case holdout. It covers single- and multi-source retrieval, arithmetic
grounding, a changed configuration, and a document-borne prompt injection.
`omnitensor-qualify-document-questions` validates one bounded evidence document
against that exact corpus, the pinned Qwen3 digest, the digest-locked BGE
source/native report, exact answer citations, all mandatory safety probes, and
the declared quality, memory, cancellation, revocation, and latency gates:

```sh
omnitensor-qualify-document-questions \
  --evidence acceptance-evidence/document-question-rx6600xt-evidence.json \
  --output /tmp/document-question-acceptance.json
```

The archived run used BGE-small/ncnn plus Qwen3-0.6B-Q8_0/llama.cpp on the named
`AMD Radeon RX 6600 XT (RADV NAVI23)`. llama.cpp reported all 29 model layers on
Vulkan with CPU fallback forbidden. The holdout recorded 1.0 retrieval recall,
1.0 grounded-term recall, 1.0 citation integrity, 2,386 ms p95 end-to-end
latency, and 1,415,577,600 peak model/runtime bytes. Regression and integration
probes require malicious and mutated documents to fail closed, revoked grants
to refuse queued work, changed source digests to suppress stale results, an
ephemeral index to recover on the next request, cancellation to terminate, and
private fragments to be discarded on every terminal path.

The 12 August 2026 installed-provider recheck used runtime 0.1.2 through the
session D-Bus API on the same RX 6600 XT. A synthetic Mars fixture completed as
`job-succeeded`; its citation retained `omni-0242-mars.txt`, canonical page/span
and both SHA-256 values. The owner-scoped public result contained no absolute
source path or raw fragment field.

The accepted report is deliberately narrow: it qualifies this frozen public
holdout on that GPU. It does not claim quality for arbitrary private corpora,
and does not qualify an NPU or TPU lane. Provider distributions must still
verify installed artifact digests and satisfy the same gate before advertising
the external workload as ready.
