# Selected-text tools

`selected-text-tools` applies one requested transformation to text captured by
an explicit user action. It never monitors the clipboard, installs a clipboard
watcher, polls clipboard ownership, or retains selection history. The desktop
host reads one selection only after the user chooses an operation and passes it
to the workload under the `clipboard:read-once` permission.

The supported operations are closed and versioned:

- `explain`
- `summarize`
- `rewrite`
- `translate`, with one explicit bounded target language
- `extract-tasks`

Language selection does not globally replace the primary model. Qwen3-8B
Q4_K_M remains the route for English, Italian, Russian, every other declared
target, and every non-translation operation. The dedicated DictaLM2 route is
selected only when the request is exactly `translate` with the explicit target
`Hebrew` (case-insensitive). Hebrew-looking source text does not select it, and
the runtime does not infer a language from clipboard content. This keeps
English as the normal path while making the one separately qualified target an
operation-specific capability rather than a system-wide language preference.

The selection is non-empty and at most 32,768 Unicode characters. OmniTensor
places a closed operation/language control and the selection into two private
ephemeral fragments. The generation request contains only their opaque
references. Both fragments are discarded after success, refusal, error, or
cancellation; progress, logs, public results, and runtime snapshots never
contain the selection.

## Result and evidence

A qualified generation worker must return the closed
`selected-text-answer.schema.json` document and cite the complete selection.
OmniTensor verifies the private reference, SHA-256 digest, and exact
`0..len(selection)` span, then emits the public
`selected-text-result.schema.json` shape. Public evidence contains only
`selectionSha256`, `textSha256`, and the span. It never exposes the private
fragment reference or selection text.

Task strings are allowed only for `extract-tasks`; all other operations must
return an empty task list. Results and extracted tasks are suggestions for
review. This workload has no edit, paste, task-creation, shell, or file-effect
port.

## Dependencies, acceptance, and accelerator policy

No additional base dependency is needed for the schemas, validation, private
fragment lifecycle, or plugin contract. A usable installation additionally
needs an external qualified `GenerationWorker`, its pinned model artifacts,
tokenizer, and native runtime. The accepted Vulkan implementation uses the
pinned Qwen3-8B Q4_K_M artifact as its primary route and the pinned DictaLM2.0
7B Instruct Q4_K_M artifact only for explicit Hebrew translation. Both models
are Apache-2.0 and are verified by exact size and SHA-256 before import. The
workload contract itself remains provider-neutral.

The frozen 16-case CC0 corpus covers all five operations, English default
routing, Italian translation through Qwen, six explicit Hebrew translations,
task extraction, and a prompt-injection translation. On 13 August 2026, the
installed worker passed on the named RX 6600 XT with 37/37 Qwen layers
and 33/33 DictaLM layers offloaded to Vulkan and no CPU fallback. The report
recorded 0.923 operation-term recall, 1.0 task recall, Hebrew-script integrity,
prompt-injection integrity, and provider-route integrity, an 18,048 ms p95,
and 802 ms cancellation. These measurements qualify the frozen cases and exact
artifact/runtime bytes; they do not claim arbitrary-selection quality or a
different GPU/runtime.

New collection never reconstructs accelerator claims from installed metadata
or collector constants. After the worker has actually loaded both models, it
atomically publishes a private, mode-0600 receipt in its service-provided state
directory. That receipt records the artifact digests, runtime, physical device,
backend, device API, total and accelerator layer counts, and CPU-fallback flag
reported by the live worker. Startup clears stale bytes, normal stop removes
them, and the collector accepts a receipt only after `describe-plugins` reports the current
worker ready. The collector requires the receipt, verifies the local model
paths against it, and copies its two measured model records verbatim into
evidence before the normal acceptance gate runs. Invalid or absent live-load
evidence aborts before the corpus or output file; no public inventory
field is added.

GPU is the only accepted route. An operator may configure an NPU-first provider only
after separately qualifying its artifact, tokenizer, native runtime, output
contract, and device behavior; pre-generation admission may then fall back to
GPU. No CPU provider or CPU fallback is accepted. Until a complete worker
advertises readiness, the workload remains disabled and the Cinnamon action is
unavailable. Readiness version 1.1.0 requires both artifacts and the complete
acceptance set; no global language setting is fixed to Hebrew or Russian.
