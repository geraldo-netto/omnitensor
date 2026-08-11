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

## Dependencies and accelerator policy

No additional base dependency is needed for the schemas, validation, private
fragment lifecycle, or plugin contract. A usable installation additionally
needs an external qualified `GenerationWorker`, its pinned model artifacts,
tokenizer, and native runtime. Qwen3 is a suitable GPU implementation, but the
contract is provider-neutral.

GPU is the default route. An operator may configure an NPU-first provider only
after separately qualifying its artifact, tokenizer, native runtime, output
contract, and device behavior; pre-generation admission may then fall back to
GPU. No CPU provider or CPU fallback is accepted. Until a complete worker
advertises readiness, the workload remains disabled and the Cinnamon action is
unavailable.
