# Structured generation providers

OmniTensor's generation boundary is task- and provider-neutral. A workload
defines a versioned prompt, accepted modalities, a closed JSON output schema,
and hard context, output-token, and output-byte ceilings. A provider implements
that contract through `omnitensor.plugins.generation.GenerationWorker`.
Neither side is coupled to Qwen, llama.cpp, OpenVINO, or a particular consumer.

## Privacy and process boundary

Source text and images are private worker data. A `GenerationRequest` carries
only opaque, bounded content references; it has no field for inline file
content. The workload worker resolves those references after consent and
ingestion. Raw content, rendered prompts, native token streams, and structured
results must not enter the public runtime snapshot, D-Bus payloads, logs, or a
shared cache.

Native model runtimes execute behind the asynchronous `GenerationWorker` port,
outside the core service process. Every implementation must provide
`terminate(request_id)`, because cancellation must be able to kill a worker
whose native SDK cannot stop cooperatively. A worker may report bounded
progress through the existing progress port; progress must name stages, never
echo source content or generated tokens.

## Task contract

`parse_generation_task` accepts exactly these version 1 fields:

- stable task and prompt identifiers with independent integer versions;
- a system prompt and instruction template containing the literal
  `{{UNTRUSTED_CONTENT}}` boundary;
- unique `text` and/or `image` modalities;
- a valid Draft 2020-12 closed object schema; and
- context-token, output-token, and encoded-output byte limits.

The marker forces prompt authors to state where untrusted data enters the
instruction. It is not a claim that a model cannot be prompt-injected; each
provider/workload still needs adversarial evaluation and grounded evidence.
The output is parsed as strict JSON and validated against the task schema before
it can become a workload result.

## Provider and routing contract

A provider descriptor records its provider ID, native runtime, accelerator,
qualification state, and immutable model provenance: model ID/version, SHA-256,
HTTPS source, and SPDX license. CPU providers are rejected.

The default preference is exactly `gpu`. An operator may configure exactly
`npu,gpu` only after installing a qualified NPU provider. The router may move
from NPU to GPU only when NPU admission or model loading fails before generation
starts. It never retries another accelerator after token generation, malformed
output, cancellation, or a runtime failure; that would duplicate expensive work
and could return a result from a different model after a partial side effect.

## Implementing a provider

1. Keep all vendor imports and model loading inside a separately supervised
   workload worker.
2. Parse a checked-in provider descriptor and verify artifact provenance before
   advertising `qualified: true`.
3. Enforce the task's token ceilings in the native SDK as well as at the shared
   contract boundary.
4. Distinguish `admission-refused` and `model-load-failed` before generation
   from failures after generation begins.
5. Validate the final structured document with `validate_structured_output`.
6. Test cancellation by proving the worker process exits and releases device
   memory, not merely that the caller stops waiting.

Provider-specific model quality, prompt-injection, latency, memory, and device
evidence remain separate acceptance gates. Implementing this common boundary
does not qualify any particular model or accelerator.

The first task-specific consumer is the [grounded event result
contract](event-results.md), which keeps model proposals separate from human
confirmation and calendar effects.
