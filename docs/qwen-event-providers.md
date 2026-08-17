# Qwen event providers

OmniTensor catalogs Qwen2.5-VL-7B-Instruct as one optional implementation of
the provider-neutral generation contract. It is not bundled, downloaded, or
marked qualified by installation. The checked-in catalog pins the upstream
Apache-2.0 revision and the ggml-org Q4_K_M model and f16 multimodal projector
by immutable revision, byte size, and SHA-256.

The default lane for this large model is GPU through a llama.cpp build with
Vulkan enabled. A load succeeds only when the native adapter reports that every
model layer is on Vulkan and that CPU fallback is false. A partial offload is a
load failure, not a slower supported mode.

The optional NPU lane uses an operator-supplied OpenVINO GenAI adapter. It is
disabled until explicitly configured and locally qualified. Detecting an NPU
does not enable or prefer this lane. Its load report must identify OpenVINO
GenAI, device `NPU`, and no CPU fallback. The shared service never imports
either native SDK; adapters live in isolated, killable plugin workers.

## Artifact setup

Read `generation-models/qwen2-5-vl-7b.json`, obtain both files from their pinned
HTTPS `uri` values, and verify `sizeBytes` and `sha256` before installing them
in a private worker-owned artifact directory. Keep the model first and the
projector as its named companion when constructing
`LlamaCppVulkanWorker`. For NPU, convert from that recorded source through
a separately versioned local toolchain; record the resulting artifact digest,
runtime version, and device before creating a provider descriptor.

No descriptor may set `qualified: true` merely because files load. Run the
same frozen `evaluation-corpora/event-extraction-v1.json` corpus on each lane
and pass the acceptance boundary in `qualify_event_provider`:

- every result validates against the grounded-event schema;
- event precision and recall are each at least 0.90;
- the prompt-injection probe produces exactly its grounded event and no extra;
- p95 generation latency is at most 30 seconds;
- peak native memory is at most 12 GiB;
- cancellation completes within 2 seconds; and
- the lane-specific load report proves no CPU fallback.

The corpus is synthetic and CC0-1.0; it contains English, Portuguese, image,
and injection-boundary cases without user documents or calendar data. Its
file SHA-256 is part of every evidence record, so edits require a new
qualification run. A passing report is evidence for creating a new qualified
descriptor; it does not mutate or bless its unqualified input.

GPU and NPU token streams need not be byte-identical. They are considered
behaviorally equivalent only when both independently pass the same grounded
event, performance, cancellation, and device-residency gates. A post-start
failure never retries on the other lane.
