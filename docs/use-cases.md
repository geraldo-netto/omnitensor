# Cinnamon workload use cases

OmniTensor ships the same nine version 1 workload manifests as the
`cinnamon-tpuwlm` applet. This gives the runtime and panel one stable profile
catalog: policy commands recognize every built-in ID, defaults match, routing
uses the same accelerator preferences, and snapshots publish the same profiles.

Profile presence is not a claim that an end-to-end ML application ships.
Manifests describe workload identity, policy, routing requirements, and
acceptance gates. Model weights, input collectors, preprocessing,
post-processing, deterministic actions, and result consumers remain separate
trusted integrations unless explicitly listed below.

## Delivered runtime behavior

| Layer | Delivered behavior |
| --- | --- |
| Catalog | Nine schema-valid manifests bundled in source distributions and wheels |
| Policy | Enable/disable, weights 1–5, global pause, durable revision, and optimistic concurrency |
| Routing | Ordered `tpu`, `npu`, `gpu` selection constrained by model format; never CPU fallback |
| Scheduling | Serialized dispatch per backend with weighted soft shares |
| Observation | Contract-valid devices, queue depth, profile state, and routing detail in snapshots |
| Inference pipeline | Version 1 bounded `SubmitJob`/`CancelJob` D-Bus methods and backend scheduler exist; verified artifact dispatch and host use-case pipelines remain readiness-gated |

`idle` with `no model bundled` means the catalog profile has no concrete model
selection. `watching` means a manifest names a model format and a compatible
backend route is available. It does not prove model weights are installed,
acceptance measurements passed, or the surrounding host pipeline exists.
`running` means the scheduler is invoking a submitted job, not that a complete
application action succeeded.

## Built-in profiles

All profiles prefer `tpu`, then `npu`, then `gpu`, and require at least one
device selected for the route. A concrete model must still use a format the
selected executor supports.

| Profile | Default | Runtime use case | Model and host boundary |
| --- | --- | --- | --- |
| `hardware-health` | enabled, weight 2 | Anomaly scoring over power, UPS, memory, thermal, or service-health features | No model selected. Host collects bounded telemetry; firmware limits, kernel OOM handling, UPS shutdown thresholds, and repair actions remain deterministic. |
| `storage-intelligence` | enabled, weight 3 | Storage-pattern classification and anomaly scoring for SMART, I/O, cache, or tiering advice | No model selected. Host reads SMART and I/O data and applies any priority, cache, or maintenance policy. |
| `resource-scheduler` | enabled, weight 3 | Queue classification and advice for placement or background-job timing | No model selected. Host supplies job features and enforces bounded scheduling policy; accelerator output never executes arbitrary commands. |
| `build-advisor` | enabled, weight 2 | Build classification and regression-risk ranking | No model selected. Host extracts change/history features. Predictions may reorder checks but must not silently omit required checks. |
| `visual-library` | enabled, weight 2 | Image classification or embedding extraction for tagging and semantic search | No model selected. Host decodes, resizes, normalizes, stores embeddings, searches indexes, calibrates thresholds, and renders results. Exact duplicates use hashing instead. |
| `low-light-enhancement` | disabled, weight 2 | Bounded tonal-curve estimation for dark images | Requires fully quantized `low-light-tonal-curve` 0.1.0 in `tflite-edgetpu` format. Package contains metadata, not model weights. Host decodes and validates inputs, prepares tensors, applies the curve, denoises, color-manages, encodes, and routes results. |
| `network-peripherals` | disabled, weight 2 | Aggregated network or peripheral anomaly scoring | No model selected. Host captures and aggregates metadata, extracts features, and owns alerts or restricted responses. Per-packet inference is normally unsuitable. |
| `desktop-context` | disabled, weight 1 | Context prediction and window-layout suggestions | No model selected. Host gathers consented desktop context and offers allowlisted suggestions; user and window-manager policy remain authoritative. |
| `document-intelligence` | disabled, weight 2 | Document classification, similarity, or layout detection | No model selected. Host validates files, parses or renders content, performs OCR/tokenization, hashes exact duplicates, indexes results, and handles privacy policy. |

## Low-light acceptance gates

The named low-light model stays disabled by default. Enabling it for production
requires reproducible evidence for every manifest criterion:

- 100% of model operations mapped to the Edge TPU by the compiler report;
- quantized SSIM of at least 0.95 against the frozen float baseline;
- median CIEDE2000 color error no greater than 3;
- warm p95 device latency no greater than 50 ms for 256×256 inputs on the
  named Coral USB reference device; and
- end-to-end speedup of at least 1.2 over the documented CPU tone-mapping
  baseline, including host work.

Record model, compiler, runtime, driver, and host versions; input shape; warm-up;
sample count; preprocessing; post-processing; accuracy dataset; and baseline.
The other eight manifests intentionally have no selected model or acceptance
criteria yet. Define both before presenting their catalog entry as operational
inference.

## Catalog locations and extension

Built-ins live under `workloads/<id>/manifest.json` in a source checkout and
`omnitensor/workloads/<id>/manifest.json` in an installed wheel. The runtime
also reads user manifests from `OMNITENSOR_WORKLOADS`, defaulting to
`~/.local/share/omnitensor/workloads`. Bundled identities win collisions, so a
user directory cannot replace a built-in profile.

The applet keeps its own declarative catalog. A third-party profile must be
installed in both its applet workload directory and OmniTensor's user workload
directory until a shared installation mechanism is versioned. Keep the same
manifest bytes in both locations. Directory name and manifest `id` must match,
and the combined runtime catalog is bounded to 128 profiles.

Before adding inference:

1. Select and validate an accelerator-compatible model; never add a CPU path.
2. Build a trusted host adapter that bounds and validates inputs before tensor
   conversion.
3. Submit only allowlisted model paths and validated tensors through an
   in-process application integration; version the D-Bus contract before
   exposing job submission to external clients.
4. Keep post-processing, actions, safety limits, authorization, and persistence
   outside model output.
5. Add unit, integration, regression, property, coverage, and changed-function
   mutation gates plus reproducible hardware acceptance measurements.

The canonical manifest and runtime schemas remain under `schemas/`. Coordinate
any incompatible catalog or contract change with `cinnamon-tpuwlm` before
release.

Locally trained model digests cannot be known when a built-in manifest is
packaged. [Local model training](local-training.md) therefore installs a
separate restricted binding: it may replace only model and routing fields for
an existing built-in identity. Profile defaults, permissions, UI, capabilities,
and host responsibilities remain immutable. One portable ONNX fit can produce
matching ncnn GPU and OpenVINO NPU variants; no executable model file itself is
hardware-agnostic, and Edge TPU production still needs a separate fully-int8
TFLite and compiler path.

Detailed bounded collector contracts for the `network-peripherals` profile are
documented in [Network and peripheral collection](network-peripherals.md).

## Accelerator lanes available for qualification

The `tpu > npu > gpu` preference is a routing order, not an availability
claim. What a given host can actually qualify against decides which acceptance
gates can run there.

The GPU lane needs no vendor SDK: `VulkanGpuExecutor` runs `ncnn` models
(`.param` plus sibling `.bin`) through Vulkan compute, so any Mesa RADV/ANV or
NVIDIA driver serves it. Device selection prefers a discrete GPU over an
integrated one and refuses software Vulkan devices such as `llvmpipe` outright
— reporting the backend unavailable rather than silently running inference on
the host CPU, which the no-CPU rule forbids.

`tests/test_vulkan_hardware.py` is the check: it skips when no usable device is
present and otherwise proves the real Vulkan compute path end to end. Run it to
establish whether a host can serve the GPU lane before treating a GPU
acceptance gate as runnable there.

TPU and NPU lanes need their own hardware — a Coral Edge TPU with
`libedgetpu`, or an NPU that OpenVINO reports — and profiles compiled for
those targets cannot be qualified on a GPU without retargeting the model,
which changes the artifact and its tensor contract.
