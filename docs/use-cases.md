# Cinnamon workload use cases

OmniTensor ships the same nine version 1 workload manifests as the
`cinnamon-xpuwlm` applet. This gives the runtime and panel one stable profile
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
| Inference pipeline | Version 1 bounded `submit-job`/`cancel-job` control-socket methods and backend scheduler exist; verified artifact dispatch and host use-case pipelines remain readiness-gated |

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
| `hardware-health` | enabled, weight 2 | Anomaly scoring over power, UPS, memory, thermal, or service-health features | No bundled model. An opt-in producer fits a portable risk source from reviewed labelled sensor windows, but native binding, pipeline, and hardware acceptance remain. Sensor identity stays private and firmware, shutdown, repair, and policy actions remain deterministic. |
| `storage-intelligence` | enabled, weight 3 | Storage-pattern classification and anomaly scoring for SMART, I/O, cache, or tiering advice | No bundled model. An opt-in Backblaze Drive Stats producer emits an identity-free portable risk source, but native binding, host validation, pipeline, and device acceptance remain. Destructive storage policy never comes from model output. |
| `resource-scheduler` | enabled, weight 3 | Queue forecasting and advice for placement or background-job timing | No bundled model. An opt-in local recipe records bounded aggregate history, fits one forecast, compiles GPU/NPU variants, installs a restricted binding, and runs it through the trusted history-only client. Reviewed Chronos and Granite sources are catalog entries only. Forecasts are advisory; host scheduling policy stays deterministic. |
| `build-advisor` | enabled, weight 2 | Build classification and regression-risk ranking | No bundled model. An opt-in approved-metadata producer emits portable build-risk and optional-check scores, but native binding, consumer, and acceptance remain. Mandatory checks are outside model authority and may never be omitted. |
| `visual-library` | enabled, weight 2 | Image classification or embedding extraction for tagging and semantic search | Ships a pinned ncnn classifier, verified weights and labels, declared preprocessing, and bounded classification output for the Vulkan GPU lane. Embedding/search remains unimplemented: a reviewed CLIP source recipe exists, but its image-only export, consumer, and target acceptance remain. Exact duplicates use hashing instead. |
| `low-light-enhancement` | disabled, weight 2 | Bounded enhanced-RGB generation for dark images | Profile 0.2.0 targets only the Vulkan GPU with Retinexformer in ncnn format and an exact float32 `[1,3,256,256]` contract. Package metadata intentionally contains no unqualified graph or weights. Portable export, licensed paired evaluation, signed native artifacts, image pipeline, and named-GPU acceptance remain. |
| `network-peripherals` | disabled, weight 2 | Aggregated network or peripheral anomaly scoring | No bundled model. An opt-in normal-only aggregate producer emits a portable reconstruction source without identities or packet payloads, but it has no attack-recall claim and native binding, pipeline, and acceptance remain. Responses stay explicitly granted and outside model control. |
| `desktop-context` | disabled, weight 1 | Context prediction and window-layout suggestions | No bundled model. An opt-in content-free producer scores allowlisted suggestions from explicit confirmations, but native binding, stale-session consumer, and acceptance remain. Every window action still requires user confirmation. |
| `document-intelligence` | disabled, weight 2 | Private semantic retrieval and grounded answers over explicitly selected documents | No large weights are bundled. MiniLM and BGE remain reviewed portable embedding sources; BGE has an explicit one-shot producer that builds and gates a digest-locked ncnn artifact on a named Vulkan GPU. The external `ask-selected-files` workload builds an ephemeral page/span index and requires exact citations from qualified BGE and Qwen providers. Its frozen public holdout and safety gate passed with BGE-small plus pinned Qwen3-8B-Q4_K_M on the named RX 6600 XT; arbitrary-corpus quality and NPU/TPU lanes are not claimed. Exact duplicates use hashing. |

Related external workflows remain manual and disabled until their complete
providers advertise readiness. `selected-text-tools` processes only the text
captured by an explicit action; it never monitors clipboard changes or stores
selection history. Its five operations share the generation-provider contract
and return reviewable output plus a digest/span evidence record. Its installed
1.1.0 Vulkan worker passed the frozen 16-case gate on the RX 6600 XT with
Qwen3-8B; the workload's configured Qwen model (default `qwen3-5-9b-iq4-xs`)
is primary for English and all routes except explicit Hebrew translation,
which alone uses pinned DictaLM2. Language is never inferred from source text
or fixed globally. Arbitrary-selection quality and NPU/TPU lanes are not
claimed.
`file-organizer` likewise reads only files selected for one manual request and
returns an evidence-bound metadata plan; it cannot move, rename, overwrite,
delete, or invoke commands.

## Low-light acceptance gates

The named low-light model stays disabled by default. Enabling it for production
requires reproducible evidence for every manifest criterion:

- zero CPU fallback during the named hardware Vulkan run;
- portable/native SSIM of at least 0.999 on the frozen holdout;
- paired-target SSIM of at least 0.8;
- zero non-finite or out-of-range native output components;
- median CIEDE2000 color error no greater than 3;
- warm p95 device latency no greater than 50 ms for 256×256 inputs on the
  named RX 6600 XT; and
- end-to-end speedup of at least 1.2 over the documented CPU enhancement
  baseline, including host work.

Record model, compiler, runtime, driver, and host versions; input shape; warm-up;
sample count; preprocessing; post-processing; accuracy dataset; and baseline.
Five additional profiles now have opt-in portable-source producers, while the
document, visual-embedding, resource-foundation-model, and low-light catalogs
contain reviewed pinned candidates. None of those source-only paths is a
qualified installed application: their profile-specific TODO rows retain the
binding, consumer, signing, dataset, and named-device acceptance blockers.
`resource-scheduler` alone has an end-to-end opt-in local forecast path, while
`visual-library` ships a runnable classifier. Neither path satisfies or
weakens these low-light gates.

### Configure the low-light folders safely

Folder configuration is available independently of the still-blocked model and
image pipeline. Create two separate folders, then persist their canonical paths:

```bash
mkdir -p "$HOME/Pictures/Low Light Input" "$HOME/Pictures/Low Light Output"
omnitensor-configure-low-light \
  --input-folder "$HOME/Pictures/Low Light Input" \
  --output-folder "$HOME/Pictures/Low Light Output"
```

Both folders must already exist. The input must be readable, the output must be
writable, and the two roots must be disjoint: they cannot be equal or nested in
either direction. This stronger rule prevents a future watcher from ingesting
its own results. The versioned configuration is stored under
`~/.config/omnitensor/workload-settings/` by default; `--settings-root` selects
another user-owned store. Loading profile 0.2.0 migrates the unchanged 0.1.0
folder document atomically, preserving its paths and incrementing its revision.

The delivery boundary creates a new, fully written output atomically in the
output folder. It refuses an existing filename, including a symlink, and never
renames, overwrites, or deletes the source image. Only its own unpublished
temporary file is cleaned after a failure. The base OmniTensor installation is
sufficient for configuration; image/model dependencies remain subject to the
acceptance gates above. This command configures the workspace—it does not claim
that the currently missing low-light model and pipeline can process images.

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
   in-process application integration; version the control-socket contract
   before exposing job submission to external clients.
4. Keep post-processing, actions, safety limits, authorization, and persistence
   outside model output.
5. Add unit, integration, regression, property, coverage, and changed-function
   mutation gates plus reproducible hardware acceptance measurements.

The canonical manifest and runtime schemas remain under `schemas/`. Coordinate
any incompatible catalog or contract change with `cinnamon-xpuwlm` before
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
