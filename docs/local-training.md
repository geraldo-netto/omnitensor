# Training local models without coupling them to one device

OmniTensor training is an offline producer workflow. The long-running service
never fits models, imports training frameworks, downloads datasets, invokes a
compiler, or changes its own manifests. A separate environment produces:

1. a canonical ONNX model and reproducible `training-report.json`;
2. one native immutable artifact for each selected accelerator lane; and
3. a restricted local model binding that changes only a built-in profile's
   model and routing fields.

This distinction is the practical meaning of hardware-agnostic models. Model
semantics, ordered features, tensor shape, version, and output meaning are
shared. Executable files are necessarily hardware-specific:

| Logical source | Native artifact | Runtime lane | Status |
| --- | --- | --- | --- |
| ONNX | ncnn `.param` + `.bin` | Vulkan GPU | Supported |
| ONNX | OpenVINO IR `.xml` + `.bin` | Intel NPU | Supported for production; execution needs an NPU |
| fully-int8 TFLite | Edge-TPU-compiled `.tflite` | Coral TPU | Not produced; separate toolchain and hardware evidence required |

Every installed variant repeats the exact same tensor and output contracts.
It also repeats a bounded `featureContract`: recipe, ordered feature names,
target, window, observation-count horizon, oldest-first observation order, and
the observations-then-features flattening order. The model binding is rejected
if variants disagree, if the target is not the first feature, or if these
semantics disagree with the enforced `float32` `NC` tensor width and raw model
output. Routing selects the first available lane with a compatible installed
artifact; it never falls back to CPU execution.

## Mandatory and optional dependencies

Use a dedicated producer environment. Do not add producer dependencies to the
service environment.

The trainers ship as their own distribution, `omnitensor-training`, built from
`packaging/omnitensor-training` in this repository. It installs into the same
`omnitensor` package as the service and pins the exact service version it was
built with, because model recipes travel with the trainers while the schemas
validating them travel with the service.

| Package or extra | Requirement | Purpose |
| --- | --- | --- |
| `omnitensor-training` | Mandatory | The trainers, recipes, exporters, and their CLIs |
| `omnitensor` (pinned) | Mandatory | Bounded recorder, fitting baseline, report and artifact contracts |
| `[train]` (`onnx`) | Mandatory for export | Validate and write canonical ONNX |
| `[model-producers]` (`torch`, `onnx`, `numpy`) | Optional | Export reviewed neural sources such as CLIP in an isolated producer environment |
| `[document-producers]` (`torch`, `transformers`, `safetensors`, `tokenizers`, `onnx`, `onnxruntime`, `pnnx`, `ncnn`, `numpy`) | Required only for the BGE GPU installer | Reconstruct the pinned BGE-small encoder, compile its ncnn graph, and compare it with the portable reference on a named Vulkan device |
| `[foundation-producers]` (`torch`, `onnx`, `numpy`, `safetensors`, `transformers`) | Optional | Export the pinned Chronos/TTM sources; a reviewed family-specific loader is also mandatory and remains an explicit producer input |
| `[retinexformer-producers]` (`torch`, `onnx`, `numpy`, `einops`) | Optional | Reconstruct the verified Retinexformer architecture/checkpoint and export its clamped fixed graph |
| `[convert]` (`pnnx`) | Optional | Build Vulkan/ncnn GPU artifact |
| `[convert-npu]` (`openvino`, `ovc`) | Optional | Build OpenVINO NPU artifact |
| `[gpu]`, `[npu]`, `[tpu]` | Not needed for training | Service-side execution runtimes only |
| `[dev]` | Not needed | Repository tests, coverage, lint, property and mutation gates |
| `edgetpu_compiler` plus int8 export toolchain | Required for future TPU production | Not distributed through pip and not automated here |

Example producer environment for portable source plus GPU and NPU variants:

```sh
python3 -m venv ~/.local/share/omnitensor-training/venv
~/.local/share/omnitensor-training/venv/bin/pip install \
  '/path/to/omnitensor[convert,convert-npu]' \
  '/path/to/omnitensor/packaging/omnitensor-training[train]'
```

`[convert]` and `[convert-npu]` stay with `omnitensor`: they are packaging-time
converters for the service's own artifact lanes, not trainers. Build both
wheels with:

```sh
python -m build --wheel --outdir dist .
python -m build --wheel --outdir dist packaging/omnitensor-training
```

Installing `[train]` does not install PyTorch or TensorFlow. The current
numeric recipes use deterministic fitting implemented by OmniTensor and need
neither. An image, embedding, or foundation recipe may declare its own
producer-only extra; it must not become a service dependency. The foundation
extra does not silently choose or download a model implementation: callers
must supply a reviewed loader matching the pinned recipe family and source
revision.

## Machine-local artifact publisher identity

A model digest proves which bytes arrived, but not who accepted those bytes,
their conversion report, or their source. OmniTensor therefore signs the exact
artifact id, version, format, SHA-256, publisher identity, source evidence, and
publication time before a qualified native model becomes active. This prevents
an unsigned or substituted file from inheriting the trust of a previously
reviewed build.

The default local identity is deliberately recognizable:

- label: `xpuwlm`;
- publisher: `xpuwlm.local`;
- key id: `xpuwlm-ed25519-v1`;
- algorithm: Ed25519.

Create it explicitly, or let an artifact-publisher command call the same
idempotent API when it first needs to sign:

```sh
omnitensor-setup-publisher-key
```

The first call creates the key; later calls verify and reuse it. The private
key defaults to
`~/.local/share/omnitensor/publisher-keys/xpuwlm/xpuwlm-ed25519-v1.private.pem`
with mode `0600` inside a `0700` directory. The public trust record defaults to
`~/.config/omnitensor/artifact-publishers/xpuwlm.json`. Output includes paths
and the SHA-256 public-key fingerprint, never private key material.

This is an unattended, machine-local producer identity, so its PKCS#8 private
key is not passphrase-encrypted. Filesystem ownership and mode are the custody
boundary. Never commit, copy into an artifact, expose to a plugin worker, or
mount it into the inference service. Back it up only if artifacts must remain
reproducibly attributable after reinstalling the workstation. A distributable
release needs a separately governed offline key and rotation/revocation policy;
do not reuse `xpuwlm.local` as a public release identity.

## Reviewed open-source source catalog

The bundled recipe catalog pins every remote byte by revision, filename, size,
and SHA-256 digest. It also records the upstream license, preprocessing,
tensor/output contract, evaluation gates, and an honest status for each target.
Inspect it before downloading anything:

```sh
TRAIN=~/.local/share/omnitensor-training/venv/bin
"$TRAIN/omnitensor-list-model-recipes"
```

| Recipe | Intended profile | Upstream license | Current deliverable |
| --- | --- | --- | --- |
| `all-minilm-l6-v2` | `document-intelligence` | [Apache-2.0 model card and weights](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/5641a7880f40ebf4035d05e60c5f9b7a9c272c84/README.md) | Fixed mean-pooling/L2 ONNX producer and source-parity gate; local reviewed holdout and runner adapters required; native lanes unqualified |
| `bge-small-en-v1-5` | `document-intelligence` | [MIT model card and weights](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/README.md) | Fixed CLS/L2 portable reference plus an explicit ncnn/Vulkan producer with bundled smoke retrieval, source/native parity, immutable installation, and a restricted local binding; NPU/TPU remain unqualified |
| `clip-vit-b-32-image` | `visual-library` | [MIT](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/LICENSE) | Fixed image-only ONNX/L2 producer, exact resize/normalization helpers, and source cosine/zero-shot gate; native lanes remain unqualified |
| `amazon-chronos-bolt-tiny` | `resource-scheduler` | [Apache-2.0 model card and weights](https://huggingface.co/amazon/chronos-bolt-tiny/blob/a0e552de83495b5c28c14c71c374f3e33280b340/README.md) | Fixed 512-value median-first-horizon ONNX producer; source parity and repeat-last/local-linear gates; native lanes unqualified |
| `ibm-granite-ttm-r2` | `resource-scheduler` | [Apache-2.0 model card and weights](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/blob/d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f/README.md) | Fixed 512-value first-point-horizon ONNX producer; source parity and repeat-last/local-linear gates; native lanes unqualified |
| `retinexformer-lol-v1` | `low-light-enhancement` | [MIT](https://github.com/caiyuanhao1998/Retinexformer/blob/1e9a0efce4b306b6701b824768370ff26066c32a/LICENSE.txt) | Strict pinned architecture/checkpoint loader, fixed clamped ONNX export, and licensed paired-image/source-parity gate; native lanes unqualified |

Downloading is explicit and fail-closed. The exact SPDX identifier shown by
the catalog must be acknowledged; redirects, byte limits, sizes, and digests
are rechecked before an atomic receipt is written:

```sh
"$TRAIN/omnitensor-fetch-model-source" all-minilm-l6-v2 \
  --accept-license Apache-2.0
"$TRAIN/omnitensor-fetch-model-source" clip-vit-b-32-image \
  --accept-license MIT
```

License acknowledgement is not legal advice and does not prove compatibility,
quality, or permission for a particular downstream dataset. Fetching also does
not execute an upstream architecture, create the missing wrapper, compile a
native artifact, install a binding, or make a profile operational. Those are
separate review and acceptance stages.

For the two sentence-embedding recipes, `produce_sentence_embedding` reopens
and rehashes an already fetched source, freezes all three token inputs to
`[1,128]`, embeds the recipe's pooling and L2 normalization into ONNX, and
writes an atomic provenance report. The caller supplies source and portable
runner factories plus a separately licensed `EmbeddingHoldout`; acceptance
requires cosine parity, top-10 retrieval overlap, and finite 384-value output.
The report contains corpus identity and counts, never query or document text.
This general producer does not compile, install, or claim GPU/NPU/TPU
evidence. BGE's narrower end-to-end GPU installer below adds those stages
without changing that reusable boundary.

### Install BGE-small for Document Intelligence on Vulkan

Keep the large producer stack out of the service environment:

```sh
python3 -m venv ~/.local/share/omnitensor-document-producer/venv
~/.local/share/omnitensor-document-producer/venv/bin/pip install \
  '/path/to/omnitensor[document-producers]'
~/.local/share/omnitensor-document-producer/venv/bin/omnitensor-install-document-model \
  --accept-license MIT
systemctl --user restart omnitensor.service
```

The command downloads only the revision-, size-, and digest-pinned MIT BGE
files after explicit acknowledgement. It preserves upstream's ONNX opset 11
graph as the portable reference. BGE's CLS selection needs no opset-13-only
operators, while the independent MiniLM mean-pooling wrapper legitimately uses
opset 13; both publish the same embedding output contract, so no second service
API or forced graph upgrade is needed.

The ONNX CPU runner is a producer-only reference oracle: its constructor is
guarded and only the one-shot installer can authorize it. The service and the
external BGE worker instantiate only the native tokenizer/Vulkan runner and
never receive that capability, so the reference cannot become a runtime
fallback.

The native path reconstructs the reviewed BERT encoder from the pinned
safetensors and exports directly to ncnn with fixed `[1,128]` token, mask, and
segment inputs. Their order, shapes, and layouts come from the portable recipe;
the explicit ncnn boundary narrows token and segment IDs to `int32` and turns
the attention mask into `float32`, matching the graph measured on Vulkan. It
refuses software Vulkan devices and CPU fallback. Before
publishing, it requires finite 384-value output, cosine parity of at least
`0.999`, top-10 overlap of at least `0.9`, maximum absolute error at most
`0.001`, and every expected top result in the bundled CC0 smoke corpus. The
report contains only corpus identity and aggregate metrics. It explicitly does
not claim production-domain retrieval quality or named-device profile
acceptance.

The binding changes only `document-intelligence` model/routing fields and
remains disabled by default. The other model-less profiles are untouched and
gain no artifact or route. The ncnn service runtime uses integer token tensors
and full-precision Vulkan arithmetic because fp16 attention on the qualified
host did not preserve embedding semantics. NPU needs a separately gated
OpenVINO artifact; TPU needs a representative int8 export, full compiler
mapping, parity, and Coral evidence.

`produce_clip_source` similarly rechecks the pinned OpenAI TorchScript bytes,
exports only `encode_image` at `[1,3,224,224]`, and puts 512-value L2
normalization inside ONNX. `clip_resize_geometry` and `normalize_clip_rgb`
define cover-resize, center-crop, RGB-to-NCHW, mean, and scale semantics without
coupling them to a service decoder. A separately licensed `ClipHoldout` and
caller-supplied source/portable runners gate cosine and zero-shot top-1 parity.
Neither image references nor prompt vectors enter the report.

`produce_foundation_forecast` reopens either pinned forecast source and calls
an explicit reviewed loader through `TorchFoundationForecastOnnxExporter`.
The ONNX boundary is fixed at `[1,512]` float context to `[1,1]` scalar and
keeps TTM's first point or Chronos' first-horizon median selection inside the
graph. A `ForecastHoldout` contains 32–512 strictly time-ordered windows from
operator-owned telemetry. Publication requires export error at most `0.0001`,
repeat-last skill at least 35%, and lower MAE than both repeat-last and the
existing local linear runner. Reports contain only corpus digest, sample count,
and aggregate errors. No portable result establishes GPU, NPU, or TPU support;
those lanes remain unqualified until their separate compiler, parity, and
named-device gates pass, and service CPU fallback remains forbidden.

`produce_retinexformer_source` rechecks the official checkpoint, architecture,
and config before `PinnedRetinexformerLoader` executes the pinned architecture,
instantiates the exact one-stage 40-feature `[1,2,2]` block layout, and loads
only the checkpoint's `params` mapping with strict key matching. The exported
`[1,3,256,256]` NCHW graph clamps enhanced RGB to `[0,1]`. A separately
licensed `RetinexformerHoldout` gates minimum portable/source whole-image SSIM
at `0.999`, minimum paired-target SSIM at `0.8`, and requires zero range or
nonfinite violations. Reports contain references to neither side of the image
pairs. The disabled `low-light-enhancement` 0.2.0 profile now declares this
enhanced-RGB contract for an ncnn/Vulkan artifact; it does not claim that such
an artifact is installed or accepted. GPU conversion remains unqualified until
native parity, signature verification, no-CPU-fallback evidence, and named RX
6600 XT execution pass. Other target conversion may follow portable acceptance.
TPU publication
additionally requires representative fully-int8 calibration, complete Edge TPU
mapping, fidelity parity, and Coral evidence. No lane may substitute CPU.

## Dataset ownership and redistribution

| Producer | Dataset/license boundary |
| --- | --- |
| Resource forecast | Operator-owned bounded telemetry; no public dataset and no raw history redistribution |
| Storage risk | Official Backblaze Drive Stats; review the linked current data-use and citation terms before each intake; the CLI acknowledgement is not a license |
| Network anomaly | Operator-approved normal-only aggregate replay; the KitNET paper inspires the design and its MIT reference implementation is not vendored |
| Build risk/ranking | Operator-approved repository metadata; no logs, source contents, or paths are placed in the report/model |
| Hardware health | Operator-reviewed observed/injected labels; sensor identities and raw labels stay out of the report/model |
| Desktop suggestions | Explicit user-confirmed, content-free local examples; raw personalized history is revocable and must not be redistributed |
| Low-light evaluation | No paired dataset is bundled; acquire and review a separately licensed train/holdout corpus before claiming fidelity |
| Document/visual embeddings | No private documents or images are bundled; each deployment owns consent, retention, and evaluation-corpus licensing |

## Inspect producer and runtime capabilities

Compiler availability is not device availability. Inspect the producer tools
for each compatible portable format without downloading or compiling a model:

```sh
"$TRAIN/python" - <<'PY'
from omnitensor.training.installation import available_targets

print("ONNX compilers:", available_targets(source_format="onnx"))
print("fully-int8 TFLite compilers:",
      available_targets(source_format="tflite", fully_quantized=True))
PY
command -v pnnx || true
command -v ovc || true
command -v edgetpu_compiler || true
```

`available_targets` checks only source compatibility and whether the compiler
executable exists. It does not probe a GPU, NPU, or TPU. Inspect the installed
runtime separately:

```sh
omnitensor-verify-install
python - <<'PY'
from pathlib import Path
import json

state = Path.home() / ".local/state/xpu-workload-manager/state.json"
snapshot = json.loads(state.read_text(encoding="utf-8"))
print(json.dumps({"devices": snapshot["devices"], "profiles": snapshot["profiles"]}, indent=2))
PY
```

An imported runtime, listed device, successful conversion, or installed
artifact is still not profile acceptance. Acceptance requires the named-device
quality, parity, latency, privacy, recovery, and end-to-end evidence declared
for that profile.

## Portability and accelerator support matrix

| Stage | Vulkan GPU | Intel NPU | Coral TPU |
| --- | --- | --- | --- |
| Shared model semantics | Device-neutral source plus exact preprocessing, tensor, output, and evaluation contracts | Same | Same |
| Native compiler port | `pnnx`: ONNX or TorchScript to ncnn | `ovc`: ONNX to OpenVINO IR | `edgetpu_compiler`: fully-int8 TFLite to Edge-TPU TFLite, accepted only with 100% Edge TPU operation mapping |
| Service executor | ncnn/Vulkan with `[gpu]` | OpenVINO with `[npu]` | TFLite with Edge TPU delegate using `[tpu]` |
| `forecast-v1` local install | Implemented when `pnnx` is present | Implemented when `ovc` is present | Not implemented: the report currently contains ONNX, not calibrated fully-int8 TFLite |
| Storage/network/build/hardware/desktop producers | Signed promotion boundary implemented; requires injected native parity evidence, then installs independently versioned ncnn artifacts | Same boundary for OpenVINO IR | Needs a separate representative int8 export; ONNX promotion refuses TPU |
| Reviewed embedding/forecast/low-light catalog | BGE has an explicit locally gated ncnn producer; every other reviewed neural source remains source-only | BGE and every other reviewed neural source remain source-only | Same, plus representative int8 calibration and full Edge TPU mapping |
| CPU fallback | Never | Never | Never |

`promote_numeric_training` is the fail-closed producer API for storage,
network, build, hardware, and desktop reports. Callers inject target compilers,
a portable/native parity runner, an Ed25519 signer, and its offline trust
verifier. Every target receives its own semantic version. Promotion records
the complete report digest, task-semantics digest, native digest, bounded
numerical evidence, and publisher provenance in the immutable store. Missing
or mismatched evidence/signatures abort before a binding is published.

This API does not claim named-device acceptance and records that fact in every
variant. Profile-specific consumers and their hardware acceptance remain
separate work. `omnitensor-install-trained-model` remains the convenience CLI
specific to `forecast-v1`; it does not silently weaken the signed promotion boundary.
Never rename ONNX to TFLite or treat compiler success as parity.

For sharing, publish the canonical recipe/report, preprocessing and result
contracts, source license/attribution, portable model digest, and evaluation
evidence. Publish each native artifact as a separate target/version with all
companion digests and compiler/runtime provenance. Do not publish an
operator's raw corpus or reuse one lane's hardware evidence for another lane.

## Resource forecast recipe

`forecast-v1` predicts one future numeric feature from a fixed window. It is
currently intended for `resource-scheduler`. It is self-supervised: later
observations are labels, so no manual labeling or public dataset is required.
The fit is time-split and refused unless held-out error beats repeating the
latest observation.

This recipe does not claim to train low-light, document, desktop, build,
hardware-fault, storage-fault, or network-anomaly models. Those use distinct
datasets, evaluation metrics, preprocessing, and host consumers. Portable
sources now exist for storage risk and aggregate network anomaly detection,
but neither is a resource forecast or an installed service pipeline.

## Storage risk portable-source recipe

`backblaze-smart-risk-v1` is a separate, offline recipe for a device-neutral
failure-risk source. Download and extract official daily Drive Stats CSVs from
the [Backblaze Drive Stats data page](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data),
review its current citation and data-use terms, then run:

```sh
~/.local/share/omnitensor-training/venv/bin/omnitensor-train-storage-model \
  --csv-root /path/to/extracted/daily-csvs \
  --accept-dataset-terms Backblaze-Drive-Stats
```

The acknowledgement token does not replace reviewing the terms. In
particular, do not redistribute the raw dataset through OmniTensor. The report
retains the official source, citation, acknowledgement, schema digest, and a
deidentified corpus digest.

Intake is bounded and streaming. It accepts canonical daily CSV filenames,
refuses missing required headers and duplicate drive rows, tolerates unrelated
new columns while recording schema drift, and never emits serial number, model,
or capacity as a feature. The six ordered inputs are raw SMART 5, 9, 187, 188,
197, and 198 values. A row becomes positive only when a later row reports that
drive failed within seven days; the failure-day row is not an input. Training
uses a purged chronological holdout, deterministic negative downsampling, an
explicit class-balance floor, and a held-out ROC-AUC gate.

Output is `model.onnx` plus `storage-training-report.json`. The graph is the
portable source of truth, not an installed service model. It uses only
`MatMul`, `Add`, and `Sigmoid`, so the same semantics and `[1,6]` contract can
be compiled and tested for Vulkan/ncnn and OpenVINO NPU lanes. A TPU release
still requires representative int8 calibration, a fully-int8 TFLite export,
100% Edge TPU compiler mapping, semantic parity, and Coral execution evidence.
No lane is silently replaced with CPU execution, and Backblaze fleet quality
does not establish quality on an operator's drives.

## Aggregate network anomaly portable-source recipe

`aggregate-reconstruction-v1` is an opt-in, normal-only recipe over the
existing consent-gated network collector. Prepare newline-delimited JSON where
each line is one complete `network-manager-link-metadata` collector document,
ordered by increasing `observedAtMs`. Audit that the interval contains normal
operation, then run:

```sh
~/.local/share/omnitensor-training/venv/bin/omnitensor-train-network-model \
  --replay /path/to/reviewed-normal-network-snapshots.jsonl \
  --confirm-normal-only I-confirm-this-replay-is-normal
```

The explicit confirmation is enforced by the Python API as well as the CLI.
It is not proof that the replay is representative. The loader accepts at most
64 MiB and 200,000 closed-schema snapshots, refuses truncated/unhealthy data,
counter regressions, duplicate or decreasing times, and never accepts packet
payloads. Stable link IDs are used transiently to compute counter deltas, then
discarded. The report retains only a digest of the exact replay bytes and
identity-free aggregate features: rates, link counts, state/connectivity
fractions, route/carrier/metered fractions, and mean signal.

The fixed reconstruction ensemble is inspired by the small-autoencoder design
in [Kitsune/KitNET](https://arxiv.org/abs/1802.09089); the
[reference implementation](https://github.com/ymirsky/Kitsune-py) is MIT
licensed. OmniTensor does not vendor that code, packet feature extractor, or
weights. It fits a small bounded linear reconstruction block for each fixed
aggregate feature group, selects a threshold from the training split, and
refuses the source when held-out reviewed-normal false positives exceed the
configured gate. With no labeled attacks it reports `anomalyRecall: null` and
must not claim attack coverage or trigger autonomous network actions.

Output is `model.onnx` plus `network-training-report.json`. The graph uses
common arithmetic, `MatMul`, and `ReduceMean` operations with a `[1,14]`
float32 contract. The same source semantics can be converted and tested for
Vulkan/ncnn GPU and OpenVINO NPU lanes. A TPU release still needs a separate
fully-int8 TFLite export with representative aggregate-feature calibration,
100% Edge TPU compiler mapping, numerical parity, and Coral execution
evidence. Until a lane passes its compiler and device gates, the report marks
it `uncompiled`; runtime never substitutes CPU execution.

## Build risk and optional-check ranking portable-source recipe

`metadata-risk-ranking-v1` fits only from an operator-approved JSONL export of
bounded build metadata. Each line is a closed object with `schemaVersion: 1`,
`buildId`, `outcome`, `durationMs`, `startedAtMs`, relative `changedPaths`,
`executedChecks`, and `failedChecks`. It accepts no logs, source contents,
absolute/traversing paths, duplicate fields, or failed checks that were not
executed. Review the export and its retention policy before running:

```sh
~/.local/share/omnitensor-training/venv/bin/omnitensor-train-build-model \
  --history /path/to/approved-build-history.jsonl \
  --mandatory-check security \
  --optional-check lint \
  --optional-check integration \
  --accept-provenance I-confirm-build-metadata-is-approved
```

Every completed build must contain every declared mandatory check. Unknown
executed checks are refused instead of silently omitted from the catalog.
Cancelled and unknown outcomes are counted but not labelled. Feature extraction
uses only fixed path-shape/category counts and fractions; build IDs and path
strings exist transiently during intake but are never persisted in the model or
report.

The chronological fit produces one build-failure probability and one risk
score per explicit optional check. It gates held-out build-risk ROC-AUC and
mean reciprocal rank for actually failed optional checks. Every optional check
must have enough executed pass and fail examples in the training split. The
model has no output for mandatory checks: their exact catalog is deterministic
report metadata with `mandatoryOmissionAllowed: false`. A future consumer may
use scores to reorder optional checks, but must always include mandatory checks
and must not treat a score as permission to skip policy.

Output is `model.onnx` plus `build-training-report.json`, with an exact
`[1,12]` float32 input contract and `[1,1+optional-check-count]` result
contract. The common arithmetic/`MatMul`/`Sigmoid` graph is a portable source
for Vulkan/ncnn GPU and OpenVINO NPU compilation. TPU production separately
requires representative metadata calibration, fully-int8 TFLite export, full
Edge TPU mapping, parity, and Coral evidence. All lanes remain `uncompiled`
until those target-specific gates pass; local history quality does not prove
portability or usefulness on another repository.

## Hardware-health portable-source recipe

`labelled-sensor-window-v1` fits a private anomaly-risk source from reviewed
outputs of the bounded `hwmon-edac-power-service` collector. An authorized
integration must export one closed JSON object per line in increasing time
order:

```json
{"label":"baseline","labelSource":"observed","snapshot":{"schemaVersion":1,"source":"hwmon-edac-power-service","sourceHealth":"ready","observedAtMs":1000,"items":[],"churn":{"added":[],"removed":[],"changed":[]},"truncatedItems":0}}
```

The example shows the wrapper, not a trainable empty sensor set. Every real
line must contain the exact same complete collector sensor set. `label` is
`baseline` or `fault`; `labelSource` is `observed` or `injected`, and only a
fault may be marked injected. Review label provenance before training. The
tool does not create faults, infer labels, or automatically read privileged
host telemetry.

Assign stable collector identities to redacted semantic roles on the command
line. Identities are used during intake and omitted from the report:

```sh
~/.local/share/omnitensor-training/venv/bin/omnitensor-train-hardware-model \
  --history /path/to/reviewed-hardware-snapshots.jsonl \
  --sensor cpu-temperature=cpu-package-0 \
  --sensor ecc=dimm-0 \
  --confirm-labels I-confirm-hardware-labels-are-reviewed
```

Intake is bounded, rejects truncated/unhealthy snapshots, sensor-set or
kind/unit drift, duplicate/decreasing timestamps, unknown fields, and
unreviewed labels. Per role it encodes the log-scaled numeric value plus exact
degraded, critical, and missing indicators over an oldest-first window. The
chronological split drops the final `window - 1` training windows so no source
observation is shared with the holdout. The holdout must meet configured
ROC-AUC, fault recall, baseline false-positive, and class-count gates; the
report records the number of purged training windows. Output is `model.onnx` plus
`hardware-training-report.json`; neither contains sensor identities or labels.

The portable graph uses common arithmetic, `MatMul`, and `Sigmoid` with one
bounded float32 input and one risk output. The report defines bounded role-only
evidence, explicitly gives the model no safety actions, and leaves firmware,
shutdown, repair, and policy decisions deterministic. GPU and NPU production
still require native compilation, numerical parity, and device acceptance.
TPU production additionally requires representative int8 calibration,
fully-int8 TFLite export, complete Edge TPU mapping, and Coral evidence. Until
those independent gates pass, every target is reported as `uncompiled`.

## Private desktop-suggestion portable-source recipe

`confirmed-layout-suggestion-v1` is an opt-in personalized scorer. Its JSONL
input is deliberately not a dump of the desktop collector. Each closed record
contains only a timestamp, bounded workspace/window/visible/role counts,
coarse focused-role and width/height buckets, and one allowlisted layout
suggestion with `confirmation: "user-confirmed"`. Titles, text, screenshots,
paths, application IDs, window IDs, and arbitrary suggestion strings have no
input field and are refused as unknown data.

An authorized user-facing integration must ask the user which suggestion they
want and produce the redacted record; OmniTensor neither observes clicks nor
guesses consent. After reviewing the local file, train with:

```sh
~/.local/share/omnitensor-training/venv/bin/omnitensor-train-desktop-model \
  --history /path/to/confirmed-content-free-desktop-examples.jsonl
```

Training uses a chronological split, requires a configured number of explicit
confirmations for every included suggestion in both partitions, and gates
held-out top-choice accuracy and macro recall. The report contains only the
history digest, aggregate counts, fixed feature names, suggestion catalog, and
quality. Its result contract requires user confirmation and forbids automatic
window actions. A future consumer must also suppress stale-session results;
this producer workflow alone does not make the disabled profile operational.

Revoke the raw personalized history with an exact-file, symlink-refusing,
locked command:

```sh
~/.local/share/omnitensor-training/venv/bin/omnitensor-revoke-desktop-training-history \
  --history /path/to/confirmed-content-free-desktop-examples.jsonl \
  --confirm-delete I-confirm-delete-desktop-training-history
```

This command deletes only that explicit corpus file and is idempotent. It does
not uninstall a model that the operator separately compiled and installed;
use the artifact revocation/removal workflow for derived installed artifacts.
The portable ONNX source uses common arithmetic, `MatMul`, and `Sigmoid`.
GPU/NPU/TPU lanes remain `uncompiled` until each has native compilation,
parity, privacy, latency, and named-device evidence; TPU additionally needs a
representative fully-int8 calibration/export and complete Edge TPU mapping.

## 1. Record bounded numeric history

Samples contain only finite numbers. Paths, titles, document text, device
serials, and other strings are refused so long-lived training history cannot
become a hidden copy of private content.

```sh
TRAIN=~/.local/share/omnitensor-training/venv/bin

"$TRAIN/omnitensor-record-training-sample" \
  --profile resource-scheduler \
  --feature queueDepth=3 \
  --feature runningProfiles=1
```

Repeat from an explicit timer or collector integration using the same feature
names. Storage rotates in bounded segments under
`~/.local/state/omnitensor/telemetry` by default. Training refuses missing
features instead of silently replacing them with zero.

For measurements the running service already publishes, prefer the snapshot
recorder. It validates the complete snapshot contract, reads at most 1 MiB,
and accepts only the required device-agnostic aggregate numeric selectors
`queueDepth` and `runningProfiles`. It never records device identities, loads,
paths, profile text, alerts, or plug-in payloads. Repeating the same snapshot
timestamp is a no-op; an older timestamp is refused.

```sh
"$TRAIN/omnitensor-record-runtime-snapshot" \
  --profile resource-scheduler \
  --selector queueDepth \
  --selector runningProfiles
```

Collection is opt-in. To sample every minute, create these user units (change
the producer virtual-environment path or `--snapshot` if your installation
uses different locations):

```ini
# ~/.config/systemd/user/omnitensor-training-recorder.service
[Unit]
Description=Record one bounded OmniTensor training sample

[Service]
Type=oneshot
ExecStart=%h/.local/share/omnitensor-training/venv/bin/omnitensor-record-runtime-snapshot --profile resource-scheduler --selector queueDepth --selector runningProfiles
```

```ini
# ~/.config/systemd/user/omnitensor-training-recorder.timer
[Unit]
Description=Opt-in OmniTensor training history sampler

[Timer]
OnBootSec=2min
OnUnitActiveSec=1min
Persistent=false

[Install]
WantedBy=timers.target
```

Enable it only after reviewing the selectors and retention location:

```sh
systemctl --user daemon-reload
systemctl --user enable --now omnitensor-training-recorder.timer
systemctl --user list-timers omnitensor-training-recorder.timer
```

No timer is installed or enabled by OmniTensor. `Persistent=false` prevents a
login from replaying missed intervals as fake history.

The number of required observations depends on `window` and `horizon`; the fit
also requires at least eight resulting windows. Real evaluation needs enough
history to span the machine's normal idle, interactive, and busy periods.

## 2. Train portable source once

Feature order is part of model identity. The predicted feature must be first
and must match `--target`.

```sh
"$TRAIN/omnitensor-train-model" \
  --profile resource-scheduler \
  --id local-resource-forecast \
  --version 1.0.0 \
  --features queueDepth,runningProfiles \
  --target queueDepth \
  --window 12 \
  --horizon 1
```

Output defaults to
`~/.local/share/omnitensor/training/resource-scheduler/1.0.0/`:

- `model.onnx`: device-neutral source graph using common `MatMul` and `Add` operations;
- `training-report.json`: corpus digest, model digest, exact feature/window
  contract, held-out metrics, and output contract.

Changing feature order, window, target, data, or model bytes requires a new
artifact version. Installation rechecks the ONNX digest against the report.
Bindings written by an older installer do not contain `featureContract` and
must be reinstalled before the trusted forecast runner can use them. The
generic job API cannot infer where caller-supplied numbers came from, so this
metadata alone does not make an opaque same-shaped inline tensor safe.

## 3. Compile and install native variants

```sh
"$TRAIN/omnitensor-install-trained-model" \
  ~/.local/share/omnitensor/training/resource-scheduler/1.0.0/training-report.json \
  --targets auto
```

`auto` means every compiler installed in the producer environment, not every
device currently attached. Use `--targets gpu`, `--targets npu`, or
`--targets npu,gpu` for an explicit release set.

The command compiles every target before publishing the binding, computes all
primary and companion digests, installs immutable versions under
`~/.local/share/omnitensor/artifacts`, then atomically writes:

```text
~/.local/share/omnitensor/model-bindings/resource-scheduler/manifest.json
```

Bindings are full canonical manifests, not unvalidated patches. At service
startup OmniTensor accepts only changes to `accelerator`,
`acceleratorPreference`, `model`, and `models`; attempts to change profile UI,
defaults, permissions, capabilities, or pipeline responsibilities are skipped.

Restart after installation:

```sh
systemctl --user restart omnitensor.service
```

`OMNITENSOR_MODEL_BINDINGS` changes the binding root. The service reads it;
training tools write it. Keep both configured to the same absolute directory.

## 4. Run a forecast from measured history

The trusted runner accepts a profile name and storage roots only. It does not
accept feature values, a tensor, or an input file: those would let a correctly
shaped but semantically reordered payload bypass the reason the feature
contract exists. It loads the installed binding as a restricted overlay of the
bundled profile, requires every artifact digest and feature contract, then
takes exactly the newest complete window from bounded recorder history.

```sh
"$TRAIN/omnitensor-run-forecast" --profile resource-scheduler
```

The base OmniTensor dependency set includes the local D-Bus client used by the
runner; no training or accelerator extra is needed to submit. The running
service still needs the runtime extra for the selected native lane. The runner
checks `DescribeContract` before submission, uses the declared oldest-first
observation and feature order, and polls at most 40 times by default. A missing
feature in any newest row is a refusal; it never searches backward for an
older complete row and silently changes the forecast time.

Use `--records-root` and `--bindings-root` only when the recorder, installer,
service, and runner are configured for the same absolute locations. The
service must be restarted after installing a new binding. Output is an exact
bounded document containing `kind`, `targetFeature`, observation-count
`horizon`, and the finite predicted `value`. It states no unit, confidence,
risk, or automatic scheduling action. Raw tensors remain in the owner-scoped
runtime job result for debugging, while the public snapshot receives only an
advisory summary.

## What installation proves

Successful installation proves reproducible fitting, portable graph validity,
native conversion, immutable digests, contract agreement, and service routing
readiness. It does not prove useful production decisions, target-hardware
latency, safety, or profile-specific end-to-end behavior. Each recipe still
needs representative holdout data, named-hardware execution, calibrated
metrics, preprocessing/postprocessing, and an authorized result consumer.

Low-light enhancement especially cannot be called complete from an exported
graph alone. It needs paired image data, fidelity/color metrics, bounded image
decoding and delivery, and either a GPU/NPU recipe or a fully-int8 Edge TPU
variant qualified on Coral hardware.
