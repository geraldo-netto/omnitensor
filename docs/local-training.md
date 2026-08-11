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

| Package or extra | Requirement | Purpose |
| --- | --- | --- |
| OmniTensor base dependencies | Mandatory | Bounded recorder, fitting baseline, report and artifact contracts |
| `[train]` (`onnx`) | Mandatory for export | Validate and write canonical ONNX |
| `[convert]` (`pnnx`) | Optional | Build Vulkan/ncnn GPU artifact |
| `[convert-npu]` (`openvino`, `ovc`) | Optional | Build OpenVINO NPU artifact |
| `[gpu]`, `[npu]`, `[tpu]` | Not needed for training | Service-side execution runtimes only |
| `[dev]` | Not needed | Repository tests, coverage, lint, property and mutation gates |
| `edgetpu_compiler` plus int8 export toolchain | Required for future TPU production | Not distributed through pip and not automated here |

Example producer environment for portable source plus GPU and NPU variants:

```sh
python3 -m venv ~/.local/share/omnitensor-training/venv
~/.local/share/omnitensor-training/venv/bin/pip install \
  '/path/to/omnitensor[train,convert,convert-npu]'
```

Installing `[train]` does not install PyTorch or TensorFlow. Current recipe is
a deterministic ridge forecaster over numeric history and needs neither. A
future image or embedding recipe may declare its own producer-only extra; it
must not become a service dependency.

## Current recipe

`forecast-v1` predicts one future numeric feature from a fixed window. It is
currently intended for `resource-scheduler`. It is self-supervised: later
observations are labels, so no manual labeling or public dataset is required.
The fit is time-split and refused unless held-out error beats repeating the
latest observation.

This recipe does not claim to train low-light, document, desktop, build,
hardware-fault, storage-fault, or network-anomaly models. Those need distinct
datasets, evaluation metrics, preprocessing, and host consumers. They can use
the same portable report, native compilation, immutable installation, and
model-binding path when their recipes exist.

## 1. Record bounded numeric history

Samples contain only finite numbers. Paths, titles, document text, device
serials, and other strings are refused so long-lived training history cannot
become a hidden copy of private content.

```sh
TRAIN=~/.local/share/omnitensor-training/venv/bin

"$TRAIN/omnitensor-record-training-sample" \
  --profile resource-scheduler \
  --feature load=0.42 \
  --feature queue=3
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
  --features load,queue \
  --target load \
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
