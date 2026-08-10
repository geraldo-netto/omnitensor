# Installing and verifying OmniTensor plus the Cinnamon applet

A successful install is not evidence of a working one. The unit can be active
while another process owns the bus name, the canonical schemas can be missing
from the installed package, the published snapshot can be stale from a previous
run, and the applet can be a half-copied tree. Each of those looks healthy in
`systemctl` and produces a desktop that silently shows nothing, so the install
sequence below ends with an automated check rather than an assumption.

## Service

Install into an environment of its own rather than the development checkout, so
the running service is a fixed snapshot and not whatever the working tree
happens to contain:

```sh
python3 -m venv ~/.local/share/omnitensor/venv
~/.local/share/omnitensor/venv/bin/pip install '/path/to/omnitensor[gpu]'
ln -sf ~/.local/share/omnitensor/venv/bin/omnitensor ~/.local/bin/omnitensor
ln -sf ~/.local/share/omnitensor/venv/bin/omnitensor-verify-install \
       ~/.local/bin/omnitensor-verify-install
ln -sf ~/.local/share/omnitensor/venv/bin/omnitensor-prepare-artifact \
       ~/.local/bin/omnitensor-prepare-artifact

install -Dm0644 systemd/omnitensor.service ~/.config/systemd/user/omnitensor.service
systemctl --user daemon-reload
systemctl --user enable --now omnitensor.service
```

Inference stays fail-closed until an artifact store is configured, because
nothing else can prove a model file is the model a manifest declares. Point
`OMNITENSOR_ARTIFACT_ROOT` at the verified store (default
`~/.local/share/omnitensor/artifacts`); with no store the service still runs,
publishes, and answers the bus, but refuses every inference job.

The unit uses `StateDirectory=`, so systemd creates the state directories on
first start; it needs no pre-existing paths. It also declares `Delegate=yes` so
each plugin worker can be accounted in its own cgroup.

### Where the applet reads the snapshot

The service publishes one file and the Cinnamon applet reads it; that file is
the only place the two meet. Each side names it independently, so moving it is
a two-sided edit and changing one side alone is silent:

| Side | What names the path | Default |
| --- | --- | --- |
| Service | `OMNITENSOR_STATE_PATH` | `~/.local/state/tpu-workload-manager/state.json` |
| Applet | The `runtime-state-path` setting, shown as "Runtime snapshot file" | the same path |

```sh
systemctl --user edit omnitensor.service
# [Service]
# Environment=OMNITENSOR_STATE_PATH=/new/path/state.json
systemctl --user restart omnitensor.service
```

Set the same path in the applet setting. An applet pointed at a file nobody
writes reports "No runtime service is publishing state", which reads exactly
like a service that is not running — so check both names before concluding the
service is down.

The service refuses to start when `org.cinnamon.OmniTensor1` is already owned,
because two instances would publish to the same snapshot path. If start fails
with `already owned`, find the other instance and stop it first — an instance
started by hand outside systemd is the usual cause:

```sh
dbus-send --session --dest=org.freedesktop.DBus --print-reply \
  /org/freedesktop/DBus org.freedesktop.DBus.GetConnectionUnixProcessID \
  string:org.cinnamon.OmniTensor1
```

## Dependencies

An install with no accelerator extra starts, publishes a contract-valid
snapshot, answers the bus, and passes every acceptance check — while being
unable to run a single inference, because no execution runtime is importable.
That is the most misleading state this service has, so choose an extra
deliberately rather than taking the default.

### Python packages

| Extra | Installs | Needed for |
| --- | --- | --- |
| *(none)* | `cryptography`, `jsonschema`, `dbus-fast` | the service, bus, schemas, policy — **no inference** |
| `[gpu]` | `ncnn`, `numpy` | GPU inference over Vulkan; the only lane that works on AMD |
| `[gpu-onnx-cuda]` | `onnxruntime-gpu` | GPU inference on NVIDIA |
| `[gpu-onnx-rocm]` | `onnxruntime-rocm` | GPU inference on AMD via ROCm, instead of Vulkan |
| `[npu]` | `openvino` | Intel NPU |
| `[tpu]` | `tflite-runtime` | Coral Edge TPU |
| `[dev]` | pytest, ruff, hypothesis, mutmut | tests and linting only |

Do not install plain `onnxruntime`. That wheel ships `CPUExecutionProvider`
only, and `executors/gpu.py` refuses it by design — OmniTensor has no CPU
backend — so it can never satisfy the GPU lane no matter how it is configured.

`[gpu]` is not small: `ncnn` pulls `opencv-python` (~72 MB), `requests`,
`tqdm`, and `portalocker` transitively. `numpy` is declared explicitly because
the Vulkan executor imports it directly; relying on it arriving with `ncnn`
would turn a future wheel change into an `ImportError` at inference time.

```sh
# Install, or add an accelerator to an existing environment
~/.local/share/omnitensor/venv/bin/pip install '/path/to/omnitensor[gpu]'

# Confirm a backend is actually importable — this is what the profile status
# reflects, and it is not covered by omnitensor-verify-install
~/.local/share/omnitensor/venv/bin/python -c 'import ncnn; print(ncnn.get_gpu_count())'
```

A profile reporting `unavailable: gpu: ncnn is not installed` means the extra
was omitted; a profile reporting `Ready on gpu` means the runtime resolved.

### System packages

| Package | Needed for | Without it |
| --- | --- | --- |
| `bubblewrap` | plugin worker sandbox | external plugin workers cannot start |
| `mesa-vulkan-drivers` (or vendor driver) | Vulkan GPU inference | `ncnn` reports zero GPUs |
| `dbus` session bus | the whole control surface | the service cannot own its bus name |
| `clang`, `bpftool`, `libbpf-dev` | building the eBPF helper only | `helpers/bpf/build.sh` refuses to run |

The kernel must expose BTF at `/sys/kernel/btf/vmlinux` for the eBPF helper;
CO-RE cannot be built without it.

```sh
sudo apt install bubblewrap mesa-vulkan-drivers        # runtime
sudo apt install clang bpftool libbpf-dev              # eBPF helper build only
```

### Installing a model

A profile with no model reports `Ready on gpu; no model bundled` and runs
nothing. Giving it one is two steps: install the artifact, then declare it in
the profile's manifest.

Formats whose weights live outside the primary file — ncnn keeps every weight
in the `.bin` beside the `.param` — are installed as a set. The companion is
discovered next to the source, copied, digested, and re-verified on every
resolve, so a weights file replaced after installation makes the artifact
unresolvable rather than loading silently.

```sh
# 1. Install. For ncnn, model.bin must sit beside model.param; it is found
#    automatically and refused if absent.
omnitensor-prepare-artifact /path/to/squeezenet.param \
    --id visual-library-classifier --version 1.1.0 --format ncnn \
    --install-root ~/.local/share/omnitensor/artifacts
```

The command prints the `requirements.model` block to paste into the profile's
manifest, with the digest computed from the file it installed:

```json
{
  "id": "visual-library-classifier",
  "version": "1.1.0",
  "format": "ncnn",
  "fullyQuantized": false,
  "minimumCompilerVersion": "validated-release",
  "minimumRuntimeVersion": "validated-release",
  "sha256": "<printed by the command>"
}
```

`requirements.accelerator` must match what the format can run on — an ncnn
artifact is GPU-only, and a manifest declaring `tpu` with an ncnn model is
refused by the schema rather than silently preferring a lane it can never
take. Restart the service and the profile moves from `idle` to `watching`.

What this does **not** give you: the manifest pins the primary file only, so
the publisher is accountable for the graph and not for the weights. Tampering
after installation is caught; a substituted weights file at publication time is
not.

#### Declaring what the model expects

`requirements.model.tensorContract` states what a caller must supply. It is
optional — a manifest without it loads and runs exactly as before — and it is
deliberately in two halves:

```json
"tensorContract": {
  "inputs": [{
    "shape": [1, 3, 227, 227],
    "dtype": "float32",
    "layout": "NCHW",
    "preprocess": {
      "channelOrder": "BGR",
      "mean": [104.0, 117.0, 123.0],
      "scale": [1.0, 1.0, 1.0]
    }
  }]
}
```

`shape` and `dtype` are **enforced at admission**. A job whose input disagrees
is refused in the submitting call with `input-contract-mismatch`, instead of
failing later inside the executor as `ncnn extraction failed for output prob`
— a message raised after the job was admitted, queued, and dispatched, with
nothing in it a caller could act on.

`preprocess` is **published and never checked**. The service decodes nothing:
it never sees a picture, only a buffer, and a buffer normalised the wrong way
is a valid tensor that infers successfully and means nothing. Stating it is
still the point — no artifact records it. An ncnn `.param` declares its input
size in its `Input` layer and is silent about channel order and mean
subtraction, so without a published contract every caller guesses.

The bundled `visual-library` values come from the artifact's identity, not
from inspection. Its `model.param` and `model.bin` are byte-identical to
upstream ncnn's `squeezenet_v1.1` (Caffe layer naming, a `Softmax prob`
output), whose documented preprocessing is 227x227 BGR with mean subtraction
of `104, 117, 123` and no scaling. The shape half is verified against the
installed `.param` by `tests/test_tensorcontract.py`; the preprocessing half
rests on that identity and would have to be restated if the artifact were
ever replaced.

#### Declaring what the output means

`requirements.model.outputContract` says how to read the result. Without it
the raw tensors are returned unchanged, which is what every profile did
before:

```json
"outputContract": { "kind": "classification", "topK": 5, "labels": "labels.txt" }
```

`kind` decides whether the output reduces at all — `classification` reduces to
the highest-scoring entries, `embedding` and `raw` are returned whole. The
reduction is added as `reading` **beside** the tensors, never in place of
them, so nothing that wanted the raw scores loses them.

`labels` names a file installed beside the weights:

```sh
omnitensor-prepare-artifact /path/to/squeezenet.param \
    --id visual-library-classifier --version 1.1.0 --format ncnn \
    --labels /path/to/labels.txt \
    --install-root ~/.local/share/omnitensor/artifacts
```

It is staged and digested by the same machinery as `model.bin`, so a label
list swapped after installation makes the artifact unresolvable rather than
silently renaming every result. Without a labels file a classification reports
indices — an honest number beats a guessed name, which is wrong in a form that
reads as right.

### Large inputs

A model input does not fit in a job submission. One 3x227x227 float image is
774,394 bytes of JSON against a 262,144-byte quota, and the argv limit refuses
it before the bus does — so an image or audio model cannot be driven by
embedding its input in the request, whatever the quota is set to.

Submit a reference instead. `payload.inputRefs` names a raw little-endian
buffer with its exact shape, dtype, and digest:

```json
{"inputRefs": [{"path": "/home/u/inputs/frame.f32",
                "shape": [3, 227, 227], "dtype": "float32",
                "sha256": "<sha256 of the file>"}]}
```

That submission is 254 bytes, and the round trip works: a 3x227x227 input
submitted this way returns a 1000-class result from the GPU in roughly 2 ms
once the model is cached, about 20 ms on the first call while it loads.

Nothing is decoded — the file is raw numbers,
never a PNG or a WAV — because decoding untrusted media in the service, before
admission, is where the memory bombs would land.

Reading a caller-named file is a capability and is **off by default**. Enable
it by naming the roots a caller may reference from:

```sh
systemctl --user edit omnitensor.service
# [Service]
# Environment=OMNITENSOR_INPUT_ROOTS=%h/inputs:%h/Pictures
```

A path outside those roots is refused, and so is a symlink that leaves them —
containment is judged by where a path resolves to, not where it sits. A file
whose size or digest disagrees with the declaration is refused rather than
truncated or padded, because both of those produce a tensor that infers
successfully and means nothing.

### Uninstalling

Removing the package leaves state behind on purpose — artifacts are expensive
to re-verify and policy is a user's own configuration — so the data is removed
separately and deliberately.

```sh
# 1. Stop and remove the service
systemctl --user disable --now omnitensor.service
rm -f ~/.config/systemd/user/omnitensor.service
systemctl --user daemon-reload

# 2. Remove the environment and its launchers
rm -f ~/.local/bin/omnitensor ~/.local/bin/omnitensor-verify-install \
      ~/.local/bin/omnitensor-prepare-artifact
rm -rf ~/.local/share/omnitensor/venv

# 3. Remove one accelerator without removing the service
~/.local/share/omnitensor/venv/bin/pip uninstall -y ncnn numpy opencv-python

# 4. State — only if you mean it. Artifacts, policy, recorded telemetry.
#    Removing an artifact also removes its companion files.
rm -rf ~/.local/share/omnitensor/artifacts     # verified models
rm -rf ~/.local/state/omnitensor               # policy, cancellation journal
rm -f  ~/.local/state/tpu-workload-manager/state.json   # published snapshot

# 5. The privileged eBPF helper, if it was installed
sudo systemctl disable --now omnitensor-bpf.service
sudo rm -f /etc/systemd/system/omnitensor-bpf.service
sudo rm -rf /usr/lib/omnitensor /sys/fs/bpf/omnitensor
sudo systemctl daemon-reload
```

Removing the snapshot while the applet is running is safe: the applet reports
an absent runtime rather than showing the last values it saw.

## Applet

Build the payload, install exactly it, and verify the installed tree against
the checksums it was built from:

```sh
cd /path/to/cinnamon-tpuwlm
npm run package
rm -rf ~/.local/share/cinnamon/applets/cinnamon-tpuwm@geraldo-netto
cp -a dist/cinnamon-tpuwm@geraldo-netto ~/.local/share/cinnamon/applets/
node scripts/package-applet.js verify \
  ~/.local/share/cinnamon/applets/cinnamon-tpuwm@geraldo-netto
```

Reload the applet in the running session without restarting Cinnamon:

```sh
dbus-send --session --dest=org.Cinnamon --type=method_call /org/Cinnamon \
  org.Cinnamon.ReloadXlet string:'cinnamon-tpuwm@geraldo-netto' string:'APPLET'
```

## Verify

```sh
omnitensor-verify-install \
  --applet-root ~/.local/share/cinnamon/applets/cinnamon-tpuwm@geraldo-netto \
  --applet-checksums /path/to/cinnamon-tpuwlm/dist/cinnamon-tpuwm@geraldo-netto.SHA256SUMS
```

It exits non-zero if any check fails and prints one line per check:

| Check | What a failure means |
| --- | --- |
| `executable` | the unit's `ExecStart` does not resolve |
| `service` | the unit is not active |
| `schemas` | a canonical schema is missing from the install, or is present but is not a valid schema |
| `workloads` | the bundled catalog does not load — a packaging defect, not a user error |
| `discovery` | plugin discovery raised instead of returning a catalog |
| `isolation` | an external worker would launch without a sandbox |
| `dbus` | the bus did not answer, or its acknowledgement violates the applet contract |
| `snapshot` | no snapshot, an invalid one, or a stale one the applet would still render |
| `applet` | the installed tree does not match the payload checksums |

The D-Bus check sends a deliberately invalid command and requires a
contract-valid rejection, so it exercises the whole transport without changing
any policy.

Run this before provisioning use cases. Which accelerator lanes the host can
then qualify against is a separate question — see
[use-cases.md](use-cases.md).
