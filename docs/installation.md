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

### Deploy the applet before, or with, the service

The applet validates a snapshot as a closed record. A field this service adds
and the installed applet does not know invalidates the *whole* document, and
the applet reports that as "No runtime service is publishing state" — which is
exactly what a service that has stopped looks like.

Making the field optional does not prevent it, and neither does merging the
applet change first: what matters is the order the two are *deployed*. Install
the applet payload built from the same contracts before restarting a service
that publishes a new field, or accept a window where the desktop reads as dead.

`omnitensor-verify-install` now checks this against the snapshot actually on
disk and the applet actually installed, so a mismatch is named at install time
rather than turning up later as an outage:

```
PASS  applet-contract: the installed applet reads the snapshot this service publishes
```

The service publishes one file and the Cinnamon applet reads it; that file is
the only place the two meet. Each side names it independently, so moving it is
a two-sided edit and changing one side alone is silent:

| Side | What names the path | Default |
| --- | --- | --- |
| Service | `OMNITENSOR_STATE_PATH` | `~/.local/state/xpu-workload-manager/state.json` |
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

### Select an accelerator on multi-device hosts

OmniTensor enumerates the device nodes that exist when the service starts and
during core-runtime rediscovery; it does not assume `renderD128`, `accel0`, or
`apex_0`. With no override it chooses the lowest numbered detected node for
each backend. The active runtime IDs are published as `devices[].id` in the
snapshot, for example `gpu-renderD129`, `npu-accel3`, or `tpu-pcie-2`.

The snapshot intentionally publishes only the active device for each backend.
List all selectable kernel nodes on a multi-device host and translate each
basename to its runtime ID with:

```sh
for node in /dev/dri/renderD*; do
  test -e "$node" && printf 'gpu-%s\n' "$(basename "$node")"
done
for node in /dev/accel/accel*; do
  test -e "$node" && printf 'npu-%s\n' "$(basename "$node")"
done
for node in /dev/apex_*; do
  test -e "$node" && printf 'tpu-pcie-%s\n' "${node##*_}"
done
```

`tpu-usb` is also selectable when the runtime detects a supported Coral USB
identity. These IDs describe host-local kernel nodes; verify the associated
device with the host's driver or `udevadm` tooling before choosing one.

To choose a different installed device, set its published runtime ID rather
than a Vulkan/OpenVINO enumeration index:

```sh
systemctl --user edit omnitensor.service
# [Service]
# Environment=OMNITENSOR_GPU_DEVICE=gpu-renderD129
# Environment=OMNITENSOR_NPU_DEVICE=npu-accel3
# Environment=OMNITENSOR_TPU_DEVICE=tpu-pcie-2
systemctl --user daemon-reload
systemctl --user restart omnitensor.service
```

Each setting is optional and independent. A stale, mistyped, or wrong-host ID
matches no device, so that backend remains unavailable instead of silently
using another accelerator. Remove the override to restore automatic selection.
Changing a Linux node number after reboot is harmless when no override is set.
Automatic discovery derives the selected node again. Core executors follow
rediscovery, but installed external workers have immutable device mounts:
restart `omnitensor.service` after accelerator hotplug or a node-number change
so their sandboxes are rebuilt from the new node, major/minor, and exact
read-only sysfs identity.

Selecting a node does not transfer model qualification to another device
family or runtime stack. The current Qwen receipt matches the native runtime
bytes and Vulkan-reported device name, not a unique card serial or PCI address;
two indistinguishable same-name devices satisfy that identity check. A
different reported GPU/NPU/TPU identity remains unavailable until the provider
has passed its native runtime, full-offload, parity, memory, and workload gates
for that identity.

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
| `[convert]` | `pnnx` | packaging only: ONNX/TorchScript to ncnn |
| `[convert-npu]` | `openvino`/`ovc` | packaging only: ONNX to OpenVINO IR |
| `[train]` | `onnx` | producer only: validate and export a portable local fit |
| `[dev]` | pytest, ruff, hypothesis, mutmut | tests and linting only |

Do not install plain `onnxruntime`. That wheel ships `CPUExecutionProvider`
only, and `executors/gpu.py` refuses it by design — OmniTensor has no CPU
backend — so it can never satisfy the GPU lane no matter how it is configured.

`[gpu]` is not small: `ncnn` pulls `opencv-python` (~72 MB), `requests`,
`tqdm`, and `portalocker` transitively. `numpy` is declared explicitly because
the Vulkan executor imports it directly; relying on it arriving with `ncnn`
would turn a future wheel change into an `ImportError` at inference time.

Training dependencies are neither mandatory nor useful in the service
environment. Install `[train]` in a separate producer venv, add `[convert]`
for ncnn GPU artifacts and/or `[convert-npu]` for OpenVINO NPU artifacts, then
install only the generated native files into the service artifact store. See
[Local model training](local-training.md) for the complete record, fit,
compile, binding, and restart workflow.

The four optional Qwen workflows have an additional, fail-closed provider
installation: a locally built llama.cpp/Vulkan wheel, four identity-isolated
workload wheels, one shared runtime wheel, pinned Qwen/BGE artifacts, and
per-plugin grants. Follow [Installing the Qwen workload providers](qwen-workload-installation.md)
rather than installing an ordinary llama.cpp wheel or copying model files into
the store by hand.

```sh
# Install, or add an accelerator to an existing environment
~/.local/share/omnitensor/venv/bin/pip install '/path/to/omnitensor[gpu]'

# Confirm a backend is actually importable — this is what the profile status
# reflects, and it is not covered by omnitensor-verify-install
~/.local/share/omnitensor/venv/bin/python -c 'import ncnn; print(ncnn.get_gpu_count())'
```

A profile reporting `unavailable: gpu: ncnn is not installed` means the extra
was omitted; a profile reporting `Ready on gpu` means the runtime resolved.

`omnitensor-verify-install` asks the executors whether each installed runtime
is usable, not merely whether it imports — so a plain `onnxruntime` is reported
as unusable with the executor's own reason rather than counted as a GPU lane.
It probes the accelerator to answer that, so ncnn prints the devices it finds
before the report; that output is the hardware talking, not a fault.

### System packages

| Package | Needed for | Without it |
| --- | --- | --- |
| `bubblewrap` | plugin worker sandbox | external plugin workers cannot start |

Plugin workers are confined by a seccomp filter built from a per-architecture
syscall table, and tables are written down for **x86_64 and aarch64 only**. On
any other machine a worker refuses to start rather than run plugin code
unconfined; `--no-seccomp` exists as a deliberate operator override and is not
selected automatically. A table cannot be extended from documentation alone —
a filter built from the wrong ABI's numbers denies unrelated syscalls — so an
architecture joins that list only once the suite has been run on it.
| `mesa-vulkan-drivers` (or vendor driver) | Vulkan GPU inference | `ncnn` reports zero GPUs |
| `dbus` session bus | the whole control surface | the service cannot own its bus name |
| `clang`, `bpftool`, `libbpf-dev` | building the eBPF helper only | `helpers/bpf/build.sh` refuses to run |

The kernel must expose BTF at `/sys/kernel/btf/vmlinux` for the eBPF helper;
CO-RE cannot be built without it.

```sh
sudo apt install bubblewrap mesa-vulkan-drivers        # runtime
sudo apt install clang bpftool libbpf-dev              # eBPF helper build only
```

### Optional privileged eBPF telemetry helper

The helper is optional. Without it, OmniTensor reports kernel telemetry as
absent and continues to serve other collectors. Installing it adds aggregate
run-queue and block-I/O latency histograms; it never exports process, user,
path, cgroup, or payload identity.

The unprivileged reader defaults to `/run/omnitensor/bpf-aggregate.sock`, a
256 KiB aggregate limit, and a 2.0-second timeout. Missing or unreachable
helpers produce an explicit unavailable state rather than invented zeroes.

Build the CO-RE object as an unprivileged user, then copy only the resulting
object, helper, and unit into root-owned locations. Never run the privileged
unit from a checkout, virtual environment, or other user-writable path.

```sh
test -r /sys/kernel/btf/vmlinux
mountpoint -q /sys/fs/bpf

bpf_build_dir="$(mktemp -d)"
helpers/bpf/build.sh "$bpf_build_dir"

sudo install -d -o root -g root -m 0755 /usr/lib/omnitensor/bpf
sudo install -o root -g root -m 0644 \
  "$bpf_build_dir/runq_latency.bpf.o" \
  /usr/lib/omnitensor/bpf/runq_latency.bpf.o
sudo install -o root -g root -m 0755 \
  helpers/bpf/omnitensor-bpf-helper.py \
  /usr/lib/omnitensor/bpf/omnitensor-bpf-helper.py
sudo install -o root -g root -m 0644 \
  helpers/bpf/omnitensor-bpf.service \
  /etc/systemd/system/omnitensor-bpf.service

sudo systemctl daemon-reload
sudo systemctl enable --now omnitensor-bpf.service
```

The unit receives only `CAP_BPF` and `CAP_PERFMON`. It loads the object with
`bpftool ... loadall ... autoattach`, pins three maps and three links below
`/sys/fs/bpf/omnitensor`, verifies every pin, and fails rather than serving an
empty result after a partial load.

Verify the privileged boundary before trusting its measurements:

```sh
sudo systemctl is-active omnitensor-bpf.service

for map in runq_latency_us block_latency_us wakeup_at; do
  sudo bpftool map show pinned "/sys/fs/bpf/omnitensor/$map"
done
for link in on_wakeup on_switch on_block_complete; do
  sudo bpftool link show pinned "/sys/fs/bpf/omnitensor/$link"
done

sudo /usr/lib/omnitensor/bpf/omnitensor-bpf-helper.py --print-once
python3 - <<'PY'
import json
import socket

with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    client.connect("/run/omnitensor/bpf-aggregate.sock")
    document = json.loads(client.recv(256 * 1024))
assert document["version"] == 1
assert {item["name"] for item in document["histograms"]} == {
    "runq_latency_us",
    "block_latency_us",
}
print(json.dumps(document, indent=2))
PY
```

Restart the user service with
`systemctl --user restart omnitensor.service`; its runtime snapshot should report
`kernelTelemetry.state` as `ready`. Stop here if any map, link, socket, or
snapshot check fails; do not interpret missing samples as an idle kernel.
These commands are the reproducible deployment and verification procedure,
not evidence that a particular host passed it. Live privileged acceptance is
tracked separately in `TODO.md` under OMNI-0184.

To roll back a helper upgrade, stop the service, remove its pins, install the
previous root-owned object and helper at the same paths, and start it again:

```sh
sudo systemctl stop omnitensor-bpf.service
sudo rm -rf /sys/fs/bpf/omnitensor
sudo install -o root -g root -m 0644 \
  /path/to/previous/runq_latency.bpf.o \
  /usr/lib/omnitensor/bpf/runq_latency.bpf.o
sudo install -o root -g root -m 0755 \
  /path/to/previous/omnitensor-bpf-helper.py \
  /usr/lib/omnitensor/bpf/omnitensor-bpf-helper.py
sudo systemctl start omnitensor-bpf.service
```

### Producing accelerator artifacts

Conversion and execution are separate claims. A producer host can build ncnn
or OpenVINO files without owning the target accelerator; that proves only that
the output was produced, parsed, digested, and installed. It does not prove
inference correctness or target-hardware operation.

| Lane | Output | Accepted source | Producer-host requirement | Status |
| --- | --- | --- | --- | --- |
| GPU | ncnn `.param` + `.bin` | ONNX `.onnx` or TorchScript `.pt` | a host where the `pnnx` binary from `[convert]` runs; no target GPU needed to convert | supported by `omnitensor-convert-model` |
| NPU | OpenVINO IR `.xml` + `.bin` | ONNX `.onnx` | a host where `ovc` from `[convert-npu]` runs; no target NPU needed to convert | produced, parsed, digested, and installed in the automated smoke test; NPU execution still needs Intel NPU hardware |
| TPU | compiled Edge TPU `.tflite` | fully-int8 TFLite `.tflite` | a host or container supported by Google's separately distributed `edgetpu_compiler` | not produced by OmniTensor; no pip extra or `ModelConverter` adapter exists |

Keep converter toolchains outside the service environment. They prepare files
for review and publication; the service installs only the runtime extra for
the lane it will execute.

```sh
# GPU packaging environment
python3 -m venv /tmp/omnitensor-gpu-converter
/tmp/omnitensor-gpu-converter/bin/pip install '/path/to/omnitensor[convert]'
/tmp/omnitensor-gpu-converter/bin/omnitensor-convert-model model.onnx \
    --format ncnn --input-shape '[1,3,224,224]' --output-dir ./ncnn-artifact

# NPU packaging environment
python3 -m venv /tmp/omnitensor-npu-converter
/tmp/omnitensor-npu-converter/bin/pip install '/path/to/omnitensor[convert-npu]'
/tmp/omnitensor-npu-converter/bin/omnitensor-convert-model model.onnx \
    --format openvino --input-shape '[1,3,224,224]' --output-dir ./openvino-artifact
```

Each command prints the exact `omnitensor-prepare-artifact` command for its
primary file. Preparation discovers the `.bin`, digests both files, and can
install the pair atomically. TPU production stays explicit and external until
the project chooses and tests a reproducible Edge TPU compiler distribution;
having a `.tflite` source alone does not mean it is compiled for the device.

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
  "sha256": "<printed by the command>",
  "companions": {
    "model.bin": "<printed by the command>"
  }
}
```

`companions` is not optional in practice for ncnn or OpenVINO IR. Those formats
keep every weight in the file beside the primary one, so a manifest that names
only the primary has vouched for the graph and not for the numbers it runs, and
dispatch refuses it with `companion-unpinned`. The store re-verifies its own
record of a companion on every resolve, which catches one replaced after
installation — it cannot catch one substituted before it, because the store
recorded the substitute and re-verifies that faithfully for ever. Only the
publisher's own digest closes it, which is what this block is.

`requirements.accelerator` must match what the format can run on — an ncnn
artifact is GPU-only, and a manifest declaring `tpu` with an ncnn model is
refused by the schema rather than silently preferring a lane it can never
take. Restart the service and the profile moves from `idle` to `watching`.

### Serving more than one accelerator

`acceleratorPreference` lists the lanes a profile would like, in order. A
profile declaring a single `model` can only ever take the lane its format
belongs to — ncnn is GPU, `tflite-edgetpu` is TPU, OpenVINO IR is NPU — so the
preference had nothing to choose between.

Declare `requirements.models` instead, one entry per lane, and drop `model`:

```json
"models": [
  {"id": "classifier",     "format": "ncnn",     "sha256": "…", "…": "…"},
  {"id": "classifier-npu", "format": "openvino", "sha256": "…", "…": "…"}
]
```

The lane is chosen first and the artifact follows it: the runtime walks the
preference, takes the first accelerator that is available *and* has an entry it
can run, then resolves that entry. A manifest declares one spelling or the
other, never both.

Every entry describes the same network in a different format, so they must
agree on `tensorContract` and `outputContract` and each format may appear once.
A manifest that breaks either is refused at load with the reason, because a
consumer reading the input contract before a lane is chosen would otherwise be
reading whichever entry happened to be first.

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
silently renaming every result.

### A model runs only if its manifest names it

`requirements.model.sha256` is what ties a profile to an exact file. Without it
the strongest available guarantee is the store's own — the active version was
digest-verified when it was installed and is re-verified on every dispatch —
and that proves the file has not changed since installation, not that it is the
file the publisher meant. Anything installed under the same id and version
satisfies an unpinned manifest, which is precisely what the digest exists to
rule out.

So an unpinned model does not run. Dispatch refuses it with `model-unpinned`
before the artifact store is consulted at all, and the snapshot reports the
profile unavailable with the same reason, so it is visible before somebody
submits a job rather than at the moment they do.

The field stays optional in the schema. A manifest for an artifact that has not
been published yet genuinely cannot pin one, and refusing to *load* such a
document would lose the policy, UI, and acceptance data it carries — the
profile simply cannot run until the digest is there. Without a labels file a classification reports
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

The refusal for a path outside the roots is identical to the refusal for a path
that does not exist, so a caller cannot probe the filesystem by trying — which
also means it cannot find the right directory by trying. The roots are
therefore published in the snapshot, beside the devices and the profiles:

```json
"inputs": {"roots": ["/home/u/inputs"], "maxBytes": 67108864}
```

Empty roots mean this service will read no referenced file at all, which is the
default. An absent block means a runtime older than the field. Both refuse every
reference, so a consumer that treats them the same is right.

### Who turns a picture into a tensor

The service does not, and will not. `inputRefs` takes raw numbers; decoding a
PNG means running an image decoder inside the process that holds the
accelerator, before admission, on bytes a caller chose.

The caller does it, and the manifest tells it how. `requirements.model
.tensorContract` states the shape, the dtype, and — separately — the channel
order, mean, and scale the model was trained with. The first two are enforced
at admission; the third is published and never checked, because the service
decodes nothing and so cannot tell a correctly normalised tensor from a wrong
one. A contract that pretended otherwise would refuse correct jobs and accept
incorrect ones with equal confidence.

For the desktop, the Cinnamon applet is that caller: it decodes with GdkPixbuf,
scales to the declared size, applies the declared normalisation, writes the
buffer into a subdirectory of one of the published roots, and submits a
reference to it. A profile that states a shape but not how to normalise a
picture for it is not offered a picture at all — the applet says what the
manifest failed to state rather than guessing RGB where the model wanted BGR,
which is a wrong answer that looks exactly like a right one.

### Granting a plugin the permissions it declares

A plugin declares what it needs in `plugin.permissions`, and the runtime
refuses its jobs with `consent-missing` until each one is granted. Nothing
could grant them: the ledger existed, the service never constructed it, and
every declared permission read as ungranted for ever — a refusal with no
remedy, which looks like a decision somebody made.

```sh
omnitensor-grant list sample-plugin
omnitensor-grant grant sample-plugin files:read --reason "indexing my pictures"
omnitensor-grant revoke sample-plugin files:read
```

This is a command and not a bus method on purpose. A grant authorises a plugin
to reach something outside itself, which is a larger thing to hand out than
enabling a profile, and on a session bus every peer runs as the same user — the
service cannot tell one from another, so a bus method would accept consent from
anything already able to talk to it and record it as yours. Running a command
requires access to this account before it starts.

What a grant means:

- **Per plugin and permission.** Granting one says nothing about any other.
- **Only what the manifest declared.** A permission the plugin never asked for
  cannot be granted, and a manifest that later drops one stops it being usable.
- **Persistent until revoked.** Re-consenting every boot trains people to say
  yes without reading.
- **Withdrawn immediately.** Enforcement re-reads the ledger before it decides,
  so revoking stops a job already sitting in a queue.

Every change records who made it, when, and why. The ledger lives at
`OMNITENSOR_GRANTS_PATH` (default `~/.local/state/omnitensor/grants.json`) and
carries its own revision, so two shells racing cannot both write.

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
rm -f  ~/.local/state/xpu-workload-manager/state.json   # published snapshot

# 5. The privileged eBPF helper, if it was installed
sudo systemctl disable --now omnitensor-bpf.service
sudo rm -f /etc/systemd/system/omnitensor-bpf.service
sudo rm -rf /usr/lib/omnitensor /sys/fs/bpf/omnitensor
sudo systemctl daemon-reload
```

Removing the snapshot while the applet is running is safe: the applet reports
an absent runtime rather than showing the last values it saw.

## What the bus boundary separates

Jobs, quotas, and results are scoped by the caller's uid. On a session bus
that separates you from another *user*, not from another program you are
running yourself — see [bus-boundary.md](bus-boundary.md) for what the
boundary is worth, why per-connection scoping is not offered, and what to
change if it ever needs to be stronger.

## Applet

Build the payload, install exactly it, and verify the installed tree against
the checksums it was built from:

The applet UUID changed with the product rename. Before installing, remove the
old `cinnamon-tpuwm@geraldo-netto` panel instance in Cinnamon Settings; the old
and new UUIDs are separate applets and must not run together. The new applet
performs a bounded one-time settings import and reads the old applet-state file
as a migration fallback. The service now publishes to
`~/.local/state/xpu-workload-manager/state.json`, so deploy the matched service
and applet builds together rather than mixing names or schema digests.

```sh
cd /path/to/cinnamon-xpuwlm
npm run package
rm -rf ~/.local/share/cinnamon/applets/cinnamon-xpuwlm@geraldo-netto
cp -a dist/cinnamon-xpuwlm@geraldo-netto ~/.local/share/cinnamon/applets/
node scripts/package-applet.js verify \
  ~/.local/share/cinnamon/applets/cinnamon-xpuwlm@geraldo-netto
```

After the new applet is added and verified, the obsolete payload can be
removed:

```sh
rm -rf ~/.local/share/cinnamon/applets/cinnamon-tpuwm@geraldo-netto
```

Reload the applet in the running session without restarting Cinnamon:

```sh
dbus-send --session --dest=org.Cinnamon --type=method_call /org/Cinnamon \
  org.Cinnamon.ReloadXlet string:'cinnamon-xpuwlm@geraldo-netto' string:'APPLET'
```

## Verify

```sh
omnitensor-verify-install \
  --applet-root ~/.local/share/cinnamon/applets/cinnamon-xpuwlm@geraldo-netto \
  --applet-checksums /path/to/cinnamon-xpuwlm/dist/cinnamon-xpuwlm@geraldo-netto.SHA256SUMS
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
