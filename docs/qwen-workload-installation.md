# Installing the Qwen workload providers

This is the operator procedure for the four local, manually triggered Qwen
workloads:

- `event-extraction` reads only files selected for that request and returns an
  evidence-backed event preview;
- `ask-selected-files` retrieves BGE spans from selected documents and answers
  with exact file, page, span, and digest citations;
- `selected-text-tools` explains, summarizes, rewrites, translates, or extracts
  tasks from one explicit clipboard selection; and
- `file-organizer` returns a review-only naming, tagging, folder, and exact
  duplicate plan. It cannot move, rename, overwrite, or delete a file.

The service does not bundle or silently download model weights. Each workload
manifest is disabled by default, and the Cinnamon action stays unavailable when
its distribution, artifact, permission, worker, or accelerator check fails.
Installing only the Python packages is therefore not a successful model
installation.

## Package and model layout

The provider is five wheels rather than one wheel with four entry points. An
OmniTensor plugin distribution may own exactly one identity and exactly one
`omnitensor-plugin.json`, so each workload needs an identity-isolated thin
wheel. All four depend on one implementation wheel:

| Wheel | Responsibility | Pinned model |
| --- | --- | --- |
| `omnitensor-qwen-vulkan-runtime` | In-process llama.cpp/Vulkan generation, the shared GPU lease, BGE/ncnn retrieval, and the four factories | runtime only |
| `omnitensor-qwen-event-extraction` | `event-extraction` entry point and manifest | Qwen3-4B Q4_K_M |
| `omnitensor-qwen-ask-selected-files` | `ask-selected-files` entry point and manifest | Qwen3-0.6B Q8_0 plus BGE-small-en-v1.5 |
| `omnitensor-qwen-selected-text-tools` | `selected-text-tools` entry point and manifest | Qwen3-0.6B Q8_0 |
| `omnitensor-qwen-file-organizer` | `file-organizer` entry point and manifest | Qwen3-4B Q4_K_M |

The smaller generation artifact is the official
`Qwen/Qwen3-0.6B-GGUF` `Qwen3-0.6B-Q8_0.gguf` at upstream revision
`1208e45d782fe18602c5eaf10e5758d5b0f24c03`. It is Apache-2.0,
804,753,088 bytes, and has SHA-256
`12fae8b8f78f0360b498d04c8db7d33aff29ab7d8080231f93a17c18119e6735`.
The checked-in [generation catalog](../generation-models/qwen3-0.6b.json)
records its source and license terms.

Event extraction and file organizer use the larger official
`Qwen/Qwen3-4B-GGUF` `Qwen3-4B-Q4_K_M.gguf` at revision
`bc640142c66e1fdd12af0bd68f40445458f3869b`. It is also Apache-2.0,
2,497,280,256 bytes, and has SHA-256
`7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`.
The checked-in [4B generation catalog](../generation-models/qwen3-4b.json)
binds those bytes to their license, provider, and deliberately narrow
qualification scope.
Review the pinned [Qwen3-4B Apache-2.0 license](https://huggingface.co/Qwen/Qwen3-4B-GGUF/blob/bc640142c66e1fdd12af0bd68f40445458f3869b/LICENSE)
before downloading it.
The split is deliberate: Ask and selected-text tools retain the smaller
accepted pin, while the two tasks currently needing stronger structured
generation pay the larger GPU memory and load cost. No checkout, package
install, or catalog entry is by itself a live readiness or quality claim;
final provider qualification and `DescribePlugins` must still pass.

Ask selected files additionally requires the accepted BGE/ncnn graph and its
two declared companions:

| File | SHA-256 |
| --- | --- |
| `model.param` | `8a27ec7beb314ba2824a8c03ced33f922f079a8d9dc8aefb712d4b1c165d33bb` |
| `model.bin` | `44034070d5e86e6be4829dc4a69c1e959ee895b07877d89b34c98d2adcf9e994` |
| `tokenizer.json` | `d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66` |

Those three files are one immutable artifact named
`bge-small-en-v1-5-ask-gpu` version `1.0.0`. The BGE source is MIT; its exact
upstream revision, portable inputs, tokenizer, and producer contract are in
[the reviewed recipe](../model-recipes/bge-small-en-v1-5.json). A generic BGE
ncnn export is not interchangeable with this graph.

## Dependencies

### Mandatory at runtime

| Dependency | Why it is mandatory |
| --- | --- |
| Linux, Python 3.11 or newer, and a matching OmniTensor 0.x installation | Plugin protocol, schemas, artifact handoff, and worker bootstrap |
| `bubblewrap` | Read-only artifact and selected-input mounts, private namespaces, and the worker filesystem boundary |
| A real Vulkan GPU, Vulkan loader, and vendor ICD | There is no CPU provider or CPU fallback; software Vulkan devices do not qualify |
| Access to the selected `/dev/dri/renderD*` device | The sandbox mounts only the device granted to that worker |
| The exact accepted `llama-cpp-python==0.3.34` Vulkan wheel | In-process Qwen execution inside the seccomp worker; startup verifies its native-library hashes, so an arbitrary rebuild or ordinary CPU wheel is not acceptable |
| All five provider wheels | The runtime implementation and all four independently discoverable workload identities |
| Both pinned Qwen GGUFs when installing all four workloads | 0.6B generation for Ask/selected text; 4B generation for events/file plans |
| `ncnn>=1.0.20260526`, `numpy>=1.24`, `tokenizers>=0.22`, and the pinned BGE files | Retrieval and tokenization for Ask selected files |
| A user D-Bus session | The OmniTensor control and job-result surface |

The worker supports the repository's x86_64 and aarch64 seccomp tables, but the
receipt shipped in this provider release records one x86_64 native runtime.
An aarch64 release needs its own native wheel, receipt, and acceptance evidence.
A different architecture fails closed unless that ABI is implemented and
tested; do not turn off seccomp merely to make a model load.

### Optional at runtime

| Dependency | Enables |
| --- | --- |
| `pymupdf>=1.24` (`omnitensor[events]`) | PDF and image extraction for event extraction, Ask selected files, and file organizer |
| Tesseract plus the required language packs | OCR through the optional PyMuPDF adapter |
| Cinnamon XPU WLM | Desktop discovery and actions; D-Bus clients can use the workloads without the applet |
| A separately qualified NPU provider | An explicit NPU route; the Vulkan wheels in this guide do not provide one |

Plain UTF-8 text and Markdown do not require PyMuPDF. Event extraction also
parses iCalendar without it. A missing optional parser produces a bounded
adapter refusal for that file type; it does not make the service use a CPU
model.

### Build- or producer-only

Keep these out of the service environment:

- a C/C++ toolchain, Python development headers, CMake, Ninja, Vulkan headers,
  and a GLSL compiler to build the Vulkan llama.cpp wheel;
- Hatchling/build tooling to build the five provider wheels; and
- `omnitensor[document-producers]` (PyTorch, Transformers, safetensors, ONNX,
  ONNX Runtime, pnnx, ncnn, NumPy, and tokenizers) to reproduce and qualify the
  BGE native artifact.

Building a model is not qualification. The resulting bytes must match the
provider manifest and pass the named-device gates before they can become
ready.

## 1. Check the host

On Debian or Ubuntu, the minimum system packages are typically:

```sh
sudo apt install bubblewrap libvulkan1 mesa-vulkan-drivers vulkan-tools \
  libglib2.0-bin build-essential python3-dev cmake ninja-build glslang-tools
```

Use the vendor Vulkan package instead of Mesa where appropriate. Confirm that
Vulkan sees a hardware device and that the service account can open its render
node:

```sh
vulkaninfo --summary
ls -l /dev/dri/renderD*
```

Do not continue with llvmpipe or another software device. Fix the driver or
render-group access, then log out and back in if group membership changed.

## 2. Install the accepted Vulkan runtime and build provider wheels

The provider receipt pins the exact accepted wheel and native libraries. Obtain
the `llama_cpp_python-0.3.34-py3-none-linux_x86_64.whl` retained with the
qualification run and verify it before installation:

```sh
OMNI_LLAMA_WHEEL=/absolute/path/to/qualified/llama_cpp_python-0.3.34-py3-none-linux_x86_64.whl
printf '%s  %s\n' \
  '8fe651093900aa4b2a21c1bd5d0ca6bc6bed619511f2962deac6a06689a19aca' \
  "$OMNI_LLAMA_WHEEL" | sha256sum --check --strict
```

Do not substitute a locally rebuilt wheel: compiler, linker, CMake, Vulkan SDK,
and source-tree differences change the native bytes even with the same Python
version and flags, and startup correctly refuses that mismatch. If the exact
wheel is unavailable or incompatible with the target host, rebuild and rerun
the complete native/workload qualification, then issue a new provider release
and receipt. Never edit the embedded receipt to make different bytes load.

Use a builder environment with the same Python major/minor version and machine
architecture as the service for the five Python provider wheels. Replace
`/absolute/path/to/omnitensor` with this checkout:

```sh
python3 -m venv ~/.local/share/omnitensor-provider-builder/venv
OMNI_BUILDER=~/.local/share/omnitensor-provider-builder/venv/bin
OMNI_SOURCE=/absolute/path/to/omnitensor
OMNI_WHEELS=~/.local/share/omnitensor-provider-wheels
mkdir -p "$OMNI_WHEELS"

"$OMNI_BUILDER/pip" install --upgrade pip build hatchling wheel

for provider in qwen-vulkan-runtime event-extraction ask-selected-files \
  selected-text-tools file-organizer; do
  "$OMNI_BUILDER/pip" wheel --no-deps \
    "$OMNI_SOURCE/providers/$provider" --wheel-dir "$OMNI_WHEELS"
done
```

Install the core service and explicit runtime dependencies first, then install
the accepted llama.cpp wheel and provider wheels with dependency resolution
disabled. That ordering prevents pip from replacing the accepted Vulkan build
with an index-provided CPU build:

```sh
OMNI_SERVICE=~/.local/share/omnitensor/venv/bin

"$OMNI_SERVICE/pip" install '/absolute/path/to/omnitensor[gpu]'
"$OMNI_SERVICE/pip" install \
  'ncnn>=1.0.20260526' 'numpy>=1.24' 'tokenizers>=0.22'
"$OMNI_SERVICE/pip" install --no-deps \
  "$OMNI_LLAMA_WHEEL"
"$OMNI_SERVICE/pip" install --no-deps \
  "$OMNI_WHEELS"/omnitensor_qwen_vulkan_runtime-0.1.1-*.whl \
  "$OMNI_WHEELS"/omnitensor_qwen_event_extraction-0.1.0-*.whl \
  "$OMNI_WHEELS"/omnitensor_qwen_ask_selected_files-0.1.0-*.whl \
  "$OMNI_WHEELS"/omnitensor_qwen_selected_text_tools-0.1.0-*.whl \
  "$OMNI_WHEELS"/omnitensor_qwen_file_organizer-0.1.0-*.whl
```

Install optional document/media parsing only when needed:

```sh
"$OMNI_SERVICE/pip" install 'pymupdf>=1.24'
```

## 3. Acquire and verify the artifacts

Downloading is an explicit operator action. Review the Apache-2.0 terms linked
in the generation catalog before running:

```sh
OMNI_MODELS=~/.local/share/omnitensor/provider-sources
mkdir -p "$OMNI_MODELS/qwen3-0.6b"

curl --fail --location \
  'https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/1208e45d782fe18602c5eaf10e5758d5b0f24c03/Qwen3-0.6B-Q8_0.gguf' \
  --output "$OMNI_MODELS/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf"

printf '%s  %s\n' \
  '12fae8b8f78f0360b498d04c8db7d33aff29ab7d8080231f93a17c18119e6735' \
  "$OMNI_MODELS/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf" | sha256sum --check --strict
test "$(stat --format=%s "$OMNI_MODELS/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf")" \
  -eq 804753088

mkdir -p "$OMNI_MODELS/qwen3-4b"
curl --fail --location \
  'https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/bc640142c66e1fdd12af0bd68f40445458f3869b/Qwen3-4B-Q4_K_M.gguf' \
  --output "$OMNI_MODELS/qwen3-4b/Qwen3-4B-Q4_K_M.gguf"

printf '%s  %s\n' \
  '7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5' \
  "$OMNI_MODELS/qwen3-4b/Qwen3-4B-Q4_K_M.gguf" | sha256sum --check --strict
test "$(stat --format=%s "$OMNI_MODELS/qwen3-4b/Qwen3-4B-Q4_K_M.gguf")" \
  -eq 2497280256
```

Produce BGE in a separate producer venv as described in
[Local model training](local-training.md#install-bge-small-for-document-intelligence-on-vulkan),
or obtain the accepted files from a trusted publisher. Before installation,
set these variables to absolute paths and check that the three hashes match the
table above:

```sh
BGE_PARAM=/absolute/path/to/qualified-bge/model.param
BGE_BIN=/absolute/path/to/qualified-bge/model.bin
BGE_TOKENIZER=/absolute/path/to/qualified-bge/tokenizer.json
sha256sum "$BGE_PARAM" "$BGE_BIN" "$BGE_TOKENIZER"
```

The one-shot installer uses one `Apache-2.0` acknowledgement for both official
Qwen artifacts and requires both license identifiers literally. It verifies
all five source files, including both Qwen byte counts and every declared digest,
before importing immutable versions into the artifact store:

```sh
OMNI_ARTIFACTS=~/.local/share/omnitensor/artifacts

"$OMNI_SERVICE/omnitensor-install-qwen-artifacts" \
  --artifact-root "$OMNI_ARTIFACTS" \
  --qwen-model "$OMNI_MODELS/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf" \
  --qwen-large-model "$OMNI_MODELS/qwen3-4b/Qwen3-4B-Q4_K_M.gguf" \
  --bge-param "$BGE_PARAM" \
  --bge-bin "$BGE_BIN" \
  --bge-tokenizer "$BGE_TOKENIZER" \
  --accept-qwen-license Apache-2.0 \
  --accept-bge-license MIT
```

Set `OMNITENSOR_ARTIFACT_ROOT` for the service to the same absolute store if it
is not using the default. Never replace a file below the store in place; a
changed primary or companion digest makes the provider unavailable.

On a host with more than one GPU, enumerate its render nodes and set
`OMNITENSOR_GPU_DEVICE` to the desired host-local runtime ID as described in
[Select an accelerator on multi-device hosts](installation.md#select-an-accelerator-on-multi-device-hosts).
Do not copy a `renderD*` value from another computer: Linux node numbers are
host-local and the service derives the selected node's major/minor and exact
sysfs identity at runtime. Restart the service after changing the selection or
after GPU hotplug so each external-worker sandbox is rebuilt. This provider
still starts only when the selected GPU's Vulkan-reported name matches its
qualification receipt and proves complete Vulkan offload; the receipt does not
distinguish two same-name cards by serial number or PCI address, and selection
is not a way to bypass hardware acceptance.

## 4. Grant only the declared access

Package installation does not imply consent. All four workers require the GPU
grant. The three selected-file workflows additionally require access to only
the files brokered for the current request; selected-text tools requires one
explicit clipboard read:

```sh
"$OMNI_SERVICE/omnitensor-grant" grant event-extraction accelerator:gpu \
  --reason 'run the explicitly requested model on my GPU'
"$OMNI_SERVICE/omnitensor-grant" grant event-extraction files:read-selected \
  --reason 'read files I explicitly select for event extraction'

"$OMNI_SERVICE/omnitensor-grant" grant ask-selected-files accelerator:gpu \
  --reason 'run the explicitly requested model on my GPU'
"$OMNI_SERVICE/omnitensor-grant" grant ask-selected-files files:read-selected \
  --reason 'answer only over files I explicitly select'

"$OMNI_SERVICE/omnitensor-grant" grant selected-text-tools accelerator:gpu \
  --reason 'run the explicitly requested model on my GPU'
"$OMNI_SERVICE/omnitensor-grant" grant selected-text-tools clipboard:read-once \
  --reason 'read one selection only after I invoke a text action'

"$OMNI_SERVICE/omnitensor-grant" grant file-organizer accelerator:gpu \
  --reason 'run the explicitly requested model on my GPU'
"$OMNI_SERVICE/omnitensor-grant" grant file-organizer files:read-selected \
  --reason 'review only files I explicitly select'
```

Grant changes are live. Revoking a declared permission refuses new and queued
work and cancels in-flight external-plugin work; a restart is not needed for a
revocation to take effect.

## 5. Restart and verify readiness

Restart after changing installed distributions or artifacts so discovery and
worker startup see one consistent release:

```sh
systemctl --user daemon-reload
systemctl --user restart omnitensor.service
systemctl --user is-active omnitensor.service
"$OMNI_SERVICE/omnitensor-verify-install"

gdbus call --session \
  --dest org.cinnamon.OmniTensor1 \
  --object-path /org/cinnamon/OmniTensor1 \
  --method org.cinnamon.OmniTensor1.DescribePlugins

journalctl --user --unit omnitensor.service --since '-5 minutes' \
  --no-pager
```

For each of the four IDs, `DescribePlugins` must report:

- `source: "external"`, the expected distribution, and `workerState: "ready"`;
- negotiated `execute`, `cancel`, `health`, and `progress` capabilities;
- every declared artifact with `ready: true`; and
- every permission with `granted: true`.

Startup loads Qwen in the worker and accepts it only when llama.cpp reports all
model layers offloaded to Vulkan. Ask selected files also preflights the exact
BGE graph, tokenizer, finite 384-value output, and its qualified named Vulkan
device. A CPU build, partial layer offload, wrong GPU, missing tokenizer,
changed model, absent grant, sandbox failure, or startup timeout leaves the
worker failed/unavailable. Do not enable a failed workload by weakening these
checks.

## Acceptance and what is not claimed

Ask selected files has a frozen six-case CC0 corpus and quantitative retrieval,
grounding, citation, latency, memory, cancellation, and grant-revocation gates.
The archived full-corpus report covers the pinned Qwen/BGE bytes and the native
runtime named inside that report; it must not be transferred to different
llama.cpp bytes. Verify a newly collected evidence document with:

```sh
"$OMNI_SERVICE/omnitensor-qualify-document-questions" \
  --evidence /absolute/path/to/document-question-evidence.json \
  --output /absolute/path/to/document-question-acceptance.json
```

Event extraction, selected-text tools, and file organizer currently have
closed schemas, privacy/safety regression coverage, worker startup checks, and
representative integration tests, but no declared model-quality acceptance
metrics. Their manifest `acceptance` arrays are intentionally empty. Keep them
disabled until an operator has reviewed representative results for the intended
language and document domain; installation must not be presented as a quality
qualification.

The provider wheel contains `qualification.json`. It binds the exact task
contract hashes, Qwen hashes, `llama-cpp-python` version and native-library
hashes, Vulkan-reported RX 6600 XT name, full 29/29 and 37/37 layer offload, and the four
representative closed-contract results run for this release. Startup verifies
that receipt before advertising a worker as ready. This receipt is a runtime
compatibility and representative-behavior gate; it does not replace the Ask
full-corpus report, uniquely identify a same-name physical card, or create
undeclared quality metrics for the other tasks.

## Runtime privacy, isolation, and cancellation

- Source text, selections, prompts, citations, model replies, and absolute
  paths never enter the public runtime snapshot. Job results remain private to
  the submitting D-Bus owner.
- Files and clipboard contents are admitted only after an explicit action and
  the matching live grant. No workflow watches a directory or clipboard.
- The worker sees only verified artifact directories, brokered selected-input
  paths, its private state directory, Vulkan metadata, and the exact render
  device through read-only or narrowly writable sandbox mounts. It receives no
  general host filesystem or network access.
- llama.cpp runs in-process because the seccomp boundary forbids spawning a
  model subprocess. A host-wide file lease serializes the four workers on the
  GPU; each unloads the model and releases the lease at the terminal path.
- Generation streams internally so cancellation is checked between tokens.
  Higher-level stages also check cancellation, and an unresponsive plugin
  remains inside a supervisor-owned, killable process.
- Private fragment stores and the Ask index are discarded on success, refusal,
  cancellation, or error. Event recovery state contains request/stage data, not
  source contents. File organizer exposes no filesystem mutation port.

## Larger GPU models and NPU providers

The current event and file-organizer manifests select the separately identified
Qwen3-4B Q4_K_M artifact; they do not replace the 0.6B bytes used by Ask and
selected-text tools. The same rule applies to a future larger or differently
quantized GPU model: it must be a new provider release rather than a
replacement under either existing identity. Such a release must:

1. pin the upstream revision, filename, byte count, SHA-256, quantization, SPDX
   license, and attribution in a reviewed catalog and manifest;
2. give the artifact a distinct ID/version and, when necessary, increase the
   current finite 3 GiB artifact-size bound to another finite tested value
   instead of making it unlimited;
3. update only the thin workload wheels that select the larger model, leaving
   unrelated workloads on their accepted pin;
4. prove complete layer offload, bounded context/output, peak GPU/system memory,
   quality on that workload's frozen corpus, cancellation, revocation, and
   recovery on every advertised GPU; and
5. remain disabled on download, digest, memory, compatibility, partial-offload,
   or acceptance failure. It must never continue on CPU.

An NPU is not enabled by installing the Vulkan provider. NPU support requires a
separate provider and immutable native artifact, with its compiler/runtime
versions, tokenizer, task schema parity, named-NPU execution, quality, memory,
cancellation, and revocation evidence recorded independently. An explicitly
configured NPU-first provider may choose the qualified GPU before generation
starts if NPU admission fails. It may not migrate to CPU, or silently switch
devices after generation has begun.

## Rollback

Keep the five provider wheels and Vulkan llama.cpp wheel for every deployed
release. To roll back, first revoke access or stop the service, then reinstall
one internally consistent prior set. Do not mix a runtime wheel, thin manifests,
or model identities from different releases:

```sh
systemctl --user stop omnitensor.service
"$OMNI_SERVICE/pip" install --force-reinstall --no-deps \
  /absolute/path/to/previous/llama_cpp_python-0.3.34-*.whl \
  /absolute/path/to/previous/omnitensor_qwen_vulkan_runtime-*.whl \
  /absolute/path/to/previous/omnitensor_qwen_event_extraction-*.whl \
  /absolute/path/to/previous/omnitensor_qwen_ask_selected_files-*.whl \
  /absolute/path/to/previous/omnitensor_qwen_selected_text_tools-*.whl \
  /absolute/path/to/previous/omnitensor_qwen_file_organizer-*.whl
systemctl --user start omnitensor.service
```

Artifacts are immutable and intentionally survive a wheel rollback. A previous
manifest resolves only its own exact artifact version and digest. If a newer
artifact version was activated, use the artifact store's verified rollback
operation together with the matching provider release; never edit the active
pointer or model files by hand. Repeat the complete readiness check after any
rollback.

## Uninstall

Revoke consent first, stop the service, and remove the four identity wheels.
Remove the shared runtime and llama.cpp only if no remaining provider uses
them:

```sh
"$OMNI_SERVICE/omnitensor-grant" revoke event-extraction accelerator:gpu \
  --reason 'uninstalling the Qwen workload provider'
"$OMNI_SERVICE/omnitensor-grant" revoke event-extraction files:read-selected \
  --reason 'uninstalling the Qwen workload provider'
"$OMNI_SERVICE/omnitensor-grant" revoke ask-selected-files accelerator:gpu \
  --reason 'uninstalling the Qwen workload provider'
"$OMNI_SERVICE/omnitensor-grant" revoke ask-selected-files files:read-selected \
  --reason 'uninstalling the Qwen workload provider'
"$OMNI_SERVICE/omnitensor-grant" revoke selected-text-tools accelerator:gpu \
  --reason 'uninstalling the Qwen workload provider'
"$OMNI_SERVICE/omnitensor-grant" revoke selected-text-tools clipboard:read-once \
  --reason 'uninstalling the Qwen workload provider'
"$OMNI_SERVICE/omnitensor-grant" revoke file-organizer accelerator:gpu \
  --reason 'uninstalling the Qwen workload provider'
"$OMNI_SERVICE/omnitensor-grant" revoke file-organizer files:read-selected \
  --reason 'uninstalling the Qwen workload provider'

systemctl --user stop omnitensor.service
"$OMNI_SERVICE/pip" uninstall -y \
  omnitensor-qwen-event-extraction \
  omnitensor-qwen-ask-selected-files \
  omnitensor-qwen-selected-text-tools \
  omnitensor-qwen-file-organizer

# Optional, only when nothing else depends on the shared provider runtime:
"$OMNI_SERVICE/pip" uninstall -y \
  omnitensor-qwen-vulkan-runtime llama-cpp-python
systemctl --user start omnitensor.service
```

The artifact store is retained on purpose: verified models are large and a
later reinstall can reuse the exact bytes. Remove the three artifact directories
only as a separate, deliberate data-retention decision after confirming that
no installed manifest references them. Likewise, remove PyMuPDF or the BGE
runtime packages only if no other workload needs them.
