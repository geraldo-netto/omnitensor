# Installing the Qwen workload providers

This is the operator procedure for the five local, manually triggered Qwen
workloads:

- `event-extraction` reads only files selected for that request and returns an
  evidence-backed event preview;
- `ask-selected-files` retrieves BGE spans from selected documents and answers
  with exact file, page, span, and digest citations;
- `selected-text-tools` explains, summarizes, rewrites, translates, or extracts
  tasks from one explicit clipboard selection;
- `document-translation` translates selected documents into one requested
  target language without writing the result back to disk; and
- `file-organizer` returns a review-only naming, tagging, folder, and exact
  duplicate plan. It cannot move, rename, overwrite, or delete a file.

The service does not bundle or silently download model weights. Each workload
manifest is disabled by default, and the Cinnamon action stays unavailable when
its distribution, artifact, permission, worker, or accelerator check fails.
Installing only the Python packages is therefore not a successful model
installation.

## Grounding trust boundary

Private fragment text is visible only inside the isolated worker. On the normal
generation path, the model chooses which opaque `sourceRef` supports each
citation or event; trusted runtime middleware never changes its semantic
fields. After that selection, middleware resolves only a reference supplied in
the current request and replaces metadata the model cannot reliably reproduce
with the store's canonical page, fragment span, and SHA-256 values. Ask also
excludes the private question fragment from the eligible citation set. Unknown,
question, malformed, or absent references remain unchanged so the downstream
canonical schema and grounding policy fail closed. The narrow deterministic
event exception is defined below. No raw fragment field or absolute path enters
the public result; an answer may still quote facts required by the user's
question.

For event extraction, a deterministic preflight adds a positive eligibility
hint only when one fragment contains a valid explicit calendar date, a clock
time, and an installed IANA timezone (including `UTC`). The hint carries no
event fields and creates no candidate; the model must still extract the event,
select its evidence reference, and satisfy the unchanged schema, timezone,
pending-confirmation, and evidence checks. If the model initially refuses such
qualified input, the runtime gives it one field-free reconsideration turn;
if that also refuses, trusted middleware can produce one pending candidate only
for the single fully matched English statement form documented by the frozen
fixture. It rejects source commands, invalid dates/zones, ambiguous local times,
backward ranges, partial matches, and multiple matches. Ambiguous or incomplete
text gets neither fallback and retains the existing refusal path.

The 13 August 2026 installed-provider recheck used runtime 0.2.0 through the
control API of the day (then session D-Bus, since replaced by the control
socket) on the qualified RX 6600 XT. The frozen English event fixture
completed in 16.7 seconds with one pending candidate, exact title, start, end,
timezone, location, and canonical full-fragment evidence hashes. Its public job
result contained no raw fragment field or absolute source path.

## Package and model layout

The provider is seven wheels rather than one wheel with five entry points. An
OmniTensor plugin distribution may own exactly one identity and exactly one
`omnitensor-plugin.json`, so each workload needs an identity-isolated wheel.
Five of them depend on the generation wheel; `ask-selected-files` also needs
the embedder wheel, which is separate because BGE on ncnn is a different
engine and no workload that only generates should install it:

| Wheel | Responsibility | Pinned models |
| --- | --- | --- |
| `omnitensor-vulkan-runtime` | In-process llama.cpp/Vulkan generation, the shared GPU lease, and four workload factories | runtime only |
| `omnitensor-ncnn-embeddings` | BGE/ncnn retrieval on the same GPU lease | runtime only |
| `omnitensor-qwen-event-extraction` | `event-extraction` entry point and manifest | Qwen3.5-9B IQ4_XS (default) and Qwen3-8B Q4_K_M |
| `omnitensor-qwen-ask-selected-files` | `ask-selected-files` entry point, manifest, and the workload it assembles from the two runtime wheels | Qwen3.5-9B IQ4_XS (default) and Qwen3-8B Q4_K_M, plus BGE-small-en-v1.5 |
| `omnitensor-qwen-selected-text-tools` | `selected-text-tools` entry point and manifest | Qwen3.5-9B IQ4_XS (default) and Qwen3-8B Q4_K_M, plus operation-specific DictaLM2 Hebrew |
| `omnitensor-qwen-file-organizer` | `file-organizer` entry point and manifest | Qwen3.5-9B IQ4_XS (default) and Qwen3-8B Q4_K_M |
| `omnitensor-qwen-document-translation` | `document-translation` entry point and manifest | Qwen3.5-9B IQ4_XS (default) and Qwen3-8B Q4_K_M, plus DictaLM2 for a Hebrew target |

Each manifest declares more than one pinned generation artifact, and a person
may choose any declared, installed model per workload (see
[Choosing the model a workload runs](choosing-a-model.md)). The default for
all five text workloads is `qwen3-5-9b-iq4-xs`, recorded in the runtime
wheel's `qualification.json` with SHA-256
`7e918aeca06c52bcb528ea6b04b4ec957e75ee8c0a73138854c0dfcf371ea429`; the
manifests declare its origin as the `unsloth/Qwen3.5-9B-GGUF`
`Qwen3.5-9B-IQ4_XS.gguf` upload at upstream revision
`3885219b6810b007914f3a7950a8d1b469d598a5`. It is Apache-2.0,
5,168,653,536 bytes, and the checked-in
[Qwen3.5-9B catalog](../generation-models/qwen3-5-9b.json) binds those bytes,
their provenance, and the qualified workload scope. Review the pinned
[Qwen3.5-9B Apache-2.0 terms](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/blob/3885219b6810b007914f3a7950a8d1b469d598a5/README.md)
before downloading it.

The shared Qwen3-8B artifact is the official
`Qwen/Qwen3-8B-GGUF` `Qwen3-8B-Q4_K_M.gguf` at upstream revision
`7c41481f57cb95916b40956ab2f0b139b296d974`. It is Apache-2.0,
5,027,783,488 bytes, and has SHA-256
`d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
The checked-in [generation catalog](../generation-models/qwen3-8b.json)
binds those bytes to their license, provider, and deliberately narrow
qualification scope. Review the pinned
[Qwen3-8B Apache-2.0 license](https://huggingface.co/Qwen/Qwen3-8B-GGUF/blob/7c41481f57cb95916b40956ab2f0b139b296d974/LICENSE)
before downloading it. Sharing immutable artifacts removes duplicate model
selection and installation paths; every workload retains an independent worker,
task contract, grant set, and acceptance status. No checkout, package install,
or catalog entry is by itself a live readiness or quality claim;
final provider qualification and `DescribePlugins` must still pass.

Selected text additionally binds the official
`dicta-il/dictalm2.0-instruct-GGUF` Q4_K_M artifact at revision
`9ed3346a1440643825287abf86e28800005ec9b3`. It is Apache-2.0,
4,374,991,808 bytes, and has SHA-256
`dc53cc29a30444677a7760af31f807a61999477c526cdbe6552803259a94c735`.
The [DictaLM catalog](../generation-models/dictalm2-hebrew.json) binds those
bytes and provenance. It is not a global language model: the selected-text
worker invokes it only for an explicit `translate` request whose target is
Hebrew. Qwen remains primary for English and every other route.

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
| All seven provider wheels | The runtime, embedder, and five independently discoverable workload identities |
| Both pinned Qwen generation GGUFs (the Qwen3.5-9B IQ4_XS default and Qwen3-8B Q4_K_M alternative) | Digest-locked generation artifacts declared by all five workload providers |
| The pinned DictaLM2.0 7B Instruct Q4_K_M GGUF | Explicit Hebrew translation for selected-text tools only |
| `ncnn>=1.0.20260526`, `numpy>=1.24`, `tokenizers>=0.22`, and the pinned BGE files | Retrieval and tokenization for Ask selected files |
| A user runtime directory (`$XDG_RUNTIME_DIR`) | The OmniTensor control socket and job-result surface |

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
| Cinnamon XPU WLM | Desktop discovery and actions; control-socket clients can use the workloads without the applet |
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
- `omnitensor-training[document-producers]` (PyTorch, Transformers, safetensors, ONNX,
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
architecture as the service for the six Python provider wheels. Replace
`/absolute/path/to/omnitensor` with this checkout:

```sh
python3 -m venv ~/.local/share/omnitensor-provider-builder/venv
OMNI_BUILDER=~/.local/share/omnitensor-provider-builder/venv/bin
OMNI_SOURCE=/absolute/path/to/omnitensor
OMNI_WHEELS=~/.local/share/omnitensor-provider-wheels
mkdir -p "$OMNI_WHEELS"

"$OMNI_BUILDER/pip" install --upgrade pip build hatchling wheel

for provider in vulkan-runtime ncnn-embeddings event-extraction ask-selected-files \
  selected-text-tools file-organizer document-translation; do
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
  "$OMNI_WHEELS"/omnitensor_vulkan_runtime-0.2.0-*.whl \
  "$OMNI_WHEELS"/omnitensor_ncnn_embeddings-0.2.0-*.whl \
  "$OMNI_WHEELS"/omnitensor_qwen_event_extraction-0.2.0-*.whl \
  "$OMNI_WHEELS"/omnitensor_qwen_ask_selected_files-0.2.0-*.whl \
  "$OMNI_WHEELS"/omnitensor_qwen_selected_text_tools-0.2.0-*.whl \
  "$OMNI_WHEELS"/omnitensor_qwen_file_organizer-0.2.0-*.whl \
  "$OMNI_WHEELS"/omnitensor_qwen_document_translation-0.2.0-*.whl
```

Reinstalling the service alone is not an upgrade. The provider wheels import
module-level names from the service package, so the two must come from the same
tree: rebuild the wheels above and repeat the whole `--no-deps` provider install
in the same step whenever `omnitensor` itself is reinstalled. Skipping it leaves
a worker that dies at entry-point load, which reaches the applet as `worker
handshake rejected: truncated-frame`; `omnitensor-verify-install` reports the
same skew as a failed `provider-imports` check, naming the distribution to
reinstall.

Install optional document/media parsing only when needed:

```sh
"$OMNI_SERVICE/pip" install 'pymupdf>=1.24'
```

## 3. Acquire and verify the artifacts

Downloading is an explicit operator action. Review the Apache-2.0 terms linked
in the generation catalogs before running:

```sh
OMNI_MODELS=~/.local/share/omnitensor/provider-sources
mkdir -p "$OMNI_MODELS/qwen3-5-9b"
mkdir -p "$OMNI_MODELS/qwen3-8b"
mkdir -p "$OMNI_MODELS/dictalm2-hebrew"

curl --fail --location \
  'https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-IQ4_XS.gguf' \
  --output "$OMNI_MODELS/qwen3-5-9b/Qwen3.5-9B-IQ4_XS.gguf"

printf '%s  %s\n' \
  '7e918aeca06c52bcb528ea6b04b4ec957e75ee8c0a73138854c0dfcf371ea429' \
  "$OMNI_MODELS/qwen3-5-9b/Qwen3.5-9B-IQ4_XS.gguf" | \
  sha256sum --check --strict
test "$(stat --format=%s "$OMNI_MODELS/qwen3-5-9b/Qwen3.5-9B-IQ4_XS.gguf")" \
  -eq 5168653536

curl --fail --location \
  'https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q4_K_M.gguf' \
  --output "$OMNI_MODELS/qwen3-8b/Qwen3-8B-Q4_K_M.gguf"

printf '%s  %s\n' \
  'd98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785' \
  "$OMNI_MODELS/qwen3-8b/Qwen3-8B-Q4_K_M.gguf" | sha256sum --check --strict
test "$(stat --format=%s "$OMNI_MODELS/qwen3-8b/Qwen3-8B-Q4_K_M.gguf")" \
  -eq 5027783488

curl --fail --location \
  'https://huggingface.co/dicta-il/dictalm2.0-instruct-GGUF/resolve/9ed3346a1440643825287abf86e28800005ec9b3/dictalm2.0-instruct-Q4_K_M.gguf' \
  --output "$OMNI_MODELS/dictalm2-hebrew/dictalm2.0-instruct-Q4_K_M.gguf"

printf '%s  %s\n' \
  'dc53cc29a30444677a7760af31f807a61999477c526cdbe6552803259a94c735' \
  "$OMNI_MODELS/dictalm2-hebrew/dictalm2.0-instruct-Q4_K_M.gguf" | \
  sha256sum --check --strict
test "$(stat --format=%s "$OMNI_MODELS/dictalm2-hebrew/dictalm2.0-instruct-Q4_K_M.gguf")" \
  -eq 4374991808
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

The one-shot installer uses one `Apache-2.0` acknowledgement for both Qwen
artifacts and requires both license identifiers literally. It verifies all five
source files, including both Qwen byte counts and every declared digest, before
importing immutable versions into the artifact store:

```sh
OMNI_ARTIFACTS=~/.local/share/omnitensor/artifacts

"$OMNI_SERVICE/omnitensor-install-generation-artifacts" \
  --artifact-root "$OMNI_ARTIFACTS" \
  --default-qwen-model "$OMNI_MODELS/qwen3-5-9b/Qwen3.5-9B-IQ4_XS.gguf" \
  --qwen-model "$OMNI_MODELS/qwen3-8b/Qwen3-8B-Q4_K_M.gguf" \
  --bge-param "$BGE_PARAM" \
  --bge-bin "$BGE_BIN" \
  --bge-tokenizer "$BGE_TOKENIZER" \
  --accept-model-license Apache-2.0 \
  --accept-bge-license MIT

"$OMNI_SERVICE/omnitensor-install-hebrew-translation-model" \
  --artifact-root "$OMNI_ARTIFACTS" \
  --model "$OMNI_MODELS/dictalm2-hebrew/dictalm2.0-instruct-Q4_K_M.gguf" \
  --accept-license Apache-2.0
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
starts only when llama.cpp proves complete Vulkan layer offload on the
selected GPU — that is the live hardware gate. The qualification receipt
records the device the measurements were taken on; a different card is
reported against that record rather than refused, and the receipt does not
distinguish two same-name cards by serial number or PCI address.

## 4. Grant only the declared access

Package installation does not imply consent. All five workers require the GPU
grant. The four selected-file workflows additionally require access to only
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

"$OMNI_SERVICE/omnitensor-grant" grant document-translation accelerator:gpu \
  --reason 'run the explicitly requested model on my GPU'
"$OMNI_SERVICE/omnitensor-grant" grant document-translation files:read-selected \
  --reason 'translate only documents I explicitly select'
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

python3 -c 'import asyncio, json; from omnitensor.socket_transport import call_control; \
  print(json.dumps(asyncio.run(call_control("describe-plugins", {})), indent=2))'

journalctl --user --unit omnitensor.service --since '-5 minutes' \
  --no-pager
```

`systemctl is-active` becomes true before model-backed workers finish their
bounded startup. `omnitensor-verify-install` therefore retries transient control-socket
startup errors for up to 30 seconds; a persistent transport or contract error
still fails the installation.
The 12 August 2026 live recheck started the verifier immediately after service
restart and passed every check after 12 seconds.

For each of the four IDs, `describe-plugins` must report:

- `source: "external"`, the expected distribution, and `workerState: "ready"`;
- negotiated `execute`, `cancel`, `health`, and `progress` capabilities;
- every declared artifact with `ready: true`; and
- every permission with `granted: true`.

Startup loads the selected model in the worker and accepts it only when
llama.cpp reports all model layers offloaded to Vulkan. Ask selected files
also preflights the exact BGE graph, tokenizer, and finite 384-value output.
A CPU build, partial layer offload, missing tokenizer, changed model, absent
grant, sandbox failure, or startup timeout leaves the worker
failed/unavailable. Do not enable a failed workload by weakening these
checks.

## Acceptance and what is not claimed

Ask selected files has a frozen six-case CC0 corpus and quantitative retrieval,
grounding, citation, latency, cancellation, and grant-revocation gates. Peak
GPU memory is measured and published but is not an eligibility limit.
The archived full-corpus report covers the pinned Qwen/BGE bytes and the native
runtime named inside that report; it must not be transferred to different
llama.cpp bytes. Verify a newly collected evidence document with:

```sh
"$OMNI_SERVICE/omnitensor-qualify-document-questions" \
  --evidence /absolute/path/to/document-question-evidence.json \
  --output /absolute/path/to/document-question-acceptance.json
```

Event extraction and file organizer currently have
closed schemas, privacy/safety regression coverage, worker startup checks, and
representative integration tests, but no declared model-quality acceptance
metrics. Their manifest `acceptance` arrays are intentionally empty.

Document translation ships before its GPU run. Its manifest declares the
acceptance metrics it will be measured against, and until that run happens the
receipt records both Qwen pairs as `unmeasured`: the workload starts, it
translates, and every answer it returns says the pair was not measured. It is
disabled by default for that reason. Do not read the shipped receipt as a
quality claim for this workload, and do not edit it into one.

Selected text has a separate frozen 16-case operational gate. The qualified
RX 6600 XT run uses Qwen for English/default, Italian, and every non-Hebrew
route, and DictaLM only for explicit Hebrew translation. It passed operation
and task term recall, target-script, injection, route, latency, cancellation,
privacy, full-offload, and no-CPU-fallback gates. Verify newly collected
evidence with:

```sh
OMNI_STATE=~/.local/state/xpu-workload-manager
OMNI_SELECTED_LOAD="$OMNI_STATE/plugin-state/selected-text-tools/selected-text-worker-load.json"

"$OMNI_SERVICE/python" "$OMNI_SOURCE/scripts/collect-selected-text-acceptance.py" \
  --worker-load-receipt "$OMNI_SELECTED_LOAD" \
  --qwen-model "$OMNI_MODELS/qwen3-8b/Qwen3-8B-Q4_K_M.gguf" \
  --hebrew-model "$OMNI_MODELS/dictalm2-hebrew/dictalm2.0-instruct-Q4_K_M.gguf" \
  --output /absolute/path/to/selected-text-evidence.json
```

The private receipt is usable only while `describe-plugins` reports the
current selected-text worker ready. The worker writes it after both actual
model loads: it records the artifact digests, runtime version, physical
device, and layer counts that actually loaded rather than enforcing a frozen
identity — complete Vulkan layer offload with no CPU fallback is the only
live load gate. Startup clears stale bytes; startup failure and normal
worker stop remove them. If `OMNITENSOR_STATE_PATH` is customized, locate
`plugin-state` beside that snapshot instead of using the default path above.
The collector refuses a missing, stale, malformed, or disagreeing receipt
before it runs the corpus or writes evidence. It then copies the measured model
records into the unchanged evidence contract.

Qualify the resulting document with:

```sh
"$OMNI_SERVICE/omnitensor-qualify-selected-text" \
  --evidence /absolute/path/to/selected-text-evidence.json \
  --output /absolute/path/to/selected-text-acceptance.json
```

That qualification is digest-bound to both model artifacts, runtime 0.3.34,
the frozen corpus, and the named GPU. It is not an arbitrary-text benchmark and
does not authorize silently inferred or globally fixed language routing. This
code change does not recollect or rewrite the archived run: a fresh evidence
artifact requires reinstalling the matching wheels and observing the current
worker on the named hardware.

The runtime wheel contains the version 2 `qualification.json`. It records the
`llama-cpp-python` version and native-library hashes, the Vulkan-reported
device the measurements were taken on, and — per workload, per model — the
task-contract digest and pass/fail result, with a default model for each
workload. Startup still verifies the native llama.cpp bytes against it and
refuses a mismatched runtime; the per-workload model coverage is *reported*
rather than enforced, so a person may run any model the manifest declares and
the client can show whether that pair was measured. The receipt does not
replace the Ask full-corpus report, uniquely identify a same-name physical
card, or create undeclared quality metrics for unmeasured pairs.

## Runtime privacy, isolation, and cancellation

- Source text, selections, prompts, citations, model replies, and absolute
  paths never enter the public runtime snapshot. Job results remain private to
  the submitting owner (the connection's peer uid).
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

## Future GPU models and NPU providers

The four current manifests declare the pinned Qwen3.5-9B IQ4_XS default and
the Qwen3-8B Q4_K_M alternative; selected text additionally selects its
dedicated Hebrew artifact only at the explicit operation boundary. A model
the manifests do not declare must arrive as a new provider release rather
than a replacement under the existing identity. Such a release must:

1. pin the upstream revision, filename, byte count, SHA-256, quantization, SPDX
   license, and attribution in a reviewed catalog and manifest;
2. give the artifact a distinct ID/version and retain finite byte-count and
   digest verification;
3. update only the thin workload wheels selecting that model, leaving unrelated
   workloads on their accepted pin;
4. prove complete layer offload, its context window, peak GPU/system memory,
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
  /absolute/path/to/previous/omnitensor_vulkan_runtime-*.whl \
  /absolute/path/to/previous/omnitensor_ncnn_embeddings-*.whl \
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
"$OMNI_SERVICE/omnitensor-grant" revoke document-translation accelerator:gpu \
  --reason 'uninstalling the Qwen workload provider'
"$OMNI_SERVICE/omnitensor-grant" revoke document-translation files:read-selected \
  --reason 'uninstalling the Qwen workload provider'

systemctl --user stop omnitensor.service
"$OMNI_SERVICE/pip" uninstall -y \
  omnitensor-qwen-event-extraction \
  omnitensor-qwen-ask-selected-files \
  omnitensor-qwen-selected-text-tools \
  omnitensor-qwen-file-organizer \
  omnitensor-qwen-document-translation

# Optional, only when nothing else depends on the shared provider runtime:
"$OMNI_SERVICE/pip" uninstall -y \
  omnitensor-vulkan-runtime omnitensor-ncnn-embeddings llama-cpp-python
systemctl --user start omnitensor.service
```

The artifact store is retained on purpose: verified models are large and a
later reinstall can reuse the exact bytes. Remove the three artifact directories
only as a separate, deliberate data-retention decision after confirming that
no installed manifest references them. Likewise, remove PyMuPDF or the BGE
runtime packages only if no other workload needs them.
