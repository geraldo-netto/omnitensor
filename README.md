# OmniTensor

Unified local inference service for heterogeneous accelerators. OmniTensor owns
device discovery, per-backend model execution, weighted workload scheduling,
runtime snapshot publishing, and the control-socket surface consumed by the
[cinnamon-xpuwlm](../cinnamon-xpuwlm) panel applet.

See [Cinnamon workload use cases](docs/use-cases.md) for the nine shared
profiles, delivered runtime behavior, model and host-pipeline boundaries, and
acceptance gates.

Installing? [docs/installation.md](docs/installation.md) has the service and
applet sequence and the `omnitensor-verify-install` gate that proves the result
is actually serving.

Installing the four local Qwen workflows? The
[Qwen workload provider guide](docs/qwen-workload-installation.md) covers the
five-wheel layout, pinned Qwen/BGE artifacts, mandatory and optional
dependencies, grants, readiness, rollback, and removal without a CPU fallback.

Building a client, or wondering which side a feature belongs on? [What this service owns, and what a client owns](docs/runtime-client-boundary.md) splits capability from intent and says why this service must stay correct with no client attached.

Writing a plugin? Start with the [extension guide](docs/extension-guide.md),
which walks packaging, discovery, schemas, permissions, artifacts, local
testing, install, upgrade, rollback, and the compatibility policy in order.

Adding a structured text or vision model? The
[generation provider contract](docs/generation-providers.md) defines versioned
tasks, private-content isolation, structured output, accelerator routing, and
the killable worker boundary without coupling workloads to one model runtime.

Importing calendar events from explicitly selected local files? The
[event extraction guide](docs/event-extraction.md) covers optional PDF/OCR
dependencies, private grounded previews, model verification, and confirmed
no-overwrite ICS export.

Asking questions over explicitly chosen documents? [Ask selected files](docs/document-questions.md)
documents the ephemeral BGE span index, grounded Qwen answer contract, exact
file/page/span citations, and GPU-first provider boundary.

Using an explicit clipboard selection? [Selected-text tools](docs/selected-text-tools.md)
documents the one-shot explain, summarize, rewrite, translate, and task-extraction
contract, its private fragment boundary, and its mandatory provider dependencies.

Organizing explicitly selected files? [File organizer](docs/file-organizer.md)
documents its evidence-bound review plan and strict no-action boundary.

Training a host-specific model? [Local model training](docs/local-training.md)
documents bounded numeric recording, reproducible portable ONNX fitting,
GPU/NPU compilation, immutable installation, restricted model bindings, and
mandatory versus optional producer dependencies.

Tuning how many plugin jobs run at once, or wondering what a profile's weight
does for a plugin workload? [Plugin admission](docs/plugin-admission.md)
documents the weighted pool in front of plugin workers and
`OMNITENSOR_PLUGIN_SLOTS`.

Publishing a qualified local artifact? The
[publisher identity section](docs/local-training.md#machine-local-artifact-publisher-identity)
explains why signatures are required and how the idempotent `xpuwlm` Ed25519
key is created, reused, protected, and exposed as public trust metadata.

## Backend hierarchy

`tpu > npu > gpu` — no CPU backend by design: the CPU is the scarcest shared
resource on the host and OmniTensor never schedules inference onto it. When no
accelerator is available a workload reports `unavailable` with a reason instead
of falling back.

| Backend | Discovery | Runtime |
| --- | --- | --- |
| `tpu` | Coral PCIe `/dev/apex_*`, Coral USB `18d1:9302` | `tflite-runtime` + `libedgetpu.so.1` delegate |
| `npu` | `/dev/accel/accel*` (kernel accel subsystem), sysfs vendor id | OpenVINO (`NPU` device) |
| `gpu` | `/dev/dri/renderD*`, sysfs vendor id | ncnn Vulkan compute (primary — any Mesa/RADV/ANV/NVIDIA Vulkan driver); ONNX Runtime CUDA/ROCm as optional second lane |

A missing runtime library or execution provider marks the backend unavailable
with an explicit reason; it never crashes the service or blocks snapshot
publishing. The GPU backend is a composite: Vulkan via ncnn is tried first
(model format `ncnn`, `.param` + `.bin` files) so no CUDA/ROCm stack is
required, with ONNX Runtime (`onnx` models) as an optional second runtime.
Software Vulkan devices (llvmpipe) are never selected — the no-CPU rule
applies inside Vulkan device enumeration too.

## Contract

The JSON Schemas under [schemas/](schemas/) are canonical; the applet ships
mirror copies. OmniTensor:

- writes the runtime snapshot atomically (temp file + rename) to the
  applet's default `~/.local/state/xpu-workload-manager/state.json` path
  (overridable with `OMNITENSOR_STATE_PATH`), validated against
  `runtime-snapshot.schema.json` before every write;
- publishes bounded, independently versioned plugin telemetry as an optional
  runtime-snapshot extension described in
  [docs/runtime-snapshot.md](docs/runtime-snapshot.md);
- serves the control socket at `$XDG_RUNTIME_DIR/omnitensor/control.sock` and
  answers `apply-command` with optimistic-concurrency revision checks,
  persisting policy state atomically.

## Routing

Each workload manifest declares `requirements.accelerator` (the backend it was
designed for) and an optional ordered `acceleratorPreference` list whose first
entry must be that same accelerator; the rest are fallbacks in order. The router
dispatches to the first preferred backend whose executor is available and whose
model format is compatible; otherwise the profile reports `unavailable`.

## Install

```sh
pip install .[dev]          # development
pip install .[tpu,gpu,npu]  # runtime extras per available hardware
pip install .[convert] 'packaging/omnitensor-training[train]' # separate producer environment
```

`systemd/omnitensor.service` runs the service as a user unit.

## Quality gates

Development requires unit, integration, regression, Hypothesis property/fuzz,
and changed-function mutation tests. Keep every changed module at or above 80%
statement coverage; CI also rejects project coverage below that floor:

```sh
.venv/bin/pytest --cov=omnitensor --cov-report=term --cov-fail-under=80
```

Mutation runs use the tracked, source-validated exact selector manifest. The
manifest is split into deterministic CI shards; its loader rejects stale,
duplicate, unsorted, wildcard, or unknown callables before mutmut starts. List
the shards, then reproduce any one locally with the same runner CI uses:

```sh
.venv/bin/python -m omnitensor.mutation_campaign mutation-selectors.json \
  --list-shards
.venv/bin/python -m omnitensor.mutation_campaign mutation-selectors.json \
  --shard scheduler --report /tmp/omnitensor-mutmut-scheduler.txt
```

The manifest currently records a partial scope blocked on OMNI-0297. Mutmut's
`only_mutate` list is the same closed set of modules named by the selector
shards, so an interim corpus cannot generate ungated mutants. Expand training,
Qwen, and document-acceptance callables in the manifest first when OMNI-0297
resumes; regenerate `mutants/` only after that reviewed expansion or when the
reorganized mutation campaign is re-enabled.

The post-split operational-acceptance shards use the same fail-closed runner;
each shard contains every mutation-bearing callable in its named owner modules:

```sh
.venv/bin/python -m omnitensor.mutation_campaign mutation-selectors.json --shard acceptance --report /tmp/omnitensor-mutmut-acceptance.txt
.venv/bin/python -m omnitensor.mutation_campaign mutation-selectors.json --shard document-binding --report /tmp/omnitensor-mutmut-document-binding.txt
.venv/bin/python -m omnitensor.mutation_campaign mutation-selectors.json --shard event-workload --report /tmp/omnitensor-mutmut-event-workload.txt
.venv/bin/python -m omnitensor.mutation_campaign mutation-selectors.json --shard grounded-answer --report /tmp/omnitensor-mutmut-grounded-answer.txt
```

The runner passes every `__mutmut_*` pattern directly as an argument, never
through shell expansion, and gates the resulting report at 80% per callable.
Its `omnitensor-mutmut` wrapper disables only mutmut's string-literal operator,
so prose case and wording changes do not manufacture brittle exact-text tests.
The development extra pins mutmut 3.7.0 because selector mangling and result
text are part of this checked contract; upgrade that pin together with the
manifest and parser tests.
To gate an already-produced report without running mutations again, use the
same manifest and shard:

```sh
.venv/bin/python -m omnitensor.mutation_quality \
  /tmp/omnitensor-mutmut-scheduler.txt \
  --selector-file mutation-selectors.json --shard scheduler
```

`mutmut` cannot instrument properties, decorated dataclass hooks, or functions
with no mutation points. Keep their regression and per-function statement
coverage in the first gate; do not broaden mutation scope to compensate.
