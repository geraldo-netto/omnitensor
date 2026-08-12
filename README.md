# OmniTensor

Unified local inference service for heterogeneous accelerators. OmniTensor owns
device discovery, per-backend model execution, weighted workload scheduling,
runtime snapshot publishing, and the D-Bus control surface consumed by the
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
- owns the D-Bus name `org.cinnamon.OmniTensor1` and answers `ApplyCommand`
  with optimistic-concurrency revision checks, persisting policy state
  atomically.

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
pip install .[train,convert] # separate producer environment, GPU artifact
```

`systemd/omnitensor.service` runs the service as a user unit.

## Quality gates

Development requires unit, integration, regression, Hypothesis property/fuzz,
and changed-function mutation tests. Enforce the 80% floor on every function,
not only on aggregate or module totals:

```sh
.venv/bin/pytest --cov=omnitensor --cov-report=term
.venv/bin/coverage json -o /tmp/omnitensor-coverage.json
.venv/bin/python -m omnitensor.quality /tmp/omnitensor-coverage.json
```

Mutation runs must remain limited to changed callables. Use mutmut's quoted
glob selector for each changed callable, then gate those same exact selectors:

```sh
.venv/bin/mutmut run 'omnitensor.module.x_changed_callable__mutmut_*'
.venv/bin/mutmut results --all true > /tmp/omnitensor-mutmut.txt
.venv/bin/python -m omnitensor.mutation_quality /tmp/omnitensor-mutmut.txt \
  --selector omnitensor.module.x_changed_callable
```

`mutmut` cannot instrument properties, decorated dataclass hooks, or functions
with no mutation points. Keep their regression and per-function statement
coverage in the first gate; do not broaden mutation scope to compensate.
