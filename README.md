# OmniTensor

Unified local inference service for heterogeneous accelerators. OmniTensor owns
device discovery, per-backend model execution, weighted workload scheduling,
runtime snapshot publishing, and the D-Bus control surface consumed by the
[cinnamon-tpuwlm](../cinnamon/cinnamon-tpuwlm) panel applet.

## Backend hierarchy

`tpu > npu > gpu` — no CPU backend by design: the CPU is the scarcest shared
resource on the host and OmniTensor never schedules inference onto it. When no
accelerator is available a workload reports `unavailable` with a reason instead
of falling back.

| Backend | Discovery | Runtime |
| --- | --- | --- |
| `tpu` | Coral PCIe `/dev/apex_*`, Coral USB `18d1:9302` | `tflite-runtime` + `libedgetpu.so.1` delegate |
| `npu` | `/dev/accel/accel*` (kernel accel subsystem), sysfs vendor id | OpenVINO (`NPU` device) |
| `gpu` | `/dev/dri/renderD*`, sysfs vendor id | ONNX Runtime (CUDA / ROCm execution providers) |

A missing runtime library or execution provider marks the backend unavailable
with an explicit reason; it never crashes the service or blocks snapshot
publishing.

## Contract

The JSON Schemas under [schemas/](schemas/) are canonical; the applet ships
mirror copies. OmniTensor:

- writes the runtime snapshot atomically (temp file + rename) to the
  applet-configured `runtime-state-path`, validated against
  `runtime-snapshot.schema.json` before every write;
- owns the D-Bus name `org.cinnamon.OmniTensor1` and answers `ApplyCommand`
  with optimistic-concurrency revision checks, persisting policy state
  atomically.

## Routing

Each workload manifest declares `requirements.accelerator` (the backend it was
designed for) and an optional ordered `acceleratorPreference` list. The router
dispatches to the first preferred backend whose executor is available and whose
model format is compatible; otherwise the profile reports `unavailable`.

## Install

```sh
pip install .[dev]          # development
pip install .[tpu,gpu,npu]  # runtime extras per available hardware
```

`systemd/omnitensor.service` runs the service as a user unit.
