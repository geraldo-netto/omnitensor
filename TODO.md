# TODO

## Findings

| id | status | severity | effort | related ids | description |
| --- | --- | --- | --- | --- | --- |
| OMNI-0015 | open | low | s | — | executors: `VulkanGpuExecutor.run` enumerates Vulkan devices twice (`require_available` → `_select_device`, then again) — TOCTOU: the chosen index can shift between enumeration and `net.set_vulkan_device` on device-set change; the second enumeration does re-filter software devices, so the no-CPU rule holds at selection time. |
| OMNI-0017 | open | low | s | — | registry: `SCHEMA_DIR` is resolved at import time preferring `<pkg>/../../schemas` over the packaged copy — in an installed layout an unrelated `schemas` directory two levels up would shadow the canonical contracts; `@cache` on `load_schema`/`_validator` means any schema change requires a restart. |
| OMNI-0023 | open | low | s | — | quality gates: mutmut 3.7 generates no mutants for methods of decorated classes — `_BackendQueue` (`@dataclass`) has zero `xǁ_BackendQueueǁ*` keys in `mutants/src/omnitensor/scheduler.py.meta`, so the stride push/pop logic is invisible to the mutation gate (covered by unit + hypothesis tests instead). |
| OMNI-0021 | open | low | m | OMNI-0024 | executors: no model caching — TPU loads the delegate and builds an interpreter per inference, NPU recompiles the model per inference; latency and device churn under steady load. |
| OMNI-0024 | open | low | s | OMNI-0021 | service: `_rediscover` rebuilds every executor each pass even when the device set is unchanged, discarding lazily-initialized runtime state (openvino `Core` is reconstructed on the next availability call every 10s) and defeating any per-executor model cache. |
| OMNI-0025 | open | medium | s | — | executors: `CompositeGpuExecutor.availability` is format-blind — with only the Vulkan lane live, `pick_backend` still selects `gpu` for an `onnx`-model workload (union `model_formats` + any-lane availability), so jobs fail at run time instead of falling through to the next preference and the snapshot claims "Serving on gpu". |
| OMNI-0026 | open | low | s | — | service: `_publisher` runs `publish`/`retract` (fsync of file and directory) on the event loop thread every 2s, stalling D-Bus `ApplyCommand` handling for the duration of each fsync. |
| OMNI-0027 | open | low | xs | — | scheduler: `_retired` cancelled-worker tasks are only cleared in `stop()`; under backend churn via `update_executors` the list grows without bound — completed retired tasks should be pruned per update. |
| OMNI-0028 | open | low | s | — | scheduler: `submit` has no queue bound — beyond 1,000,000 queued jobs `queueDepth` violates the snapshot schema maximum, `build_snapshot` raises, and the publisher crashes the whole service; a per-backend depth cap with a clean rejection is missing. |

## Rejected / Won't fix

| id | status | severity | effort | related ids | description |
| --- | --- | --- | --- | --- | --- |
