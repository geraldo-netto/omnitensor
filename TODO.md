# TODO

## Findings

| id | status | severity | effort | related ids | description |
| --- | --- | --- | --- | --- | --- |
| OMNI-0002 | open | medium | xs | — | state: `PolicyStore.load` accepts JSON `true`/`false` as `revision` (`bool` passes `isinstance(int)`); every later acknowledgement then violates the contract (`revision: True is not of type 'integer'`) and raises. |
| OMNI-0004 | open | low | xs | — | snapshot: `build_snapshot` raises `TypeError`, not `ValueError`, for non-numeric metrics (`int(None)`, `int(complex)`), breaking its documented "schema-valid or ValueError" contract. |
| OMNI-0005 | open | low | xs | — | registry: `load_workloads` raises raw `json.JSONDecodeError` for malformed manifest JSON instead of `ManifestError`, unlike every other manifest failure path. |
| OMNI-0006 | open | low | xs | — | discovery: `device_utilization` parses sysfs content `nan`/`inf` as load `100.0` (clamp of NaN returns the bound) instead of `None`. |
| OMNI-0007 | open | high | m | OMNI-0013 | scheduler: `stop()` abandons queued jobs — their futures never resolve so awaiting callers hang; the in-flight job's `CancelledError` is converted to `RuntimeError('')` on its future and the cancellation is swallowed (verified: f2/f3 pending after stop, f1 gets `RuntimeError('')`). |
| OMNI-0008 | open | medium | m | — | scheduler: stride newcomer burst — a profile first submitting after others advanced their pass starts at pass 0 and monopolizes the device until it catches up (verified: `BBBBBBBBBBAAAAAAAAAA` with equal weights); `passes`/`order`/`profiles` entries are also never pruned for departed profiles. |
| OMNI-0009 | open | high | m | OMNI-0010 | service: `_rediscover` replaces `self._executors`/`self._devices` but the `Scheduler` keeps the dict it was constructed with — dispatch uses stale executors forever, and `profile_statuses` (new executors) disagrees with `scheduler.stats()` (old queues). |
| OMNI-0010 | open | high | m | OMNI-0009 | policy has no runtime effect: `paused` and per-profile `enabled` are persisted and acknowledged but scheduler dispatch and `profile_statuses` never consult them; the snapshot never reports the `paused` status the schema defines. |
| OMNI-0011 | open | medium | s | — | service: `_publisher` skips publishing when no devices are detected (snapshot schema requires ≥1 device), so the last snapshot file goes silently stale after all accelerators vanish — the applet has no staleness/absence signal from the service. |
| OMNI-0012 | open | medium | s | — | service: an exception escaping `_publisher` (e.g. `write_snapshot` OSError) propagates out of `asyncio.gather`, leaving the sibling `_rediscover` task running while `finally` stops the scheduler and disconnects the bus (orphan pending task at shutdown). |
| OMNI-0013 | open | low | s | OMNI-0007 | scheduler: `_running` is a set of workload ids — the same profile running on two backends concurrently is counted once and discarded when the first job finishes, undercounting `runningProfiles`. |
| OMNI-0014 | open | low | s | — | executors: `NpuExecutor._ensure_core` lazy init races between `availability()` (event loop thread) and `run()` (worker thread via `asyncio.to_thread`), possibly constructing two `openvino.Core` objects; `TpuExecutor._delegate_error` is likewise written in the worker thread and read from the loop thread. |
| OMNI-0015 | open | low | s | — | executors: `VulkanGpuExecutor.run` enumerates Vulkan devices twice (`require_available` → `_select_device`, then again) — TOCTOU: the chosen index can shift between enumeration and `net.set_vulkan_device` on device-set change; the second enumeration does re-filter software devices, so the no-CPU rule holds at selection time. |
| OMNI-0016 | open | low | xs | — | durability: `write_snapshot` and `PolicyStore.save` fsync the temp file but not the parent directory after `os.replace`; the rename itself can be lost on power failure (matters for `policy.json` revision durability). |
| OMNI-0017 | open | low | s | — | registry: `SCHEMA_DIR` is resolved at import time preferring `<pkg>/../../schemas` over the packaged copy — in an installed layout an unrelated `schemas` directory two levels up would shadow the canonical contracts; `@cache` on `load_schema`/`_validator` means any schema change requires a restart. |
| OMNI-0018 | open | low | xs | — | systemd: unit is `WantedBy=default.target` but only ordered `After=graphical-session.target`; in a non-graphical session the session bus may be absent, producing a restart loop. |
| OMNI-0019 | open | medium | s | OMNI-0010 | service: `profile_statuses` hardcodes `"queued": 0` and derives `running`/`watching` from the global `runningProfiles` count rather than per-profile queue depth/running state — the snapshot misleads the applet under load. |
| OMNI-0020 | open | medium | s | — | service: `bus.request_name` reply is not checked; a second instance keeps running without owning `org.cinnamon.OmniTensor1` while still publishing to the same snapshot path (two writers). |
| OMNI-0021 | open | low | m | — | executors: no model caching — TPU loads the delegate and builds an interpreter per inference, NPU recompiles the model per inference; latency and device churn under steady load. |

## Rejected / Won't fix

| id | status | severity | effort | related ids | description |
| --- | --- | --- | --- | --- | --- |
