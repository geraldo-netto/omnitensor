# TODO

## Findings

| id | status | severity | effort | related ids | description |
| --- | --- | --- | --- | --- | --- |
| OMNI-0026 | open | low | s | — | service: `_publisher` runs `publish`/`retract` (fsync of file and directory) on the event loop thread every 2s, stalling D-Bus `ApplyCommand` handling for the duration of each fsync. |
| OMNI-0028 | open | low | s | — | scheduler: `submit` has no queue bound — beyond 1,000,000 queued jobs `queueDepth` violates the snapshot schema maximum, `build_snapshot` raises, and the publisher crashes the whole service; a per-backend depth cap with a clean rejection is missing. |

## Rejected / Won't fix

| id | status | severity | effort | related ids | description |
| --- | --- | --- | --- | --- | --- |
