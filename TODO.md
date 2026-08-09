# TODO

## Findings

| id | status | severity | effort | related ids | description |
| --- | --- | --- | --- | --- | --- |
| OMNI-0028 | open | low | s | — | scheduler: `submit` has no queue bound — beyond 1,000,000 queued jobs `queueDepth` violates the snapshot schema maximum, `build_snapshot` raises, and the publisher crashes the whole service; a per-backend depth cap with a clean rejection is missing. |

## Rejected / Won't fix

| id | status | severity | effort | related ids | description |
| --- | --- | --- | --- | --- | --- |
