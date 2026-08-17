# Benchmark results

Accuracy is whole answers that passed every rule their case asked for —
deliberately strict, because a summary with an invented citation is worse
than no summary. The rules column is the partial-credit view; when the two
disagree, read the failures column, which says what actually broke.

Speed is measured on the same generations, at the task's own context and
token budget. Nothing was capped or shortened to make a number look better.

| workload | model | device | correct | accuracy | rules | s/case | chars/s | what failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| event-extraction | qwen3-5-9b-iq4-xs | AMD Radeon RX 6600 XT (RADV NAVI23) | 8/16 | 50% | 61% | 19.3 | 39 | answers-rather-than-refusing (5), at_least (5), event_starts (5), contains (2), event_needs (2), produced-a-usable-document (1) |
