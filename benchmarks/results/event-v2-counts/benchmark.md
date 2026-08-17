# Benchmark results

Accuracy is whole answers that passed every rule their case asked for —
deliberately strict, because a summary with an invented citation is worse
than no summary. The rules column is the partial-credit view; when the two
disagree, read the failures column, which says what actually broke.

Speed is measured on the same generations, at the task's own context and
token budget. Nothing was capped or shortened to make a number look better.

| workload | model | device | correct | accuracy | rules | s/case | chars/s | what failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| event-extraction | qwen3-5-9b-iq4-xs | AMD Radeon RX 6600 XT (RADV NAVI23) | 2/16 | 12% | 4% | 7.3 | 34 | answers-rather-than-refusing (14), at_least (14), event_starts (12), contains (8), event_needs (4) |
