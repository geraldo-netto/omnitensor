# Benchmark results

Accuracy is whole answers that passed every rule their case asked for —
deliberately strict, because a summary with an invented citation is worse
than no summary. The rules column is the partial-credit view; when the two
disagree, read the failures column, which says what actually broke.

Speed is measured on the same generations, at the task's own context and
token budget. Nothing was capped or shortened to make a number look better.

| workload | model | device | correct | accuracy | rules | s/case | chars/s | what failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ask-selected-files | qwen3-8b-q4-k-m | AMD Radeon RX 6600 XT (RADV NAVI23) | 9/10 | 90% | 97% | 11.0 | 35 | no_invented_numbers (1) |
| event-extraction | qwen3-8b-q4-k-m | AMD Radeon RX 6600 XT (RADV NAVI23) | 3/10 | 30% | 14% | 5.5 | 52 | answers-rather-than-refusing (7), at_least (7), event_starts (5) |
| file-organizer | qwen3-8b-q4-k-m | AMD Radeon RX 6600 XT (RADV NAVI23) | 8/10 | 80% | 93% | 28.0 | 45 | contains (2) |
| selected-text-tools | qwen3-8b-q4-k-m | AMD Radeon RX 6600 XT (RADV NAVI23) | 9/10 | 90% | 97% | 13.3 | 36 | no_foreign_script (1) |
