# Benchmark results

Accuracy is whole answers that passed every rule their case asked for —
deliberately strict, because a summary with an invented citation is worse
than no summary. The rules column is the partial-credit view; when the two
disagree, read the failures column, which says what actually broke.

Speed is measured on the same generations, at the task's own context and
token budget. Nothing was capped or shortened to make a number look better.

| workload | model | device | correct | accuracy | rules | s/case | chars/s | what failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ask-selected-files | qwen3-4b-q4-k-m | AMD Radeon 610M (RADV RAPHAEL_MENDOCINO) | 9/10 | 90% | 97% | 55.7 | 7 | no_invented_numbers (1) |
| ask-selected-files | qwen3-8b-q4-k-m | AMD Radeon 610M (RADV RAPHAEL_MENDOCINO) | 9/10 | 90% | 97% | 82.7 | 5 | no_invented_numbers (1) |
| event-extraction | qwen3-4b-q4-k-m | AMD Radeon 610M (RADV RAPHAEL_MENDOCINO) | 3/10 | 30% | 14% | 23.2 | 11 | answers-rather-than-refusing (7), at_least (7), event_starts (5) |
| event-extraction | qwen3-8b-q4-k-m | AMD Radeon 610M (RADV RAPHAEL_MENDOCINO) | 3/10 | 30% | 14% | 36.6 | 8 | answers-rather-than-refusing (7), at_least (7), event_starts (5) |
| file-organizer | qwen3-4b-q4-k-m | AMD Radeon 610M (RADV RAPHAEL_MENDOCINO) | 10/10 | 100% | 100% | 153.9 | 7 | — |
| file-organizer | qwen3-8b-q4-k-m | AMD Radeon 610M (RADV RAPHAEL_MENDOCINO) | 8/10 | 80% | 93% | 193.9 | 6 | contains (2) |
| selected-text-tools | qwen3-4b-q4-k-m | AMD Radeon 610M (RADV RAPHAEL_MENDOCINO) | 9/10 | 90% | 97% | 46.3 | 10 | no_foreign_script (1) |
| selected-text-tools | qwen3-8b-q4-k-m | AMD Radeon 610M (RADV RAPHAEL_MENDOCINO) | 9/10 | 90% | 97% | 108.2 | 4 | no_foreign_script (1) |
