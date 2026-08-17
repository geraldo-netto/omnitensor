# Benchmark results

Accuracy is whole answers that passed every rule their case asked for —
deliberately strict, because a summary with an invented citation is worse
than no summary. The rules column is the partial-credit view; when the two
disagree, read the failures column, which says what actually broke.

Speed is measured on the same generations, at the task's own context and
token budget. Nothing was capped or shortened to make a number look better.

| workload | model | device | correct | accuracy | rules | s/case | chars/s | what failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ask-selected-files | qwen3-5-35b-a3b-q4-k-m | AMD Radeon 610M (RADV RAPHAEL_MENDOCINO) | 9/10 | 90% | 97% | 86.1 | 5 | no_invented_numbers (1) |
| event-extraction | qwen3-5-35b-a3b-q4-k-m | AMD Radeon 610M (RADV RAPHAEL_MENDOCINO) | 3/10 | 30% | 14% | 61.9 | 5 | answers-rather-than-refusing (7), at_least (7), event_starts (5) |
| file-organizer | qwen3-5-35b-a3b-q4-k-m | AMD Radeon 610M (RADV RAPHAEL_MENDOCINO) | 9/10 | 90% | 96% | 226.6 | 5 | produced-a-usable-document (1) |
| selected-text-tools | qwen3-5-35b-a3b-q4-k-m | AMD Radeon 610M (RADV RAPHAEL_MENDOCINO) | 10/10 | 100% | 100% | 125.7 | 4 | — |
