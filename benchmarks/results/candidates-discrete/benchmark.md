# Benchmark results

Accuracy is whole answers that passed every rule their case asked for —
deliberately strict, because a summary with an invented citation is worse
than no summary. The rules column is the partial-credit view; when the two
disagree, read the failures column, which says what actually broke.

Speed is measured on the same generations, at the task's own context and
token budget. Nothing was capped or shortened to make a number look better.

| workload | model | device | correct | accuracy | rules | s/case | chars/s | what failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ask-selected-files | qwen3-5-9b-deepseek-v4-flash-q4-k-m | AMD Radeon RX 6600 XT (RADV NAVI23) | 9/10 | 90% | 97% | 19.7 | 22 | no_invented_numbers (1) |
| ask-selected-files | qwen3-5-9b-iq4-xs | AMD Radeon RX 6600 XT (RADV NAVI23) | 9/10 | 90% | 97% | 18.5 | 23 | no_invented_numbers (1) |
| ask-selected-files | qwen3-5-9b-q4-k-m | AMD Radeon RX 6600 XT (RADV NAVI23) | 9/10 | 90% | 97% | 17.7 | 23 | no_invented_numbers (1) |
| ask-selected-files | qwen3-8b-q5-k-m | AMD Radeon RX 6600 XT (RADV NAVI23) | 9/10 | 90% | 97% | 11.1 | 34 | no_invented_numbers (1) |
| event-extraction | qwen3-5-9b-deepseek-v4-flash-q4-k-m | AMD Radeon RX 6600 XT (RADV NAVI23) | 3/10 | 30% | 14% | 8.2 | 40 | answers-rather-than-refusing (7), at_least (7), event_starts (5) |
| event-extraction | qwen3-5-9b-iq4-xs | AMD Radeon RX 6600 XT (RADV NAVI23) | 3/10 | 30% | 14% | 7.7 | 39 | answers-rather-than-refusing (7), at_least (7), event_starts (5) |
| event-extraction | qwen3-5-9b-q4-k-m | AMD Radeon RX 6600 XT (RADV NAVI23) | 3/10 | 30% | 14% | 7.9 | 39 | answers-rather-than-refusing (7), at_least (7), event_starts (5) |
| event-extraction | qwen3-8b-q5-k-m | AMD Radeon RX 6600 XT (RADV NAVI23) | 3/10 | 30% | 14% | 5.6 | 49 | answers-rather-than-refusing (7), at_least (7), event_starts (5) |
| file-organizer | qwen3-5-9b-deepseek-v4-flash-q4-k-m | AMD Radeon RX 6600 XT (RADV NAVI23) | 8/10 | 80% | 92% | 36.9 | 29 | at_least (1), produced-a-usable-document (1) |
| file-organizer | qwen3-5-9b-iq4-xs | AMD Radeon RX 6600 XT (RADV NAVI23) | 8/10 | 80% | 92% | 39.5 | 28 | at_least (1), produced-a-usable-document (1) |
| file-organizer | qwen3-5-9b-q4-k-m | AMD Radeon RX 6600 XT (RADV NAVI23) | 8/10 | 80% | 92% | 38.2 | 30 | at_least (1), produced-a-usable-document (1) |
| file-organizer | qwen3-8b-q5-k-m | AMD Radeon RX 6600 XT (RADV NAVI23) | 9/10 | 90% | 96% | 31.0 | 45 | produced-a-usable-document (1) |
| selected-text-tools | qwen3-5-9b-deepseek-v4-flash-q4-k-m | AMD Radeon RX 6600 XT (RADV NAVI23) | 10/10 | 100% | 100% | 18.7 | 25 | — |
| selected-text-tools | qwen3-5-9b-iq4-xs | AMD Radeon RX 6600 XT (RADV NAVI23) | 10/10 | 100% | 100% | 19.4 | 24 | — |
| selected-text-tools | qwen3-5-9b-q4-k-m | AMD Radeon RX 6600 XT (RADV NAVI23) | 10/10 | 100% | 100% | 18.1 | 27 | — |
| selected-text-tools | qwen3-8b-q5-k-m | AMD Radeon RX 6600 XT (RADV NAVI23) | 9/10 | 90% | 97% | 13.5 | 35 | no_foreign_script (1) |
