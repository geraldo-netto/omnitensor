# Benchmark results

Accuracy is whole answers that passed every rule their case asked for —
deliberately strict, because a summary with an invented citation is worse
than no summary. The rules column is the partial-credit view; when the two
disagree, read the failures column, which says what actually broke.

Speed is measured on the same generations, at the task's own context and
token budget. Nothing was capped or shortened to make a number look better.

| workload | model | device | correct | accuracy | rules | s/case | chars/s | what failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ask-selected-files | qwen3-8b-q4-k-m | vulkan-0 | 8/10 | 80% | 93% | 87.1 | 4 | refuses (2) |
| event-extraction | qwen3-8b-q4-k-m | vulkan-0 | 3/10 | 30% | 14% | 31.7 | 9 | answers-rather-than-refusing (7), at_least (7), event_starts (5) |
| file-organizer | qwen3-8b-q4-k-m | vulkan-0 | 0/10 | 0% | 56% | 168.5 | 7 | at_least (10), contains (2) |
| selected-text-tools | qwen3-8b-q4-k-m | vulkan-0 | 10/10 | 100% | 100% | 81.6 | 6 | — |
