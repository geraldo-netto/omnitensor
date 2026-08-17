# Benchmark results

Accuracy is whole answers that passed every rule their case asked for —
deliberately strict, because a summary with an invented citation is worse
than no summary. The rules column is the partial-credit view; when the two
disagree, read the failures column, which says what actually broke.

Speed is measured on the same generations, at the task's own context and
token budget. Nothing was capped or shortened to make a number look better.

| workload | model | device | correct | accuracy | rules | s/case | chars/s | what failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ask-selected-files | qwen3-4b-q4-k-m | vulkan-0 | 9/10 | 90% | 97% | 37.1 | 11 | no_invented_numbers (1) |
| ask-selected-files | qwen3-8b-q4-k-m | vulkan-0 | 9/10 | 90% | 97% | 68.1 | 6 | no_invented_numbers (1) |
| file-organizer | qwen3-4b-q4-k-m | vulkan-0 | 10/10 | 100% | 100% | 80.7 | 13 | — |
| file-organizer | qwen3-8b-q4-k-m | vulkan-0 | 8/10 | 80% | 93% | 168.6 | 7 | contains (2) |
