# Benchmark results

Accuracy is whole answers that passed every rule their case asked for —
deliberately strict, because a summary with an invented citation is worse
than no summary. The rules column is the partial-credit view; when the two
disagree, read the failures column, which says what actually broke.

Speed is measured on the same generations, at the task's own context and
token budget. Nothing was capped or shortened to make a number look better.

| workload | model | device | correct | accuracy | rules | s/case | chars/s | what failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selected-text-tools | dictalm2-hebrew-q4-k-m | vulkan-0 | 0/10 | 0% | 0% | 1.2 | 0 | produced-a-usable-document (10) |
| selected-text-tools | qwen3-8b-q4-k-m | vulkan-0 | 9/10 | 90% | 97% | 81.4 | 6 | no_foreign_script (1) |
