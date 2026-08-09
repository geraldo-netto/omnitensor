# TODO

## Findings

| id | status | severity | effort | related ids | description |
| --- | --- | --- | --- | --- | --- |
| OMNI-0032 | open | low | xs | OMNI-0031 | quality gates: `tests/test_documentation.py` reads `docs/use-cases.md`, but mutmut does not copy `docs/` into its sandbox, so every changed-function mutation run fails during clean-test collection before mutants execute. |

## Rejected / Won't fix

| id | status | severity | effort | related ids | description |
| --- | --- | --- | --- | --- | --- |
