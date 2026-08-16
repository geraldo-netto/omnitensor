"""Write down what a run found, for somebody deciding months later.

Two files, because they answer different questions. The JSON holds every case,
every failed rule and every timing, so a surprising summary can be taken apart.
The Markdown is the table a person actually reads when choosing between two
models.

Both are written after each model finishes rather than at the end: a run that
is interrupted after two hours should leave two hours of findings behind.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

RESULT_JSON = "benchmark.json"
RESULT_MARKDOWN = "benchmark.md"


def as_document(runs: Sequence) -> dict:
    """Every run, in full, as data rather than as prose."""
    return {
        "version": 1,
        "runs": [
            {
                "workload": run.workload,
                "modelId": run.model_id,
                "device": run.device,
                "contextTokens": run.context_tokens,
                "cache": run.cache,
                "layersOffloaded": run.layers_offloaded,
                "loadSeconds": round(run.load_seconds, 2),
                "accuracy": round(run.tally.accuracy, 4),
                "ruleScore": round(run.tally.rule_score, 4),
                "correct": run.tally.correct,
                "cases": run.tally.total,
                "errors": run.errors,
                "secondsPerCase": round(run.seconds_per_case, 2),
                "charactersPerSecond": round(run.characters_per_second, 1),
                "failures": run.tally.failures(),
                "outcomes": [
                    {
                        "caseId": outcome.case_id,
                        "correct": outcome.judgement.correct,
                        "passed": list(outcome.judgement.passed),
                        "failed": list(outcome.judgement.failed),
                        "seconds": round(outcome.timing.seconds, 2),
                        "outputCharacters": outcome.timing.output_characters,
                        "error": outcome.error,
                        "document": outcome.document,
                    }
                    for outcome in run.outcomes
                ],
            }
            for run in runs
        ],
    }


def as_table(runs: Sequence) -> str:
    """The comparison, one row per (workload, model)."""
    lines = [
        "| workload | model | device | correct | accuracy | rules | s/case | chars/s "
        "| what failed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in sorted(runs, key=lambda item: (item.workload, item.model_id)):
        tally = run.tally
        failed = ", ".join(f"{name} ({count})" for name, count in tally.failures().items())
        lines.append(
            f"| {run.workload} | {run.model_id} | {run.device} | "
            f"{tally.correct}/{tally.total} | {tally.accuracy:.0%} | "
            f"{tally.rule_score:.0%} | {run.seconds_per_case:.1f} | "
            f"{run.characters_per_second:.0f} | {failed or '—'} |"
        )
    return "\n".join(lines)


def as_markdown(runs: Sequence) -> str:
    header = (
        "# Benchmark results\n\n"
        "Accuracy is whole answers that passed every rule their case asked for —\n"
        "deliberately strict, because a summary with an invented citation is worse\n"
        "than no summary. The rules column is the partial-credit view; when the two\n"
        "disagree, read the failures column, which says what actually broke.\n\n"
        "Speed is measured on the same generations, at the task's own context and\n"
        "token budget. Nothing was capped or shortened to make a number look better.\n\n"
    )
    return header + as_table(runs) + "\n"


def write_results(runs: Sequence, directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / RESULT_JSON).write_text(
        json.dumps(as_document(runs), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown = directory / RESULT_MARKDOWN
    markdown.write_text(as_markdown(runs), encoding="utf-8")
    return markdown


__all__ = ["as_document", "as_markdown", "as_table", "write_results"]
