"""A variant may reword a task; it may not quietly become a different one."""

from __future__ import annotations

import json

import pytest

from omnitensor.benchmark.prompt_trial import (
    VariantError,
    differences,
    load_variant,
    varied,
)
from omnitensor.plugins.event_workload import event_generation_task

SHIPPED = event_generation_task()


def write(tmp_path, document):
    path = tmp_path / "variant.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_a_variant_replaces_only_what_it_states(tmp_path):
    tried = varied(SHIPPED, load_variant(write(tmp_path, {"system": "a different framing"})))

    assert tried.system_prompt == "a different framing"
    assert tried.instruction_template == SHIPPED.instruction_template
    assert tried.limits.output_tokens == SHIPPED.limits.output_tokens
    assert tried.output_schema == SHIPPED.output_schema
    assert differences(SHIPPED, tried) == ("system",)


def test_the_output_budget_is_part_of_the_wording_being_tried(tmp_path):
    tried = varied(SHIPPED, load_variant(write(tmp_path, {"outputTokens": 2048})))

    assert tried.limits.output_tokens == 2048
    assert tried.limits.context_tokens == SHIPPED.limits.context_tokens
    assert differences(SHIPPED, tried) == ("outputTokens 1024 to 2048",)


def test_a_variant_that_changes_nothing_is_visible_as_such(tmp_path):
    tried = varied(SHIPPED, load_variant(write(tmp_path, {"name": "no-op"})))

    assert differences(SHIPPED, tried) == ()


def test_an_instruction_without_the_content_placeholder_is_refused(tmp_path):
    # The model would be asked to work on nothing, and every case would fail
    # for a reason that has nothing to do with the wording being tried.
    path = write(tmp_path, {"instructionTemplate": "Extract the events."})

    with pytest.raises(VariantError, match="UNTRUSTED_CONTENT"):
        load_variant(path)


def test_a_variant_cannot_reach_past_the_wording(tmp_path):
    path = write(tmp_path, {"outputSchema": {}, "taskId": "something-else"})

    with pytest.raises(VariantError, match="does not apply"):
        load_variant(path)


def test_the_shipped_variants_are_readable_and_each_changes_something(tmp_path):
    from pathlib import Path

    trials = Path(__file__).resolve().parents[1] / "benchmarks" / "trials"
    for path in sorted(trials.glob("*.json")):
        tried = varied(SHIPPED, load_variant(path))
        assert differences(SHIPPED, tried), f"{path.name} changes nothing"
