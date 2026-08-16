"""The receipt this provider ships, against the tasks it will actually run.

Every workload here refuses to start unless the task it is about to run is the
one the qualification receipt says passed on this hardware. That binding is the
point: a prompt, a token budget or a schema changed without re-qualifying is a
model doing something nobody measured.

What was missing is the other half. `test_runtime.py` proves the *mechanism*
against a receipt it builds itself, so a task edited in `omnitensor` — a
different distribution, in the same repository — left the shipped receipt
describing a task that no longer exists, and nothing said so until a person
clicked the button and got "worker-unavailable: plugin worker is not ready".

That is precisely what happened to `ask-selected-files`: its output budget was
raised from 1,024 to 2,048 tokens to stop long answers being truncated, the
digest moved, the receipt did not, and the worker had been failing to start
ever since — silently, because a worker that never starts logs nothing a person
sees.

These tests compare the two directly. When one fails, the workload named in it
must be re-qualified on the GPU and the receipt re-recorded; the failure is not
the test's to fix.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omnitensor_qwen_runtime import factories, qualification

RECEIPT = (
    Path(__file__).parents[1] / "src/omnitensor_qwen_runtime/qualification.json"
)


def shipped() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


@pytest.mark.parametrize("plugin_id", sorted(factories._TASKS))
def test_the_shipped_receipt_describes_the_task_that_will_run(plugin_id):
    """A digest that has moved means the task changed without re-qualification.

    Re-qualify the workload on the GPU and re-record the receipt; do not edit
    the expectation to match the code, which is the one change that turns this
    gate into decoration.
    """
    recorded = shipped()["workloads"].get(plugin_id)
    assert recorded is not None, f"{plugin_id} has no qualification entry at all"

    current = qualification.task_sha256(factories._TASKS[plugin_id]())

    assert current == recorded["taskSha256"], (
        f"{plugin_id}: the task this provider will run has digest {current}, "
        f"while the receipt records {recorded['taskSha256']} as qualified. "
        "The workload needs re-qualifying on the GPU."
    )


@pytest.mark.parametrize("plugin_id", sorted(factories._TASKS))
def test_every_workload_the_provider_offers_is_recorded_as_passing(plugin_id):
    recorded = shipped()["workloads"][plugin_id]

    assert recorded["result"] == "passed"
    assert recorded["modelId"] in shipped()["models"]


def test_the_receipt_covers_exactly_the_workloads_this_provider_creates():
    """A receipt entry with no workload, or a workload with no entry, is a
    gate that will fail at start rather than here."""
    assert set(shipped()["workloads"]) == set(factories._TASKS)
