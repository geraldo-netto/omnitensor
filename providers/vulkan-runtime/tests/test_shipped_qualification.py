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

Since receipt version 2 a workload may list several models, so every *pair* is
checked, not the workload's only entry: a task change moves the digest for all
of them at once, and a pair whose digest was missed is a worker that refuses to
start for whoever chose that model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omnitensor_vulkan_runtime import factories, qualification

RECEIPT = Path(__file__).parents[1] / "src/omnitensor_vulkan_runtime/qualification.json"


def shipped() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def pairs() -> list[tuple[str, str]]:
    """Every (workload, model) the shipped receipt records."""
    return sorted(
        (plugin_id, model_id)
        for plugin_id, entry in shipped()["workloads"].items()
        for model_id in entry["models"]
    )


def measured_pairs() -> list[tuple[str, str]]:
    """The pairs somebody actually ran, which are the ones with a digest.

    A pair recorded `unmeasured` has no task digest to have drifted: it is a
    workload that ships before its GPU run, and the answer it produces says
    "was not measured" rather than claiming a receipt it does not have.
    """
    return [
        (plugin_id, model_id)
        for plugin_id, model_id in pairs()
        if shipped()["workloads"][plugin_id]["models"][model_id]["result"] != "unmeasured"
    ]


@pytest.mark.parametrize(("plugin_id", "model_id"), measured_pairs())
def test_the_shipped_receipt_describes_the_task_that_will_run(plugin_id, model_id):
    """A digest that has moved means the task changed without re-qualification.

    Re-qualify the pair on the GPU and re-record the receipt; do not edit the
    expectation to match the code, which is the one change that turns this gate
    into decoration.
    """
    recorded = shipped()["workloads"][plugin_id]["models"][model_id]

    current = qualification.task_sha256(factories._TASKS[plugin_id]())

    assert current == recorded["taskSha256"], (
        f"{plugin_id} on {model_id}: the task this provider will run has digest "
        f"{current}, while the receipt records {recorded['taskSha256']}. "
        "The pair needs re-qualifying on the GPU."
    )


@pytest.mark.parametrize(("plugin_id", "model_id"), pairs())
def test_every_recorded_pair_names_a_model_the_receipt_qualified(plugin_id, model_id):
    assert model_id in shipped()["models"]


@pytest.mark.parametrize(("plugin_id", "model_id"), pairs())
def test_a_pair_that_failed_says_why(plugin_id, model_id):
    """A recorded failure without its reason is a note that says only "no", and
    the next person re-runs the same hours to learn the same thing."""
    recorded = shipped()["workloads"][plugin_id]["models"][model_id]

    assert recorded["result"] in ("passed", "failed", "unmeasured")
    if recorded["result"] == "failed":
        assert recorded.get("reason", "").strip()
    if recorded["result"] == "unmeasured":
        # Nothing ran, so there is no digest to record; one here would be a
        # measurement nobody took.
        assert "taskSha256" not in recorded


@pytest.mark.parametrize("plugin_id", sorted(factories._TASKS))
def test_no_workload_defaults_to_a_model_that_failed(plugin_id):
    """A default that failed hands every job that did not choose a model to a
    model somebody measured and rejected."""
    entry = shipped()["workloads"][plugin_id]

    assert entry["default"] in entry["models"]
    assert entry["models"][entry["default"]]["result"] != "failed"


def test_every_workload_this_provider_creates_is_measured_or_says_it_is_not():
    """Which workloads still owe a GPU run, named rather than assumed."""
    unmeasured = sorted(
        plugin_id
        for plugin_id, entry in shipped()["workloads"].items()
        if all(record["result"] == "unmeasured" for record in entry["models"].values())
    )

    # OMNI-0523: `document-translation` ships before its receipt because the
    # acceptance run needs the installed distribution to run against. Remove it
    # from here in the commit that records the measurement.
    assert unmeasured == ["document-translation"]


def test_the_receipt_covers_exactly_the_workloads_this_provider_creates():
    """A receipt entry with no workload, or a workload with no entry, is a
    gate that will fail at start rather than here."""
    assert set(shipped()["workloads"]) == set(factories._TASKS)
