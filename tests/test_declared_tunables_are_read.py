"""A knob a manifest declares is a knob something has to turn.

`describe-plugins` publishes each workload's `configurationSchema`, and the
desktop client draws a control for every property in it. So a key declared
here and read by nothing is a control that changes nothing — which is the
worst kind of setting: a person sets it, sees it saved, and gets the same
answers back with no way to tell why.

The check is deliberately crude and hard to fool: for every declared key,
some module in this repository has to name it. What that module does with it
is its own tests' business; that *nothing* does is what this catches, and it
is the direction the mistake actually goes — a manifest is edited to offer a
setting before, or instead of, the code that honours it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
MANIFESTS = ROOT / "plugin-manifests"
# Where a key may be honoured: this package, and the provider distributions
# that implement the workloads. Tests are excluded on purpose — a key named
# only by a test is a key nothing reads at run time.
SOURCES = (ROOT / "src", *sorted((ROOT / "providers").glob("*/src")))


def declared_keys(workload_id: str) -> tuple[str, ...]:
    document = json.loads((MANIFESTS / f"{workload_id}.json").read_text(encoding="utf-8"))
    schema = document["plugin"]["schemas"]["configuration"]
    return tuple(schema.get("properties", {}))


def readers_of(key: str) -> tuple[str, ...]:
    """Every module that names this key, by relative path."""
    found = []
    for root in SOURCES:
        for path in root.rglob("*.py"):
            if key in path.read_text(encoding="utf-8"):
                found.append(str(path.relative_to(ROOT)))
    return tuple(found)


WORKLOADS = sorted(path.stem for path in MANIFESTS.glob("*.json"))


@pytest.mark.parametrize("workload_id", WORKLOADS)
def test_every_tunable_a_manifest_declares_is_named_by_something_that_runs(workload_id):
    for key in declared_keys(workload_id):
        assert readers_of(key), (
            f"{workload_id} declares {key!r} and no module reads it: the client will draw a "
            "control that changes nothing"
        )


@pytest.mark.parametrize("workload_id", WORKLOADS)
def test_a_declared_tunable_is_read_outside_the_manifest_it_is_declared_in(workload_id):
    """Named in a schema and nowhere else is the same as not read at all."""
    for key in declared_keys(workload_id):
        modules = readers_of(key)
        assert any(not module.startswith("plugin-manifests") for module in modules), key


def test_the_check_would_notice_a_knob_nobody_turns():
    """The gate's own failure mode, asserted rather than assumed."""
    assert readers_of("answerGuidance"), "a key that is read must be found"
    assert not readers_of("aKnobNobodyDeclaredOrRead"), "an unread key must not be found"
