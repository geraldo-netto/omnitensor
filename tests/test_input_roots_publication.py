"""The one thing a caller needs and cannot discover.

Referencing a file is a capability: it is off unless configured on, and a
buffer written outside the configured roots is refused with the same answer as
a path that does not exist — deliberately, so refusals cannot be used to probe
the filesystem.  That leaves a caller unable to tell "I staged it in the wrong
place" from "the service is not reading files at all", and unable to find the
right place by trying.

So the roots are published.  Optional in the contract, because a snapshot
written before the field is still a valid snapshot; and empty rather than
absent when nothing is configured, because "reads nothing" is a fact worth
stating rather than a silence to interpret.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnitensor.discovery import Device
from omnitensor.registry import _schema_path, validate_document
from omnitensor.snapshot import (
    MAX_PUBLISHED_INPUT_ROOTS,
    build_snapshot,
    input_roots_document,
)
from omnitensor.tensorref import DEFAULT_MAX_TENSOR_BYTES

METRICS = {"queueDepth": 0, "runningProfiles": 0}


def devices() -> list[Device]:
    return [Device("gpu-renderD128", "gpu", "AMD GPU", "dri", "0x1002")]


def snapshot(**overrides) -> dict:
    return build_snapshot(devices(), METRICS, {}, **overrides)


def test_the_configured_roots_are_published_as_the_service_sees_them():
    document = input_roots_document([Path("/home/user/omnitensor-inputs")])

    assert document == {
        "roots": ["/home/user/omnitensor-inputs"],
        "maxBytes": DEFAULT_MAX_TENSOR_BYTES,
    }


def test_a_service_that_reads_nothing_says_so_rather_than_staying_silent():
    """Empty is a fact; absent is a runtime too old to have one."""
    assert input_roots_document(()) == {"roots": [], "maxBytes": DEFAULT_MAX_TENSOR_BYTES}
    assert "inputs" not in snapshot()
    assert snapshot(inputs=input_roots_document(()))["inputs"]["roots"] == []


def test_the_published_list_is_bounded_like_every_other_collection():
    roots = [f"/root{index}" for index in range(MAX_PUBLISHED_INPUT_ROOTS + 4)]

    assert len(input_roots_document(roots)["roots"]) == MAX_PUBLISHED_INPUT_ROOTS


def test_a_snapshot_carrying_the_block_is_contract_valid():
    document = snapshot(inputs=input_roots_document([Path("/home/user/inputs")]))

    assert validate_document("runtime-snapshot.schema.json", document) == []
    assert document["inputs"]["maxBytes"] == DEFAULT_MAX_TENSOR_BYTES


def test_the_block_is_optional_so_an_older_snapshot_still_validates():
    document = snapshot()
    document.pop("inputs", None)

    assert validate_document("runtime-snapshot.schema.json", document) == []


def test_a_root_list_the_schema_refuses_never_reaches_a_reader():
    for broken in (
        {"roots": [f"/r{index}" for index in range(MAX_PUBLISHED_INPUT_ROOTS + 1)], "maxBytes": 1},
        {"roots": ["/a", "/a"], "maxBytes": 1},
        {"roots": [""], "maxBytes": 1},
        {"roots": ["/a"], "maxBytes": -1},
        {"roots": ["/a"]},
        {"roots": ["/a"], "maxBytes": 1, "extra": True},
    ):
        with pytest.raises(ValueError, match="violates contract"):
            snapshot(inputs=broken)


def test_the_service_publishes_the_roots_it_was_configured_with(tmp_path):
    """The check that matters: what the service publishes is what it will read.

    Both come from the same tuple, so a service configured with a root it does
    not honour — or honouring one it does not publish — is a construction bug
    rather than a drift between two lists.
    """
    from omnitensor.service import OmniTensorService

    root = tmp_path / "inputs"
    root.mkdir()
    service = OmniTensorService(
        snapshot_path=tmp_path / "state.json",
        policy_path=tmp_path / "policy.json",
        workloads_path=tmp_path / "workloads",
        artifact_root=tmp_path / "artifacts",
        input_roots=(root,),
    )

    published = service.publish_once()

    assert published["inputs"]["roots"] == [str(root)]
    assert service._input_roots == (root,)
    written = json.loads((tmp_path / "state.json").read_text())
    assert written["inputs"] == published["inputs"]


def test_the_shipped_schema_states_why_the_block_exists():
    """A consumer reading only the schema has to learn not to guess."""
    schema = json.loads(_schema_path("runtime-snapshot.schema.json").read_text())
    description = schema["properties"]["inputs"]["description"]

    assert "must not guess" in description
    assert "capability" in description
