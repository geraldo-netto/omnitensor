"""Where this service can read a source somebody selected.

`../xpuwlm` F2: the client refuses a source outside the published roots before
a job exists, and nothing published them — so the check sat dormant and every
malformed source still cost a whole submission to find out about.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from omnitensor.composition import ServiceEnvironment, _env_selected_file_roots
from omnitensor.discovery import Device
from omnitensor.registry import validate_document
from omnitensor.snapshot import (
    MAX_PUBLISHED_INPUT_ROOTS,
    MAX_SELECTED_SOURCE_BYTES,
    build_snapshot,
    selected_files_document,
)

DEVICES = [Device("gpu-renderD128", "gpu", "GPU 0", "dri")]
METRICS = {"queueDepth": 0, "runningProfiles": 0}


def test_a_runtime_that_reads_no_selected_file_says_exactly_that():
    """Empty is the honest answer for a service nobody configured roots on."""
    assert selected_files_document(()) == {"roots": [], "maxBytes": MAX_SELECTED_SOURCE_BYTES}


def test_the_published_roots_are_the_ones_configured(tmp_path):
    document = selected_files_document([tmp_path / "documents", str(tmp_path / "pictures")])

    assert document["roots"] == [
        str(tmp_path / "documents"),
        str(tmp_path / "pictures"),
    ]


def test_more_roots_than_can_be_published_is_a_configuration_error(tmp_path):
    """A root the snapshot cannot name is one no client can check against."""
    roots = [tmp_path / f"root-{index}" for index in range(MAX_PUBLISHED_INPUT_ROOTS + 1)]

    with pytest.raises(ValueError, match="at most"):
        selected_files_document(roots)


def test_the_snapshot_carries_the_block_and_still_matches_the_contract(tmp_path):
    snapshot = build_snapshot(
        DEVICES,
        METRICS,
        {},
        selected_files=selected_files_document([tmp_path]),
    )

    assert snapshot["selectedFiles"] == {
        "roots": [str(tmp_path)],
        "maxBytes": MAX_SELECTED_SOURCE_BYTES,
    }
    assert validate_document("runtime-snapshot.schema.json", snapshot) == []


def test_a_snapshot_without_the_block_is_still_valid():
    """Absent means a runtime older than the field, and a client submits as before."""
    snapshot = build_snapshot(DEVICES, METRICS, {})

    assert "selectedFiles" not in snapshot
    assert validate_document("runtime-snapshot.schema.json", snapshot) == []


def test_where_a_caller_stages_a_buffer_is_a_different_fact(tmp_path):
    """Confusing the two refuses valid sources on any sandboxed machine."""
    snapshot = build_snapshot(
        DEVICES,
        METRICS,
        {},
        inputs={"roots": [str(tmp_path / "staging")], "maxBytes": 1024},
        selected_files=selected_files_document([tmp_path / "home"]),
    )

    assert snapshot["inputs"]["roots"] == [str(tmp_path / "staging")]
    assert snapshot["selectedFiles"]["roots"] == [str(tmp_path / "home")]
    assert validate_document("runtime-snapshot.schema.json", snapshot) == []


def test_the_roots_come_from_the_environment_that_runs_the_unit(tmp_path):
    first = tmp_path / "documents"
    second = tmp_path / "pictures"
    raw = os.pathsep.join((str(first), str(second), str(first)))

    assert _env_selected_file_roots({"OMNITENSOR_SELECTED_FILE_ROOTS": raw}) == (first, second)
    assert _env_selected_file_roots({}) == ()


def test_too_many_configured_roots_stop_the_service_starting(tmp_path):
    raw = os.pathsep.join(
        str(tmp_path / f"root-{index}") for index in range(MAX_PUBLISHED_INPUT_ROOTS + 1)
    )

    with pytest.raises(ValueError, match="OMNITENSOR_SELECTED_FILE_ROOTS"):
        ServiceEnvironment.read({"OMNITENSOR_SELECTED_FILE_ROOTS": raw})


def test_the_service_options_carry_the_roots_to_the_runtime(tmp_path):
    options = ServiceEnvironment.read(
        {"OMNITENSOR_SELECTED_FILE_ROOTS": str(tmp_path / "documents")}
    ).service_options()

    assert options["selected_file_roots"] == (tmp_path / "documents",)


def test_a_service_publishes_the_roots_it_was_given(fake_nodes, tmp_path):
    from omnitensor.service import OmniTensorService

    service = OmniTensorService(
        snapshot_path=tmp_path / "state/runtime-snapshot.json",
        policy_path=tmp_path / "state/policy.json",
        workloads_path=tmp_path / "workloads",
        discovery_paths=fake_nodes,
        selected_file_roots=(Path.home(),),
    )

    assert service._selected_file_roots == (Path.home(),)
