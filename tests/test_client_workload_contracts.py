"""The workloads the desktop client drives, checked against their manifests.

`../xpuwlm` builds a payload per workload from a spec of its own: which fields
exist, how many files are allowed, which suffixes a chooser offers. Those specs
are transcribed from the manifests in this repository, and a transcription is a
copy — it agrees until one side changes.

This is the cheap half of that agreement, and it runs anywhere: the payload the
client builds for each workload is validated against the manifest's own input
schema, and the bounds the client enforces before submitting are checked against
the bounds the manifest states. The expensive half — that the provider behind
each manifest actually answers — is `test_client_workload_e2e.py`, which needs a
running service.

A drift caught here costs nothing. The same drift caught there costs a job, and
in a client it costs a person a minute of watching a spinner for a refusal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).parents[1]
MANIFESTS = ROOT / "plugin-manifests"

# What the client sends for each workload, in the shape its own spec builds.
# Transcribed from `xpuwlm/workflows/specs/catalog.py`: a payload builder per
# workload, and the field bounds beside them.
CLIENT_PAYLOADS = {
    "ask-selected-files": {
        "sources": ["/home/person/omnitensor-inputs/invoice.txt"],
        "question": "What is the total due?",
    },
    "selected-text-tools": {
        "selection": "The mitochondrion is the powerhouse of the cell.",
        "operation": "explain",
    },
    "file-organizer": {
        "sources": [
            "/home/person/omnitensor-inputs/invoice.txt",
            "/home/person/omnitensor-inputs/notes.md",
        ],
    },
    "document-translation": {
        "sources": ["/home/person/omnitensor-inputs/contract.txt"],
        "targetLanguage": "Português",
    },
    "event-extraction": {"sources": ["/home/person/omnitensor-inputs/meeting.md"]},
    "media-transcription": {"sources": ["/home/person/omnitensor-inputs/clip.wav"]},
    "media-similarity": {
        "sources": [
            "/home/person/omnitensor-inputs/library/a.mp4",
            "/home/person/omnitensor-inputs/library/b.mkv",
        ],
        "relativePaths": ["a.mp4", "b.mkv"],
        "minimumSimilarity": 0.8,
    },
}

# Workloads this repository ships that the client does not drive yet, with the
# reason. Named rather than omitted: the check below is in both directions, so
# a shipped workload is either transcribed above or listed here on purpose.
# Empty is the state to be in, and `document-translation` was the last entry:
# it left when the client shipped a spec, a window and a chooser for it.
NOT_CLIENT_DRIVEN: dict[str, str] = {}

# The bounds the client refuses on before it submits anything, so a person is
# told at the chooser rather than a minute later by a failed job.
CLIENT_SOURCE_BOUNDS = {
    "ask-selected-files": 16,
    "file-organizer": 16,
    "event-extraction": 32,
    "media-transcription": 1,
}


def manifest(workload_id: str) -> dict:
    return json.loads((MANIFESTS / f"{workload_id}.json").read_text(encoding="utf-8"))


def input_schema(workload_id: str) -> dict:
    return manifest(workload_id)["plugin"]["schemas"]["input"]


@pytest.mark.parametrize("workload_id", sorted(CLIENT_PAYLOADS))
def test_the_payload_the_client_builds_satisfies_the_manifest(workload_id):
    """The client's spec is a transcription of this manifest; this is the check
    that the transcription still says the same thing."""
    errors = sorted(
        Draft202012Validator(input_schema(workload_id)).iter_errors(CLIENT_PAYLOADS[workload_id]),
        key=lambda error: error.path,
    )

    assert errors == [], f"{workload_id}: {[error.message for error in errors]}"


@pytest.mark.parametrize("workload_id", sorted(CLIENT_SOURCE_BOUNDS))
def test_the_client_refuses_at_the_bound_the_manifest_states(workload_id):
    """A chooser that offers seventeen files for a workload that takes sixteen
    spends a job to say so."""
    declared = input_schema(workload_id)["properties"]["sources"]["maxItems"]

    assert declared == CLIENT_SOURCE_BOUNDS[workload_id]


@pytest.mark.parametrize("workload_id", sorted(CLIENT_PAYLOADS))
def test_a_payload_missing_a_required_field_is_refused_by_the_manifest(workload_id):
    """The client validates before submitting; this proves the runtime would
    have refused the same thing, so the two agree about what is required."""
    schema = input_schema(workload_id)
    required = schema.get("required") or []
    if not required:
        pytest.skip(f"{workload_id} requires nothing")
    for field in required:
        incomplete = {
            name: value for name, value in CLIENT_PAYLOADS[workload_id].items() if name != field
        }
        errors = list(Draft202012Validator(schema).iter_errors(incomplete))

        assert errors, f"{workload_id} accepted a payload without {field}"


def test_the_operations_the_client_offers_are_the_ones_the_manifest_allows():
    """Five buttons in a window, one enum in a manifest. A sixth on either side
    is a refusal a person only sees after waiting for it."""
    schema = input_schema("selected-text-tools")
    declared = set(schema["properties"]["operation"]["enum"])
    offered = {"explain", "summarize", "rewrite", "translate", "extract-tasks"}

    assert declared == offered


def test_the_client_workloads_and_the_shipped_manifests_agree_in_both_directions():
    """Only one direction was checked, and the other is where the gap was.

    A window for a workload this repository no longer ships is a button that
    fails on click — that is the direction this had. A workload this
    repository ships that the client never drives is the opposite: a manifest
    published to a client that cannot reach it, and nothing said so when
    `document-translation` landed.
    """
    shipped = {path.stem for path in MANIFESTS.glob("*.json")}

    assert set(CLIENT_PAYLOADS) <= shipped
    assert set(NOT_CLIENT_DRIVEN) <= shipped
    assert set(CLIENT_PAYLOADS) | set(NOT_CLIENT_DRIVEN) == shipped
    assert set(CLIENT_PAYLOADS).isdisjoint(NOT_CLIENT_DRIVEN)
    assert all(reason.strip() for reason in NOT_CLIENT_DRIVEN.values())
