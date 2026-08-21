"""Which pinned artifact a person may choose as a workload's model.

`../xpuwlm` G6: a manifest pins every artifact a workload uses, so a dropdown
built from that list offered `ask-selected-files` its embedder, and
`set-profile-model` accepted it — ending at a worker that cannot answer a
question with BGE.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
MANIFESTS = ROOT / "plugin-manifests"

# What the shipped providers actually do with each pinned artifact, read from
# their code rather than assumed: `bootstrap.chosen_or(...)` is a choice a
# person makes, `bootstrap.require_artifact(...)` is one the workload makes for
# itself.
FIXED_ROLE = {
    # The embedder and the OCR extraction pair (OMNI-0615) have fixed
    # roles; the person chooses only the generation model.
    "ask-selected-files": {
        "bge-small-en-v1-5-ask-gpu",
        "ppocrv6-medium-det",
        "ppocrv6-medium-rec",
    },
    "document-translation": {"dictalm2-hebrew-q4-k-m"},
    "selected-text-tools": {"dictalm2-hebrew-q4-k-m"},
    # The vision model became choosable in OMNI-0587/0588: the 9B is the
    # default and the 7B the alternative, so only the speech model is fixed.
    # Speech and the OCR enrichment pair (OMNI-0607) have fixed roles: the
    # person chooses the vision model; whisper and OCR are not alternatives.
    "media-transcription": {
        "whisper-small-multilingual",
        "ppocrv6-medium-det",
        "ppocrv6-medium-rec",
    },
    "event-extraction": set(),
    "file-organizer": set(),
}


def manifest(workload_id: str) -> dict:
    return json.loads((MANIFESTS / f"{workload_id}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("workload_id", sorted(FIXED_ROLE))
def test_an_artifact_the_workload_picks_for_itself_is_not_offered(workload_id):
    artifacts = manifest(workload_id)["plugin"]["artifacts"]
    unselectable = {
        artifact["id"] for artifact in artifacts if artifact.get("selectable", True) is False
    }

    assert unselectable == FIXED_ROLE[workload_id]


@pytest.mark.parametrize("workload_id", sorted(FIXED_ROLE))
def test_the_runtime_offers_exactly_what_the_manifest_says_may_be_chosen(workload_id):
    from omnitensor.plugins.discovery import PluginSource
    from omnitensor.plugins.identity import ResolvedPlugin
    from omnitensor.plugins.loading import InstalledPluginRuntime

    document = manifest(workload_id)
    plugin = ResolvedPlugin(
        workload_id,
        document["version"],
        PluginSource.EXTERNAL,
        f"omnitensor-{workload_id}",
        "0.2.0",
        workload_id,
        "module:create",
        MANIFESTS / f"{workload_id}.json",
        document,
    )
    runtime = InstalledPluginRuntime(ROOT / "workloads")
    runtime._plugin = lambda plugin_id: plugin if plugin_id == workload_id else None

    offered = runtime.declared_artifacts(workload_id)

    declared = [artifact["id"] for artifact in document["plugin"]["artifacts"]]
    assert list(offered) == [name for name in declared if name not in FIXED_ROLE[workload_id]]
    assert set(offered).isdisjoint(FIXED_ROLE[workload_id])


def test_a_manifest_that_says_nothing_keeps_every_artifact_choosable():
    """Absent means selectable, so an older manifest means what it meant."""
    from omnitensor.plugins.discovery import PluginSource
    from omnitensor.plugins.identity import ResolvedPlugin
    from omnitensor.plugins.loading import InstalledPluginRuntime

    document = manifest("ask-selected-files")
    for artifact in document["plugin"]["artifacts"]:
        artifact.pop("selectable", None)
    plugin = ResolvedPlugin(
        "ask-selected-files",
        document["version"],
        PluginSource.EXTERNAL,
        "omnitensor-qwen-ask-selected-files",
        "0.2.0",
        "ask-selected-files",
        "module:create",
        MANIFESTS / "ask-selected-files.json",
        document,
    )
    runtime = InstalledPluginRuntime(ROOT / "workloads")
    runtime._plugin = lambda plugin_id: plugin if plugin_id == "ask-selected-files" else None

    assert "bge-small-en-v1-5-ask-gpu" in runtime.declared_artifacts("ask-selected-files")


def test_the_inventory_says_of_every_artifact_whether_it_may_be_chosen():
    from omnitensor.inspection import ArtifactReadiness, plugin_inventory_entry
    from omnitensor.plugins.artifacts import ArtifactResolution
    from omnitensor.plugins.discovery import PluginSource
    from omnitensor.plugins.identity import ResolvedPlugin

    document = manifest("ask-selected-files")
    plugin = ResolvedPlugin(
        "ask-selected-files",
        document["version"],
        PluginSource.EXTERNAL,
        "omnitensor-qwen-ask-selected-files",
        "0.2.0",
        "ask-selected-files",
        "module:create",
        MANIFESTS / "ask-selected-files.json",
        document,
    )

    entry = plugin_inventory_entry(
        plugin,
        resolve_artifact=lambda _reference: ArtifactResolution(True, None, "", 0),
    )

    published = {item["id"]: item["selectable"] for item in entry["artifacts"]}
    assert published == {
        "qwen3-5-9b-iq4-xs": True,
        "qwen3-8b-q4-k-m": True,
        "bge-small-en-v1-5-ask-gpu": False,
        "ppocrv6-medium-det": False,
        "ppocrv6-medium-rec": False,
    }
    assert ArtifactReadiness("a", "1.0.0", "gguf", True, "").selectable is True


def test_choosing_an_artifact_the_workload_picks_for_itself_is_refused():
    """The other half: `set-profile-model` accepted it before this."""
    import asyncio
    import json as json_module
    import tempfile

    from omnitensor.control import CONTROL_VERSION, build_control_service
    from omnitensor.state import ProfilePolicy

    with tempfile.TemporaryDirectory() as directory:
        service = build_control_service(
            Path(directory) / "policy.json",
            {"ask-selected-files": ProfilePolicy(enabled=True, weight=2)},
            profile_models=lambda _profile_id: ("qwen3-5-9b-iq4-xs", "qwen3-8b-q4-k-m"),
        )
        command = json_module.dumps(
            {
                "version": CONTROL_VERSION,
                "id": "xpuwlm-1",
                "issuedAt": 1_700_000_000_000,
                "expectedRevision": 0,
                "operation": "set-profile-model",
                "profileId": "ask-selected-files",
                "value": "bge-small-en-v1-5-ask-gpu",
            }
        )
        acknowledgement = json_module.loads(asyncio.run(service.apply_command_text(command)))

    assert acknowledgement["status"] == "rejected"
    assert "not declared by this workload" in acknowledgement["message"]
