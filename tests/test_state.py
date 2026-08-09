from __future__ import annotations

import json

import pytest

from omnitensor.state import PolicyState, PolicyStore, ProfilePolicy

DEFAULTS = {
    "hardware-health": ProfilePolicy(enabled=True, weight=2),
    "visual-library": ProfilePolicy(enabled=False, weight=3),
}


def store(tmp_path) -> PolicyStore:
    return PolicyStore(tmp_path / "policy.json", dict(DEFAULTS))


def test_load_missing_file_returns_defaults(tmp_path):
    state = store(tmp_path).load()
    assert state.paused is False
    assert state.revision == 0
    assert state.profiles["hardware-health"] == ProfilePolicy(enabled=True, weight=2)


@pytest.mark.parametrize(
    "document",
    ["{nope", '"scalar"', "[]", "17", "null",
     '{"profiles": 3, "paused": "yes", "revision": -4}'],
)
def test_load_sanitizes_malformed_documents(tmp_path, document):
    (tmp_path / "policy.json").write_text(document)
    state = store(tmp_path).load()
    assert state.paused is False
    assert state.revision == 0
    assert set(state.profiles) == set(DEFAULTS)


@pytest.mark.parametrize("revision", [True, False, -1, "7", 1.5, None])
def test_load_rejects_non_integer_revisions(tmp_path, revision):
    (tmp_path / "policy.json").write_text(json.dumps({"revision": revision}))
    state = store(tmp_path).load()
    assert state.revision == 0
    assert type(state.revision) is int


def test_load_keeps_valid_revision_and_clamps_weights(tmp_path):
    (tmp_path / "policy.json").write_text(json.dumps({
        "revision": 12,
        "paused": True,
        "profiles": {
            "hardware-health": {"enabled": False, "weight": 99},
            "visual-library": {"enabled": "maybe", "weight": True},
            "unknown-profile": {"enabled": True, "weight": 4},
        },
    }))
    state = store(tmp_path).load()
    assert state.revision == 12
    assert state.paused is True
    assert state.profiles["hardware-health"] == ProfilePolicy(enabled=False, weight=5)
    assert state.profiles["visual-library"] == ProfilePolicy(enabled=False, weight=3)
    assert "unknown-profile" not in state.profiles


def test_save_round_trips_and_leaves_no_temp_files(tmp_path):
    policy_store = store(tmp_path)
    state = PolicyState(
        paused=True,
        profiles={"hardware-health": ProfilePolicy(enabled=False, weight=4)},
        revision=3,
    )
    policy_store.save(state)
    raw = json.loads((tmp_path / "policy.json").read_text())
    assert raw == {
        "paused": True,
        "profiles": {"hardware-health": {"enabled": False, "weight": 4}},
        "revision": 3,
    }
    leftovers = [entry for entry in tmp_path.iterdir() if entry.name.startswith(".policy-")]
    assert leftovers == []
