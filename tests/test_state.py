from __future__ import annotations

import json

import pytest

from omnitensor.state import (
    MAX_DEVICE_CHOICES,
    MAX_PROFILES,
    MAX_WEIGHT,
    PolicyState,
    PolicyStore,
    ProfilePolicy,
)

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
    ["{nope", '"scalar"', "[]", "17", "null", '{"profiles": 3, "paused": "yes", "revision": -4}'],
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
    (tmp_path / "policy.json").write_text(
        json.dumps(
            {
                "revision": 12,
                "paused": True,
                "profiles": {
                    "hardware-health": {"enabled": False, "weight": 99},
                    "visual-library": {"enabled": "maybe", "weight": True},
                    "unknown-profile": {"enabled": True, "weight": 4},
                },
            }
        )
    )
    state = store(tmp_path).load()
    assert state.revision == 12
    assert state.paused is True
    assert state.profiles["hardware-health"] == ProfilePolicy(enabled=False, weight=5)
    assert state.profiles["visual-library"] == ProfilePolicy(enabled=False, weight=3)
    # Kept, not dropped: a profile with no default here is a plugin discovered
    # after this store was built, and its policy is somebody's decision.
    assert state.profiles["unknown-profile"] == ProfilePolicy(enabled=True, weight=4)


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
        "deviceChoices": {},
        "modelChoices": {},
        "revision": 3,
    }
    leftovers = [entry for entry in tmp_path.iterdir() if entry.name.startswith(".policy-")]
    assert leftovers == []


def test_device_choices_round_trip_and_invalid_entries_are_dropped(tmp_path):
    choices = {
        "hardware-health": "gpu-renderD129",
        "external-plugin": "gpu-renderD2048",
        "bad-device": "gpu-card0",
        "": "gpu-renderD128",
        "x" * 81: "gpu-renderD128",
    }
    (tmp_path / "policy.json").write_text(json.dumps({"deviceChoices": choices}), encoding="utf-8")

    state = store(tmp_path).load()

    assert state.device_choices == {
        "external-plugin": "gpu-renderD2048",
        "hardware-health": "gpu-renderD129",
    }
    store(tmp_path).save(state)
    assert json.loads((tmp_path / "policy.json").read_text())["deviceChoices"] == {
        "external-plugin": "gpu-renderD2048",
        "hardware-health": "gpu-renderD129",
    }


def test_device_choice_load_is_bounded_deterministically(tmp_path):
    choices = {
        f"profile-{index:03d}": f"gpu-renderD{128 + index}"
        for index in range(MAX_DEVICE_CHOICES + 5)
    }
    (tmp_path / "policy.json").write_text(json.dumps({"deviceChoices": choices}), encoding="utf-8")

    loaded = store(tmp_path).load().device_choices

    assert len(loaded) == MAX_DEVICE_CHOICES
    assert tuple(loaded) == tuple(sorted(choices)[:MAX_DEVICE_CHOICES])


def test_policy_for_a_profile_the_defaults_never_knew_survives_a_reload(tmp_path):
    """Plugins are discovered after the store is built; their policy is still theirs."""
    (tmp_path / "policy.json").write_text(
        json.dumps(
            {
                "profiles": {
                    "hardware-health": {"enabled": False, "weight": 1},
                    "file-organizer": {"enabled": False, "weight": 4},
                }
            }
        ),
        encoding="utf-8",
    )

    profiles = store(tmp_path).load().profiles

    assert profiles["file-organizer"] == ProfilePolicy(enabled=False, weight=4)
    assert profiles["hardware-health"] == ProfilePolicy(enabled=False, weight=1)


def test_an_adopted_profile_with_junk_values_falls_back_rather_than_failing(tmp_path):
    (tmp_path / "policy.json").write_text(
        json.dumps({"profiles": {"file-organizer": {"enabled": "yes", "weight": 99}}}),
        encoding="utf-8",
    )

    assert store(tmp_path).load().profiles["file-organizer"] == ProfilePolicy(
        enabled=True, weight=MAX_WEIGHT
    )


def test_a_stored_profile_id_the_contract_could_not_publish_is_ignored(tmp_path):
    (tmp_path / "policy.json").write_text(
        json.dumps({"profiles": {"../escape": {"enabled": True, "weight": 1}}}),
        encoding="utf-8",
    )

    assert "../escape" not in store(tmp_path).load().profiles


def test_stored_policy_is_kept_for_every_profile_however_many_there_are(tmp_path):
    """The 128-profile ceiling is gone from the contract and from here."""
    stored = {f"plugin-{index:03d}": {"enabled": True, "weight": 1} for index in range(200)}
    (tmp_path / "policy.json").write_text(json.dumps({"profiles": stored}), encoding="utf-8")

    assert MAX_PROFILES is None
    assert len(store(tmp_path).load().profiles) == 200 + len(DEFAULTS)
