"""The policy this runtime enforces, published where a client can read it.

The snapshot told a client everything except the one thing a client changes.
So the desktop client kept its own copy of enabled, weight and device — and
two stores for one fact drift: on the desk this was written for, one profile
was weight 2 in the store the scheduler obeys and weight 1 in the copy the
window drew, at the same moment.

Publishing it from the store that answers `apply-command` makes that
impossible by construction rather than by discipline. The revision travels with
it for a second reason: a client that has read a snapshot never has to guess
what to send as `expectedRevision`, which is what a cold client used to do.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from conftest import add_npu, add_pcie_tpu, sample_manifest, write_workload

from omnitensor.discovery import Device
from omnitensor.registry import validate_document
from omnitensor.service import OmniTensorService
from omnitensor.snapshot import build_snapshot
from omnitensor.state import PolicyState, ProfilePolicy

PROFILE = "sample-workload"


def device():
    return Device(id="tpu-pcie-0", backend="tpu", name="Coral PCIe Edge TPU", kind="pcie")


def service(fake_nodes, tmp_path):
    workloads_root = tmp_path / "workloads"
    workloads_root.mkdir(exist_ok=True)
    write_workload(workloads_root, sample_manifest())
    return OmniTensorService(
        snapshot_path=tmp_path / "state/runtime-snapshot.json",
        policy_path=tmp_path / "state/policy.json",
        workloads_path=workloads_root,
        discovery_paths=fake_nodes,
    )


def published(fake_nodes, tmp_path, command=None):
    """One published snapshot, after optionally applying one command."""

    async def scenario():
        add_pcie_tpu(fake_nodes)
        add_npu(fake_nodes)
        built = service(fake_nodes, tmp_path)
        built._scheduler.start()
        if command is not None:
            reply = await built.control.apply_command_text(json.dumps(command))
            assert json.loads(reply)["status"] == "applied", reply
        snapshot = built.publish_once()
        await built._scheduler.stop()
        return snapshot, built.control.state

    return asyncio.run(scenario())


def command(operation, profile_id=None, value=None, revision=0):
    return {
        "version": 2,
        "id": f"test-{operation}",
        "issuedAt": 1_700_000_000_000,
        "expectedRevision": revision,
        "operation": operation,
        "profileId": profile_id,
        "value": value,
    }


class TestTheDocument:
    def test_a_state_publishes_its_revision_pause_and_every_profile(self):
        state = PolicyState(
            paused=True,
            profiles={
                PROFILE: ProfilePolicy(enabled=False, weight=3),
                "hardware-health": ProfilePolicy(enabled=True, weight=1),
            },
            device_choices={PROFILE: "gpu-renderD128"},
            revision=7,
        )

        assert state.snapshot_document() == {
            "revision": 7,
            "paused": True,
            "profiles": {
                "hardware-health": {"enabled": True, "weight": 1},
                PROFILE: {
                    "enabled": False,
                    "weight": 3,
                    "deviceId": "gpu-renderD128",
                },
            },
        }

    def test_the_device_sits_on_the_profile_it_belongs_to(self):
        """A client draws one row per profile, so the choice travels on the row
        rather than in a second map it would have to join."""
        state = PolicyState(
            profiles={"a": ProfilePolicy(enabled=True, weight=1)},
            device_choices={"a": "gpu-renderD129"},
        )

        assert state.snapshot_document()["profiles"]["a"]["deviceId"] == "gpu-renderD129"

    def test_a_profile_nobody_pinned_carries_no_device(self):
        """Absent means the runtime chooses, which is not the same as a client
        having chosen the runtime's current pick."""
        state = PolicyState(profiles={"a": ProfilePolicy(enabled=True, weight=1)})

        assert "deviceId" not in state.snapshot_document()["profiles"]["a"]


class TestTheContract:
    def test_a_snapshot_carrying_policy_is_contract_valid(self):
        snapshot = build_snapshot(
            devices=[device()],
            metrics={"queueDepth": 0, "runningProfiles": 0},
            profiles={},
            policy={"revision": 3, "paused": False, "profiles": {}},
        )

        assert validate_document("runtime-snapshot.schema.json", snapshot) == []
        assert snapshot["policy"]["revision"] == 3

    def test_policy_is_optional_so_an_older_reader_is_unaffected(self):
        """Additive on purpose: the panel applet reads this file and knows
        nothing about the block."""
        snapshot = build_snapshot(
            devices=[device()],
            metrics={"queueDepth": 0, "runningProfiles": 0},
            profiles={},
        )

        assert "policy" not in snapshot
        assert validate_document("runtime-snapshot.schema.json", snapshot) == []

    @pytest.mark.parametrize(
        "policy",
        [
            {"paused": False, "profiles": {}},
            {"revision": 1, "profiles": {}},
            {"revision": 1, "paused": False},
            {"revision": 1, "paused": False, "profiles": {"a": {"enabled": True}}},
            {"revision": 1, "paused": False, "profiles": {"a": {"enabled": True, "weight": 9}}},
            {
                "revision": 1,
                "paused": False,
                "profiles": {"a": {"enabled": True, "weight": 1, "deviceId": "card0"}},
            },
        ],
    )
    def test_a_policy_block_that_says_less_than_the_contract_is_refused(self, policy):
        """A half-published policy is worse than none: a client cannot tell a
        profile it may govern from one the field forgot."""
        snapshot = {
            "version": 1,
            "generatedAt": 1_700_000_000_000,
            "devices": [device().snapshot_entry()],
            "metrics": {"queueDepth": 0, "runningProfiles": 0},
            "profiles": {},
            "alerts": [],
            "policy": policy,
        }

        assert validate_document("runtime-snapshot.schema.json", snapshot) != []


class TestWhatTheServicePublishes:
    def test_the_published_policy_is_the_store_that_answers_commands(
        self, fake_nodes, tmp_path
    ):
        """One source by construction. Anything else is the drift this exists
        to remove."""
        snapshot, state = published(fake_nodes, tmp_path)

        assert snapshot["policy"] == state.snapshot_document()

    def test_an_applied_command_is_visible_in_the_next_snapshot(
        self, fake_nodes, tmp_path
    ):
        snapshot, state = published(
            fake_nodes,
            tmp_path,
            command("set-profile-weight", PROFILE, 4),
        )

        assert snapshot["policy"]["profiles"][PROFILE]["weight"] == 4
        assert snapshot["policy"]["revision"] == state.revision

    def test_pausing_everything_is_published_as_runtime_state(
        self, fake_nodes, tmp_path
    ):
        """A client that closed does not un-pause a runtime, so the pause has
        to be legible to the next client that opens."""
        snapshot, _state = published(fake_nodes, tmp_path, command("set-paused", value=True))

        assert snapshot["policy"]["paused"] is True

    def test_the_revision_a_client_must_send_next_is_the_one_published(
        self, fake_nodes, tmp_path
    ):
        """The reason a cold client had to guess, and then retry on the
        rejection its guess earned."""
        snapshot, state = published(
            fake_nodes, tmp_path, command("set-profile-enabled", PROFILE, False)
        )

        assert snapshot["policy"]["revision"] == state.revision
        assert snapshot["policy"]["profiles"][PROFILE]["enabled"] is False


class TestChoosingAModel:
    """A person choosing which model a workload runs.

    The reason this exists: `dictalm2-hebrew-q4-k-m` is installed and
    qualified, `HebrewTranslationRuntime` is written and working, and until the
    receipt could hold more than one model per workload there was no way to ask
    for either. Choosing is intent, so it is a command and a published fact,
    exactly like the card a profile is pinned to.
    """

    def control(self, tmp_path, declared=("qwen3-8b-q4-k-m", "qwen3-4b-q4-k-m")):
        from omnitensor.control import ControlService
        from omnitensor.state import PolicyStore

        store = PolicyStore(
            tmp_path / "policy.json",
            {PROFILE: ProfilePolicy(enabled=True, weight=1)},
        )
        return ControlService(
            store,
            profile_exists=lambda profile_id: profile_id == PROFILE,
            profile_models=lambda profile_id: declared if profile_id == PROFILE else (),
        )

    def apply(self, control, value, revision=0, profile_id=PROFILE):
        text = json.dumps(command("set-profile-model", profile_id, value, revision))
        return json.loads(asyncio.run(control.apply_command_text(text)))

    def test_a_declared_model_is_accepted_and_published(self, tmp_path):
        control = self.control(tmp_path)

        reply = self.apply(control, "qwen3-4b-q4-k-m")

        assert reply["status"] == "applied"
        assert control.state.snapshot_document()["profiles"][PROFILE]["modelId"] == (
            "qwen3-4b-q4-k-m"
        )

    def test_a_model_the_manifest_never_pinned_is_refused(self, tmp_path):
        """Every declared artifact has a digest this service verifies; anything
        else has none, so choosing it could only end at a worker that refuses
        to start. Better refused where the refusal can be read."""
        control = self.control(tmp_path)

        reply = self.apply(control, "some-model-from-the-internet")

        assert reply["status"] == "rejected"
        assert "not declared by this workload" in reply["message"]
        assert control.state.model_choices == {}

    def test_clearing_the_choice_returns_the_workload_to_its_default(self, tmp_path):
        """Absent is not the same fact as having chosen what the receipt
        defaults to, so it is published as absent."""
        control = self.control(tmp_path)
        self.apply(control, "qwen3-4b-q4-k-m")

        reply = self.apply(control, None, revision=control.state.revision)

        assert reply["status"] == "applied"
        assert "modelId" not in control.state.snapshot_document()["profiles"][PROFILE]

    def test_a_profile_nobody_installed_is_refused(self, tmp_path):
        control = self.control(tmp_path)

        reply = self.apply(control, "qwen3-4b-q4-k-m", profile_id="not-installed")

        assert reply["status"] == "rejected"
        assert "Unknown workload profile" in reply["message"]

    def test_the_choice_survives_a_restart(self, tmp_path):
        """A policy this service enforces with no client attached, so it is
        stored here rather than in whichever window happened to set it."""
        from omnitensor.state import PolicyStore

        control = self.control(tmp_path)
        self.apply(control, "qwen3-4b-q4-k-m")

        reloaded = PolicyStore(tmp_path / "policy.json", {}).load()

        assert reloaded.model_choices == {PROFILE: "qwen3-4b-q4-k-m"}

    def test_a_snapshot_carrying_a_chosen_model_is_contract_valid(self):
        snapshot = build_snapshot(
            devices=[device()],
            metrics={"queueDepth": 0, "runningProfiles": 0},
            profiles={},
            policy={
                "revision": 4,
                "paused": False,
                "profiles": {
                    PROFILE: {
                        "enabled": True,
                        "weight": 1,
                        "modelId": "dictalm2-hebrew-q4-k-m",
                    }
                },
            },
        )

        assert validate_document("runtime-snapshot.schema.json", snapshot) == []
