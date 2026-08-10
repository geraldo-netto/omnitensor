"""Consent that can actually be given, and that actually takes effect.

The ledger could always record a grant and nothing could create one: the
service never constructed it, so `active_permissions` came from a deny-all
stub and every declared permission read as ungranted for ever.  A refusal with
no remedy is worse than no gate, because it looks like a decision somebody
made.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnitensor import consent
from omnitensor.plugins.grants import GrantError, GrantLedger


@pytest.fixture
def ledger_path(tmp_path) -> Path:
    return tmp_path / "grants.json"


@pytest.fixture
def declared(monkeypatch):
    """A plugin declaring two permissions, without installing one."""
    monkeypatch.setattr(
        consent, "declared_permissions", lambda plugin_id: {"files:read", "net:connect"}
    )


def run(ledger_path, *arguments):
    status, output = consent.run(["--grants-path", str(ledger_path), *arguments])
    return status, json.loads(output)


def test_a_permission_starts_ungranted_and_says_so(ledger_path, declared):
    status, document = run(ledger_path, "list", "sample-plugin")

    assert status == 0
    assert document["permissions"] == [
        {"name": "files:read", "granted": False},
        {"name": "net:connect", "granted": False},
    ]


def test_consent_is_recorded_and_survives_a_new_reader(ledger_path, declared):
    status, document = run(
        ledger_path, "grant", "sample-plugin", "files:read", "--reason", "photos"
    )

    assert status == 0
    assert document["granted"] == ["files:read"]

    # A separate process reading the ledger sees it, which is the only way the
    # service ever will.
    _status, listed = run(ledger_path, "list", "sample-plugin")
    assert listed["permissions"][0] == {"name": "files:read", "granted": True}


def test_withdrawing_consent_takes_it_away(ledger_path, declared):
    run(ledger_path, "grant", "sample-plugin", "files:read")

    status, document = run(ledger_path, "revoke", "sample-plugin", "files:read")

    assert status == 0
    assert document["granted"] == []


def test_consent_cannot_be_given_for_something_never_asked_for(ledger_path, declared):
    """The declaration bounds what may be granted, so a manifest that drops a
    permission stops it being usable rather than leaving a grant behind."""
    status, document = run(ledger_path, "grant", "sample-plugin", "disk:format")

    assert status == 1
    assert document["error"] == "permission-undeclared"


def test_an_unknown_plugin_is_named_rather_than_traced(ledger_path):
    status, document = run(ledger_path, "list", "no-such-plugin")

    assert status == 1
    assert document["error"] == "plugin-unknown"


def test_the_change_records_who_made_it_and_why(ledger_path, declared):
    run(ledger_path, "grant", "sample-plugin", "files:read", "--reason", "indexing my pictures")

    stored = json.loads(ledger_path.read_text())
    audit = stored["audit"][-1]

    assert audit["pluginId"] == "sample-plugin"
    assert audit["permission"] == "files:read"
    assert audit["action"] == "granted"
    assert audit["provenance"]["reason"] == "indexing my pictures"
    assert audit["provenance"]["origin"] == "user"
    assert audit["provenance"]["actorId"]


def test_a_reason_is_bounded_like_everything_else_a_user_types(ledger_path, declared):
    run(ledger_path, "grant", "sample-plugin", "files:read", "--reason", "x" * 5_000)

    stored = json.loads(ledger_path.read_text())

    assert len(stored["audit"][-1]["provenance"]["reason"]) == consent.MAX_REASON_CHARS


def test_granting_twice_is_not_an_error_and_not_a_second_grant(ledger_path, declared):
    run(ledger_path, "grant", "sample-plugin", "files:read")
    status, document = run(ledger_path, "grant", "sample-plugin", "files:read")

    assert status == 0
    assert document["granted"] == ["files:read"]


def test_the_ledger_path_follows_the_environment_when_nothing_names_one(monkeypatch, tmp_path):
    monkeypatch.setenv("OMNITENSOR_GRANTS_PATH", str(tmp_path / "from-env.json"))

    assert consent._ledger_path(None) == tmp_path / "from-env.json"
    assert consent._ledger_path(str(tmp_path / "explicit.json")) == tmp_path / "explicit.json"

    monkeypatch.delenv("OMNITENSOR_GRANTS_PATH")
    assert consent._ledger_path(None).is_absolute(), "the default expands to a real path"


def test_the_declaration_is_read_from_the_manifest_not_from_the_plugin():
    """Asking what a plugin requires must never mean running it."""
    with pytest.raises(GrantError) as excinfo:
        consent.declared_permissions("no-such-plugin")

    assert excinfo.value.code == "plugin-unknown"
    assert consent.declared_permissions("visual-library") == set()


def test_the_service_consults_the_ledger_rather_than_a_deny_all_stub(tmp_path):
    """The gate refused everything because nothing was ever wired to it."""
    from omnitensor.service import OmniTensorService

    grants = tmp_path / "grants.json"
    service = OmniTensorService(
        snapshot_path=tmp_path / "state.json",
        policy_path=tmp_path / "policy.json",
        workloads_path=tmp_path / "workloads",
        artifact_root=tmp_path / "artifacts",
        grants_path=grants,
    )

    assert isinstance(service._grants, GrantLedger)
    assert service._permitted("visual-library", "files:read") is False

    # No installed profile declares a permission, so nothing is permitted — and
    # that is a fact about the catalogue rather than about the wiring.
    assert service._plugin_runtime._grant_source is service._grants


def test_a_withdrawn_grant_stops_a_job_already_queued(tmp_path, monkeypatch):
    """Enforcement re-reads before it decides, so revoking means revoking now."""
    from omnitensor.registry import Workload
    from omnitensor.service import OmniTensorService

    grants = tmp_path / "grants.json"
    ledger = GrantLedger(grants)
    service = OmniTensorService(
        snapshot_path=tmp_path / "state.json",
        policy_path=tmp_path / "policy.json",
        workloads_path=tmp_path / "workloads",
        artifact_root=tmp_path / "artifacts",
        grants=ledger,
    )
    declaring = Workload(
        id="declaring-profile",
        manifest={"plugin": {"permissions": ["files:read"]}},
    )
    service._workloads["declaring-profile"] = declaring

    assert service._permitted("declaring-profile", "files:read") is False

    provenance = consent._provenance("test", "req-1")
    ledger.grant(
        "declaring-profile",
        "files:read",
        {"files:read"},
        provenance,
        expected_revision=ledger.revision,
    )

    assert service._permitted("declaring-profile", "files:read") is True

    ledger.revoke(
        "declaring-profile",
        "files:read",
        {"files:read"},
        consent._provenance("test", "req-2"),
        expected_revision=ledger.revision,
    )

    assert service._permitted("declaring-profile", "files:read") is False, (
        "a withdrawal takes effect at once"
    )


def test_consent_given_to_one_profile_does_not_authorise_another(tmp_path):
    """A permission name is not an identity.

    Answering "does any installed plugin hold a grant for this string" made
    consent transitive: granting `files:read` to a photo indexer also
    satisfied the gate of anything else that asked for `files:read`, which is
    the opposite of what granting one thing means.
    """
    from omnitensor.registry import Workload
    from omnitensor.service import OmniTensorService

    ledger = GrantLedger(tmp_path / "grants.json")
    service = OmniTensorService(
        snapshot_path=tmp_path / "state.json",
        policy_path=tmp_path / "policy.json",
        workloads_path=tmp_path / "workloads",
        artifact_root=tmp_path / "artifacts",
        grants=ledger,
    )
    for profile_id in ("consented-profile", "other-profile"):
        service._workloads[profile_id] = Workload(
            id=profile_id, manifest={"plugin": {"permissions": ["files:read"]}}
        )

    ledger.grant(
        "consented-profile",
        "files:read",
        {"files:read"},
        consent._provenance("indexing my pictures", "req-1"),
        expected_revision=ledger.revision,
    )

    assert service._permitted("consented-profile", "files:read") is True
    assert service._permitted("other-profile", "files:read") is False


def test_a_permission_a_profile_never_declared_is_never_permitted(tmp_path):
    """The manifest bounds what consent can be given for."""
    from omnitensor.registry import Workload
    from omnitensor.service import OmniTensorService

    service = OmniTensorService(
        snapshot_path=tmp_path / "state.json",
        policy_path=tmp_path / "policy.json",
        workloads_path=tmp_path / "workloads",
        artifact_root=tmp_path / "artifacts",
        grants=GrantLedger(tmp_path / "grants.json"),
    )
    service._workloads["quiet"] = Workload(id="quiet", manifest={"plugin": {"permissions": []}})

    assert service._permitted("quiet", "files:read") is False
    assert service._permitted("no-such-profile", "files:read") is False
