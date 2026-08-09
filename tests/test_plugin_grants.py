from __future__ import annotations

import json
import multiprocessing
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins import (
    MAX_DECLARED_PERMISSIONS,
    ActiveGrant,
    GrantAction,
    GrantAuditEvent,
    GrantError,
    GrantLedger,
    GrantOrigin,
    GrantProvenance,
    GrantSnapshot,
)

READ_SENSOR = "read:/sys/class/hwmon/*"
RUN_ACTION = "action:notify-user"
DECLARED = {READ_SENSOR, RUN_ACTION}


def provenance(
    actor_id="user:1000",
    origin=GrantOrigin.USER,
    reason="enabled in settings",
    recorded_at_ms=10,
    request_id="request-1",
):
    return GrantProvenance(actor_id, origin, reason, recorded_at_ms, request_id)


def grant_once(ledger, permission=READ_SENSOR, declared=DECLARED, at=10):
    return ledger.grant(
        "hardware-health",
        permission,
        declared,
        provenance(recorded_at_ms=at, request_id=f"request-{at}"),
        expected_revision=ledger.revision,
    )


def test_missing_ledger_is_empty_and_does_not_create_state(tmp_path):
    path = tmp_path / "grants.json"
    ledger = GrantLedger(path)

    assert ledger.revision == 0
    assert ledger.snapshot("hardware-health", DECLARED) == GrantSnapshot(
        "hardware-health",
        0,
        (),
        (),
    )
    assert not path.exists()


def test_declared_grant_persists_active_provenance_and_audit(tmp_path):
    path = tmp_path / "grants.json"
    ledger = GrantLedger(path)
    source = provenance()

    snapshot = ledger.grant(
        "hardware-health",
        READ_SENSOR,
        DECLARED,
        source,
        expected_revision=0,
    )

    event = GrantAuditEvent(1, "hardware-health", READ_SENSOR, GrantAction.GRANTED, source)
    assert snapshot == GrantSnapshot(
        "hardware-health",
        1,
        (ActiveGrant(READ_SENSOR, source),),
        (event,),
    )
    assert ledger.is_granted("hardware-health", READ_SENSOR, DECLARED) is True
    ledger.require("hardware-health", READ_SENSOR, DECLARED)
    assert GrantLedger(path).snapshot("hardware-health", DECLARED) == snapshot


def test_undeclared_permission_is_denied_before_persistence(tmp_path):
    path = tmp_path / "grants.json"
    ledger = GrantLedger(path)

    with pytest.raises(GrantError) as excinfo:
        ledger.grant(
            "hardware-health",
            "read:/etc/shadow",
            DECLARED,
            provenance(),
            expected_revision=0,
        )

    assert str(excinfo.value) == (
        "permission-undeclared: hardware-health does not declare read:/etc/shadow"
    )
    assert ledger.revision == 0
    assert not path.exists()


def test_active_permissions_returns_only_declared_current_grants(tmp_path):
    ledger = GrantLedger(tmp_path / "grants.json")
    grant_once(ledger)
    assert ledger.active_permissions("hardware-health", DECLARED) == frozenset(
        {READ_SENSOR}
    )
    assert ledger.active_permissions("hardware-health", set()) == frozenset()


def test_manifest_removal_immediately_denies_a_stale_persisted_grant(tmp_path):
    ledger = GrantLedger(tmp_path / "grants.json")
    grant_once(ledger)

    assert ledger.is_granted("hardware-health", READ_SENSOR, set()) is False
    assert ledger.snapshot("hardware-health", set()).active == ()
    with pytest.raises(GrantError) as excinfo:
        ledger.require("hardware-health", READ_SENSOR, set())
    assert excinfo.value.code == "permission-denied"


def test_revoke_removes_access_and_records_independent_provenance(tmp_path):
    path = tmp_path / "grants.json"
    ledger = GrantLedger(path)
    granted = provenance()
    ledger.grant("hardware-health", READ_SENSOR, DECLARED, granted, expected_revision=0)
    revoked = provenance(
        origin=GrantOrigin.ADMIN,
        reason="incident response",
        recorded_at_ms=20,
        request_id="request-2",
    )

    snapshot = ledger.revoke(
        "hardware-health",
        READ_SENSOR,
        set(),
        revoked,
        expected_revision=1,
    )

    assert snapshot.active == ()
    assert snapshot.revision == 2
    assert snapshot.audit[-1] == GrantAuditEvent(
        2,
        "hardware-health",
        READ_SENSOR,
        GrantAction.REVOKED,
        revoked,
    )
    assert ledger.is_granted("hardware-health", READ_SENSOR, DECLARED) is False
    assert json.loads(path.read_text())["grants"] == {}


def test_duplicate_grant_and_absent_revoke_are_revision_stable_noops(tmp_path):
    ledger = GrantLedger(tmp_path / "grants.json")
    first = grant_once(ledger)

    duplicate = ledger.grant(
        "hardware-health",
        READ_SENSOR,
        DECLARED,
        provenance(recorded_at_ms=99, request_id="duplicate"),
        expected_revision=1,
    )
    absent = ledger.revoke(
        "hardware-health",
        RUN_ACTION,
        DECLARED,
        provenance(request_id="absent"),
        expected_revision=1,
    )

    assert duplicate == absent == first
    assert ledger.revision == 1


def test_revision_mismatch_preserves_current_state(tmp_path):
    path = tmp_path / "grants.json"
    ledger = GrantLedger(path)
    grant_once(ledger)
    before = path.read_bytes()

    with pytest.raises(GrantError) as excinfo:
        ledger.revoke(
            "hardware-health",
            READ_SENSOR,
            DECLARED,
            provenance(request_id="stale"),
            expected_revision=0,
        )

    assert str(excinfo.value) == "revision-mismatch: expected 0; current revision is 1"
    assert path.read_bytes() == before


def test_audit_history_is_bounded_to_the_newest_events(tmp_path):
    path = tmp_path / "grants.json"
    ledger = GrantLedger(path, audit_limit=2)
    grant_once(ledger, at=1)
    ledger.revoke(
        "hardware-health",
        READ_SENSOR,
        DECLARED,
        provenance(recorded_at_ms=2, request_id="request-2"),
        expected_revision=1,
    )
    grant_once(ledger, at=3)

    snapshot = GrantLedger(path, audit_limit=2).snapshot("hardware-health", DECLARED)
    assert [event.revision for event in snapshot.audit] == [2, 3]
    assert [event.action for event in snapshot.audit] == [
        GrantAction.REVOKED,
        GrantAction.GRANTED,
    ]


def test_atomic_write_failure_does_not_publish_candidate_state(tmp_path, monkeypatch):
    path = tmp_path / "grants.json"
    ledger = GrantLedger(path)
    grant_once(ledger)
    before = path.read_bytes()

    def fail_write(_path, _document, prefix):
        assert prefix == ".plugin-grants-"
        raise OSError("disk full")

    monkeypatch.setattr("omnitensor.plugins.grants.write_json_atomic", fail_write)
    with pytest.raises(OSError, match="disk full"):
        ledger.grant(
            "hardware-health",
            RUN_ACTION,
            DECLARED,
            provenance(request_id="request-2"),
            expected_revision=1,
        )
    assert ledger.revision == 1
    assert ledger.is_granted("hardware-health", RUN_ACTION, DECLARED) is False
    assert path.read_bytes() == before


@pytest.mark.parametrize("plugin_id", ["", "../escape", "Upper", None])
def test_plugin_ids_are_strict(plugin_id, tmp_path):
    ledger = GrantLedger(tmp_path / "grants.json")
    with pytest.raises(GrantError) as excinfo:
        ledger.snapshot(plugin_id, set())
    assert excinfo.value.code == "invalid-plugin-id"


@pytest.mark.parametrize(
    "permission",
    ["", "ambient-root", "Read:/tmp", "read:", "read:/bad path", "x:" + "a" * 159, None],
)
def test_permissions_use_the_manifest_grammar(permission, tmp_path):
    ledger = GrantLedger(tmp_path / "grants.json")
    with pytest.raises(GrantError) as excinfo:
        ledger.is_granted("hardware-health", permission, set())
    assert excinfo.value.code == "invalid-permission"


def test_declared_permissions_are_a_bounded_validated_set(tmp_path):
    ledger = GrantLedger(tmp_path / "grants.json")
    with pytest.raises(GrantError) as wrong_type:
        ledger.snapshot("hardware-health", [])
    assert wrong_type.value.code == "invalid-declarations"

    too_many = {f"read:/source/{index}" for index in range(MAX_DECLARED_PERMISSIONS + 1)}
    with pytest.raises(GrantError) as count:
        ledger.snapshot("hardware-health", too_many)
    assert count.value.code == "invalid-declarations"

    with pytest.raises(GrantError) as malformed:
        ledger.snapshot("hardware-health", {"malformed"})
    assert malformed.value.code == "invalid-permission"


@pytest.mark.parametrize(
    "candidate",
    [
        None,
        provenance(actor_id="bad actor"),
        provenance(origin="user"),
        provenance(reason="x" * 241),
        provenance(recorded_at_ms=-1),
        provenance(recorded_at_ms=True),
        provenance(request_id="bad request"),
    ],
)
def test_grant_provenance_is_explicit_and_bounded(candidate, tmp_path):
    ledger = GrantLedger(tmp_path / "grants.json")
    with pytest.raises(GrantError) as excinfo:
        ledger.grant(
            "hardware-health",
            READ_SENSOR,
            DECLARED,
            candidate,
            expected_revision=0,
        )
    assert excinfo.value.code in {"invalid-provenance", "invalid-integer"}


@pytest.mark.parametrize("revision", [-1, True, 1.5, "1", None])
def test_expected_revision_is_a_non_negative_integer(revision, tmp_path):
    with pytest.raises(GrantError) as excinfo:
        GrantLedger(tmp_path / "grants.json").grant(
            "hardware-health",
            READ_SENSOR,
            DECLARED,
            provenance(),
            expected_revision=revision,
        )
    assert excinfo.value.code == "invalid-integer"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_bytes": 0},
        {"max_bytes": True},
        {"audit_limit": 0},
        {"audit_limit": True},
    ],
)
def test_ledger_limits_are_positive_integers(tmp_path, kwargs):
    with pytest.raises(ValueError):
        GrantLedger(tmp_path / "grants.json", **kwargs)


def test_read_and_write_byte_limit_is_exact(tmp_path):
    path = tmp_path / "grants.json"
    ledger = GrantLedger(path)
    grant_once(ledger)
    size = path.stat().st_size

    assert GrantLedger(path, max_bytes=size).revision == 1
    with pytest.raises(GrantError) as read_limit:
        GrantLedger(path, max_bytes=size - 1)
    assert read_limit.value.code == "grants-too-large"

    other = tmp_path / "other.json"
    with pytest.raises(GrantError) as write_limit:
        grant_once(GrantLedger(other, max_bytes=size - 1))
    assert write_limit.value.code == "grants-too-large"
    assert not other.exists()


def valid_document():
    source = provenance()
    return {
        "documentVersion": 1,
        "revision": 1,
        "grants": {
            "hardware-health": {
                READ_SENSOR: {
                    "actorId": source.actor_id,
                    "origin": str(source.origin),
                    "reason": source.reason,
                    "recordedAtMs": source.recorded_at_ms,
                    "requestId": source.request_id,
                }
            }
        },
        "audit": [
            {
                "revision": 1,
                "pluginId": "hardware-health",
                "permission": READ_SENSOR,
                "action": "granted",
                "provenance": {
                    "actorId": source.actor_id,
                    "origin": "user",
                    "reason": source.reason,
                    "recordedAtMs": source.recorded_at_ms,
                    "requestId": source.request_id,
                },
            }
        ],
    }


def load_document(tmp_path, document, **kwargs):
    path = tmp_path / "grants.json"
    path.write_text(json.dumps(document))
    return GrantLedger(path, **kwargs)


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda document: document.update({"extra": True}), "invalid-grants"),
        (lambda document: document.update(documentVersion=2), "grants-version-incompatible"),
        (lambda document: document.update(documentVersion=True), "grants-version-incompatible"),
        (lambda document: document.update(revision=-1), "invalid-integer"),
        (lambda document: document.update(grants=[]), "invalid-grants"),
        (
            lambda document: document["grants"].update({"../escape": {}}),
            "invalid-plugin-id",
        ),
        (
            lambda document: document["grants"].update({"other-plugin": []}),
            "invalid-grants",
        ),
        (
            lambda document: document["grants"]["hardware-health"].update({"bad": {}}),
            "invalid-permission",
        ),
        (lambda document: document.update(audit={}), "invalid-grants"),
        (lambda document: document["audit"].append({}), "invalid-grants"),
        (lambda document: document["audit"][0].update(revision=0), "invalid-grants"),
        (lambda document: document["audit"][0].update(revision=2), "invalid-grants"),
        (
            lambda document: document["audit"][0].update(pluginId="../bad"),
            "invalid-plugin-id",
        ),
        (lambda document: document["audit"][0].update(permission="bad"), "invalid-permission"),
        (lambda document: document["audit"][0].update(action="future"), "invalid-grants"),
        (
            lambda document: document["audit"][0]["provenance"].update(origin="future"),
            "invalid-grants",
        ),
        (
            lambda document: document["audit"][0].update(provenance={}),
            "invalid-grants",
        ),
    ],
)
def test_persisted_grant_contract_fails_closed(tmp_path, mutate, code):
    document = valid_document()
    mutate(document)
    with pytest.raises(GrantError) as excinfo:
        load_document(tmp_path, document)
    assert excinfo.value.code == code


@pytest.mark.parametrize("content", ["not-json", "[]", "\ud800"])
def test_unreadable_or_non_object_grant_state_fails_closed(tmp_path, content):
    path = tmp_path / "grants.json"
    path.write_text(content, errors="surrogatepass")
    with pytest.raises(GrantError) as excinfo:
        GrantLedger(path)
    expected = "invalid-grants" if content == "[]" else "grants-unreadable"
    assert excinfo.value.code == expected


def test_loaded_empty_plugin_grant_map_is_canonicalized_away(tmp_path):
    document = valid_document()
    document["grants"]["empty-plugin"] = {}
    ledger = load_document(tmp_path, document)
    assert ledger.snapshot("empty-plugin", set()).active == ()


def test_loaded_audit_count_cannot_exceed_local_bound(tmp_path):
    document = valid_document()
    document["audit"].append(dict(document["audit"][0]))
    with pytest.raises(GrantError, match="audit must contain at most 1 events"):
        load_document(tmp_path, document, audit_limit=1)


@given(
    declared=st.sets(
        st.from_regex(r"[a-z][a-z0-9-]{0,8}:/[a-zA-Z0-9._/-]{1,20}", fullmatch=True),
        max_size=10,
    ),
)
def test_undeclared_arbitrary_permissions_are_never_authorized(declared):
    with tempfile.TemporaryDirectory() as directory:
        ledger = GrantLedger(Path(directory) / "grants.json")
        assert ledger.is_granted("hardware-health", READ_SENSOR, declared) is False


def test_a_stale_ledger_cannot_resurrect_a_permission_revoked_elsewhere(tmp_path):
    """The classic lost-update: commit from a snapshot taken before the revoke."""
    path = tmp_path / "grants.json"
    stale = GrantLedger(path)
    grant_once(stale)
    assert stale.revision == 1

    elsewhere = GrantLedger(path)
    elsewhere.revoke(
        "hardware-health", READ_SENSOR, DECLARED, provenance(), expected_revision=1
    )

    with pytest.raises(GrantError) as excinfo:
        stale.grant(
            "hardware-health", RUN_ACTION, DECLARED, provenance(), expected_revision=1
        )
    assert excinfo.value.code == "revision-mismatch"
    reread = GrantLedger(path)
    assert reread.snapshot("hardware-health", DECLARED).active == ()


def test_a_stale_ledger_observes_a_grant_committed_elsewhere(tmp_path):
    path = tmp_path / "grants.json"
    stale = GrantLedger(path)
    GrantLedger(path).grant(
        "hardware-health", READ_SENSOR, DECLARED, provenance(), expected_revision=0
    )
    with pytest.raises(GrantError) as excinfo:
        stale.grant(
            "hardware-health", RUN_ACTION, DECLARED, provenance(), expected_revision=0
        )
    assert excinfo.value.code == "revision-mismatch"
    assert stale.revision == 1


def _race_grant(path: str, ready, results, permission: str) -> None:
    ledger = GrantLedger(Path(path))
    ready.wait(timeout=30)
    try:
        ledger.grant("hardware-health", permission, DECLARED, provenance(), expected_revision=0)
    except GrantError as error:
        results.put(error.code)
    else:
        results.put("ok")


def test_concurrent_grants_cannot_both_commit_the_same_revision(tmp_path):
    path = tmp_path / "grants.json"
    context = multiprocessing.get_context("spawn")
    ready = context.Barrier(2)
    results = context.Queue()
    workers = [
        context.Process(target=_race_grant, args=(str(path), ready, results, permission))
        for permission in (READ_SENSOR, RUN_ACTION)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=60)
    assert sorted(results.get() for _ in workers) == ["ok", "revision-mismatch"]
    assert GrantLedger(path).revision == 1
