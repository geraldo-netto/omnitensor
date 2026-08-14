from __future__ import annotations

import asyncio
import threading

import pytest

from omnitensor.plugins import (
    MAX_REVOCATION_POLL_SECONDS,
    ConsentGuard,
    GrantLedger,
    GrantOrigin,
    GrantProvenance,
    LiveGrantView,
    RevocationError,
)

READ_SENSOR = "read:/sys/class/hwmon/*"
RUN_ACTION = "action:notify-user"
DECLARED = {READ_SENSOR, RUN_ACTION}
PLUGIN = "hardware-health"


def provenance(reason="enabled in settings"):
    return GrantProvenance("user:1000", GrantOrigin.USER, reason, 10, "request-1")


def grant(ledger, permission=READ_SENSOR):
    return ledger.grant(
        PLUGIN, permission, DECLARED, provenance(), expected_revision=ledger.revision
    )


def revoke(ledger, permission=READ_SENSOR):
    return ledger.revoke(
        PLUGIN, permission, DECLARED, provenance("withdrawn"), expected_revision=ledger.revision
    )


def view(path):
    return LiveGrantView(GrantLedger(path), PLUGIN, DECLARED)


def test_a_view_fails_closed_before_anything_is_granted(tmp_path):
    gate = view(tmp_path / "grants.json")
    assert gate.allows(READ_SENSOR) is False
    assert gate.granted == frozenset()
    with pytest.raises(RevocationError) as excinfo:
        gate.require(READ_SENSOR)
    assert excinfo.value.code == "permission-denied"


def test_a_view_observes_a_grant_committed_by_another_process(tmp_path):
    path = tmp_path / "grants.json"
    gate = view(path)
    assert gate.allows(READ_SENSOR) is False

    grant(GrantLedger(path))

    assert gate.allows(READ_SENSOR) is True
    assert gate.granted == frozenset({READ_SENSOR})


def test_a_view_observes_a_revocation_committed_by_another_process(tmp_path):
    """A snapshot captured at start could never see the withdrawal."""
    path = tmp_path / "grants.json"
    grant(GrantLedger(path))
    gate = view(path)
    assert gate.allows(READ_SENSOR) is True

    revoke(GrantLedger(path))

    assert gate.allows(READ_SENSOR) is False
    with pytest.raises(RevocationError):
        gate.require(READ_SENSOR)


def test_an_undeclared_permission_is_never_allowed(tmp_path):
    path = tmp_path / "grants.json"
    gate = LiveGrantView(GrantLedger(path), PLUGIN, {READ_SENSOR})
    assert gate.allows(RUN_ACTION) is False
    assert gate.declared == frozenset({READ_SENSOR})


def test_a_view_requires_a_real_ledger():
    with pytest.raises(TypeError, match="GrantLedger"):
        LiveGrantView(object(), PLUGIN, DECLARED)


def test_guarded_work_runs_to_completion_while_consent_holds(tmp_path):
    path = tmp_path / "grants.json"
    grant(GrantLedger(path))
    guard = ConsentGuard(view(path), poll_seconds=0.01)

    async def work():
        await asyncio.sleep(0.02)
        return "collected"

    assert asyncio.run(guard.run(work, {READ_SENSOR})) == "collected"


def test_guard_refreshes_grants_away_from_the_event_loop(monkeypatch, tmp_path):
    path = tmp_path / "grants.json"
    grant(GrantLedger(path))
    guard = ConsentGuard(view(path), poll_seconds=0.005)
    event_loop_thread = threading.get_ident()
    refresh_threads = []
    revoked = guard.revoked

    def record_refresh(permissions):
        refresh_threads.append(threading.get_ident())
        return revoked(permissions)

    monkeypatch.setattr(guard, "revoked", record_refresh)

    async def work():
        await asyncio.sleep(0.02)
        return "collected"

    assert asyncio.run(guard.run(work, {READ_SENSOR})) == "collected"
    assert len(refresh_threads) >= 2, "the polling refresh did not run"
    assert all(thread != event_loop_thread for thread in refresh_threads)


def test_guarded_work_is_refused_when_consent_was_never_given(tmp_path):
    guard = ConsentGuard(view(tmp_path / "grants.json"), poll_seconds=0.01)
    started = []

    async def work():
        started.append(1)

    with pytest.raises(RevocationError) as excinfo:
        asyncio.run(guard.run(work, {READ_SENSOR}))

    assert excinfo.value.code == "permission-denied"
    assert started == [], "work started without consent"


def test_guarded_work_is_cancelled_promptly_when_consent_is_revoked(tmp_path):
    """In-flight collection used to run to completion on withdrawn consent."""
    path = tmp_path / "grants.json"
    grant(GrantLedger(path))
    guard = ConsentGuard(view(path), poll_seconds=0.01)
    finished = []
    cancelled = []

    async def scenario():
        async def work():
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                cancelled.append(1)
                raise
            finished.append(1)

        async def withdraw():
            await asyncio.sleep(0.05)
            revoke(GrantLedger(path))

        withdrawal = asyncio.create_task(withdraw())
        try:
            await guard.run(work, {READ_SENSOR})
        finally:
            await withdrawal

    with pytest.raises(RevocationError) as excinfo:
        asyncio.run(asyncio.wait_for(scenario(), timeout=10))

    assert excinfo.value.code == "consent-revoked"
    assert excinfo.value.detail == f"stopped after {READ_SENSOR} was revoked"
    assert cancelled == [1]
    assert finished == [], "work completed after consent was withdrawn"


def test_revoking_an_unrelated_permission_does_not_stop_the_work(tmp_path):
    path = tmp_path / "grants.json"
    ledger = GrantLedger(path)
    grant(ledger)
    grant(ledger, RUN_ACTION)
    guard = ConsentGuard(view(path), poll_seconds=0.01)

    async def scenario():
        async def work():
            await asyncio.sleep(0.05)
            return "done"

        revoke(GrantLedger(path), RUN_ACTION)
        return await guard.run(work, {READ_SENSOR})

    assert asyncio.run(asyncio.wait_for(scenario(), timeout=10)) == "done"


def test_a_failing_operation_reports_its_own_error(tmp_path):
    path = tmp_path / "grants.json"
    grant(GrantLedger(path))
    guard = ConsentGuard(view(path), poll_seconds=0.01)

    async def work():
        raise ValueError("plugin defect")

    with pytest.raises(ValueError, match="plugin defect"):
        asyncio.run(guard.run(work, {READ_SENSOR}))


def test_cancelling_the_caller_cancels_the_guarded_work(tmp_path):
    path = tmp_path / "grants.json"
    grant(GrantLedger(path))
    guard = ConsentGuard(view(path), poll_seconds=0.01)
    cancelled = []

    async def scenario():
        async def work():
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                cancelled.append(1)
                raise

        guarded = asyncio.create_task(guard.run(work, {READ_SENSOR}))
        await asyncio.sleep(0.03)
        guarded.cancel()
        await asyncio.gather(guarded, return_exceptions=True)

    asyncio.run(asyncio.wait_for(scenario(), timeout=10))

    assert cancelled == [1], "the guarded operation outlived its caller"


def test_revoked_reports_every_missing_permission_in_stable_order(tmp_path):
    path = tmp_path / "grants.json"
    grant(GrantLedger(path))
    guard = ConsentGuard(view(path), poll_seconds=0.01)
    assert guard.revoked({READ_SENSOR, RUN_ACTION}) == (RUN_ACTION,)
    assert guard.revoked({READ_SENSOR}) == ()


@pytest.mark.parametrize(
    "poll_seconds", [0, -1, True, MAX_REVOCATION_POLL_SECONDS + 1, "1"]
)
def test_the_poll_interval_is_bounded(tmp_path, poll_seconds):
    with pytest.raises(ValueError, match="poll_seconds"):
        ConsentGuard(view(tmp_path / "grants.json"), poll_seconds=poll_seconds)


def test_the_operation_must_be_callable(tmp_path):
    path = tmp_path / "grants.json"
    grant(GrantLedger(path))
    guard = ConsentGuard(view(path), poll_seconds=0.01)
    with pytest.raises(TypeError, match="callable"):
        asyncio.run(guard.run("not-callable", {READ_SENSOR}))
