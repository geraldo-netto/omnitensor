from __future__ import annotations

import asyncio
import json

import pytest

from omnitensor.plugins.collection import CollectionError, ReplaySource, SourceSnapshot
from omnitensor.plugins.desktop_collection import (
    DESKTOP_CONSENT_PERMISSION,
    DESKTOP_METADATA_PERMISSION,
    DESKTOP_PLUGIN_ID,
    DesktopContextCollector,
    DesktopSession,
    WindowRole,
    WindowSample,
    WindowState,
    window_identity,
    window_sample_error,
)
from omnitensor.plugins.triggers import SourceStatus, Trigger, TriggerKind
from omnitensor.sdk.helpers import PermissionView

SESSION = DesktopSession("session-a", 4)
WINDOW = window_identity("session-a", "0x1001")


def permissions(*, consent=True, metadata=True):
    declared = frozenset({DESKTOP_CONSENT_PERMISSION, DESKTOP_METADATA_PERMISSION})
    granted = frozenset(
        {
            *({DESKTOP_CONSENT_PERMISSION} if consent else set()),
            *({DESKTOP_METADATA_PERMISSION} if metadata else set()),
        }
    )
    return PermissionView(declared, granted)


def sample(stable_id=WINDOW, **changes):
    values = {
        "stable_id": stable_id,
        "application_id": "org.gnome.TextEditor",
        "role": WindowRole.NORMAL,
        "state": WindowState.FOCUSED,
        "workspace": 1,
        "width": 1280,
        "height": 720,
        "observed_at_ms": 9,
    }
    values.update(changes)
    return WindowSample(**values)


def snapshot(*samples):
    return SourceSnapshot(SourceStatus.READY, 10, tuple(samples) or (sample(),))


def collector(*snapshots, session=SESSION, **changes):
    source = ReplaySource(list(snapshots) or [snapshot()], label="desktop")
    return DesktopContextCollector(source, permissions(**changes), session, ())


def trigger(plugin_id=DESKTOP_PLUGIN_ID):
    return Trigger(plugin_id, "desktop-1", TriggerKind.EVENT, {}, 1)


def collect(subject):
    return asyncio.run(subject.collect(trigger())).payload


def test_window_geometry_and_role_are_emitted():
    payload = collect(collector())
    [item] = payload["items"]
    assert item["role"] == "normal"
    assert item["width"] == 1280
    assert item["workspace"] == 1


def test_no_window_content_or_title_can_be_emitted():
    """Titles carry document names and ticket numbers; none of it leaves here."""
    payload = collect(collector())
    encoded = json.dumps(payload)

    assert "title" not in encoded
    assert set(payload["items"][0]) == {
        "id",
        "applicationId",
        "role",
        "state",
        "workspace",
        "width",
        "height",
        "observedAtMs",
    }


def test_a_source_that_grew_a_content_field_is_refused_not_redacted():
    class Titled(WindowSample):
        title = "Quarterly results - private"

    bad = Titled(WINDOW, "app", WindowRole.NORMAL, WindowState.FOCUSED, 1, 100, 100, 9)
    assert "must not carry title" in window_sample_error(bad)


def test_collection_without_consent_is_refused_even_with_the_metadata_grant():
    subject = DesktopContextCollector(
        ReplaySource([snapshot()], label="desktop"),
        permissions(consent=False),
        SESSION,
        (),
    )
    with pytest.raises(CollectionError, match="consent-missing"):
        collect(subject)


def test_collection_without_the_metadata_grant_is_refused():
    subject = DesktopContextCollector(
        ReplaySource([snapshot()], label="desktop"),
        permissions(metadata=False),
        SESSION,
        (),
    )
    with pytest.raises(CollectionError, match="permission-denied"):
        collect(subject)


def test_a_trigger_for_another_plugin_is_refused():
    with pytest.raises(CollectionError) as caught:
        asyncio.run(collector().collect(trigger("other-plugin")))

    assert caught.value.code == "trigger-invalid"
    assert caught.value.detail == ("desktop context metadata requires a desktop-context trigger")


def test_identities_are_per_session_so_they_cannot_be_correlated():
    """The same native window in two sessions must not share an identity."""
    assert window_identity("session-a", "0x1001") != window_identity("session-b", "0x1001")
    assert window_identity("session-a", "0x1001") == window_identity("session-a", "0x1001")
    assert "0x1001" not in window_identity("session-a", "0x1001")


def test_a_session_change_discards_the_previous_desktop():
    """Carrying state across a login attributes one person's desktop to another."""
    subject = collector(snapshot(sample()), snapshot(sample()))
    asyncio.run(subject.collect(trigger()))

    subject.session_changed(DesktopSession("session-b", 2))
    payload = asyncio.run(subject.collect(trigger())).payload

    assert subject.session.session_id == "session-b"
    assert [item["id"] for item in payload["churn"]["added"]] == [WINDOW]


def test_a_session_must_be_a_desktop_session():
    with pytest.raises(TypeError, match="DesktopSession"):
        collector().session_changed("session-b")
    with pytest.raises(TypeError, match="DesktopSession"):
        DesktopContextCollector(ReplaySource([snapshot()]), permissions(), "session-a", ())


def test_a_window_moving_workspace_is_reported_as_a_change():
    subject = collector(snapshot(sample(workspace=1)), snapshot(sample(workspace=3)))
    asyncio.run(subject.collect(trigger()))
    second = asyncio.run(subject.collect(trigger())).payload
    assert second["churn"]["changed"] == [{"id": WINDOW, "fields": ["workspace"]}]


def test_a_closed_window_is_reported_as_removed():
    other = window_identity("session-a", "0x1002")
    subject = collector(snapshot(sample(), sample(other)), snapshot(sample()))
    asyncio.run(subject.collect(trigger()))
    second = asyncio.run(subject.collect(trigger())).payload
    assert [item["id"] for item in second["churn"]["removed"]] == [other]


@pytest.mark.parametrize(
    "changes",
    [
        {"role": "normal"},
        {"state": "focused"},
        {"application_id": ""},
        {"workspace": -1},
        {"width": 70_000},
        {"height": True},
    ],
)
def test_a_malformed_sample_is_refused(changes):
    assert window_sample_error(sample(**changes)) != ""


def test_emission_is_bounded():
    samples = [sample(window_identity("session-a", f"0x{index}")) for index in range(5)]
    subject = DesktopContextCollector(
        ReplaySource([snapshot(*samples)], label="desktop"),
        permissions(),
        SESSION,
        (),
        max_items=2,
    )
    payload = collect(subject)
    assert len(payload["items"]) == 2
    assert payload["truncatedItems"] == 3


def test_withdrawn_consent_is_visible_in_readiness_not_only_at_collection():
    """OMNI-0429: READY was answered by a collector that refused every collect."""
    subject = collector(consent=False)

    readiness = asyncio.run(subject.readiness())

    assert readiness.status is SourceStatus.UNAVAILABLE
    assert readiness.detail == "desktop context consent has not been given"
    with pytest.raises(CollectionError) as refused:
        collect(subject)
    assert refused.value.code == "consent-missing"

    # The metadata grant is still reported in its own words.
    without_metadata = collector(metadata=False)
    assert asyncio.run(without_metadata.readiness()).detail == (
        "desktop context metadata permission is not granted"
    )


def test_one_shared_sample_check_serves_every_collector_family():
    """OMNI-0428: four profiles carried the identical validation loop."""
    from omnitensor.plugins.collection import BoundedCollector
    from omnitensor.plugins.hardware_collection import HardwareHealthCollector
    from omnitensor.plugins.resource_collection import ResourceSchedulerCollector
    from omnitensor.plugins.storage_collection import StorageIntelligenceCollector

    families = (
        DesktopContextCollector,
        HardwareHealthCollector,
        ResourceSchedulerCollector,
        StorageIntelligenceCollector,
    )
    for family in families:
        # Declared as data, not re-implemented: none of them overrides the loop.
        assert family.sample_error is not BoundedCollector.sample_error
        assert "_validate_snapshot" not in vars(family)

    subject = collector(SourceSnapshot(SourceStatus.READY, 10, ("not a window sample",)))
    with pytest.raises(CollectionError) as refused:
        collect(subject)
    assert refused.value.code == "source-invalid"
