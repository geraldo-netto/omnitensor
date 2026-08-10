"""Consented desktop context: where windows are, never what is in them.

This is the most invasive thing the service can be asked to observe, so the
rules are the strictest.  Window *content* is never collected and there is no
field that could carry it.  Titles are the one risky field — people put
document names, ticket numbers, and account names in them — so a title is
reduced to a coarse category before it is emitted, and the raw string never
leaves this module.

Consent is per session and revocable, and a session change invalidates it: a
grant given on one login is not a grant for the next, because the person at the
keyboard may not be the same one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from .collection import BoundedCollector, CollectionError, SourceSnapshot

DESKTOP_PLUGIN_ID = "desktop-context"
DESKTOP_CONSENT_PERMISSION = "consent:desktop-context"
DESKTOP_METADATA_PERMISSION = "read:desktop-context-metadata"
DEFAULT_MAX_WINDOWS = 64
MAX_WORKSPACES = 64
MAX_DIMENSION = 65_536


class WindowRole(StrEnum):
    """A coarse category standing in for the title, never the title itself."""

    NORMAL = "normal"
    DIALOG = "dialog"
    UTILITY = "utility"
    NOTIFICATION = "notification"
    UNKNOWN = "unknown"


class WindowState(StrEnum):
    FOCUSED = "focused"
    VISIBLE = "visible"
    MINIMISED = "minimised"
    HIDDEN = "hidden"


@dataclass(frozen=True, slots=True)
class WindowSample:
    """One window's geometry and role — deliberately no title, no content."""

    stable_id: str
    application_id: str
    role: WindowRole
    state: WindowState
    workspace: int
    width: int
    height: int
    observed_at_ms: int


@dataclass(frozen=True, slots=True)
class DesktopSession:
    """The session a snapshot belongs to; consent does not cross sessions."""

    session_id: str
    workspace_count: int


def window_identity(session_id: str, native_id: str) -> str:
    """A per-session opaque identity, so ids cannot be correlated across logins."""
    digest = hashlib.sha256(f"{session_id}\x00{native_id}".encode()).hexdigest()
    return f"win-{digest[:32]}"


def window_sample_error(sample: object) -> str:
    """One stable validation failure for an untrusted source sample."""
    if not isinstance(sample, WindowSample):
        return "window sample has an invalid type"
    if not isinstance(sample.role, WindowRole):
        return "window sample role is invalid"
    if not isinstance(sample.state, WindowState):
        return "window sample state is invalid"
    if not isinstance(sample.application_id, str) or not 1 <= len(sample.application_id) <= 120:
        return "window application identity is invalid"
    geometry = _geometry_error(sample)
    if geometry:
        return geometry
    for attribute in ("title", "text", "content", "screenshot"):
        if hasattr(sample, attribute):
            # A source that grew a content field is a contract break, not a
            # field to redact: refusing is the only safe response.
            return f"window sample must not carry {attribute}"
    return ""


def _geometry_error(sample: WindowSample) -> str:
    for value, name, maximum in (
        (sample.workspace, "workspace", MAX_WORKSPACES),
        (sample.width, "width", MAX_DIMENSION),
        (sample.height, "height", MAX_DIMENSION),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
            return f"window {name} is out of range"
    return ""


class DesktopContextCollector(BoundedCollector[WindowSample]):
    """Emit consented, content-free window metadata for one session."""

    metadata_permission = DESKTOP_METADATA_PERMISSION
    label = "desktop context metadata"
    source_name = "cinnamon-window-metadata"

    def __init__(
        self,
        source,
        permissions,
        session: DesktopSession,
        allowed_ids=(),
        *,
        max_items: int = DEFAULT_MAX_WINDOWS,
        **changes,
    ) -> None:
        if not isinstance(session, DesktopSession):
            raise TypeError("session must be a DesktopSession")
        super().__init__(source, permissions, allowed_ids, max_items=max_items, **changes)
        self._session = session

    @property
    def session(self) -> DesktopSession:
        return self._session

    def identity_of(self, item: WindowSample) -> str:
        return item.stable_id

    def document_of(self, item: WindowSample) -> dict:
        return {
            "id": item.stable_id,
            "applicationId": item.application_id,
            "role": str(item.role),
            "state": str(item.state),
            "workspace": item.workspace,
            "width": item.width,
            "height": item.height,
            "observedAtMs": item.observed_at_ms,
        }

    def changed_fields(self, previous: WindowSample, current: WindowSample) -> list[str]:
        fields = (
            ("applicationId", previous.application_id, current.application_id),
            ("role", previous.role, current.role),
            ("state", previous.state, current.state),
            ("workspace", previous.workspace, current.workspace),
            ("width", previous.width, current.width),
            ("height", previous.height, current.height),
        )
        return [name for name, before, after in fields if before != after]

    async def collect(self, trigger):
        # Consent is checked separately from the metadata grant: a user can
        # keep the plugin installed and withdraw only the desktop consent.
        if not self._permissions.allows(DESKTOP_CONSENT_PERMISSION):
            raise CollectionError(
                "consent-missing", "desktop context consent has not been given"
            )
        return await super().collect(trigger)

    def session_changed(self, session: DesktopSession) -> None:
        """Adopt a new session, discarding everything the old one produced.

        Carrying window state across a session boundary would attribute one
        person's desktop to whoever logs in next.
        """
        if not isinstance(session, DesktopSession):
            raise TypeError("session must be a DesktopSession")
        self._session = session
        self._previous = {}

    def _validate_snapshot(self, snapshot: object) -> None:
        if isinstance(snapshot, SourceSnapshot):
            for sample in snapshot.items:
                error = window_sample_error(sample)
                if error:
                    raise CollectionError("source-invalid", error)
        super()._validate_snapshot(snapshot)
