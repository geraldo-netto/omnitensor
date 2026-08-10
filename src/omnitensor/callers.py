"""Who asked: resolving the caller behind a bus message, and scoping by it.

``dbus_fast`` hands an exported method its arguments and nothing else — no
sender, no credentials — so ownership could not be enforced end to end no
matter how carefully the stores were written.  The sender is available one
layer down: a message handler sees every incoming message before it is
dispatched, and it runs synchronously in the frame that then schedules the
method, so a value bound there is inherited by the method's task.  Binding is
therefore always explicit, including binding *nothing* when a message carries
no sender, so a stale identity from the previous message can never be read as
the current caller's.

Ownership is scoped by uid rather than by unique bus name.  A unique name lasts
only as long as one connection, so a client that reconnects would be locked out
of the jobs it just submitted — an owner token nobody can present is not a
security boundary, it is a bug that looks like one.

Be clear about what this is worth.  On a session bus the peers are usually all
the same user, so uid scoping separates almost nothing; it earns its keep when
the socket is reachable by another uid, and it never claims isolation it does
not have.  The unique name is carried alongside for audit, where "which
connection did this" is the question being asked.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass

ANONYMOUS_OWNER = "anonymous"
DEFAULT_MAX_CACHED_CALLERS = 512
_UNIQUE_NAME = re.compile(r"^:[0-9]+\.[0-9]+$")
_WELL_KNOWN = re.compile(r"^[A-Za-z_-][A-Za-z0-9_-]*(\.[A-Za-z_-][A-Za-z0-9_-]*)+$")


class CallerError(ValueError):
    """Stable identity failure, safe to report over the bus."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class CallerIdentity:
    """One caller, as the bus daemon describes it."""

    unique_name: str
    uid: int | None = None

    @property
    def owner_token(self) -> str:
        """The value stores scope by; stable across a caller's reconnects."""
        if self.uid is None:
            # Without credentials the connection itself is the most that can
            # honestly be claimed, and it does not survive a reconnect.
            return f"name:{self.unique_name}" if self.unique_name else ANONYMOUS_OWNER
        return f"uid:{self.uid}"

    def document(self) -> dict:
        return {"uniqueName": self.unique_name, "uid": self.uid, "owner": self.owner_token}


ANONYMOUS = CallerIdentity("", None)

# Bound by the transport for the duration of one incoming message.  A method
# runs in a task created after this is set, so it inherits the value; nothing
# the method itself sets can leak back out to the next message.
_CURRENT_SENDER: ContextVar[str] = ContextVar("omnitensor_current_sender", default="")


def bind_sender(sender: object) -> None:
    """Bind the sender of the message being dispatched, or clear it.

    Always assigns, even when there is no sender: leaving the previous value
    in place would attribute one caller's message to another.
    """
    _CURRENT_SENDER.set(sender if isinstance(sender, str) and sender else "")


def current_sender() -> str:
    return _CURRENT_SENDER.get()


def valid_bus_name(name: object) -> bool:
    """Whether ``name`` is a bus name the daemon could have produced."""
    if not isinstance(name, str) or not 1 <= len(name) <= 255:
        return False
    return bool(_UNIQUE_NAME.match(name) or _WELL_KNOWN.match(name))


def caller_capture_handler(
    on_message: Callable[[str], None] | None = None,
) -> Callable[[object], None]:
    """A ``dbus_fast`` message handler that binds the sender and handles nothing.

    It returns ``None`` for every message so normal dispatch continues; a
    handler that swallowed messages here would take the whole service down
    with it.
    """

    def handler(message: object) -> None:
        sender = getattr(message, "sender", None)
        bind_sender(sender if valid_bus_name(sender) else "")
        if on_message is not None:
            on_message(current_sender())
        return None

    return handler


class CallerIdentityResolver:
    """Resolve a sender's uid through the bus daemon, with a bounded cache.

    The lookup is a round trip per call otherwise, on a path that runs for
    every method; the cache is bounded because the key is attacker-influenced —
    a peer can reconnect repeatedly and mint a new unique name each time.
    """

    def __init__(
        self,
        unix_user: Callable[[str], Awaitable[int]] | None = None,
        *,
        max_cached_callers: int = DEFAULT_MAX_CACHED_CALLERS,
    ) -> None:
        if (
            isinstance(max_cached_callers, bool)
            or not isinstance(max_cached_callers, int)
            or max_cached_callers < 1
        ):
            raise CallerError("bounds-invalid", "max_cached_callers must be a positive integer")
        self._unix_user = unix_user
        self._max_cached_callers = max_cached_callers
        self._cache: OrderedDict[str, CallerIdentity] = OrderedDict()

    def attach(self, unix_user: Callable[[str], Awaitable[int]]) -> None:
        """Supply the credential lookup once a bus connection exists.

        The resolver is constructed with the service, before there is a bus to
        ask, so until this is called every caller resolves as anonymous rather
        than as some default identity.
        """
        if not callable(unix_user):
            raise CallerError("lookup-invalid", "unix_user must be callable")
        self._unix_user = unix_user
        # Identities resolved before the lookup existed are all anonymous, and
        # keeping them would outlive the reason they were anonymous.
        self._cache.clear()

    def forget(self, unique_name: str) -> None:
        """Drop a cached identity, for when a name changes owner."""
        self._cache.pop(unique_name, None)

    async def resolve(self, sender: str | None = None) -> CallerIdentity:
        """The identity behind ``sender``, defaulting to the bound one."""
        name = sender if sender is not None else current_sender()
        if not valid_bus_name(name):
            return ANONYMOUS
        cached = self._cache.get(name)
        if cached is not None:
            self._cache.move_to_end(name)
            return cached
        identity = CallerIdentity(name, await self._uid_of(name))
        self._cache[name] = identity
        self._cache.move_to_end(name)
        while len(self._cache) > self._max_cached_callers:
            self._cache.popitem(last=False)
        return identity

    async def owner_token(self, sender: str | None = None) -> str:
        return (await self.resolve(sender)).owner_token

    def cached_owner_token(self, sender: str | None = None) -> str:
        """The owner token available without a round trip.

        A synchronous method cannot ask the daemon for credentials, so it gets
        the uid token when one is already cached and the connection-scoped one
        otherwise.  That is narrower than the uid token, never wider, so a
        caller can never gain reach by arriving through a synchronous method.
        """
        name = sender if sender is not None else current_sender()
        if not valid_bus_name(name):
            return ANONYMOUS_OWNER
        cached = self._cache.get(name)
        return cached.owner_token if cached is not None else CallerIdentity(name).owner_token

    async def _uid_of(self, name: str) -> int | None:
        if self._unix_user is None:
            return None
        try:
            uid = await self._unix_user(name)
        except Exception:  # noqa: BLE001 - an unresolvable caller is anonymous
            # The daemon refuses for a peer that has already disconnected, and
            # a job must not be attributed to whoever asks next.
            return None
        if isinstance(uid, bool) or not isinstance(uid, int) or uid < 0:
            return None
        return uid


def unix_user_lookup(bus) -> Callable[[str], Awaitable[int]]:
    """``GetConnectionUnixUser`` against the bus daemon, for a connected bus."""

    async def lookup(name: str) -> int:
        reply = await bus.call(
            _get_connection_unix_user_message(bus, name)
        )
        if reply is None or not getattr(reply, "body", None):
            raise CallerError("caller-unknown", f"the bus did not describe {name}")
        return int(reply.body[0])

    return lookup


def _get_connection_unix_user_message(bus, name: str):
    from dbus_fast import Message

    return Message(
        destination="org.freedesktop.DBus",
        path="/org/freedesktop/DBus",
        interface="org.freedesktop.DBus",
        member="GetConnectionUnixUser",
        signature="s",
        body=[name],
    )
