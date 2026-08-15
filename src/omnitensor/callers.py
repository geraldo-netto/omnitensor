"""Who asked: the caller behind a control-socket connection, and scoping by it.

The transport reads ``SO_PEERCRED`` when it accepts a connection — the kernel
stamps the peer's uid, no daemon round trip, nothing the peer can forge — and
binds a sender token of the form ``peer:<uid>:<serial>`` for the lifetime of
that connection's requests.  Binding is explicit, including binding *nothing*
when credentials could not be read, so a stale identity from a previous
connection can never be read as the current caller's.

Ownership is scoped by uid rather than by connection.  A connection lasts only
as long as one socket, so a client that reconnects would be locked out of the
jobs it just submitted — an owner token nobody can present is not a security
boundary, it is a bug that looks like one.  The serial exists for audit, where
"which connection did this" is the question being asked; it never widens or
narrows what a caller may touch.

Be clear about what this is worth.  The socket lives in the user's runtime
directory, which is mode 0700, so the peers are normally all the same user and
uid scoping separates almost nothing; it earns its keep if the socket is ever
reachable by another uid, and it never claims isolation it does not have.
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from dataclasses import dataclass

ANONYMOUS_OWNER = "anonymous"
_PEER_SENDER = re.compile(r"^peer:(0|[1-9][0-9]{0,9}):(0|[1-9][0-9]{0,18})$")


@dataclass(frozen=True, slots=True)
class CallerIdentity:
    """One caller, as the kernel described it at accept time."""

    sender: str
    uid: int | None = None

    @property
    def owner_token(self) -> str:
        """The value stores scope by; stable across a caller's reconnects."""
        return ANONYMOUS_OWNER if self.uid is None else f"uid:{self.uid}"

    def document(self) -> dict:
        return {"sender": self.sender, "uid": self.uid, "owner": self.owner_token}


ANONYMOUS = CallerIdentity("", None)

# Bound by the transport for the duration of one connection.  Each connection
# is served in its own task, so requests on it — and the tasks they spawn —
# inherit the value; nothing a method sets can leak to another connection.
_CURRENT_SENDER: ContextVar[str] = ContextVar("omnitensor_current_sender", default="")


def bind_sender(sender: object) -> None:
    """Bind the sender of the connection being served, or clear it.

    Always assigns, even when there is no sender: leaving the previous value
    in place would attribute one caller's requests to another.
    """
    _CURRENT_SENDER.set(sender if isinstance(sender, str) and sender else "")


def current_sender() -> str:
    return _CURRENT_SENDER.get()


def valid_sender(name: object) -> bool:
    """Whether ``name`` is a token the transport could have minted."""
    return isinstance(name, str) and bool(_PEER_SENDER.match(name))


class CallerIdentityResolver:
    """Resolve a sender token to the identity it encodes.

    The uid travels inside the token, stamped by the transport from
    ``SO_PEERCRED``, so resolution is a parse: no round trip, no cache, and
    therefore no cache to poison.  A token this module would not itself mint
    resolves as anonymous rather than as some default identity.
    """

    def resolve(self, sender: str | None = None) -> CallerIdentity:
        """The identity behind ``sender``, defaulting to the bound one."""
        name = sender if sender is not None else current_sender()
        match = _PEER_SENDER.match(name) if isinstance(name, str) else None
        if match is None:
            return ANONYMOUS
        return CallerIdentity(name, int(match.group(1)))

    def owner_token(self, sender: str | None = None) -> str:
        return self.resolve(sender).owner_token
