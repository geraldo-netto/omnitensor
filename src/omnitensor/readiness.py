"""Tell the process manager when this service can actually answer.

Without this, everything downstream guesses. `omnitensor-verify-install` run
straight after `systemctl --user restart` polled the control socket blind for
its whole thirty-second budget, because `systemctl` returns as soon as the
process is spawned and nothing said when the socket began accepting. A service
that reports readiness turns that poll into a single successful call.

The protocol is systemd's `sd_notify`: one datagram to the path in
`$NOTIFY_SOCKET`, whose first character may be `@` (or a NUL) for an abstract
socket. It is deliberately best-effort — a service started by hand has no
`NOTIFY_SOCKET` at all, and that is not an error.
"""

from __future__ import annotations

import logging
import os
import socket

LOGGER = logging.getLogger(__name__)

READY = "READY=1"
STOPPING = "STOPPING=1"


def notify(state: str, *, environ=None) -> bool:
    """Send one notification. ``False`` means nobody was listening."""
    if not isinstance(state, str) or not state:
        raise ValueError("notification state must be a non-empty string")
    address = (environ if environ is not None else os.environ).get("NOTIFY_SOCKET", "")
    if not address:
        return False
    # An abstract socket is named with a leading NUL; systemd writes it as '@'.
    path = "\0" + address[1:] if address[0] in "@\0" else address
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM | socket.SOCK_CLOEXEC) as sender:
            sender.connect(path)
            sender.sendall(state.encode("utf-8"))
    except OSError as error:
        # Readiness is a courtesy to whatever started us; failing to deliver it
        # must never stop a service that is, in fact, ready.
        LOGGER.debug("Could not notify %r: %s", state, error)
        return False
    return True


def notify_ready(*, environ=None) -> bool:
    """Say the control socket is accepting and the schedulers are running."""
    return notify(READY, environ=environ)


def notify_stopping(*, environ=None) -> bool:
    """Say the service has begun shutting down."""
    return notify(STOPPING, environ=environ)


__all__ = ["READY", "STOPPING", "notify", "notify_ready", "notify_stopping"]
