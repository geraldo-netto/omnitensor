"""Kernel device-change notifications over ``NETLINK_KOBJECT_UEVENT``.

Discovery used to answer "did the hardware change?" by rereading sysfs on a
timer.  On a machine whose hardware never changes that is the whole cost: the
answer is "no" every time, and the runtime paid for asking.  The kernel already
broadcasts the event, so this module listens for it instead.

Nothing here is a dependency: a raw ``AF_NETLINK`` datagram socket is in the
standard library, and the alternative (``pyudev``) would pull libudev into a
service that needs one bit of information from it.

The socket is deliberately the only thing this module knows about.  Consumers
see the :class:`~omnitensor.ports.DeviceChangeSource` protocol; the service
never learns that netlink exists, and a host that cannot open the socket
degrades to polling, loudly, rather than silently.
"""

from __future__ import annotations

import asyncio
import logging
import socket
from collections.abc import Callable, Collection

LOGGER = logging.getLogger(__name__)

NETLINK_KOBJECT_UEVENT = 15
# Group 1 is the kernel's own uevent broadcast; group 2 is the copy
# systemd-udevd re-sends once it has applied its rules (and the one an
# unprivileged process is reliably allowed to join).  Both are subscribed
# where the kernel permits it, because a host without udev running still has
# group 1 and a hardened host may refuse it.
KERNEL_UEVENT_GROUP = 1
UDEV_UEVENT_GROUP = 2
UEVENT_GROUPS = KERNEL_UEVENT_GROUP | UDEV_UEVENT_GROUP

# The subsystems :mod:`omnitensor.discovery` actually reads: DRM render nodes
# for GPUs, the accel class for NPUs, and PCIe/USB for Coral TPUs.  Everything
# else the kernel broadcasts — input devices, block devices, network links —
# would wake a rediscovery that could not possibly find anything new.
WATCHED_SUBSYSTEMS = frozenset({"accel", "apex", "drm", "pci", "usb"})

# One netlink datagram. This is the protocol's frame, not a ceiling on an
# answer: a uevent that did not fit in it was never sent in the first place.
UEVENT_FRAME_BYTES = 65536

_SUBSYSTEM_KEY = b"SUBSYSTEM="


def open_uevent_socket(groups: int = UEVENT_GROUPS) -> socket.socket:
    """Bind a non-blocking uevent netlink socket to ``groups``.

    Raises ``OSError`` when the family is unavailable or the kernel refuses
    the multicast group, which is the caller's signal to fall back.
    """
    family = getattr(socket, "AF_NETLINK", None)
    if family is None:  # pragma: no cover - Linux-only service
        raise OSError("AF_NETLINK is not available on this platform")
    sock = socket.socket(family, socket.SOCK_DGRAM | socket.SOCK_CLOEXEC, NETLINK_KOBJECT_UEVENT)
    try:
        sock.setblocking(False)
        sock.bind((0, groups))
    except OSError:
        sock.close()
        raise
    return sock


class UeventDeviceChanges:
    """Wake when the kernel says an accelerator appeared or went away."""

    def __init__(
        self,
        sock: socket.socket,
        *,
        subsystems: Collection[str] = WATCHED_SUBSYSTEMS,
        frame_bytes: int = UEVENT_FRAME_BYTES,
    ) -> None:
        self._socket = sock
        self._subsystems = frozenset(subsystems)
        self._frame_bytes = frame_bytes
        self._closed = False
        # The waiter and the fd it is registered against live on the instance
        # so that :meth:`aclose` can unregister and resolve them.  Closing the
        # socket alone drops the fd from the selector without ever waking the
        # reader, which would leave ``wait_for_change`` awaiting forever.
        self._waiter: asyncio.Future | None = None
        self._reader: tuple[asyncio.AbstractEventLoop, int] | None = None

    async def wait_for_change(self) -> bool:
        """Sleep on the socket until a watched subsystem reports an event.

        Returns ``False`` once the event source is gone, which tells the
        caller it must fall back to polling; there is no other way for a
        reader to learn that the kernel stopped talking to it.
        """
        loop = asyncio.get_running_loop()
        while not self._closed:
            try:
                fileno = self._socket.fileno()
            except OSError:
                return False
            if fileno < 0:
                return False
            waiter: asyncio.Future = loop.create_future()
            try:
                loop.add_reader(fileno, _wake, waiter)
            except (OSError, ValueError):
                self._closed = True
                return False
            self._waiter = waiter
            self._reader = (loop, fileno)
            try:
                await waiter
            finally:
                self._unregister_reader()
                self._waiter = None
            if self._closed:
                # ``aclose`` woke us; the socket is gone, so do not read it.
                return False
            relevant, alive = self._drain()
            if not alive:
                self._closed = True
                return False
            if relevant:
                return True
        return False

    def _drain(self) -> tuple[bool, bool]:
        """Read every pending datagram, coalescing a burst into one answer."""
        relevant = False
        while True:
            try:
                payload = self._socket.recv(self._frame_bytes, socket.MSG_DONTWAIT)
            except BlockingIOError:
                return relevant, True
            except OSError:
                return relevant, False
            if not payload:
                # A closed peer reports readable forever; treat it as the end
                # of the event source rather than spinning on it.
                return relevant, False
            relevant = relevant or self.concerns_devices(payload)

    def concerns_devices(self, payload: bytes) -> bool:
        """Whether one uevent names a subsystem discovery reads.

        Both wire formats are NUL-separated ``KEY=VALUE`` runs — the udev copy
        simply prefixes a binary header — so one scan reads either.
        """
        return any(
            field[len(_SUBSYSTEM_KEY) :].decode("utf-8", "replace").strip() in self._subsystems
            for field in payload.split(b"\0")
            if field.startswith(_SUBSYSTEM_KEY)
        )

    def _unregister_reader(self) -> None:
        """Drop the selector registration while the fd is still valid."""
        reader, self._reader = self._reader, None
        if reader is None:
            return
        loop, fileno = reader
        try:
            loop.remove_reader(fileno)
        except (OSError, ValueError, RuntimeError):  # pragma: no cover - loop already gone
            LOGGER.debug("uevent reader removal failed", exc_info=True)

    async def aclose(self) -> None:
        """Release the socket; safe to call more than once.

        The reader is unregistered before the fd is closed — afterwards the
        number may already name a different file — and any pending waiter is
        resolved so an in-flight :meth:`wait_for_change` returns instead of
        awaiting a socket nobody will ever read again.
        """
        self._closed = True
        self._unregister_reader()
        waiter, self._waiter = self._waiter, None
        if waiter is not None:
            _wake(waiter)
        try:
            self._socket.close()
        except OSError:  # pragma: no cover - close rarely fails
            LOGGER.debug("uevent socket close failed", exc_info=True)


def _wake(waiter: asyncio.Future) -> None:
    if not waiter.done():
        waiter.set_result(None)


def open_device_changes(
    *,
    open_socket: Callable[[int], socket.socket] = open_uevent_socket,
    subsystems: Collection[str] = WATCHED_SUBSYSTEMS,
) -> UeventDeviceChanges | None:
    """Open the kernel event source, or return ``None`` and say why.

    Both group masks are tried because binding group 1 needs a privilege a
    user service may not have, while group 2 alone still carries every event
    udev has processed.
    """
    failure: OSError | None = None
    for groups in (UEVENT_GROUPS, UDEV_UEVENT_GROUP):
        try:
            sock = open_socket(groups)
        except OSError as error:
            failure = error
            continue
        return UeventDeviceChanges(sock, subsystems=subsystems)
    LOGGER.warning(
        "Could not subscribe to kernel device events (%s); device discovery must poll instead",
        failure,
    )
    return None


__all__ = [
    "KERNEL_UEVENT_GROUP",
    "NETLINK_KOBJECT_UEVENT",
    "UDEV_UEVENT_GROUP",
    "UEVENT_FRAME_BYTES",
    "UEVENT_GROUPS",
    "WATCHED_SUBSYSTEMS",
    "UeventDeviceChanges",
    "open_device_changes",
    "open_uevent_socket",
]
