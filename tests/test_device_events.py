"""Event-driven device discovery: the netlink adapter, the host wrapper, and
the service loop that must stop polling an idle machine (OMNI-0478)."""

from __future__ import annotations

import asyncio
import errno
import logging
import socket

import pytest

from omnitensor import device_events
from omnitensor.device_events import (
    UDEV_UEVENT_GROUP,
    UEVENT_GROUPS,
    UeventDeviceChanges,
    open_device_changes,
    open_uevent_socket,
)
from omnitensor.discovery import Device
from omnitensor.host import EventDrivenDeviceDiscovery, SysfsDeviceDiscovery, build_host_ports
from omnitensor.plugins.loading import InstalledPluginRuntime
from omnitensor.ports import DeviceChangeSource
from omnitensor.service import OmniTensorService


def uevent(action: str = "add", subsystem: str = "drm", devpath: str = "/devices/pci/drm/card0"):
    """One kernel-format uevent datagram: header line, then NUL-separated keys."""
    fields = [
        f"{action}@{devpath}".encode(),
        b"ACTION=" + action.encode(),
        b"DEVPATH=" + devpath.encode(),
        b"SUBSYSTEM=" + subsystem.encode(),
    ]
    return b"\0".join(fields) + b"\0"


def udev_uevent(subsystem: str = "drm"):
    """The udev re-broadcast: the same keys behind a binary magic header."""
    return b"libudev\0\xfe\xed\xca\xfe" + uevent(subsystem=subsystem)


@pytest.fixture
def socket_pair():
    # Datagrams, as netlink delivers them: one uevent per recv.
    reader, writer = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    reader.setblocking(False)
    try:
        yield reader, writer
    finally:
        reader.close()
        writer.close()


# --------------------------------------------------------------------------
# The netlink adapter


async def test_a_watched_subsystem_event_wakes_the_waiter(socket_pair):
    reader, writer = socket_pair
    changes = UeventDeviceChanges(reader)
    waiting = asyncio.ensure_future(changes.wait_for_change())
    await asyncio.sleep(0)

    writer.send(uevent(subsystem="drm"))

    assert await asyncio.wait_for(waiting, timeout=5) is True


async def test_an_unwatched_subsystem_never_wakes_a_rediscovery(socket_pair):
    reader, writer = socket_pair
    changes = UeventDeviceChanges(reader)
    waiting = asyncio.ensure_future(changes.wait_for_change())
    await asyncio.sleep(0)

    writer.send(uevent(subsystem="block"))
    writer.send(uevent(subsystem="input"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert not waiting.done()

    writer.send(uevent(subsystem="accel"))
    assert await asyncio.wait_for(waiting, timeout=5) is True


async def test_the_udev_rebroadcast_format_is_read_too(socket_pair):
    reader, writer = socket_pair
    changes = UeventDeviceChanges(reader)
    waiting = asyncio.ensure_future(changes.wait_for_change())
    await asyncio.sleep(0)

    writer.send(udev_uevent(subsystem="usb"))

    assert await asyncio.wait_for(waiting, timeout=5) is True


async def test_a_burst_of_events_is_coalesced_into_one_answer(socket_pair):
    reader, writer = socket_pair
    changes = UeventDeviceChanges(reader)
    for _ in range(8):
        writer.send(uevent(subsystem="drm"))

    assert await asyncio.wait_for(changes.wait_for_change(), timeout=5) is True
    # Every datagram was drained, so the next wait has nothing left to read.
    pending = asyncio.ensure_future(changes.wait_for_change())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not pending.done()
    pending.cancel()
    await asyncio.gather(pending, return_exceptions=True)


async def test_an_event_source_at_end_of_stream_reports_that_polling_must_resume(socket_pair):
    """A readable socket that yields nothing is a source that has gone away."""
    reader, writer = socket_pair

    class AtEnd:
        def __init__(self, sock):
            self._sock = sock

        def fileno(self):
            return self._sock.fileno()

        def recv(self, *_args, **_kwargs):
            return b""

        def close(self):
            return None

    changes = UeventDeviceChanges(AtEnd(reader))
    waiting = asyncio.ensure_future(changes.wait_for_change())
    await asyncio.sleep(0)
    writer.send(uevent())

    assert await asyncio.wait_for(waiting, timeout=5) is False


async def test_waiting_after_aclose_reports_no_event_source(socket_pair):
    reader, _writer = socket_pair
    changes = UeventDeviceChanges(reader)
    await changes.aclose()
    await changes.aclose()

    assert await changes.wait_for_change() is False


async def test_aclose_wakes_an_in_flight_wait_instead_of_stranding_it(socket_pair):
    reader, _writer = socket_pair
    changes = UeventDeviceChanges(reader)
    waiting = asyncio.ensure_future(changes.wait_for_change())
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    await changes.aclose()

    assert await asyncio.wait_for(waiting, timeout=5) is False


async def test_aclose_unregisters_the_reader_before_the_descriptor_is_closed(socket_pair):
    reader, _writer = socket_pair
    changes = UeventDeviceChanges(reader)
    waiting = asyncio.ensure_future(changes.wait_for_change())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    fileno = reader.fileno()

    await changes.aclose()
    await asyncio.wait_for(waiting, timeout=5)

    # A descriptor still in the selector after close would be re-registered
    # against whatever file the kernel hands out next.
    assert fileno not in asyncio.get_running_loop()._selector.get_map()


async def test_a_socket_that_cannot_be_polled_falls_back_rather_than_raising():
    class Detached:
        def fileno(self):
            return -1

        def close(self):
            return None

    assert await UeventDeviceChanges(Detached()).wait_for_change() is False


async def test_a_socket_whose_fileno_raises_falls_back():
    class Broken:
        def fileno(self):
            raise OSError(errno.EBADF, "bad file descriptor")

        def close(self):
            return None

    assert await UeventDeviceChanges(Broken()).wait_for_change() is False


async def test_a_read_error_ends_the_event_source(socket_pair):
    reader, writer = socket_pair

    class Failing:
        def __init__(self, sock):
            self._sock = sock

        def fileno(self):
            return self._sock.fileno()

        def recv(self, *_args, **_kwargs):
            raise OSError(errno.ECONNRESET, "reset")

        def close(self):
            return None

    changes = UeventDeviceChanges(Failing(reader))
    waiting = asyncio.ensure_future(changes.wait_for_change())
    await asyncio.sleep(0)
    writer.send(uevent())

    assert await asyncio.wait_for(waiting, timeout=5) is False


def test_a_payload_without_a_subsystem_key_names_no_device(socket_pair):
    reader, _writer = socket_pair
    changes = UeventDeviceChanges(reader)

    assert changes.concerns_devices(b"remove@/devices/virtual\0ACTION=remove\0") is False


def test_the_watched_subsystems_are_exactly_what_discovery_reads():
    assert frozenset({"accel", "apex", "drm", "pci", "usb"}) == device_events.WATCHED_SUBSYSTEMS


# --------------------------------------------------------------------------
# Opening the source


def test_the_broad_group_mask_is_preferred_and_the_udev_group_is_the_fallback():
    attempted = []

    def refuse_kernel_group(groups):
        attempted.append(groups)
        if groups == UEVENT_GROUPS:
            raise PermissionError(errno.EPERM, "operation not permitted")
        return socket.socketpair()[0]

    changes = open_device_changes(open_socket=refuse_kernel_group)

    assert attempted == [UEVENT_GROUPS, UDEV_UEVENT_GROUP]
    assert isinstance(changes, UeventDeviceChanges)


def test_no_usable_group_returns_no_source_and_says_why(caplog):
    def refuse(_groups):
        raise PermissionError(errno.EPERM, "operation not permitted")

    with caplog.at_level(logging.WARNING, logger="omnitensor.device_events"):
        assert open_device_changes(open_socket=refuse) is None

    assert "Could not subscribe to kernel device events" in caplog.text


def test_the_real_socket_binds_or_refuses_without_leaking_a_descriptor():
    try:
        sock = open_uevent_socket()
    except OSError:
        return
    try:
        assert sock.fileno() >= 0
        assert sock.gettimeout() == 0.0
    finally:
        sock.close()


# --------------------------------------------------------------------------
# The host wrapper


class RecordingChanges:
    def __init__(self, answers):
        self.answers = list(answers)
        self.closed = 0

    async def wait_for_change(self) -> bool:
        if not self.answers:
            await asyncio.Event().wait()
        return self.answers.pop(0)

    async def aclose(self) -> None:
        self.closed += 1


async def test_the_event_source_opens_on_first_use_not_at_construction(fake_nodes):
    opened = []

    def opener():
        opened.append(1)
        return RecordingChanges([True])

    discovery = EventDrivenDeviceDiscovery(SysfsDeviceDiscovery(fake_nodes), open_changes=opener)
    assert opened == []
    assert discovery.detect() == []
    assert opened == []

    assert await discovery.wait_for_change() is True
    assert opened == [1]


async def test_a_host_without_an_event_source_asks_the_caller_to_poll(fake_nodes):
    discovery = EventDrivenDeviceDiscovery(
        SysfsDeviceDiscovery(fake_nodes), open_changes=lambda: None
    )

    assert await discovery.wait_for_change() is False
    assert await discovery.wait_for_change() is False
    await discovery.aclose()


async def test_closing_releases_the_source_once(fake_nodes):
    source = RecordingChanges([True])
    discovery = EventDrivenDeviceDiscovery(
        SysfsDeviceDiscovery(fake_nodes), open_changes=lambda: source
    )
    await discovery.wait_for_change()

    await discovery.aclose()
    await discovery.aclose()

    assert source.closed == 1


def test_the_wrapper_delegates_detection_and_utilization(fake_nodes):
    class Base:
        def detect(self):
            return [Device("gpu-renderD128", "gpu", "GPU", "dri")]

        def utilization(self, device):
            return 42.0

    discovery = EventDrivenDeviceDiscovery(Base(), open_changes=lambda: None)

    assert [device.id for device in discovery.detect()] == ["gpu-renderD128"]
    assert discovery.utilization(Device("gpu-renderD128", "gpu", "GPU", "dri")) == 42.0


def test_production_wiring_gives_the_service_an_event_capable_discovery(tmp_path):
    ports = build_host_ports(snapshot_path=tmp_path / "state.json")

    assert isinstance(ports.discovery, EventDrivenDeviceDiscovery)
    assert isinstance(ports.discovery, DeviceChangeSource)


def test_an_injected_discovery_is_never_wrapped(tmp_path):
    injected = SysfsDeviceDiscovery()
    ports = build_host_ports(snapshot_path=tmp_path / "state.json", discovery=injected)

    assert ports.discovery is injected
    assert not isinstance(injected, DeviceChangeSource)


# --------------------------------------------------------------------------
# The service loop


class EventDiscovery:
    """A DeviceDiscovery whose changes are delivered rather than polled."""

    def __init__(self, devices):
        self.devices = list(devices)
        self.detect_calls = 0
        self.closed = 0
        self._events = asyncio.Queue()

    def detect(self):
        self.detect_calls += 1
        return list(self.devices)

    def utilization(self, _device):
        return None

    async def wait_for_change(self) -> bool:
        return await self._events.get()

    async def aclose(self) -> None:
        self.closed += 1

    def announce(self, alive: bool = True) -> None:
        self._events.put_nowait(alive)


def build_service(tmp_path, **kwargs):
    workloads_root = tmp_path / "workloads"
    workloads_root.mkdir(exist_ok=True)
    kwargs.setdefault(
        "plugin_runtime",
        InstalledPluginRuntime(workloads_root, entry_points_provider=lambda **_k: ()),
    )
    return OmniTensorService(
        snapshot_path=tmp_path / "state/runtime-snapshot.json",
        policy_path=tmp_path / "state/policy.json",
        workloads_path=workloads_root,
        **kwargs,
    )


async def test_an_idle_machine_never_rereads_sysfs(tmp_path):
    discovery = EventDiscovery([Device("gpu-renderD128", "gpu", "GPU", "dri")])
    service = build_service(tmp_path, discovery=discovery, discovery_interval_s=0.001)
    at_start = discovery.detect_calls

    loop = asyncio.ensure_future(service._rediscover())
    for _ in range(50):
        await asyncio.sleep(0)

    # No timer fired, so nothing was read: the whole idle cost is one socket
    # wait, however long the machine sits there.
    assert discovery.detect_calls == at_start

    service._stopping.set()
    discovery.announce()
    await asyncio.wait_for(loop, timeout=5)


async def test_a_kernel_event_rediscovers_and_asks_for_a_publish(tmp_path):
    gpu = Device("gpu-renderD128", "gpu", "GPU", "dri")
    discovery = EventDiscovery([gpu])
    service = build_service(tmp_path, discovery=discovery)
    published = []
    service.request_publish = lambda: published.append(1)

    loop = asyncio.ensure_future(service._rediscover())
    await asyncio.sleep(0)
    discovery.devices = [gpu, Device("npu-accel0", "npu", "NPU", "accel")]
    discovery.announce()
    for _ in range(50):
        await asyncio.sleep(0)
        if published:
            break

    assert published == [1]
    assert [device.id for device in service._devices] == ["gpu-renderD128", "npu-accel0"]

    service._stopping.set()
    discovery.announce()
    await asyncio.wait_for(loop, timeout=5)


async def test_an_event_source_that_closes_falls_back_to_polling_and_says_the_cost(
    tmp_path, caplog
):
    discovery = EventDiscovery([Device("gpu-renderD128", "gpu", "GPU", "dri")])
    service = build_service(tmp_path, discovery=discovery, discovery_interval_s=0.01)

    with caplog.at_level(logging.WARNING, logger="omnitensor.service"):
        loop = asyncio.ensure_future(service._rediscover())
        await asyncio.sleep(0)
        discovery.announce(alive=False)
        for _ in range(200):
            await asyncio.sleep(0.001)
            if discovery.detect_calls > 1:
                break
        service._stopping.set()
        await asyncio.wait_for(loop, timeout=5)

    assert "Device discovery is polling sysfs every" in caplog.text
    assert "the kernel event source closed" in caplog.text
    assert "wakeups a day" in caplog.text
    assert discovery.detect_calls > 1


async def test_a_discovery_without_events_polls_and_announces_it_at_once(tmp_path, caplog):
    class Plain:
        def __init__(self):
            self.detect_calls = 0

        def detect(self):
            self.detect_calls += 1
            return [Device("gpu-renderD128", "gpu", "GPU", "dri")]

        def utilization(self, _device):
            return None

    discovery = Plain()
    service = build_service(tmp_path, discovery=discovery, discovery_interval_s=10.0)

    with caplog.at_level(logging.WARNING, logger="omnitensor.service"):
        loop = asyncio.ensure_future(service._rediscover())
        await asyncio.sleep(0)
        service._stopping.set()
        await asyncio.wait_for(loop, timeout=5)

    assert "Device discovery is polling sysfs every 10.0 s" in caplog.text
    assert "this host exposes no device-change events" in caplog.text
    # 10 s between reads is 8640 wakeups a day, which the log states rather
    # than leaving a reader to work out.
    assert "about 8640 wakeups a day" in caplog.text


async def test_a_failing_event_source_falls_back_rather_than_stopping_discovery(tmp_path, caplog):
    class Exploding:
        def __init__(self):
            self.detect_calls = 0

        def detect(self):
            self.detect_calls += 1
            return [Device("gpu-renderD128", "gpu", "GPU", "dri")]

        def utilization(self, _device):
            return None

        async def wait_for_change(self) -> bool:
            raise RuntimeError("netlink went away")

        async def aclose(self) -> None:
            return None

    discovery = Exploding()
    service = build_service(tmp_path, discovery=discovery, discovery_interval_s=0.01)

    with caplog.at_level(logging.WARNING, logger="omnitensor.service"):
        loop = asyncio.ensure_future(service._rediscover())
        for _ in range(200):
            await asyncio.sleep(0.001)
            if discovery.detect_calls > 1:
                break
        service._stopping.set()
        await asyncio.wait_for(loop, timeout=5)

    assert "Kernel device-event source failed" in caplog.text
    assert discovery.detect_calls > 1


async def test_shutdown_closes_the_event_source(tmp_path):
    discovery = EventDiscovery([Device("gpu-renderD128", "gpu", "GPU", "dri")])
    service = build_service(tmp_path, discovery=discovery)

    await service._close_discovery()

    assert discovery.closed == 1


async def test_closing_a_pollable_discovery_is_a_no_op(tmp_path):
    class Plain:
        def detect(self):
            return [Device("gpu-renderD128", "gpu", "GPU", "dri")]

        def utilization(self, _device):
            return None

    service = build_service(tmp_path, discovery=Plain())

    await service._close_discovery()


async def test_a_failing_close_is_logged_rather_than_raised(tmp_path, caplog):
    class Stubborn:
        def detect(self):
            return [Device("gpu-renderD128", "gpu", "GPU", "dri")]

        def utilization(self, _device):
            return None

        async def wait_for_change(self) -> bool:
            return False

        async def aclose(self) -> None:
            raise RuntimeError("socket is wedged")

    service = build_service(tmp_path, discovery=Stubborn())

    with caplog.at_level(logging.WARNING, logger="omnitensor.service"):
        await service._close_discovery()

    assert "Could not close the device-event source" in caplog.text
