"""Ports: the pure contracts the service core depends on.

Consumers (:mod:`omnitensor.service`, :mod:`omnitensor.control`) depend on
these Protocols, never on concrete adapters.  Implementations are injected at
construction time; the production wiring supplies filesystem, D-Bus, and
sysfs adapters as defaults.  Only domain value types are imported here — no
implementation modules.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .job_ports import JobAdmission, JobDispatcher

if TYPE_CHECKING:
    from .discovery import Device
    from .state import PolicyState


class DeviceDiscovery(Protocol):
    """Detects accelerator devices and reads their utilization."""

    def detect(self) -> list[Device]:
        """Return present devices in gpu > npu > tpu hierarchy order."""

    def utilization(self, device: Device) -> float | None:
        """Kernel-reported utilization percentage, ``None`` when unavailable."""


@runtime_checkable
class DeviceChangeSource(Protocol):
    """Optional :class:`DeviceDiscovery` capability: say when to look again.

    A discovery adapter that the kernel can wake costs an idle machine
    nothing: no timer, no sysfs read, no wakeup at all while the hardware is
    not changing.  Adapters without it are polled, which is a real cost the
    service states rather than absorbs.
    """

    async def wait_for_change(self) -> bool:
        """Block until devices may have changed.

        Returns ``True`` when something happened and ``False`` when the event
        source is unavailable or has closed, which is the caller's instruction
        to fall back to polling.
        """

    async def aclose(self) -> None:
        """Release the event source; safe to call when never opened."""


class SnapshotPublisher(Protocol):
    """Publishes a contract-valid snapshot document atomically."""

    def publish(self, snapshot: dict) -> None:
        """Publish ``snapshot`` so readers never observe a partial document."""

    def retract(self) -> None:
        """Withdraw any published snapshot so readers observe absence rather
        than a silently stale document; a no-op when nothing is published."""


class PolicyStorage(Protocol):
    """Loads and durably persists :class:`~omnitensor.state.PolicyState`."""

    def load(self) -> PolicyState:
        """Return the persisted state, sanitized, or defaults."""

    def save(self, state: PolicyState) -> None:
        """Persist ``state`` atomically; raises ``OSError`` on failure."""


class CommandHandler(Protocol):
    """Applies one command document and returns acknowledgement JSON text."""

    async def apply_command_text(self, text: str) -> str:
        """Never raises; always returns contract-valid acknowledgement JSON."""


class RuntimeHandler(CommandHandler, Protocol):
    """Versioned runtime control and bounded job boundary.

    Every method returns its own schema's document, or raises
    :class:`omnitensor.guard.GuardRefusedError` when the call is refused at
    the boundary before the method runs.  A transport renders that refusal as
    an envelope error, never as the method's result.
    """

    async def submit_job_text(self, text: str) -> str:
        """Submit one versioned job request and return an acknowledgement."""

    async def cancel_job_text(self, text: str) -> str:
        """Cancel one active job and return an acknowledgement."""

    async def job_result_text(self, text: str) -> str:
        """Read one job's current or terminal result document."""

    def describe_plugins_text(self) -> str:
        """Return the installed-plugin inventory document."""

    def describe_contract_text(self) -> str:
        """Return the runtime's version and schema handshake document."""


class ControlTransport(Protocol):
    """Exposes a :class:`RuntimeHandler` to external callers (e.g. D-Bus)."""

    async def start(self, handler: RuntimeHandler) -> None:
        """Connect the transport and begin serving ``handler``."""

    async def stop(self) -> None:
        """Disconnect the transport; safe to call when never started."""


class PluginRuntime(Protocol):
    """Own metadata discovery and isolated worker lifecycles."""

    async def start(self) -> object:
        """Discover and start every accepted installed plugin."""

    async def stop(self) -> object:
        """Stop all plugin workers; safe to call after partial startup."""


@runtime_checkable
class PluginIdentitySource(Protocol):
    """Announces the installed workload IDs owned by a plugin runtime."""

    def plugin_ids(self) -> Collection[str]: ...


@runtime_checkable
class PluginJobRuntime(PluginIdentitySource, JobAdmission, JobDispatcher, Protocol):
    """Optional installed-plugin admission and dispatch capability."""


@runtime_checkable
class PluginCatalogSnapshot(Protocol):
    """Accepted plugin metadata exposed without importing plugin code."""

    plugins: Collection[object]


@runtime_checkable
class PluginRuntimeSnapshot(Protocol):
    """Inventory and worker states returned or retained by a plugin runtime."""

    catalog: PluginCatalogSnapshot
    workers: Collection[object]


@runtime_checkable
class PluginSnapshotSource(Protocol):
    """Optional retained plugin discovery snapshot capability."""

    @property
    def snapshot(self) -> PluginRuntimeSnapshot: ...


@runtime_checkable
class PluginPermissionSource(Protocol):
    """Permissions captured when an installed plugin worker started."""

    def granted_permissions(self, plugin_id: str) -> Collection[str]: ...


@runtime_checkable
class PluginSettingsSource(Protocol):
    """Stored per-plugin settings, for a runtime that persists them.

    The answer carries ``revision`` and ``configuration``; ``None`` means the
    plugin declares no configuration contract, or the store refused and the
    honest answer is no answer.
    """

    def stored_settings(self, plugin_id: str) -> object | None: ...


@runtime_checkable
class AcceleratorReloadableRuntime(Protocol):
    """A plugin runtime whose worker sandboxes can adopt new device leases."""

    async def reload_accelerator_devices(self) -> object: ...


@runtime_checkable
class DeviceAwareExecutors(Protocol):
    """An executor collection that can answer about one physical device.

    An executor set assembled by an older build (or by a test) is a plain
    mapping keyed by backend name, which cannot tell two cards apart.  This
    Protocol is what separates the two, so the distinction is a declared
    contract rather than four independent ``getattr(..., "for_device")``
    probes.
    """

    def for_device(self, gpu_device_id: str | None) -> dict: ...

    def lane_key(self, backend: str, gpu_device_id: str | None) -> str: ...

    def device_id(self, backend: str, gpu_device_id: str | None) -> str | None: ...

    def executor_for_device(self, backend: str, device_id: str) -> object | None: ...
