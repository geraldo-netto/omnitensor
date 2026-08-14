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
        """Return present devices in tpu > npu > gpu hierarchy order."""

    def utilization(self, device: Device) -> float | None:
        """Kernel-reported utilization percentage, ``None`` when unavailable."""


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
    """Versioned runtime control and bounded job boundary."""

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
class AcceleratorReloadableRuntime(Protocol):
    """A plugin runtime whose worker sandboxes can adopt new device leases."""

    async def reload_accelerator_devices(self) -> object: ...
