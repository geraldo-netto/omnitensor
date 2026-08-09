"""Ports: the pure contracts the service core depends on.

Consumers (:mod:`omnitensor.service`, :mod:`omnitensor.control`) depend on
these Protocols, never on concrete adapters.  Implementations are injected at
construction time; the production wiring supplies filesystem, D-Bus, and
sysfs adapters as defaults.  Only domain value types are imported here — no
implementation modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

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


class PolicyStorage(Protocol):
    """Loads and durably persists :class:`~omnitensor.state.PolicyState`."""

    def load(self) -> PolicyState:
        """Return the persisted state, sanitized, or defaults."""

    def save(self, state: PolicyState) -> None:
        """Persist ``state`` atomically; raises ``OSError`` on failure."""


class CommandHandler(Protocol):
    """Applies one command document and returns acknowledgement JSON text."""

    def apply_command_text(self, text: str) -> str:
        """Never raises; always returns contract-valid acknowledgement JSON."""


class ControlTransport(Protocol):
    """Exposes a :class:`CommandHandler` to external callers (e.g. D-Bus)."""

    async def start(self, handler: CommandHandler) -> None:
        """Connect the transport and begin serving ``handler``."""

    async def stop(self) -> None:
        """Disconnect the transport; safe to call when never started."""
