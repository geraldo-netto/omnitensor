"""Lifecycle base class for separately packaged workload plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..plugins.protocol import (
    CancellationToken,
    PluginContext,
    PluginHealth,
    PluginHealthStatus,
    PluginRequest,
    PluginResult,
    ProgressReporter,
)
from .helpers import ConfigurationView, PermissionView, SDKContractError


class ManagedPlugin(ABC):
    """Idempotent lifecycle shell with typed configuration and permission views."""

    plugin_id: str

    def __init__(self) -> None:
        self._context: PluginContext | None = None

    @property
    def context(self) -> PluginContext:
        if self._context is None:
            raise SDKContractError("plugin-not-started", "plugin context is unavailable")
        return self._context

    @property
    def configuration(self) -> ConfigurationView:
        return ConfigurationView(self.context.configuration)

    @property
    def permissions(self) -> PermissionView:
        return PermissionView(self.context.permissions, self.context.permissions)

    async def start(self, context: PluginContext) -> None:
        if self._context is not None:
            raise SDKContractError("plugin-already-started", "plugin is already started")
        if context.plugin_id != self.plugin_id:
            raise SDKContractError(
                "plugin-identity-mismatch",
                f"expected {self.plugin_id}; received {context.plugin_id}",
            )
        self._context = context
        try:
            await self.on_start()
        except BaseException:
            self._context = None
            raise

    async def health(self) -> PluginHealth:
        if self._context is None:
            return PluginHealth(PluginHealthStatus.STOPPED, "plugin is stopped", 0)
        return await self.on_health()

    async def stop(self) -> None:
        if self._context is None:
            return
        try:
            await self.on_stop()
        finally:
            self._context = None

    async def on_start(self) -> None:
        """Optional resource acquisition hook."""
        return None

    async def on_health(self) -> PluginHealth:
        return PluginHealth(PluginHealthStatus.READY, "plugin is ready", 0)

    async def on_stop(self) -> None:
        """Optional resource release hook."""
        return None

    @abstractmethod
    async def execute(
        self,
        request: PluginRequest,
        cancellation: CancellationToken,
        progress: ProgressReporter,
    ) -> PluginResult:
        """Execute one bounded request."""
