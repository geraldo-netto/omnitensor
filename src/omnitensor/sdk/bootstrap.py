"""Trusted host resources supplied before an external plugin factory runs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .helpers import SDKContractError

_IDENTIFIER = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_CURRENT: PluginBootstrap | None = None


@dataclass(frozen=True, slots=True)
class BootstrapArtifact:
    """One digest-verified artifact mounted read-only in this worker."""

    id: str
    version: str
    format: str
    sha256: str
    path: Path
    companions: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class PluginBootstrap:
    """Host-owned resources for exactly one isolated plugin worker."""

    plugin_id: str
    artifacts: tuple[BootstrapArtifact, ...]
    state_path: Path | None
    accelerator_lease_path: Path | None = None

    def require_artifact(self, artifact_id: str) -> BootstrapArtifact:
        matches = tuple(item for item in self.artifacts if item.id == artifact_id)
        if len(matches) != 1:
            raise SDKContractError(
                "artifact-unavailable", f"exactly one mounted artifact must be named {artifact_id}"
            )
        return matches[0]


def current_plugin_bootstrap(plugin_id: str) -> PluginBootstrap:
    """Return the immutable resources for the factory currently being loaded."""
    if _CURRENT is None or _CURRENT.plugin_id != plugin_id:
        raise SDKContractError("bootstrap-unavailable", "plugin bootstrap identity is unavailable")
    return _CURRENT


def configure_plugin_bootstrap(bootstrap: PluginBootstrap) -> None:
    """Install bootstrap data once; called only by the trusted worker launcher."""
    global _CURRENT
    if _CURRENT is not None:
        if bootstrap == _CURRENT:
            return
        raise SDKContractError("bootstrap-duplicate", "plugin bootstrap is already configured")
    if (
        not isinstance(bootstrap, PluginBootstrap)
        or _IDENTIFIER.fullmatch(bootstrap.plugin_id) is None
    ):
        raise SDKContractError("bootstrap-invalid", "plugin bootstrap is invalid")
    _CURRENT = bootstrap
