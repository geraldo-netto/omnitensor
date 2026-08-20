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
    # Stated by the manifest, empty when it stated nothing. A provider builds
    # its provenance from these rather than from a constant of its own: a
    # constant describes the model that shipped first, not the model that ran.
    source_uri: str = ""
    license_spdx: str = ""


@dataclass(frozen=True, slots=True)
class PluginBootstrap:
    """Host-owned resources for exactly one isolated plugin worker."""

    plugin_id: str
    artifacts: tuple[BootstrapArtifact, ...]
    state_path: Path | None
    accelerator_lease_path: Path | None = None
    # Which model a person chose for this workload, empty when nobody chose.
    # It arrives here rather than being read from policy inside the factory
    # because a worker is sandboxed and has no policy to read: the host decides
    # what it may use, as it already does for artifacts and devices.
    model_choice: str = ""

    def chosen_or(self, default_artifact_id: str) -> BootstrapArtifact:
        """The artifact a person chose, or this workload's default.

        Refuses rather than falling back when the choice is not mounted: a
        person who chose a model and silently got the other one would have no
        reason to believe anything the window says about what ran.
        """
        return self.require_artifact(self.model_choice or default_artifact_id)

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
    global _CURRENT  # noqa: PLW0603 - a one-entry memo of a pure check
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
