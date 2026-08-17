"""Manifest migration and runtime protocol compatibility gates."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from enum import StrEnum

from ..registry import validate_workload_document
from .discovery import PluginSource
from .identity import (
    PluginCatalog,
    PluginRejection,
    PluginRejectionCode,
    ResolvedPlugin,
)

CURRENT_MANIFEST_VERSION = 2
CURRENT_RUNTIME_API = 1
DEFAULT_PROTOCOL_MINIMUM = 1
DEFAULT_PROTOCOL_MAXIMUM = 1
MAX_PROTOCOL_CAPABILITIES = 32
MAX_PROTOCOL_CAPABILITY_CHARS = 64
DEFAULT_PROTOCOL_CAPABILITIES = frozenset({"cancel", "execute", "health", "progress"})


class ManifestCompatibilityCode(StrEnum):
    COMPATIBLE = "compatible"
    ADAPTED_V1 = "adapted-v1"
    MANIFEST_VERSION = "manifest-version-incompatible"
    RUNTIME_API = "runtime-api-incompatible"
    PROTOCOL_VERSION = "protocol-version-incompatible"
    ADAPTER_INVALID = "manifest-adapter-invalid"


@dataclass(frozen=True, slots=True)
class ManifestCompatibilityDecision:
    compatible: bool
    code: ManifestCompatibilityCode
    detail: str
    plugin: ResolvedPlugin | None
    protocol_version: int | None
    capabilities: frozenset[str]
    adapted: bool


class PluginManifestCompatibilityGate:
    """Adapt bundled v1 metadata and negotiate one host protocol contract."""

    def __init__(
        self,
        *,
        runtime_api: int = CURRENT_RUNTIME_API,
        protocol_minimum: int = DEFAULT_PROTOCOL_MINIMUM,
        protocol_maximum: int = DEFAULT_PROTOCOL_MAXIMUM,
        capabilities: frozenset[str] = DEFAULT_PROTOCOL_CAPABILITIES,
    ) -> None:
        _positive_integer(runtime_api, "runtime_api")
        _positive_integer(protocol_minimum, "protocol_minimum")
        _positive_integer(protocol_maximum, "protocol_maximum")
        if protocol_minimum > protocol_maximum:
            raise ValueError("protocol range must be ordered")
        _validate_capabilities(capabilities)
        self._runtime_api = runtime_api
        self._protocol_minimum = protocol_minimum
        self._protocol_maximum = protocol_maximum
        self._capabilities = capabilities

    def check(self, plugin: ResolvedPlugin) -> ManifestCompatibilityDecision:
        manifest = plugin.manifest
        version = manifest.get("manifestVersion")
        adapted = version == 1 and plugin.source is PluginSource.BUNDLED
        if version == 1 and not adapted:
            return _rejected(
                ManifestCompatibilityCode.MANIFEST_VERSION,
                "manifest v1 migration is limited to bundled plugins",
            )
        if version not in (1, CURRENT_MANIFEST_VERSION) or isinstance(version, bool):
            return _rejected(
                ManifestCompatibilityCode.MANIFEST_VERSION,
                f"runtime requires manifest version {CURRENT_MANIFEST_VERSION}",
            )
        effective = adapt_bundled_manifest_v1(manifest) if adapted else copy.deepcopy(manifest)
        declared_api = effective["requirements"]["runtimeApi"]
        if type(declared_api) is not int or declared_api != self._runtime_api:
            return _rejected(
                ManifestCompatibilityCode.RUNTIME_API,
                f"runtime API {declared_api!r} is incompatible with {self._runtime_api}",
            )
        if validate_workload_document(effective):
            return _rejected(
                ManifestCompatibilityCode.ADAPTER_INVALID,
                "effective manifest violates the runtime schema",
            )
        protocol = effective["plugin"]["protocol"]
        minimum = max(protocol["minimum"], self._protocol_minimum)
        maximum = min(protocol["maximum"], self._protocol_maximum)
        if minimum > maximum:
            return _rejected(
                ManifestCompatibilityCode.PROTOCOL_VERSION,
                (
                    f"plugin protocol {protocol['minimum']}-{protocol['maximum']} "
                    f"does not overlap runtime {self._protocol_minimum}-{self._protocol_maximum}"
                ),
            )
        declared_capabilities = frozenset(protocol.get("capabilities", ()))
        negotiated = declared_capabilities & self._capabilities
        effective_plugin = replace(plugin, manifest=effective)
        code = (
            ManifestCompatibilityCode.ADAPTED_V1
            if adapted
            else ManifestCompatibilityCode.COMPATIBLE
        )
        detail = (
            "bundled manifest v1 adapted without changing policy fields"
            if adapted
            else "manifest and runtime protocols are compatible"
        )
        return ManifestCompatibilityDecision(
            True,
            code,
            detail,
            effective_plugin,
            maximum,
            negotiated,
            adapted,
        )


def declared_model_artifacts(manifest: dict) -> list[dict]:
    """The model a v1 workload declares, as a plugin artifact declaration.

    A v1 manifest states its model under ``requirements.model`` and knows
    nothing about ``plugin.artifacts``.  Adapting it to zero artifacts made
    the inventory answer "what does this need, and does it have it" with
    silence for every bundled profile — including the one whose model is
    installed, digest-verified, and serving.

    A declaration without ``sha256`` is dropped rather than invented: an
    artifact entry is an attestation to an exact file, and one that named a
    version without saying which bytes would let the inventory report ready
    for a file the publisher never vouched for.
    """
    model = manifest.get("requirements", {}).get("model")
    if not isinstance(model, dict) or not model.get("sha256"):
        return []
    return [
        {
            "id": model["id"],
            "version": model["version"],
            "format": model["format"],
            "sha256": model["sha256"],
        }
    ]


def adapt_bundled_manifest_v1(manifest: dict) -> dict:
    """Create fail-closed execution metadata while preserving all v1 policy data."""
    effective = copy.deepcopy(manifest)
    effective["manifestVersion"] = CURRENT_MANIFEST_VERSION
    effective["plugin"] = {
        "entryPoint": effective["id"],
        "protocol": {
            "minimum": DEFAULT_PROTOCOL_MINIMUM,
            "maximum": DEFAULT_PROTOCOL_MAXIMUM,
            "capabilities": [],
        },
        "schemas": {
            name: {"type": "object", "additionalProperties": False}
            for name in ("configuration", "input", "output")
        },
        "triggers": ["manual"],
        "artifacts": declared_model_artifacts(effective),
        "permissions": [],
    }
    return effective


def resolve_plugin_compatibility(
    catalog: PluginCatalog,
    gate: PluginManifestCompatibilityGate | None = None,
) -> PluginCatalog:
    """Remove incompatible identities and retain stable bounded rejections."""
    checker = gate or PluginManifestCompatibilityGate()
    accepted = []
    rejected = list(catalog.rejections)
    for plugin in catalog.plugins:
        decision = checker.check(plugin)
        if decision.compatible:
            accepted.append(decision.plugin)
        else:
            rejected.append(
                PluginRejection(
                    plugin.entry_point_name,
                    plugin.distribution_name,
                    PluginRejectionCode.INCOMPATIBLE,
                    f"{decision.code}: {decision.detail}",
                )
            )
    rejected.sort(key=lambda item: (item.entry_point_name, item.distribution_name or "", item.code))
    return PluginCatalog(tuple(accepted), tuple(rejected))


def _rejected(
    code: ManifestCompatibilityCode,
    detail: str,
) -> ManifestCompatibilityDecision:
    return ManifestCompatibilityDecision(False, code, detail, None, None, frozenset(), False)


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_capabilities(capabilities: frozenset[str]) -> None:
    if not isinstance(capabilities, frozenset):
        raise TypeError("capabilities must be a frozenset")
    if len(capabilities) > MAX_PROTOCOL_CAPABILITIES or any(
        not isinstance(capability, str)
        or not capability
        or len(capability) > MAX_PROTOCOL_CAPABILITY_CHARS
        for capability in capabilities
    ):
        raise ValueError(
            f"capabilities must contain at most {MAX_PROTOCOL_CAPABILITIES} "
            f"names of 1-{MAX_PROTOCOL_CAPABILITY_CHARS} characters"
        )
