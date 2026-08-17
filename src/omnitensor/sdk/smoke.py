"""Installed-wheel smoke runner for external plugin CI."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib import metadata

from ..plugins.discovery import PLUGIN_ENTRY_POINT_GROUP
from ..plugins.protocol import (
    PluginContext,
    PluginRequest,
    PluginResultStatus,
    WorkloadPlugin,
)
from .conformance import run_plugin_contract

MAX_SMOKE_DOCUMENT_BYTES = 64 * 1024


class PluginSmokeError(RuntimeError):
    """An installed plugin cannot complete the bounded smoke contract."""


async def run_installed_plugin_smoke(
    plugin_id: str,
    payload: Mapping,
    configuration: Mapping | None = None,
    *,
    entry_points_provider: Callable[..., Iterable] = metadata.entry_points,
) -> dict:
    """Load one installed entry point and execute a complete contract job."""
    if not isinstance(plugin_id, str) or not plugin_id:
        raise ValueError("plugin_id must be a non-empty string")
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    if configuration is not None and not isinstance(configuration, Mapping):
        raise TypeError("configuration must be a mapping")
    plugin = _load_installed_plugin(plugin_id, entry_points_provider)
    context = PluginContext(plugin_id, 1, dict(configuration or {}), frozenset())
    request = PluginRequest(
        "installed-smoke-job",
        plugin_id,
        "manual",
        dict(payload),
        0,
        30_000,
    )
    report = await run_plugin_contract(plugin, context, request)
    document = {
        "pluginId": plugin_id,
        "status": report.result.status.value,
        "output": dict(report.result.output),
        "detail": report.result.detail,
        "progress": [{"stage": item.stage, "fraction": item.fraction} for item in report.progress],
    }
    if report.result.status is not PluginResultStatus.SUCCEEDED:
        raise PluginSmokeError(f"plugin smoke job ended with status {report.result.status.value}")
    return document


def _load_installed_plugin(
    plugin_id: str,
    entry_points_provider: Callable[..., Iterable],
) -> WorkloadPlugin:
    try:
        candidates = tuple(entry_points_provider(group=PLUGIN_ENTRY_POINT_GROUP))
    except Exception as error:
        raise PluginSmokeError(f"entry-point enumeration failed: {type(error).__name__}") from error
    matches = [candidate for candidate in candidates if candidate.name == plugin_id]
    if len(matches) != 1:
        raise PluginSmokeError("plugin entry point is missing or ambiguous")
    try:
        factory = matches[0].load()
        plugin = factory()
    except Exception as error:
        raise PluginSmokeError(f"plugin load failed: {type(error).__name__}") from error
    if not isinstance(plugin, WorkloadPlugin) or plugin.plugin_id != plugin_id:
        raise PluginSmokeError("entry point does not implement the declared plugin")
    return plugin


def parse_smoke_document(value: str, name: str) -> dict:
    """Parse one finite bounded JSON object supplied on the command line."""
    encoded = value.encode("utf-8")
    if len(encoded) > MAX_SMOKE_DOCUMENT_BYTES:
        raise ValueError(f"{name} exceeds {MAX_SMOKE_DOCUMENT_BYTES} bytes")
    try:
        document = json.loads(
            value,
            parse_constant=lambda constant: (_ for _ in ()).throw(ValueError(constant)),
        )
    except (UnicodeError, ValueError) as error:
        raise ValueError(f"{name} must be finite JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{name} must be a JSON object")
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute one installed OmniTensor plugin smoke job",
    )
    parser.add_argument("--plugin-id", required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--configuration", default="{}")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    payload = parse_smoke_document(arguments.payload, "payload")
    configuration = parse_smoke_document(arguments.configuration, "configuration")
    result = asyncio.run(
        run_installed_plugin_smoke(
            arguments.plugin_id,
            payload,
            configuration,
        )
    )
    print(json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - module process entry point
    main()
