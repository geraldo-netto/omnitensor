"""Snapshot building and atomic publishing.

Every snapshot is validated against the canonical
``runtime-snapshot.schema.json`` before it is written, and the write is
atomic (temp file + ``os.replace``) so the applet's hardened reader never
observes a partial document.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

from .atomicio import remove_durable, write_json_atomic
from .discovery import Device
from .registry import validate_document
from .tensorref import DEFAULT_MAX_TENSOR_BYTES

SNAPSHOT_VERSION = 1
PLUGIN_TELEMETRY_VERSION = 1
MAX_DEVICE_ENTRIES = 16


def _now_ms() -> int:
    return max(1, int(time.time() * 1000))


def _metric(metrics: dict, key: str) -> int:
    try:
        return int(metrics.get(key, 0))
    except (TypeError, ValueError) as error:
        raise ValueError(f"metrics {key} is not an integer: {metrics.get(key)!r}") from error


MAX_PUBLISHED_INPUT_ROOTS = 8


def input_roots_document(
    roots: Sequence[Path | str], max_bytes: int = DEFAULT_MAX_TENSOR_BYTES
) -> dict:
    """Where a caller may stage a referenced input, as this service sees it.

    A caller cannot infer these paths and must not guess them.  Referencing a
    file is a capability that is off unless configured on, and a buffer written
    anywhere else is refused with the same answer as a path that does not exist
    — deliberately, so a caller cannot probe the filesystem through refusals.
    That makes the roots the one thing a caller needs and the one thing it has
    no way to discover, which is why they are published.

    Empty is the honest answer for the default configuration: this service will
    read no referenced file at all.
    """
    return {
        "roots": [str(root) for root in list(roots)[:MAX_PUBLISHED_INPUT_ROOTS]],
        "maxBytes": int(max_bytes),
    }


def build_snapshot(
    devices: list[Device],
    metrics: dict,
    profiles: dict[str, dict],
    alerts: list[dict] | None = None,
    plugin_telemetry: list[dict] | None = None,
    generated_at_ms: int | None = None,
    inputs: dict | None = None,
) -> dict:
    """Build a contract-valid snapshot document.

    ``metrics`` must carry integer ``queueDepth`` and ``runningProfiles``;
    per-device load rides on the device entries themselves.
    """
    snapshot = {
        "version": SNAPSHOT_VERSION,
        "generatedAt": generated_at_ms if generated_at_ms is not None else _now_ms(),
        "devices": [device.snapshot_entry() for device in devices[:MAX_DEVICE_ENTRIES]],
        "metrics": {
            "queueDepth": _metric(metrics, "queueDepth"),
            "runningProfiles": _metric(metrics, "runningProfiles"),
        },
        "profiles": profiles,
        "alerts": alerts or [],
    }
    if inputs is not None:
        snapshot["inputs"] = inputs
    if plugin_telemetry is not None:
        snapshot["pluginTelemetry"] = {
            "version": PLUGIN_TELEMETRY_VERSION,
            "plugins": plugin_telemetry,
        }
    violations = validate_document("runtime-snapshot.schema.json", snapshot)
    if violations:
        raise ValueError(f"snapshot violates contract: {'; '.join(violations)}")
    return snapshot


def write_snapshot(path: Path, snapshot: dict) -> None:
    """Atomically publish ``snapshot`` to ``path``."""
    write_json_atomic(path, snapshot, prefix=".snapshot-")


def remove_snapshot(path: Path) -> None:
    """Durably remove a published snapshot so readers observe absence
    instead of a stale document; a no-op when nothing is published."""
    remove_durable(path)
