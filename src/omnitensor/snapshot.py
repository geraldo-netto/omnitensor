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

from omnitensor.plugins.telemetry import PLUGIN_TELEMETRY_CONTRACT_VERSION

from .atomicio import remove_durable, write_json_atomic
from .discovery import Device
from .registry import validate_document
from .tensorref import DEFAULT_MAX_TENSOR_BYTES

SNAPSHOT_VERSION = 1
PLUGIN_TELEMETRY_VERSION = PLUGIN_TELEMETRY_CONTRACT_VERSION


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
    declared = list(roots)
    if len(declared) > MAX_PUBLISHED_INPUT_ROOTS:
        # Truncating would leave the service reading from a directory it never
        # told anyone about: the applet would not list that root's pictures for
        # a path the runtime would happily accept, which is the discoverability
        # gap this block exists to close. A configuration nobody can publish is
        # a configuration error, said at startup rather than half-honoured.
        raise ValueError(
            f"{len(declared)} input roots are configured; at most "
            f"{MAX_PUBLISHED_INPUT_ROOTS} can be published, and a root the "
            "snapshot cannot name is one no consumer can use"
        )
    return {
        "roots": [str(root) for root in declared],
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
    kernel_telemetry: dict | None = None,
    policy: dict | None = None,
) -> dict:
    """Build a contract-valid snapshot document.

    ``metrics`` must carry integer ``queueDepth`` and ``runningProfiles``;
    per-device load rides on the device entries themselves.
    """
    snapshot = {
        "version": SNAPSHOT_VERSION,
        "generatedAt": generated_at_ms if generated_at_ms is not None else _now_ms(),
        "devices": [device.snapshot_entry() for device in devices],
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
    if kernel_telemetry is not None:
        snapshot["kernelTelemetry"] = kernel_telemetry
    if policy is not None:
        # What a client changes, published from the store that enforces it, so
        # a client renders policy rather than remembering its own copy of it.
        snapshot["policy"] = policy
    _validate_snapshot(snapshot)
    return snapshot


# The last document that passed, apart from when it was built. An idle runtime
# rebuilds a byte-identical snapshot every tick and revalidating it cost 1.9 ms
# of one core per tick to reach the answer it reached the tick before. Content,
# never identity: the memo is a pure function of the document.
_LAST_VALIDATED: dict | None = None


def _validate_snapshot(snapshot: dict) -> None:
    global _LAST_VALIDATED  # noqa: PLW0603 - a one-entry memo of a pure check
    content = {key: value for key, value in snapshot.items() if key != "generatedAt"}
    if content == _LAST_VALIDATED:
        return
    violations = validate_document("runtime-snapshot.schema.json", snapshot)
    if violations:
        raise ValueError(f"snapshot violates contract: {'; '.join(violations)}")
    _LAST_VALIDATED = content


def write_snapshot(path: Path, snapshot: dict) -> None:
    """Atomically publish ``snapshot`` to ``path``."""
    write_json_atomic(path, snapshot, prefix=".snapshot-")


def remove_snapshot(path: Path) -> None:
    """Durably remove a published snapshot so readers observe absence
    instead of a stale document; a no-op when nothing is published."""
    remove_durable(path)
