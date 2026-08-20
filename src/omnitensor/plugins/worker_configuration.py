"""How a worker is told the configuration a person chose for it.

The obvious channel is an argument beside `--model-choice`, and it is the
wrong one: a tunable is free text somebody typed into a local-only assistant,
and `/proc/<pid>/cmdline` is world-readable on a default Linux, so every other
user on the host would be able to read it. The handshake is not available
either — `parse_handshake` requires the field sets to match exactly in both
directions, so adding one is a protocol-version change.

What needs neither is the worker's own private state directory: created per
plugin at mode 0700, already passed as `--state-path`, and already the
worker's only trusted write path. The host writes the document there before
the worker starts and the worker reads it once in `start()`, which is the
whole contract — a configuration change restarts the worker that holds it, so
there is nothing to reread and no watcher to keep.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

from ..atomicio import JsonTooLargeError, read_json_bounded, write_bytes_atomic
from .protocol import JsonObject
from .settings import DEFAULT_MAX_SETTINGS_BYTES

CONFIGURATION_FILENAME = "configuration.json"
CONFIGURATION_MODE = 0o600


def configuration_path(state_path: Path) -> Path:
    return Path(state_path) / CONFIGURATION_FILENAME


def publish_worker_configuration(
    state_path: Path | None,
    configuration: Mapping[str, object] | None,
) -> None:
    """Put the stored configuration where the worker will read it, or remove it.

    Removal matters as much as writing: a configuration cleared back to its
    defaults would otherwise leave the previous document in place, and the
    next worker would start with settings nobody holds any more.
    """
    if state_path is None:
        return
    path = configuration_path(state_path)
    if not configuration:
        with contextlib.suppress(OSError):
            os.unlink(path)
        return
    payload = json.dumps(dict(configuration), allow_nan=False, separators=(",", ":")).encode()
    write_bytes_atomic(path, payload, CONFIGURATION_MODE, prefix=f".{CONFIGURATION_FILENAME}.")


def read_worker_configuration(state_path: Path | None) -> JsonObject:
    """The configuration this worker was given, or an empty one.

    Anything unreadable is an empty configuration rather than a refusal to
    start: the document is the host's to write, a plugin that has never been
    configured has none, and a worker that will not run because a tunable is
    unreadable is a workload lost to a setting it does not need.
    """
    if state_path is None:
        return {}
    try:
        document = read_json_bounded(configuration_path(state_path), DEFAULT_MAX_SETTINGS_BYTES)
    except (OSError, JsonTooLargeError, UnicodeError, ValueError):
        return {}
    return document if isinstance(document, dict) else {}
