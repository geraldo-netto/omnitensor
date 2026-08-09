"""Persistent policy state: profile intent, pause flag, and revision.

The revision implements the optimistic-concurrency contract shared with the
applet: every applied command increments it, and a command whose
``expectedRevision`` does not match is rejected so the applet refreshes and
retries.  Persistence is atomic (temp file + ``os.replace``) so a crashed
write can never corrupt the state the next start loads.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

MIN_WEIGHT = 1
MAX_WEIGHT = 5


def _clamp_weight(value: object, fallback: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        return fallback
    return max(MIN_WEIGHT, min(MAX_WEIGHT, value))


@dataclass
class ProfilePolicy:
    enabled: bool
    weight: int


@dataclass
class PolicyState:
    paused: bool = False
    profiles: dict[str, ProfilePolicy] = field(default_factory=dict)
    revision: int = 0

    def portfolio(self) -> dict:
        return {
            "paused": self.paused,
            "profiles": {
                profile_id: {"enabled": policy.enabled, "weight": policy.weight}
                for profile_id, policy in sorted(self.profiles.items())
            },
        }


class PolicyStore:
    """Loads, sanitizes, and atomically persists :class:`PolicyState`."""

    def __init__(self, path: Path, defaults: dict[str, ProfilePolicy]):
        self._path = path
        self._defaults = defaults

    def load(self) -> PolicyState:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        profiles: dict[str, ProfilePolicy] = {}
        supplied = raw.get("profiles")
        supplied = supplied if isinstance(supplied, dict) else {}
        for profile_id, default in self._defaults.items():
            entry = supplied.get(profile_id)
            entry = entry if isinstance(entry, dict) else {}
            enabled = entry.get("enabled")
            profiles[profile_id] = ProfilePolicy(
                enabled=enabled if isinstance(enabled, bool) else default.enabled,
                weight=_clamp_weight(entry.get("weight"), default.weight),
            )
        revision = raw.get("revision")
        return PolicyState(
            paused=raw.get("paused") is True,
            profiles=profiles,
            revision=revision if isinstance(revision, int) and revision >= 0 else 0,
        )

    def save(self, state: PolicyState) -> None:
        payload = {
            "paused": state.paused,
            "profiles": {
                profile_id: {"enabled": policy.enabled, "weight": policy.weight}
                for profile_id, policy in sorted(state.profiles.items())
            },
            "revision": state.revision,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(dir=self._path.parent, prefix=".policy-")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self._path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temp_name)
            raise
