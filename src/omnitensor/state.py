"""Persistent policy state: profile intent, pause flag, and revision.

The revision implements the optimistic-concurrency contract shared with the
applet: every applied command increments it, and a command whose
``expectedRevision`` does not match is rejected so the applet refreshes and
retries.  Persistence is atomic (temp file + ``os.replace``) so a crashed
write can never corrupt the state the next start loads.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .atomicio import write_json_atomic

MIN_WEIGHT = 1
MAX_WEIGHT = 5
MAX_DEVICE_CHOICES = 128
# No ceiling on stored policy: the snapshot contract no longer caps how many
# profiles it publishes, so policy for a profile that exists is policy this
# file keeps.
MAX_PROFILES = None
GPU_DEVICE_ID_PATTERN = re.compile(r"^gpu-renderD[0-9]{1,6}$")
PROFILE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")


def _clamp_weight(value: object, fallback: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        return fallback
    return max(MIN_WEIGHT, min(MAX_WEIGHT, value))


def _sanitized_revision(value: object) -> int:
    # bool is an int subclass; JSON true/false must not become a revision.
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


@dataclass
class ProfilePolicy:
    enabled: bool
    weight: int


@dataclass
class PolicyState:
    paused: bool = False
    profiles: dict[str, ProfilePolicy] = field(default_factory=dict)
    device_choices: dict[str, str] = field(default_factory=dict)
    revision: int = 0

    def portfolio(self) -> dict:
        return {
            "paused": self.paused,
            "profiles": {
                profile_id: {"enabled": policy.enabled, "weight": policy.weight}
                for profile_id, policy in sorted(self.profiles.items())
            },
            "deviceChoices": dict(sorted(self.device_choices.items())),
        }


def _adopt_stored_profiles(
    profiles: dict[str, ProfilePolicy], supplied: dict
) -> None:
    """Keep policy for profiles this store has no default for.

    Defaults come from the workload catalog, which is read at construction and
    knows nothing of the plugins discovered afterwards.  Seeding from defaults
    alone therefore dropped every adopted plugin profile on restart, and the
    runtime re-adopted it at its default — so a plugin someone had disabled or
    weighted came back enabled at weight one on every start.  An entry that is
    stored, well-formed and not a default is that person's decision.
    """
    for profile_id, entry in sorted(supplied.items()):
        if MAX_PROFILES is not None and len(profiles) >= MAX_PROFILES:
            return
        if profile_id in profiles or not isinstance(entry, dict):
            continue
        if not isinstance(profile_id, str) or not PROFILE_ID_PATTERN.fullmatch(profile_id):
            continue
        enabled = entry.get("enabled")
        profiles[profile_id] = ProfilePolicy(
            enabled=enabled if isinstance(enabled, bool) else True,
            weight=_clamp_weight(entry.get("weight"), MIN_WEIGHT),
        )


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
        _adopt_stored_profiles(profiles, supplied)
        device_choices = raw.get("deviceChoices")
        device_choices = device_choices if isinstance(device_choices, dict) else {}
        sanitized_choices = {
            profile_id: device_id
            for profile_id, device_id in sorted(device_choices.items())
            if isinstance(profile_id, str)
            and 1 <= len(profile_id) <= 80
            and isinstance(device_id, str)
            and GPU_DEVICE_ID_PATTERN.fullmatch(device_id)
        }
        return PolicyState(
            paused=raw.get("paused") is True,
            profiles=profiles,
            device_choices=dict(
                list(sanitized_choices.items())[:MAX_DEVICE_CHOICES]
            ),
            revision=_sanitized_revision(raw.get("revision")),
        )

    def save(self, state: PolicyState) -> None:
        payload = {
            "paused": state.paused,
            "profiles": {
                profile_id: {"enabled": policy.enabled, "weight": policy.weight}
                for profile_id, policy in sorted(state.profiles.items())
            },
            "deviceChoices": dict(sorted(state.device_choices.items())),
            "revision": state.revision,
        }
        write_json_atomic(self._path, payload, prefix=".policy-")
