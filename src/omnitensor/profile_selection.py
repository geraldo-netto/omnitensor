"""Profile state selection for runtime snapshots."""

from __future__ import annotations

import sys
from collections.abc import Callable

from .execution import ExecutorSet
from .registry import Workload
from .scheduler import Scheduler, select_backend
from .state import PolicyState

# The snapshot contract publishes at most 128 profiles; past that the document
# is invalid and nothing is published at all, so extra plugins are dropped from
# the list rather than costing every reader the whole snapshot.
MAX_PUBLISHED_PROFILES = 128
PAUSED_BY_POLICY = "paused-by-policy"
PROFILE_DISABLED = "profile-disabled"
NO_MODEL = "no-model"
ARTIFACT_UNAVAILABLE = "artifact-unavailable"
SERVING = "serving"
CONSENT_MISSING = "consent-missing"


def _reason(name: str, fallback: str) -> str:
    facade = sys.modules.get("omnitensor.service")
    return getattr(facade, name, fallback) if facade is not None else fallback


def profile_statuses(
    workloads: dict[str, Workload],
    executors: dict,
    scheduler: Scheduler,
    policy: PolicyState,
    artifact_ready: Callable[[Workload], tuple[bool, str]] | None = None,
    permissions_missing: Callable[[Workload], tuple[str, ...]] | None = None,
) -> dict[str, dict]:
    """Runtime status per profile for the snapshot document."""
    per_profile = scheduler.profile_stats()
    return {
        workload_id: profile_status(
            workload,
            executors,
            per_profile.get(workload_id, {"queued": 0, "running": 0}),
            policy,
            artifact_ready,
            permissions_missing,
        )
        for workload_id, workload in workloads.items()
    }


def profile_status(
    workload: Workload,
    executors: dict,
    counts: dict,
    policy: PolicyState,
    artifact_ready: Callable[[Workload], tuple[bool, str]] | None = None,
    permissions_missing: Callable[[Workload], tuple[str, ...]] | None = None,
) -> dict:
    queued = counts["queued"]
    profile_policy = policy.profiles.get(workload.id)
    if policy.paused:
        return {
            "status": "paused",
            "queued": queued,
            "detail": "Runtime paused by policy",
            "reason": _reason("PAUSED_BY_POLICY", PAUSED_BY_POLICY),
        }
    if profile_policy is not None and not profile_policy.enabled:
        return {
            "status": "paused",
            "queued": queued,
            "detail": "Profile disabled by policy",
            "reason": _reason("PROFILE_DISABLED", PROFILE_DISABLED),
        }
    routed_executors = (
        executors.for_device(policy.device_choices.get(workload.id))
        if isinstance(executors, ExecutorSet)
        else executors
    )
    choice = select_backend(workload, routed_executors)
    if choice.backend is None:
        return {
            "status": "unavailable",
            "queued": queued,
            "detail": choice.reason[:240],
            "reason": choice.code,
        }
    if not workload.models:
        return {
            "status": "idle",
            "queued": queued,
            "detail": f"Ready on {choice.backend}; no model bundled",
            "reason": _reason("NO_MODEL", NO_MODEL),
        }
    ungranted = permissions_missing(workload) if permissions_missing is not None else ()
    if ungranted:
        return {
            "status": "unavailable",
            "queued": queued,
            "detail": f"needs consent for {', '.join(ungranted)}"[:240],
            "reason": _reason("CONSENT_MISSING", CONSENT_MISSING),
        }
    if artifact_ready is not None:
        ready, reason = artifact_ready(workload)
        if not ready:
            return {
                "status": "unavailable",
                "queued": queued,
                "detail": f"{choice.backend}: {reason}"[:240],
                "reason": _reason("ARTIFACT_UNAVAILABLE", ARTIFACT_UNAVAILABLE),
            }
    return {
        "status": "running" if counts["running"] > 0 else "watching",
        "queued": queued,
        "detail": f"Serving on {choice.backend}",
        "reason": _reason("SERVING", SERVING),
    }


def plugin_profile_statuses(
    profile_ids,
    counts_of: dict[str, dict],
    policy: PolicyState,
    limit: int = MAX_PUBLISHED_PROFILES,
) -> dict[str, dict]:
    """Status per installed plugin profile, for the snapshot document.

    Plugins are not in the workload catalog, so they had no entry here and the
    published profile set never named them — which is what let a surface tell
    enabled from disabled for catalog profiles and nothing at all for the ones
    a person actually runs.  They are governed the same way, and now say so.
    """
    return {
        profile_id: plugin_profile_status(
            profile_id,
            counts_of.get(profile_id, {"queued": 0, "running": 0}),
            policy,
        )
        for profile_id in sorted(profile_ids)[:limit]
    }


def plugin_profile_status(profile_id: str, counts: dict, policy: PolicyState) -> dict:
    """One plugin profile's status.

    A plugin has no model contract and no scheduler lane to be unavailable on:
    what can be said about it is whether policy lets it run, and whether it is
    running now or waiting for a worker slot.
    """
    queued = counts["queued"]
    profile_policy = policy.profiles.get(profile_id)
    if policy.paused:
        return {
            "status": "paused",
            "queued": queued,
            "detail": "Runtime paused by policy",
            "reason": _reason("PAUSED_BY_POLICY", PAUSED_BY_POLICY),
        }
    if profile_policy is not None and not profile_policy.enabled:
        return {
            "status": "paused",
            "queued": queued,
            "detail": "Profile disabled by policy",
            "reason": _reason("PROFILE_DISABLED", PROFILE_DISABLED),
        }
    running = counts["running"]
    weight = profile_policy.weight if profile_policy is not None else 1
    detail = (
        f"Running in a plugin worker; {queued} waiting at weight {weight}"
        if running
        else "Ready; runs in a plugin worker when asked"
    )
    return {
        "status": "running" if running else "idle",
        "queued": queued,
        "detail": detail[:240],
        "reason": _reason("SERVING", SERVING),
    }


profile_statuses.__module__ = "omnitensor.service"


__all__ = [
    "ARTIFACT_UNAVAILABLE",
    "CONSENT_MISSING",
    "MAX_PUBLISHED_PROFILES",
    "NO_MODEL",
    "PAUSED_BY_POLICY",
    "PROFILE_DISABLED",
    "SERVING",
    "plugin_profile_status",
    "plugin_profile_statuses",
    "profile_status",
    "profile_statuses",
]
