"""Profile state selection for runtime snapshots."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .execution import ExecutorSet
from .registry import Workload
from .scheduler import Scheduler, select_backend
from .state import PolicyState

# No ceiling: the snapshot contract no longer caps how many profiles it
# carries, so a plugin is never dropped from the published set to fit a number.
# The parameter stays on the helper for a caller that wants a bound of its own.
MAX_PUBLISHED_PROFILES = None
PAUSED_BY_POLICY = "paused-by-policy"
PROFILE_DISABLED = "profile-disabled"
NO_MODEL = "no-model"
ARTIFACT_UNAVAILABLE = "artifact-unavailable"
SERVING = "serving"
CONSENT_MISSING = "consent-missing"
# The published reason vocabulary belongs to the sibling service's snapshot
# schema, so a profile whose accelerator lease could not be re-established
# reports the closest published code rather than inventing one.
DEVICE_UNAVAILABLE = "device-absent"


@dataclass(frozen=True)
class ReasonCodes:
    """The machine-readable reason codes a status document may carry.

    These used to be read back off ``omnitensor.service`` through
    ``sys.modules`` at call time, so this module -- a lower layer -- depended
    on the facade above it through the module registry.  The codes are data:
    a caller that wants its own set passes them in.
    """

    paused_by_policy: str = PAUSED_BY_POLICY
    profile_disabled: str = PROFILE_DISABLED
    no_model: str = NO_MODEL
    artifact_unavailable: str = ARTIFACT_UNAVAILABLE
    serving: str = SERVING
    consent_missing: str = CONSENT_MISSING
    device_unavailable: str = DEVICE_UNAVAILABLE


DEFAULT_REASONS = ReasonCodes()


def profile_statuses(
    workloads: dict[str, Workload],
    executors: dict,
    scheduler: Scheduler,
    policy: PolicyState,
    artifact_ready: Callable[[Workload], tuple[bool, str]] | None = None,
    permissions_missing: Callable[[Workload], tuple[str, ...]] | None = None,
    reasons: ReasonCodes = DEFAULT_REASONS,
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
            reasons,
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
    reasons: ReasonCodes = DEFAULT_REASONS,
) -> dict:
    queued = counts["queued"]
    profile_policy = policy.profiles.get(workload.id)
    if policy.paused:
        return {
            "status": "paused",
            "queued": queued,
            "detail": "Runtime paused by policy",
            "reason": reasons.paused_by_policy,
        }
    if profile_policy is not None and not profile_policy.enabled:
        return {
            "status": "paused",
            "queued": queued,
            "detail": "Profile disabled by policy",
            "reason": reasons.profile_disabled,
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
            "detail": choice.reason,
            "reason": choice.code,
        }
    if not workload.models:
        return {
            "status": "idle",
            "queued": queued,
            "detail": f"Ready on {choice.backend}; no model bundled",
            "reason": reasons.no_model,
        }
    ungranted = permissions_missing(workload) if permissions_missing is not None else ()
    if ungranted:
        return {
            "status": "unavailable",
            "queued": queued,
            "detail": f"needs consent for {', '.join(ungranted)}",
            "reason": reasons.consent_missing,
        }
    if artifact_ready is not None:
        ready, reason = artifact_ready(workload)
        if not ready:
            return {
                "status": "unavailable",
                "queued": queued,
                "detail": f"{choice.backend}: {reason}",
                "reason": reasons.artifact_unavailable,
            }
    return {
        "status": "running" if counts["running"] > 0 else "watching",
        "queued": queued,
        "detail": f"Serving on {choice.backend}",
        "reason": reasons.serving,
    }


def plugin_profile_statuses(
    profile_ids,
    counts_of: dict[str, dict],
    policy: PolicyState,
    limit: int | None = MAX_PUBLISHED_PROFILES,
    reasons: ReasonCodes = DEFAULT_REASONS,
    unavailable: frozenset[str] | set[str] = frozenset(),
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
            reasons,
            unavailable=profile_id in unavailable,
        )
        for profile_id in sorted(profile_ids)[:limit]
    }


def plugin_profile_status(
    profile_id: str,
    counts: dict,
    policy: PolicyState,
    reasons: ReasonCodes = DEFAULT_REASONS,
    *,
    unavailable: bool = False,
) -> dict:
    """One plugin profile's status.

    A plugin has no model contract and no scheduler lane to be unavailable on:
    what can be said about it is whether policy lets it run, and whether it is
    running now or waiting for a worker slot.
    """
    queued = counts["queued"]
    profile_policy = policy.profiles.get(profile_id)
    if unavailable:
        return {
            "status": "unavailable",
            "queued": queued,
            "detail": "Accelerator lease could not be re-established; not accepting work",
            "reason": reasons.device_unavailable,
        }
    if policy.paused:
        return {
            "status": "paused",
            "queued": queued,
            "detail": "Runtime paused by policy",
            "reason": reasons.paused_by_policy,
        }
    if profile_policy is not None and not profile_policy.enabled:
        return {
            "status": "paused",
            "queued": queued,
            "detail": "Profile disabled by policy",
            "reason": reasons.profile_disabled,
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
        "detail": detail,
        "reason": reasons.serving,
    }


__all__ = [
    "ARTIFACT_UNAVAILABLE",
    "DEVICE_UNAVAILABLE",
    "DEFAULT_REASONS",
    "CONSENT_MISSING",
    "MAX_PUBLISHED_PROFILES",
    "NO_MODEL",
    "PAUSED_BY_POLICY",
    "PROFILE_DISABLED",
    "SERVING",
    "ReasonCodes",
    "plugin_profile_status",
    "plugin_profile_statuses",
    "profile_status",
    "profile_statuses",
]
