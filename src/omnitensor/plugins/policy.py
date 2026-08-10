"""One policy decision, asked the same way at every stage of a job.

A job that was admissible when it was queued is not necessarily admissible
when it reaches inference: the user can pause the runtime, disable the profile,
or withdraw consent while it sits in a queue, and an artifact can be collected
out from under it.  Checking only at admission means a job that the user
already stopped keeps consuming a device on its way to delivering a result
nobody is allowed to receive.

So the gate is the same object at every stage, and it is asked again at each
one.  The four conditions it composes each already fail closed on their own;
what this adds is that they are applied together, in the same order, with the
same stable reason, wherever the job happens to be.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from enum import StrEnum

from .pipeline import PipelineStage


class PolicyRefusal(StrEnum):
    """Why a job may not proceed, stable enough to show a user."""

    PAUSED = "runtime-paused"
    PROFILE_DISABLED = "profile-disabled"
    CONSENT_MISSING = "consent-missing"
    ARTIFACT_NOT_READY = "artifact-not-ready"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """An admission answer carrying the reason when it is a refusal."""

    admitted: bool
    refusal: PolicyRefusal | None = None
    detail: str = ""

    @property
    def refused(self) -> bool:
        return not self.admitted


ADMITTED = PolicyDecision(True)


class PipelinePolicyGate:
    """Apply pause, enablement, consent, and readiness at every stage."""

    def __init__(
        self,
        profile_id: str,
        *,
        is_paused: Callable[[], bool],
        is_enabled: Callable[[str], bool],
        required_permissions: Collection[str] = (),
        allows_permission: Callable[[str, str], bool] = lambda _profile, _permission: True,
        artifact_ready: Callable[[str], tuple[bool, str]] = lambda _p: (True, ""),
    ) -> None:
        self._profile_id = profile_id
        self._is_paused = is_paused
        self._is_enabled = is_enabled
        self._required_permissions = tuple(sorted(required_permissions))
        self._allows_permission = allows_permission
        self._artifact_ready = artifact_ready

    def evaluate(self, stage: PipelineStage = PipelineStage.QUEUED) -> PolicyDecision:
        """Decide admissibility now, for a job about to enter ``stage``.

        Order matters for the message a user sees: a paused runtime explains
        every profile at once, so it is reported before a per-profile reason.
        """
        if self._is_paused():
            return PolicyDecision(
                False, PolicyRefusal.PAUSED, "the runtime is paused by policy"
            )
        if not self._is_enabled(self._profile_id):
            return PolicyDecision(
                False,
                PolicyRefusal.PROFILE_DISABLED,
                f"{self._profile_id} is disabled by policy",
            )
        for permission in self._required_permissions:
            # Asked of *this* profile: a permission name alone cannot say who
            # was consented to, and answering "does anybody hold this" let a
            # grant issued to one plugin satisfy another plugin's gate.
            if not self._allows_permission(self._profile_id, permission):
                return PolicyDecision(
                    False,
                    PolicyRefusal.CONSENT_MISSING,
                    f"{self._profile_id} has no active grant for {permission}",
                )
        # Readiness is checked last because it is the only condition a user
        # cannot fix from the applet, so the actionable reasons come first.
        if stage in _READINESS_STAGES:
            ready, reason = self._artifact_ready(self._profile_id)
            if not ready:
                return PolicyDecision(
                    False,
                    PolicyRefusal.ARTIFACT_NOT_READY,
                    reason or f"{self._profile_id} has no ready artifact",
                )
        return ADMITTED

    def require(self, stage: PipelineStage = PipelineStage.QUEUED) -> None:
        """Raise :class:`PolicyRefusedError` when the job may not proceed."""
        decision = self.evaluate(stage)
        if decision.refused:
            raise PolicyRefusedError(decision)


# Readiness only constrains the stages that actually need the model.  Asking
# before collection would refuse work that never touches an artifact.
_READINESS_STAGES = frozenset(
    {
        PipelineStage.QUEUED,
        PipelineStage.RESOLVE,
        PipelineStage.INFER,
    }
)


class PolicyRefusedError(RuntimeError):
    """A stage refused because policy no longer admits the job."""

    def __init__(self, decision: PolicyDecision):
        self.decision = decision
        self.code = str(decision.refusal)
        self.detail = decision.detail
        super().__init__(f"{self.code}: {self.detail}")
