from __future__ import annotations

import pytest

from omnitensor.plugins.pipeline import PipelineStage
from omnitensor.plugins.policy import (
    ADMITTED,
    PipelinePolicyGate,
    PolicyDecision,
    PolicyRefusal,
    PolicyRefusedError,
)

PROFILE = "hardware-health"
READ_SENSOR = "read:/sys/class/hwmon/*"

EVERY_STAGE = [stage for stage in PipelineStage]


def gate(**changes):
    options = {
        "is_paused": lambda: False,
        "is_enabled": lambda _profile: True,
        "required_permissions": (),
        "allows_permission": lambda _profile, _permission: True,
        "artifact_ready": lambda _profile: (True, ""),
    }
    options.update(changes)
    return PipelinePolicyGate(PROFILE, **options)


def test_an_admissible_job_is_admitted_at_every_stage():
    decisions = [gate().evaluate(stage) for stage in EVERY_STAGE]
    assert all(decision == ADMITTED for decision in decisions)
    assert ADMITTED.refused is False


@pytest.mark.parametrize("stage", EVERY_STAGE)
def test_a_paused_runtime_refuses_at_every_stage(stage):
    """A job queued before the pause must not keep advancing through it."""
    decision = gate(is_paused=lambda: True).evaluate(stage)
    assert decision.refused
    assert decision.refusal is PolicyRefusal.PAUSED
    assert "paused" in decision.detail


@pytest.mark.parametrize("stage", EVERY_STAGE)
def test_a_disabled_profile_refuses_at_every_stage(stage):
    decision = gate(is_enabled=lambda _profile: False).evaluate(stage)
    assert decision.refusal is PolicyRefusal.PROFILE_DISABLED
    assert PROFILE in decision.detail


@pytest.mark.parametrize("stage", EVERY_STAGE)
def test_withdrawn_consent_refuses_at_every_stage(stage):
    """Consent withdrawn mid-flight must stop the job wherever it is."""
    decision = gate(
        required_permissions=(READ_SENSOR,),
        allows_permission=lambda _profile, _permission: False,
    ).evaluate(stage)
    assert decision.refusal is PolicyRefusal.CONSENT_MISSING
    assert READ_SENSOR in decision.detail


def test_the_pause_reason_is_reported_before_a_per_profile_reason():
    """A paused runtime explains every profile at once, so it comes first."""
    decision = gate(
        is_paused=lambda: True,
        is_enabled=lambda _profile: False,
        required_permissions=(READ_SENSOR,),
        allows_permission=lambda _profile, _permission: False,
        artifact_ready=lambda _profile: (False, "no active version"),
    ).evaluate()
    assert decision.refusal is PolicyRefusal.PAUSED


def test_an_actionable_reason_is_reported_before_readiness():
    """Readiness is the one condition a user cannot fix from the applet."""
    decision = gate(
        required_permissions=(READ_SENSOR,),
        allows_permission=lambda _profile, _permission: False,
        artifact_ready=lambda _profile: (False, "no active version"),
    ).evaluate()
    assert decision.refusal is PolicyRefusal.CONSENT_MISSING


@pytest.mark.parametrize(
    "stage", [PipelineStage.QUEUED, PipelineStage.RESOLVE, PipelineStage.INFER]
)
def test_readiness_is_required_by_the_stages_that_need_the_model(stage):
    decision = gate(artifact_ready=lambda _profile: (False, "no active version")).evaluate(stage)
    assert decision.refusal is PolicyRefusal.ARTIFACT_NOT_READY
    assert decision.detail == "no active version"


@pytest.mark.parametrize(
    "stage",
    [
        PipelineStage.COLLECT,
        PipelineStage.PREPROCESS,
        PipelineStage.POSTPROCESS,
        PipelineStage.DELIVER,
        PipelineStage.TERMINAL,
    ],
)
def test_readiness_does_not_constrain_stages_that_never_touch_the_model(stage):
    """Refusing collection for a missing model would stop work that never needs it."""
    assert gate(artifact_ready=lambda _profile: (False, "gone")).evaluate(stage) == ADMITTED


def test_an_unready_artifact_without_a_reason_still_explains_itself():
    decision = gate(artifact_ready=lambda _profile: (False, "")).evaluate()
    assert PROFILE in decision.detail


def test_every_required_permission_is_checked_in_stable_order():
    asked = []

    def allows(profile_id, permission):
        asked.append((profile_id, permission))
        return False

    decision = gate(required_permissions=("z:last", "a:first"), allows_permission=allows).evaluate()

    assert asked == [(PROFILE, "a:first")], "asked of this profile, not of the catalogue"
    assert "a:first" in decision.detail


def test_require_raises_with_the_refusal_attached():
    with pytest.raises(PolicyRefusedError) as excinfo:
        gate(is_paused=lambda: True).require()

    assert excinfo.value.code == str(PolicyRefusal.PAUSED)
    assert excinfo.value.decision.refusal is PolicyRefusal.PAUSED
    assert "runtime-paused" in str(excinfo.value)


def test_require_returns_quietly_when_admitted():
    assert gate().require(PipelineStage.INFER) is None


def test_a_decision_reports_refusal_consistently():
    refused = PolicyDecision(False, PolicyRefusal.PAUSED, "paused")
    assert refused.refused is True
    assert ADMITTED.refusal is None


def test_policy_is_re_evaluated_rather_than_cached():
    """The whole point is that a decision made at queue time can change."""
    paused = [False]
    subject = gate(is_paused=lambda: paused[0])

    assert subject.evaluate(PipelineStage.COLLECT) == ADMITTED
    paused[0] = True
    assert subject.evaluate(PipelineStage.COLLECT).refusal is PolicyRefusal.PAUSED
    paused[0] = False
    assert subject.evaluate(PipelineStage.COLLECT) == ADMITTED
