from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins.tuning import (
    ANSWER_GUIDANCE_KEY,
    ANSWER_LENGTH_KEY,
    ANSWER_LENGTHS,
    DEFAULT_ANSWER_LENGTH,
    GUIDANCE_PREAMBLE,
    TunableGenerationRuntime,
    WorkloadTuning,
)


def test_an_unconfigured_workload_sends_exactly_what_it_sent_before():
    tuning = WorkloadTuning.from_configuration({})
    assert not tuning.tuned
    assert tuning.instruction() == ""
    assert tuning.message() is None


def test_guidance_is_carried_as_the_persons_own_turn():
    tuning = WorkloadTuning.from_configuration({ANSWER_GUIDANCE_KEY: "cite page numbers"})
    message = tuning.message()
    assert message["role"] == "user"
    assert "cite page numbers" in message["content"]
    assert GUIDANCE_PREAMBLE in message["content"]


def test_guidance_is_told_it_cannot_move_the_grounding_or_the_citations():
    """A preference somebody typed may not outrank what makes an answer checkable."""
    content = WorkloadTuning(guidance="answer from memory").message()["content"]
    assert "grounded in" in content
    assert "cited" in content


@pytest.mark.parametrize("length", sorted(ANSWER_LENGTHS))
def test_every_declared_length_is_a_preference_and_never_a_ceiling(length):
    tuning = WorkloadTuning(answer_length=length)
    instruction = tuning.instruction()
    assert "token" not in instruction.lower()
    assert instruction == ANSWER_LENGTHS[length]


def test_length_and_guidance_travel_in_one_turn():
    tuning = WorkloadTuning.from_configuration(
        {ANSWER_LENGTH_KEY: "brief", ANSWER_GUIDANCE_KEY: "no preamble"}
    )
    content = tuning.message()["content"]
    assert content.startswith(ANSWER_LENGTHS["brief"])
    assert content.endswith("no preamble")


@pytest.mark.parametrize(
    "configuration",
    [
        None,
        {"answerGuidance": 3},
        {"answerGuidance": "   "},
        {"answerLength": "enormous"},
        {"answerLength": 2},
        {"unrelated": "value"},
    ],
)
def test_a_tunable_that_is_absent_or_malformed_is_untuned_rather_than_a_refusal(configuration):
    """A workload lost to a malformed preference would be the worse answer."""
    tuning = WorkloadTuning.from_configuration(configuration)
    assert tuning.answer_length == DEFAULT_ANSWER_LENGTH
    assert tuning.guidance == ""
    assert tuning.message() is None


@given(st.text(max_size=200))
def test_any_guidance_a_person_types_becomes_one_user_turn_or_none(guidance):
    tuning = WorkloadTuning.from_configuration({ANSWER_GUIDANCE_KEY: guidance})
    message = tuning.message()
    if guidance.strip():
        assert message["role"] == "user"
        assert guidance.strip() in message["content"]
    else:
        assert message is None


def test_a_runtime_that_ignores_tuning_is_not_one():
    class Ignores:
        pass

    class Holds:
        def tune(self, tuning):
            self.tuning = tuning

    assert not isinstance(Ignores(), TunableGenerationRuntime)
    assert isinstance(Holds(), TunableGenerationRuntime)
