from __future__ import annotations

import asyncio
import json

import pytest

from omnitensor.plugins.event_workload import (
    EVENT_GROUNDING_HINT,
    EVENT_RECONSIDERATION,
    EventPrompting,
    MemoryFragmentStore,
    event_generation_task,
)
from omnitensor.plugins.fragments import SourceFragment
from omnitensor.plugins.generation import GenerationRequest
from omnitensor.plugins.prompting import NO_PROMPTING, TaskPrompting
from omnitensor.plugins.selected_text import SelectedTextPrompting, selected_text_task


def published(store, *fragments):
    asyncio.run(store.publish("job-1", fragments))


def test_a_workload_that_says_nothing_extra_is_still_a_prompting_port():
    assert isinstance(NO_PROMPTING, TaskPrompting)
    assert NO_PROMPTING.hint(event_generation_task(), None, None) == ""
    assert NO_PROMPTING.reconsideration(event_generation_task(), "hint", "raw", None, None) is None


@pytest.mark.parametrize("prompting", [EventPrompting(), SelectedTextPrompting()])
def test_every_workload_policy_satisfies_the_port(prompting):
    assert isinstance(prompting, TaskPrompting)


def test_the_event_preflight_says_so_only_for_the_shape_it_is_about():
    store = MemoryFragmentStore()
    fragment = SourceFragment(
        "private:job-1:source:1",
        "a" * 64,
        1,
        "Release planning on 12 August 2026 at 10:00 Europe/Rome.",
        "b" * 64,
    )
    published(store, fragment)
    request = GenerationRequest("job-1", "event-extraction", (fragment.reference,))

    assert EventPrompting().hint(event_generation_task(), request, store) == EVENT_GROUNDING_HINT


def test_a_refusal_the_preflight_contradicts_earns_one_re_prompt():
    refused = json.dumps(
        {"outcome": "refused", "confirmationState": "refused", "events": []},
    )
    prompting = EventPrompting()

    assert prompting.reconsideration(event_generation_task(), "hint", refused, None, None) == (
        EVENT_RECONSIDERATION
    )
    # Nothing re-prompts a refusal nobody contradicted.
    assert prompting.reconsideration(event_generation_task(), "", refused, None, None) is None
    assert prompting.reconsideration(event_generation_task(), "hint", "{}", None, None) is None


def test_a_transformation_that_refuses_has_said_what_it_has_to_say():
    assert (
        SelectedTextPrompting().reconsideration(selected_text_task(), "hint", "{}", None, None)
        is None
    )


def test_the_selected_operation_reaches_the_model_and_the_selection_never_does():
    store = MemoryFragmentStore()
    control = SourceFragment(
        "private:control",
        "a" * 64,
        1,
        json.dumps({"language": "French", "operation": "translate"}),
        "b" * 64,
    )
    selection = SourceFragment("private:selection", "c" * 64, 1, "private selection", "d" * 64)
    published(store, control, selection)
    request = GenerationRequest(
        "job-1", "selected-text-tools", (control.reference, selection.reference)
    )

    hint = SelectedTextPrompting().hint(selected_text_task(), request, store)

    assert 'Trusted selected-text operation is "translate"' in hint
    assert '"French"' in hint
    assert selection.text not in hint


def test_a_workload_policy_answers_nothing_for_another_workloads_task():
    """Each states its own; none of them speaks for the rest."""
    store = MemoryFragmentStore()
    fragment = SourceFragment("private:job-1:source:1", "a" * 64, 1, "text", "b" * 64)
    published(store, fragment)
    request = GenerationRequest("job-1", "file-organizer", (fragment.reference,))

    assert EventPrompting().hint(selected_text_task(), request, store) == ""
    assert SelectedTextPrompting().hint(event_generation_task(), request, store) == ""
