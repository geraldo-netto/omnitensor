"""What each workload asks the model, pinned where a change is readable.

The workload half of every prompt is built here — the `GenerationTask` and the
`TaskPrompting` beside it — and assembled into messages by a provider wheel
this suite cannot import. So a refactor in this package can drop an operation
instruction or a grounding hint and every test in `tests/` stays green, which
is the class of breakage that has actually happened: the hints moved modules
twice in one day, and only a provider suite nobody runs by default touched
them.

Pinned as text rather than as a digest. A hash that moves tells a reviewer
that something changed and nothing about what, which is a line to rubber-stamp;
a golden file makes the same change a diff somebody reads — "the summarize
instruction lost the word *concisely*" is visible, and deciding it is fine is
then a real decision.

Every rule below is a property the workload must keep whatever the wording is,
so the golden file carries the words and the assertions carry the meaning.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from omnitensor.plugins.document_answer import document_question_task
from omnitensor.plugins.event_workload import (
    EventPrompting,
    MemoryFragmentStore,
    event_generation_task,
)
from omnitensor.plugins.file_operations import FileOperationsPrompting
from omnitensor.plugins.file_organizer import file_organizer_task
from omnitensor.plugins.fragments import SourceFragment
from omnitensor.plugins.generation import GenerationRequest
from omnitensor.plugins.prompting import NO_PROMPTING
from omnitensor.plugins.selected_text import SelectedTextPrompting, selected_text_task

GOLDEN = Path(__file__).parent / "golden" / "workload-prompts"
UPDATE_HINT = (
    "run `.venv/bin/python -m pytest tests/test_workload_prompt_contract.py "
    "--update-workload-prompts` and read the diff before committing it"
)

# One canonical request per workload, in the production order of its fragments:
# whatever a workload's own plugin publishes before it asks for a generation.
CANONICAL = {
    "selected-text-tools": (
        SelectedTextPrompting(),
        (
            ("private:job-1:control", json.dumps({"operation": "summarize", "language": None})),
            ("private:job-1:selection", "The migration moved 1.4 million rows overnight."),
        ),
    ),
    "event-extraction": (
        EventPrompting(),
        (
            (
                "private:job-1:source:1:page:1",
                "Release planning is in Room 2 on 12 August 2026 from 10:00 to 11:00 Europe/Rome.",
            ),
        ),
    ),
    # The ask request stays canonical for the golden file: the operations
    # prompting contributes nothing to it (the first fragment is a question,
    # not a closed control), so the reviewed ask prompt is what it was.
    # The operation hint is asserted separately below.
    "ask-selected-files": (
        FileOperationsPrompting(),
        (
            ("private:job-1:question", "What is the total due?"),
            ("private:job-1:span:0-42", "The invoice totals 412.90 EUR, due on 30 September."),
        ),
    ),
    "file-organizer": (
        NO_PROMPTING,
        (("private:job-1:source:1", "invoice-2026-08.pdf"),),
    ),
}

TASKS = {
    "ask-selected-files": document_question_task,
    "selected-text-tools": selected_text_task,
    "file-organizer": file_organizer_task,
    "event-extraction": event_generation_task,
}


def _request(workload_id: str):
    prompting, fragments = CANONICAL[workload_id]
    store = MemoryFragmentStore()
    published = tuple(
        SourceFragment(reference, "a" * 64, 1, text, "b" * 64) for reference, text in fragments
    )
    asyncio.run(store.publish("job-1", published))
    references = tuple(fragment.reference for fragment in published)
    return prompting, store, GenerationRequest("job-1", workload_id, references)


def _rendered(workload_id: str) -> str:
    """Everything this package contributes to one workload's prompt."""
    task = TASKS[workload_id]()
    prompting, store, request = _request(workload_id)
    hint = prompting.hint(task, request, store)
    return "\n".join(
        (
            f"task            {task.task_id} v{task.task_version}",
            f"prompt          {task.prompt_id} v{task.prompt_version}",
            f"modalities      {', '.join(task.modalities)}",
            f"context tokens  {task.limits.context_tokens}",
            f"output tokens   {task.limits.output_tokens}",
            f"output bytes    {task.limits.output_bytes}",
            "",
            "--- system prompt ---",
            task.system_prompt,
            "--- instruction template ---",
            task.instruction_template,
            "--- workload hint ---",
            hint,
            "--- output schema ---",
            json.dumps(task.output_schema, indent=2, sort_keys=True, ensure_ascii=False),
            "",
        )
    )


@pytest.mark.parametrize("workload_id", sorted(TASKS))
def test_the_prompt_this_workload_builds_is_the_one_that_was_reviewed(workload_id, request):
    golden = GOLDEN / f"{workload_id}.txt"
    rendered = _rendered(workload_id)
    if request.config.getoption("--update-workload-prompts", default=False):
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(rendered, encoding="utf-8")
        pytest.skip(f"rewrote {golden.name}")
    assert golden.is_file(), f"{golden} is missing; {UPDATE_HINT}"
    assert rendered == golden.read_text(encoding="utf-8"), (
        f"{workload_id} asks the model something other than what was reviewed; {UPDATE_HINT}"
    )


@pytest.mark.parametrize("workload_id", sorted(TASKS))
def test_every_task_states_what_the_untrusted_content_is(workload_id):
    """The template is where private text lands, and it lands in one place."""
    task = TASKS[workload_id]()

    assert task.system_prompt.strip()
    assert task.instruction_template.count("{{UNTRUSTED_CONTENT}}") == 1


@pytest.mark.parametrize("workload_id", sorted(TASKS))
def test_no_task_puts_a_ceiling_on_the_answer(workload_id):
    """The standing rule: the context window is the only legitimate bound."""
    limits = TASKS[workload_id]().limits

    assert limits.output_tokens is None, "an output ceiling truncates the answer somebody needed"
    assert limits.output_bytes is None
    assert limits.context_tokens > 0


@pytest.mark.parametrize("workload_id", sorted(TASKS))
def test_the_answer_contract_is_closed_and_carries_the_request_it_answers(workload_id):
    """An open schema lets a model return a field nothing validates."""
    schema = TASKS[workload_id]().output_schema

    assert schema.get("additionalProperties") is False
    assert "requestId" in schema.get("properties", {})
    assert "requestId" in schema.get("required", [])


@pytest.mark.parametrize("workload_id", sorted(CANONICAL))
def test_no_workload_hint_carries_private_text(workload_id):
    """A hint is references and declared operations, never the source itself."""
    prompting, store, request = _request(workload_id)
    hint = prompting.hint(TASKS[workload_id](), request, store)

    for reference in request.content_references:
        assert store.resolve("job-1", reference).text not in hint


@pytest.mark.parametrize(
    ("operation", "required"),
    [
        ("explain", "Explain the selection"),
        ("summarize", "Summarize the selection"),
        ("rewrite", "Rewrite the selection"),
        ("extract-tasks", "List every explicit actionable item"),
    ],
)
def test_every_selected_text_operation_reaches_the_model_as_an_instruction(operation, required):
    """The five buttons are one prompt plus this sentence.

    Without it the model is asked to apply an operation it was never told,
    and answers by echoing the selection back — which is what a person sees.
    """
    store = MemoryFragmentStore()
    control = SourceFragment(
        "private:job-1:control",
        "a" * 64,
        1,
        json.dumps({"operation": operation, "language": None}),
        "b" * 64,
    )
    selection = SourceFragment("private:job-1:selection", "c" * 64, 1, "Some text.", "d" * 64)
    asyncio.run(store.publish("job-1", (control, selection)))
    request = GenerationRequest(
        "job-1", "selected-text-tools", (control.reference, selection.reference)
    )

    hint = SelectedTextPrompting().hint(selected_text_task(), request, store)

    assert f'operation is "{operation}"' in hint
    assert required in hint


def test_a_translation_names_the_language_it_was_asked_for():
    store = MemoryFragmentStore()
    control = SourceFragment(
        "private:job-1:control",
        "a" * 64,
        1,
        json.dumps({"operation": "translate", "language": "Português"}),
        "b" * 64,
    )
    selection = SourceFragment("private:job-1:selection", "c" * 64, 1, "Some text.", "d" * 64)
    asyncio.run(store.publish("job-1", (control, selection)))
    request = GenerationRequest(
        "job-1", "selected-text-tools", (control.reference, selection.reference)
    )

    hint = SelectedTextPrompting().hint(selected_text_task(), request, store)

    assert '"Português"' in hint
    assert "only the target language" in hint


def _answer(operation: str, result: str) -> str:
    return json.dumps(
        {
            "version": 1,
            "requestId": "job-1",
            "operation": operation,
            "result": result,
            "tasks": [],
            "evidence": {
                "sourceRef": "private:job-1:selection",
                "sourceSha256": "c" * 64,
                "span": {"start": 0, "end": 4},
                "textSha256": "d" * 64,
            },
        }
    )


def _selected_text_request(selection_text: str):
    store = MemoryFragmentStore()
    control = SourceFragment(
        "private:job-1:control",
        "a" * 64,
        1,
        json.dumps({"operation": "explain", "language": None}),
        "b" * 64,
    )
    selection = SourceFragment("private:job-1:selection", "c" * 64, 1, selection_text, "d" * 64)
    asyncio.run(store.publish("job-1", (control, selection)))
    return store, GenerationRequest(
        "job-1", "selected-text-tools", (control.reference, selection.reference)
    )


SENTENCE = "The mitochondrion is the powerhouse of the cell."


@pytest.mark.parametrize("operation", ["explain", "summarize", "rewrite"])
def test_an_answer_that_is_the_selection_earns_one_re_ask(operation):
    """OMNI-0571, measured on the desk: `explain` returned the sentence itself.

    Saying "never return the selection unchanged" in the instruction did not
    stop it, so the workload checks its own answer.
    """
    store, request = _selected_text_request(SENTENCE)

    reprompt = SelectedTextPrompting().reconsideration(
        selected_text_task(), "hint", _answer(operation, SENTENCE), request, store
    )

    assert reprompt is not None
    assert "repeats the selection" in reprompt


@pytest.mark.parametrize("operation", ["translate", "extract-tasks"])
def test_an_operation_that_may_answer_with_the_same_words_is_left_alone(operation):
    """A translation into the language it is already in is a translation."""
    store, request = _selected_text_request(SENTENCE)

    assert (
        SelectedTextPrompting().reconsideration(
            selected_text_task(), "hint", _answer(operation, SENTENCE), request, store
        )
        is None
    )


def test_a_real_answer_is_not_re_asked():
    store, request = _selected_text_request(SENTENCE)
    answered = _answer("explain", "Mitochondria make most of a cell's usable energy.")

    assert (
        SelectedTextPrompting().reconsideration(
            selected_text_task(), "hint", answered, request, store
        )
        is None
    )


def test_surrounding_blanks_do_not_make_an_echo_a_different_answer():
    store, request = _selected_text_request(SENTENCE)

    assert (
        SelectedTextPrompting().reconsideration(
            selected_text_task(), "hint", _answer("explain", f"  {SENTENCE}\n"), request, store
        )
        is not None
    )


def test_a_rewrite_that_changed_something_is_a_rewrite():
    """Exact, not fuzzy: two changed words are a rewrite, not an echo."""
    store, request = _selected_text_request(SENTENCE)
    reworded = "The mitochondrion is the power plant of the cell."

    assert (
        SelectedTextPrompting().reconsideration(
            selected_text_task(), "hint", _answer("rewrite", reworded), request, store
        )
        is None
    )


@pytest.mark.parametrize("raw", ["not json", "[]", '{"operation":"explain"}', "null"])
def test_an_answer_that_is_not_the_contract_is_not_this_check_s_business(raw):
    store, request = _selected_text_request(SENTENCE)

    assert (
        SelectedTextPrompting().reconsideration(selected_text_task(), "hint", raw, request, store)
        is None
    )


def test_without_the_request_there_is_nothing_to_compare_against():
    """A backend that passes neither gets the behaviour it had before."""
    assert (
        SelectedTextPrompting().reconsideration(
            selected_text_task(), "hint", _answer("explain", SENTENCE), None, None
        )
        is None
    )


def _file_operation_request(operation: str, language: str | None, content: str):
    store = MemoryFragmentStore()
    control = SourceFragment(
        "private:job-1:control",
        "a" * 64,
        1,
        json.dumps({"operation": operation, "language": language}),
        "b" * 64,
    )
    fragment = SourceFragment("private:job-1:operation-content-1", "c" * 64, 1, content, "d" * 64)
    asyncio.run(store.publish("job-1", (control, fragment)))
    return store, GenerationRequest(
        "job-1", "ask-selected-files", (control.reference, fragment.reference)
    )


@pytest.mark.parametrize(
    ("operation", "required"),
    [
        ("explain", "Explain the selection"),
        ("summarize", "Summarize the selection"),
        ("rewrite", "Rewrite the selection"),
        ("extract-tasks", "List every explicit actionable item"),
    ],
)
def test_every_file_side_operation_reaches_the_model_as_an_instruction(operation, required):
    """OMNI-0617 stage 2: the file side runs the same instruction table."""
    store, request = _file_operation_request(operation, None, "Some text.")

    hint = FileOperationsPrompting().hint(document_question_task(), request, store)

    assert f'operation is "{operation}"' in hint
    assert required in hint


def test_a_file_side_translation_names_the_language_it_was_asked_for():
    store, request = _file_operation_request("translate", "Português", "Some text.")

    hint = FileOperationsPrompting().hint(document_question_task(), request, store)

    assert '"Português"' in hint
    assert "only the target language" in hint


def test_an_ask_request_gets_no_operation_hint():
    """The question is not a control object, so the core contributes nothing
    and the reviewed ask prompt stays exactly what it was."""
    prompting, store, request = _request("ask-selected-files")

    assert prompting.hint(document_question_task(), request, store) == ""


def _file_answer(operation: str, answer: str) -> str:
    return json.dumps(
        {
            "version": 1,
            "requestId": "job-1",
            "operation": operation,
            "answer": answer,
            "citations": [
                {
                    "sourceRef": "private:job-1:operation-content-1",
                    "sourceSha256": "c" * 64,
                    "span": {"start": 0, "end": len(SENTENCE)},
                    "page": 1,
                    "textSha256": "d" * 64,
                }
            ],
        }
    )


@pytest.mark.parametrize("operation", ["explain", "summarize", "rewrite"])
def test_a_file_side_answer_that_is_the_content_earns_one_re_ask(operation):
    store, request = _file_operation_request(operation, None, SENTENCE)

    reprompt = FileOperationsPrompting().reconsideration(
        document_question_task(), "hint", _file_answer(operation, SENTENCE), request, store
    )

    assert reprompt is not None
    assert "repeats the selection" in reprompt


@pytest.mark.parametrize("operation", ["translate", "extract-tasks"])
def test_a_file_side_operation_that_may_repeat_the_words_is_left_alone(operation):
    store, request = _file_operation_request(operation, None, SENTENCE)

    assert (
        FileOperationsPrompting().reconsideration(
            document_question_task(), "hint", _file_answer(operation, SENTENCE), request, store
        )
        is None
    )


def test_a_real_file_side_answer_is_not_re_asked():
    store, request = _file_operation_request("explain", None, SENTENCE)
    answered = _file_answer("explain", "Mitochondria make most of a cell's usable energy.")

    assert (
        FileOperationsPrompting().reconsideration(
            document_question_task(), "hint", answered, request, store
        )
        is None
    )
