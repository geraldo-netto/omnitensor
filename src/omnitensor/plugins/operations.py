"""The operations core: one enum, one instruction table, one routing rule.

The operation set lived in four places — the selected-text module, its
manifest, and both of its schemas — with the client's dropdown as a fifth,
and the translation-route validation was duplicated verbatim between the
inline and file-translate workloads (OMNI-0617, stage 0; the staged plan is
docs/operations-core-migration.md). This module is where those facts live
once. It is a NEW module on purpose: document_qa/document_spans/
document_answer/fragments are pinned by mutation-selector paths and stay
where they are.

The five transformation instructions retain their reviewed bytes. Adding
inline ``ask`` widens the selected-text output schema, so that task's complete
digest moves and its qualification receipt records the pair as unmeasured.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from .fragments import FragmentStoreError
from .generation import GenerationRouter
from .target_language import MAX_LANGUAGE_CHARACTERS, valid_target_language

TRANSFORM_OPERATIONS = frozenset({"explain", "summarize", "rewrite", "translate", "extract-tasks"})
OPERATIONS = frozenset({"ask", *TRANSFORM_OPERATIONS})
MAX_QUESTION_CHARACTERS = 4_096
# The three that promise something other than what was handed in. A
# translation into the language the selection is already in is legitimately
# the same words, and an extraction that finds nothing says so in `tasks`.
RESTATEMENT_IS_A_NON_ANSWER = frozenset({"explain", "summarize", "rewrite"})

# What to say when an answer is the question. Not a refusal and not a rule
# about length: the model is told what it did and asked once more.
ECHOED_SELECTION_REPROMPT = (
    "That answer repeats the selection word for word, which is not the operation you were "
    "asked for. Answer again in your own words: keep the same closed JSON contract, the same "
    "operation, and the same evidence, and make result something other than the selection.\n"
    "/no_think"
)


def operation_instruction(operation: object, language: object, question: object = None) -> str:
    translate = (
        # `ensure_ascii=False`: a person asking for "Português" was quoting a
        # name the model then read as `"Portugu\u00eas"`, which is not the
        # language they asked for and not a word in any of them.
        f"Translate the selection into {json.dumps(language, ensure_ascii=False)} in result; "
        "preserve the exact "
        "meaning of every noun, verb, number, and name; use only the target language and its "
        "script; transliterate proper names; do not add a label; tasks must be empty."
        if isinstance(language, str) and language
        else ""
    )
    # Each of the three says not to hand the selection back. A one-sentence
    # factoid — "The mitochondrion is the powerhouse of the cell." — came back
    # verbatim as its own explanation, which reads to the person exactly like
    # an answer and is none: the labelled cases pass because their selections
    # are long enough that repeating one is obviously not a summary. Saying it
    # costs a clause and refuses nothing (OMNI-0571).
    return {
        "ask": (
            f"Answer {json.dumps(question, ensure_ascii=False)} using only the selection in "
            "result; when the selection does not answer it, say so; tasks must be empty."
            if isinstance(question, str) and question
            else ""
        ),
        "explain": (
            "Explain the selection clearly in result, in your own words; never return the "
            "selection unchanged, even when it is one sentence; tasks must be empty."
        ),
        "summarize": (
            "Summarize the selection concisely in result; never return the selection "
            "unchanged; tasks must be empty."
        ),
        "rewrite": (
            "Rewrite the selection while preserving its meaning; never return it unchanged, "
            "even when it already reads well; tasks must be empty."
        ),
        "translate": translate,
        "extract-tasks": (
            "List every explicit actionable item in tasks. If any action is present, tasks "
            "must not be empty. Result must briefly introduce the extracted tasks, not echo "
            "the selection."
        ),
    }.get(operation, "")


class OperationPrompting:
    """What an operations workload tells the model beyond its task prompt.

    The operation a person chose is in a trusted control fragment, not in
    the task: one prompt serves every operation, and which one was asked for
    decides what a correct answer looks like. Parameterised by task id and by
    the field the workload's answer contract puts the prose in — ``result``
    inline, ``answer`` on the file side — so the file-side workload (stage 2)
    builds its own prompting from the same core the inline side uses.
    """

    __slots__ = ("_answer_field", "_task_id")

    def __init__(self, task_id: str, *, answer_field: str = "result") -> None:
        self._task_id = task_id
        self._answer_field = answer_field

    def hint(self, task, request, store) -> str:
        if task.task_id != self._task_id or len(request.content_references) != 2:
            return ""
        try:
            control = json.loads(
                store.resolve(request.request_id, request.content_references[0]).text
            )
        except (FragmentStoreError, UnicodeError, json.JSONDecodeError):
            return ""
        if not isinstance(control, dict):
            return ""
        operation = control.get("operation")
        expected = (
            {"language", "operation", "question"}
            if operation == "ask"
            else {
                "language",
                "operation",
            }
        )
        if set(control) != expected:
            return ""
        instruction = operation_instruction(
            operation,
            control.get("language"),
            control.get("question"),
        )
        if not instruction:
            return ""
        return (
            f"Trusted selected-text operation is {json.dumps(control['operation'])}. "
            f"{instruction} Return operation exactly as named.\n"
        )

    def reconsideration(self, task, hint: str, raw: str, request=None, store=None) -> str | None:
        """One re-ask when the answer is the selection handed straight back.

        Measured on the live desk: `explain` on "The mitochondrion is the
        powerhouse of the cell." returned that sentence as its own
        explanation. Telling the model not to, in the instruction, did not
        stop it — so the workload checks. Explaining, summarising and
        rewriting all promise something *other* than the input; translating
        into the language it is already in, or extracting nothing when there
        is nothing, do not, which is why only three are checked.

        A re-ask rather than a refusal: the second answer is returned whatever
        it is, so this can only replace a non-answer with an answer, never
        take one away (OMNI-0571).
        """
        if task.task_id != self._task_id or request is None or store is None:
            return None
        selection = _selection_text(request, store)
        answered = _answer_result(raw, self._answer_field)
        if not selection or not answered:
            return None
        if _restates(answered, selection):
            return ECHOED_SELECTION_REPROMPT
        return None


def _selection_text(request, store) -> str:
    """The selection this request was about, or nothing readable."""
    references = getattr(request, "content_references", ())
    if len(references) != 2:
        return ""
    try:
        return store.resolve(request.request_id, references[1]).text or ""
    except Exception:  # noqa: BLE001 - a fragment this cannot read is one it cannot judge
        return ""


def _answer_result(raw: str, answer_field: str) -> str:
    """The answer text the model returned, or nothing when it returned no JSON."""
    try:
        document = json.loads(raw)
    except (UnicodeError, ValueError):
        return ""
    if not isinstance(document, dict):
        return ""
    if document.get("operation") not in RESTATEMENT_IS_A_NON_ANSWER:
        return ""
    result = document.get(answer_field)
    return result if isinstance(result, str) else ""


def _restates(answered: str, selection: str) -> str:
    """Whether the answer is the selection, ignoring surrounding blanks.

    Deliberately exact rather than fuzzy: a rewrite that changed two words is
    a rewrite, and a similarity threshold would start refusing answers for
    being close to a short input.
    """
    return answered.strip() == selection.strip()


def validated_language(value: object, error_type, *, noun: str) -> str:
    """The target a person named, in the shape every manifest declares."""
    if not valid_target_language(value):
        raise error_type(
            "language-invalid",
            f"{noun} must contain 1-{MAX_LANGUAGE_CHARACTERS} letters",
        )
    return value.strip()


def validated_translation_routes(
    value: Mapping[str, GenerationRouter] | None,
    error_type,
    *,
    noun: str = "translation language",
) -> dict[str, GenerationRouter]:
    """Language-specific routes, keyed by the same normalisation requests use."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise error_type("provider-invalid", "translation routes must be a language mapping")
    routes: dict[str, GenerationRouter] = {}
    for language, router in value.items():
        try:
            normalized = validated_language(language, error_type, noun=noun).casefold()
        except error_type as error:
            raise error_type("provider-invalid", "translation route language is invalid") from error
        if normalized in routes or not isinstance(router, GenerationRouter):
            raise error_type(
                "provider-invalid", "translation routes must be unique generation routers"
            )
        routes[normalized] = router
    return routes


def translation_route(default, routes: Mapping[str, GenerationRouter], language: str | None):
    """The router a translate request rides: language-specific, or the default."""
    if language is None:
        return default
    return routes.get(language.casefold(), default)


__all__ = [
    "ECHOED_SELECTION_REPROMPT",
    "MAX_QUESTION_CHARACTERS",
    "OPERATIONS",
    "OperationPrompting",
    "RESTATEMENT_IS_A_NON_ANSWER",
    "TRANSFORM_OPERATIONS",
    "operation_instruction",
    "translation_route",
    "validated_language",
    "validated_translation_routes",
]
