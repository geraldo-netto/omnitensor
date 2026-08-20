"""What a workload tells the model beyond its own prompt.

A generation backend loads a model and decodes tokens. None of that changes
when a workload is added — but the backend used to branch on four workload ids
to assemble prompts, so a GPU adapter knew what an event is, what a
selected-text operation is, and which refusal is worth putting back to the
model once.

This is the port that ends that. Each workload states its own prompt-side
policy beside the task it belongs to, the factory hands it to the adapter, and
the adapter asks rather than knows.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .generation import GenerationRequest, GenerationTask


@runtime_checkable
class TaskPrompting(Protocol):
    """One workload's prompt-side policy, as the backend sees it."""

    def hint(
        self,
        task: GenerationTask,
        request: GenerationRequest,
        store: object,
    ) -> str:
        """Whatever this workload adds to the system prompt.

        Never private text: only opaque references, what a trusted control
        fragment says the person asked for, and whether a preflight matched.
        """
        ...

    def reconsideration(
        self,
        task: GenerationTask,
        hint: str,
        raw: str,
        request: GenerationRequest,
        store: object,
    ) -> str | None:
        """The one re-prompt an answer earns, or nothing to take it as final.

        The request and the fragment store are here because judging an answer
        often means comparing it with what was asked: a selection handed back
        unchanged is not an explanation of itself, and only the workload knows
        that. The backend stays ignorant of what any of it means.
        """
        ...


class NoPrompting:
    """A workload whose task prompt says everything it has to say."""

    __slots__ = ()

    def hint(
        self,
        task: GenerationTask,
        request: GenerationRequest,
        store: object,
    ) -> str:
        return ""

    def reconsideration(
        self,
        task: GenerationTask,
        hint: str,
        raw: str,
        request: GenerationRequest,
        store: object,
    ) -> str | None:
        return None


NO_PROMPTING = NoPrompting()

__all__ = ["NO_PROMPTING", "NoPrompting", "TaskPrompting"]
