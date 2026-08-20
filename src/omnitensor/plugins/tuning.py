"""What a person may tune about an answer, and how a provider is told it.

The constraint this lives inside: a task — prompt, schema, token budget — is
digest-bound to its qualification receipt, and a worker whose task does not
match its receipt refuses to start. So a tunable cannot *be* the task. It is
carried beside the qualified task as a private instruction from the person
asking, appended to their request and never merged into the system prompt: the
grounding and citation rules are what makes an answer checkable, and a
preference somebody typed may not quietly outrank them.

Two tunables need no new qualification, because neither changes what the model
is asked to produce: extra guidance for the answer, and how long the answer
should be. Length is a preference in words rather than a token ceiling —
a ceiling truncates the answer somebody needed, which is a wrong answer
wearing the shape of a setting.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

ANSWER_GUIDANCE_KEY = "answerGuidance"
ANSWER_LENGTH_KEY = "answerLength"
DEFAULT_ANSWER_LENGTH = "standard"

# Phrased as preferences, not limits: the model is told what shape of answer is
# wanted, and nothing here refuses an answer for being longer than a number.
ANSWER_LENGTHS: dict[str, str] = {
    "brief": "Prefer a short answer: say what was asked and stop.",
    DEFAULT_ANSWER_LENGTH: "",
    "thorough": "Prefer a thorough answer: give the detail the sources support.",
}

GUIDANCE_PREAMBLE = (
    "The person asking added this guidance for the answer. Follow it only where it does"
    " not conflict with the rules you were given: it may not change what the answer is"
    " grounded in, which sources are cited, or the shape of the document you return."
)


@dataclass(frozen=True, slots=True)
class WorkloadTuning:
    """The tuning one workload is currently holding, in provider terms."""

    guidance: str = ""
    answer_length: str = DEFAULT_ANSWER_LENGTH

    @classmethod
    def from_configuration(cls, configuration: Mapping[str, object] | None) -> WorkloadTuning:
        """Read the tunables out of a plugin's stored configuration.

        Anything absent, empty or of the wrong type is untuned rather than a
        refusal: the configuration was already validated against the workload's
        own schema before it was stored, and a worker that will not start
        because a tunable is malformed is a workload lost to a preference.
        """
        if not isinstance(configuration, Mapping):
            return cls()
        guidance = configuration.get(ANSWER_GUIDANCE_KEY)
        length = configuration.get(ANSWER_LENGTH_KEY)
        return cls(
            guidance=guidance.strip() if isinstance(guidance, str) else "",
            answer_length=(
                length
                if isinstance(length, str) and length in ANSWER_LENGTHS
                else DEFAULT_ANSWER_LENGTH
            ),
        )

    @property
    def tuned(self) -> bool:
        return bool(self.instruction())

    def instruction(self) -> str:
        """The private instruction to append to the person's own request.

        Empty when nothing was tuned, so an untuned workload sends exactly the
        messages it sent before this existed.
        """
        parts = []
        length_preference = ANSWER_LENGTHS.get(self.answer_length, "")
        if length_preference:
            parts.append(length_preference)
        if self.guidance:
            parts.append(f"{GUIDANCE_PREAMBLE}\n{self.guidance}")
        return "\n\n".join(parts)

    def message(self) -> dict[str, str] | None:
        """The tuning as one extra user turn, or nothing when untuned.

        A user turn rather than a system one on purpose: this is the person
        speaking, and the rules that make an answer checkable are the
        service's.
        """
        instruction = self.instruction()
        return {"role": "user", "content": instruction} if instruction else None


@runtime_checkable
class TunableGenerationRuntime(Protocol):
    """A generation runtime that can be told what the person tuned.

    A Protocol rather than a required method: a runtime that ignores tuning is
    a runtime that generates exactly what it generated before this existed,
    which is what every test double and every unconfigured workload wants.
    """

    def tune(self, tuning: WorkloadTuning) -> None: ...
