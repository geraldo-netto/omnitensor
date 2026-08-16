"""Translate a whole document by translating every span of it.

The half of OMNI-0360 that does the work. :mod:`omnitensor.plugins.spans`
decides where a document breaks; this walks those pieces, asks the generation
router for each one, and puts the answers back together in order.

Three properties, each of which was a way the naive version would have lied:

* **Every span is translated or the whole thing fails.** A document missing its
  fourth paragraph, delivered as if complete, is worse than an error. A span
  that cannot be translated stops the job and names where it was.
* **Progress is per span, and honest.** A document is many generations, and a
  spinner that sits at "working" for forty minutes tells somebody nothing. The
  fraction is spans finished over spans total, which is a real proportion of
  the work rather than a guess about time.
* **Cancellation lands between spans.** A person who changes their mind after
  three of two hundred spans should wait for one generation, not all of them.

The task's own output budget is a *sizing input* here, not a limit on the
answer: spans are chosen small enough that their translations fit inside it
comfortably, so no translation is ever cut off. Making a document fit by
shortening its translation is the thing this module exists to avoid.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .spans import OUTPUT_HEADROOM, Span, covers, reassemble, split

# The room a span's translation is given inside the task's fixed output budget.
# A translation can legitimately run longer than its source, so a span is sized
# well under the budget rather than up to it: the cost of a span being smaller
# than necessary is a few seconds, and the cost of one being too large is a
# truncated paragraph.
SPAN_SAFETY = 1.25


class TranslationError(ValueError):
    """A refusal that names what could not be translated, never its content."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def span_budget_tokens(output_tokens: int) -> int:
    """How large a span may be, given what one generation may produce.

    Derived rather than chosen: the task's budget is fixed by its qualification
    receipt, so the free variable is how much source is handed to it at once.
    """
    budget = int(output_tokens / (OUTPUT_HEADROOM * SPAN_SAFETY))
    if budget < 1:
        raise TranslationError(
            "budget-too-small", "the task's output budget cannot hold any span"
        )
    return budget


def plan(text: str, *, output_tokens: int) -> tuple[Span, ...]:
    """The spans this document will be translated in.

    Cheap, and worth doing before anything is loaded: the number of spans is
    what tells a person whether this is a coffee or an afternoon.
    """
    spans = split(text, budget_tokens=span_budget_tokens(output_tokens))
    if not covers(text, spans):
        # Belt and braces. The splitter is tested for this, and a document
        # silently losing a paragraph is severe enough to check again here
        # rather than trust a test written months earlier.
        raise TranslationError("split-incomplete", "the document did not split cleanly")
    return spans


@dataclass(frozen=True, slots=True)
class SpanTranslation:
    """One span's answer, and what it cost."""

    span: Span
    text: str


@dataclass(frozen=True, slots=True)
class Translation:
    """A whole document, translated."""

    text: str
    spans: tuple[SpanTranslation, ...]

    @property
    def span_count(self) -> int:
        return len(self.spans)


async def translate_document(
    text: str,
    *,
    output_tokens: int,
    translate_span: Callable,
    on_progress: Callable[[int, int], object] | None = None,
    raise_if_cancelled: Callable[[], object] | None = None,
) -> Translation:
    """Translate ``text`` span by span and reassemble it.

    ``translate_span`` is awaited with one :class:`Span` and returns its
    translation; everything about *how* — which model, which route, which
    prompt — belongs to the caller, so this stays testable without a GPU and
    honest about what it guarantees.
    """
    spans = plan(text, output_tokens=output_tokens)
    if not spans:
        return Translation("", ())
    translated: list[SpanTranslation] = []
    for span in spans:
        if raise_if_cancelled is not None:
            # Between spans, never inside one: a half-written span is not
            # something anybody can use, so the generation is allowed to end.
            raise_if_cancelled()
        answer = await translate_span(span)
        if not isinstance(answer, str) or not answer.strip():
            raise TranslationError(
                "span-untranslated",
                f"span {span.index + 1} of {len(spans)} came back empty",
            )
        translated.append(SpanTranslation(span, answer))
        if on_progress is not None:
            on_progress(len(translated), len(spans))
    return Translation(
        text=reassemble([item.text for item in translated]),
        spans=tuple(translated),
    )


def progress_fraction(done: int, total: int) -> float:
    """Spans finished over spans total.

    A real proportion of the work rather than a guess about time — which is
    what a person watching a two-hour job needs, and what a fraction that
    reaches 74% in three seconds and then stops emphatically is not.
    """
    if total < 1:
        return 1.0
    return min(1.0, max(0.0, done / total))


def estimated_seconds(spans: Sequence[Span], seconds_per_span: float) -> float:
    """How long this document is likely to take, from a measured rate.

    Stated as a number a person can act on before they wait for it. The rate
    comes from what this machine actually did, not from a constant.
    """
    return max(0.0, len(spans) * max(0.0, seconds_per_span))


__all__ = [
    "SPAN_SAFETY",
    "SpanTranslation",
    "Translation",
    "TranslationError",
    "estimated_seconds",
    "plan",
    "progress_fraction",
    "span_budget_tokens",
    "translate_document",
]
