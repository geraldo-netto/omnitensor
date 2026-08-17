"""Extract from as many sources as a person selected, one pass at a time.

Nothing bounds how many events an answer may hold, which is the rule. What is
bounded is the context: prompt, sources and the answer all share it, so twenty
long documents cannot be read in one generation however much the contract
allows. The resolution is the one translation reached with spans — split the
*work*, never the answer.

So the fragments are grouped into passes that fit, each pass is generated on
its own, and the events accumulate. A meeting named on three pages is one
event, so accumulation deduplicates on the same key the parser uses; without
that, splitting the work would multiply the answer.
"""

from __future__ import annotations

from collections.abc import Sequence

from .events import EventCandidate, GroundedEventResult
from .spans import estimate_tokens

# What is left for the sources after the prompt and the answer. An event costs
# 400-600 tokens to write, mostly the two digests each evidence entry carries,
# so a pass that filled the context with input would have nowhere to put its
# output and would stop mid-object.
SOURCE_SHARE = 0.4

# A pass always carries at least one fragment. A source too large to share a
# pass still gets one to itself rather than being dropped or trimmed.
MINIMUM_PER_PASS = 1


def plan_passes(
    fragments: Sequence[tuple[str, str]],
    *,
    context_tokens: int,
    prompt_tokens: int = 0,
) -> tuple[tuple[str, ...], ...]:
    """Group ``(reference, text)`` pairs into passes that fit the context.

    Returns the references per pass, in order, every reference exactly once.
    """
    budget = max(int(context_tokens * SOURCE_SHARE) - prompt_tokens, 1)
    passes: list[tuple[str, ...]] = []
    current: list[str] = []
    spent = 0
    for reference, text in fragments:
        cost = estimate_tokens(text)
        if current and spent + cost > budget:
            passes.append(tuple(current))
            current, spent = [], 0
        current.append(reference)
        spent += cost
    if current:
        passes.append(tuple(current))
    return tuple(passes)


def merge(results: Sequence[GroundedEventResult], request_id: str) -> GroundedEventResult:
    """One result from many passes, without counting an event twice.

    The outcome is the honest combination: succeeded when anything was found,
    refused only when every pass refused — a source that held nothing does not
    erase what another source held. ``partial`` survives from any pass, since
    it means some material could not be read at all.
    """
    events: list[EventCandidate] = []
    seen: set[tuple] = set()
    ids: set[str] = set()
    duplicates = 0
    outcomes = {result.outcome for result in results}
    for result in results:
        for event in result.events:
            key = _key(event)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            events.append(_with_unique_id(event, ids))
    if not events:
        detail = next((result.detail for result in results if result.detail), "")
        return GroundedEventResult(request_id, "refused", "no-event-found", detail, (), duplicates)
    outcome = "partial" if "partial" in outcomes else "succeeded"
    detail = "" if outcome == "succeeded" else "some selected material could not be read"
    return GroundedEventResult(
        request_id, outcome, "events-extracted", detail, tuple(events), duplicates
    )


def _key(event: EventCandidate) -> tuple:
    label = tuple(event.label.text.casefold().split())
    place = ()
    if event.where is not None:
        stated = event.where.address.full if event.where.address else event.where.venue
        stated = stated or event.where.url
        place = () if stated is None else tuple(stated.casefold().split())
    return label, event.when.date, event.when.time, place


def _with_unique_id(event: EventCandidate, taken: set[str]) -> EventCandidate:
    """Keep candidate ids unique across passes, which number independently."""
    from dataclasses import replace  # noqa: PLC0415 - only needed on collision

    identity = event.candidate_id
    if identity not in taken:
        taken.add(identity)
        return event
    suffix = 2
    while f"{identity}-{suffix}" in taken:
        suffix += 1
    identity = f"{identity}-{suffix}"
    taken.add(identity)
    return replace(event, candidate_id=identity)


__all__ = ["MINIMUM_PER_PASS", "SOURCE_SHARE", "merge", "plan_passes"]
