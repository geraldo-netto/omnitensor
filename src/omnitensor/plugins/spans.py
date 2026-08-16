"""Split a document into pieces a model can hold, without losing any of it.

Translation is where this became necessary. A 32,768-token context and a fixed
output budget meant a long document came back truncated — an answer that looks
finished and is not, which is the worst kind. The fix is not a bigger cap: ten
megabytes of text is roughly three million tokens, so no cap makes it fit. The
fix is to stop asking one generation to do it.

So: split on the boundaries a document already has, translate each piece with a
budget derived from that piece, and put them back in order. Three rules hold
the whole thing up.

* **Nothing is dropped.** Every character of the input appears in exactly one
  span, in order. The test suite reassembles and compares, because a splitter
  that quietly loses a paragraph is indistinguishable from a good one until
  somebody reads the output.
* **Splits go where a reader would put them.** Blank lines first, then
  sentence ends, then whitespace. A span that starts mid-word costs the model
  the sentence it was in the middle of.
* **A span that cannot be split is still emitted whole.** A 40,000-character
  line with no space in it — minified JSON, a base64 blob — is handed over as
  it is rather than cut. It may fail; failing loudly on one span beats
  silently mangling it.

Token counts here are estimates, deliberately conservative. The real tokenizer
lives inside the worker, and this has to run before a model is loaded to decide
how many spans there will be.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Characters per token, conservatively low so an estimate errs toward smaller
# spans. English prose runs about four; code, Hebrew and heavily accented text
# run lower, and a span that turns out smaller than budgeted costs nothing
# while one that turns out larger costs a refusal.
CHARACTERS_PER_TOKEN = 2.6

# Translation output can legitimately be longer than its input — Hebrew and
# German both do it — so the budget for a span is its own estimate with room.
# Never used to *cut* a translation: it is what the span is sized against so
# that the model is never asked for more than it can produce.
OUTPUT_HEADROOM = 1.9

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?;:])\s+")
_WHITESPACE = re.compile(r"\s+")


def estimate_tokens(text: str) -> int:
    """About how many tokens this text will become. Rounded up, never down."""
    return int(len(text) / CHARACTERS_PER_TOKEN) + 1


@dataclass(frozen=True, slots=True)
class Span:
    """One piece of a document, and where it came from.

    ``start`` and ``end`` are character offsets into the original, so a
    reassembled document can be checked against the one that went in, and so a
    failure can name the part of the document it happened in.
    """

    index: int
    start: int
    end: int
    text: str

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)

    @property
    def output_budget(self) -> int:
        """Tokens to allow this span's answer.

        Derived from the span, not fixed: that is the whole difference between
        this and the 1,024-token cap it replaces. Generous, because a
        translation that runs out of budget is a truncated answer, and this
        codebase does not produce those.
        """
        return int(self.tokens * OUTPUT_HEADROOM) + 64


def split(text: str, *, budget_tokens: int) -> tuple[Span, ...]:
    """``text`` as spans of at most ``budget_tokens`` estimated tokens.

    Empty input yields no spans — a document with nothing in it has nothing to
    translate, and an empty span would be a generation nobody needs.
    """
    if budget_tokens < 1:
        raise ValueError("a span budget of no tokens cannot hold anything")
    if not text:
        return ()
    limit = int(budget_tokens * CHARACTERS_PER_TOKEN)
    pieces = _split_to_limit(text, limit)
    spans = []
    at = 0
    for index, piece in enumerate(pieces):
        start = text.index(piece, at)
        at = start + len(piece)
        spans.append(Span(index=index, start=start, end=at, text=piece))
    return tuple(spans)


def _split_to_limit(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    for pattern in (_PARAGRAPH, _SENTENCE, _WHITESPACE):
        pieces = _gather(text, pattern, limit)
        if pieces is not None:
            return pieces
    # Nothing to split on at all: one long unbroken run. Handed over whole
    # rather than cut mid-token — failing on one span is recoverable, mangling
    # it silently is not.
    return [text]


def _gather(text: str, pattern: re.Pattern[str], limit: int) -> list[str] | None:
    """Greedily group ``pattern``-delimited parts into pieces under ``limit``.

    Returns ``None`` when this boundary cannot produce pieces small enough, so
    the caller can try a finer one. Separators stay attached to the part before
    them, which is what keeps reassembly exact.
    """
    parts = _parts(text, pattern)
    if len(parts) < 2:
        return None
    pieces: list[str] = []
    current = ""
    for part in parts:
        if current and len(current) + len(part) > limit:
            pieces.append(current)
            current = part
        else:
            current += part
    if current:
        pieces.append(current)
    if any(len(piece) > limit for piece in pieces):
        # A single part is already too long; a finer boundary has to try.
        return None
    return pieces


def _parts(text: str, pattern: re.Pattern[str]) -> list[str]:
    """``text`` cut at every match, with each separator kept on its left part."""
    parts = []
    at = 0
    for match in pattern.finditer(text):
        parts.append(text[at : match.end()])
        at = match.end()
    if at < len(text):
        parts.append(text[at:])
    return parts


def reassemble(pieces: list[str] | tuple[str, ...]) -> str:
    """Put translated spans back together in the order they were split.

    Nothing is inserted between them: the separators travelled with the spans,
    so the shape of the document — its paragraphs, its line breaks — is the
    input's shape rather than one this invented.
    """
    return "".join(pieces)


def covers(text: str, spans: tuple[Span, ...]) -> bool:
    """Whether these spans account for every character of ``text``.

    Cheap enough to assert before spending an hour of GPU time on a document,
    and the property that everything else here depends on.
    """
    if not spans:
        return not text
    return reassemble([span.text for span in spans]) == text


__all__ = [
    "CHARACTERS_PER_TOKEN",
    "OUTPUT_HEADROOM",
    "Span",
    "covers",
    "estimate_tokens",
    "reassemble",
    "split",
]
