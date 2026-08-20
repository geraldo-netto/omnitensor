"""How extracted text from a page, a slide or a frame is put together."""

from omnitensor.plugins.media_transcription import MediaTranscriptionError


def joined_visible_text(parts) -> str:
    """Every part that says something, in order, each said once.

    There used to be a ceiling of 16,384 characters here, and everything went
    through it — document pages, slides, frames — so a dense page of a book or
    a two-column paper failed the whole transcription, with a message about a
    presentation slide. The published result schema bounds neither
    `visibleText` nor `description`: the ceiling was this file's own, and a
    ceiling on what somebody is told back is a wrong answer waiting for a
    large enough page.

    The NUL check stays, because that is not about size: a transcript with a
    NUL in it is not text, and it would travel through the contract as a
    string that nothing downstream can render.

    Repeats are dropped. A page is read twice — once from its text layer and
    once by the vision model looking at the rendered picture — and on a page
    whose picture says exactly what the text layer already said, the answer
    carried the same sentences twice. What the model saw and extraction
    missed is still kept: only a part identical to one already taken is
    dropped, never a part that merely overlaps.
    """
    kept: list[str] = []
    for part in parts:
        text = str(part).strip()
        if text and text not in kept:
            kept.append(text)
    joined = "\n".join(kept)
    if "\x00" in joined:
        raise MediaTranscriptionError(
            "presentation-invalid", "transcribed text is not text: it contains a NUL"
        )
    return joined


__all__ = ["joined_visible_text"]
