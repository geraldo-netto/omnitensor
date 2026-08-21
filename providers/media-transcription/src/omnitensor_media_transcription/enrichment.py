"""The merge seam: the vision model is mandatory, OCR can only enrich.

Owner decision (2026-08-22, OMNI-0609): Qwen runs on every visual frame
regardless of what — or how much — the OCR lane found. The guarantee is
structural rather than disciplinary: ``transcribe`` has no code path that
returns before the vision model answered, and ``_merged`` takes the vision
transcript as a non-optional argument. The tempting shortcut — skip the slow
model when OCR looks complete — is exactly the gate this module exists to
make unwritable, and the always-runs test pins it.

``structured_text`` (a rendered SVG's own <text> nodes) keeps its standing
precedence in the vision transcriber: it is exact source data, a stronger
provenance than either model's reading, and this seam neither displaces it
nor lets OCR displace anything.
"""

from __future__ import annotations

import asyncio

from omnitensor.plugins.media_transcription import (
    RecognisedLine,
    VisualFrame,
    VisualTranscript,
)
from omnitensor.plugins.protocol import CancellationToken

from .ocr import OcrRefusal, VulkanOcr


class EnrichedVisualTranscriber:
    """A VisualTranscriber whose answers carry the OCR lane's coordinates."""

    def __init__(self, ocr: VulkanOcr, vision) -> None:
        self._ocr = ocr
        self._vision = vision
        # The last refusal, for the receipt: "no lines" and "no lane" must
        # not read the same.
        self.last_refusal: OcrRefusal | None = None

    async def transcribe(
        self, frame: VisualFrame, cancellation: CancellationToken
    ) -> VisualTranscript:
        # Concurrent, not sequential: on a two-device desk the lanes hold
        # different silicon, and running OCR first cost the dense-page read
        # 3.6 s of iGPU time the vision model spent waiting for (measured,
        # OMNI-0612). gather never drops the vision half: an OCR thread
        # failure is impossible by ocr.read's contract, and a vision failure
        # propagates exactly as it did before the lane existed.
        reading, transcript = await asyncio.gather(
            asyncio.to_thread(self._ocr.read, frame.path),
            self._vision.transcribe(frame, cancellation),
        )
        return self._merged(reading, transcript)

    def _merged(self, reading, transcript: VisualTranscript) -> VisualTranscript:
        """One answer from both lanes; ``transcript`` is not optional."""
        if isinstance(reading, OcrRefusal):
            self.last_refusal = reading
            return transcript
        self.last_refusal = None
        lines, remainder = _validated_lines(reading.lines, transcript.visible_text)
        return VisualTranscript(
            transcript.timestamp_ms,
            transcript.visible_text,
            transcript.description,
            transcript.slide_number,
            transcript.page_number,
            lines,
            remainder,
        )

    async def preflight(self) -> None:
        await self._vision.preflight()

    async def release(self) -> None:
        await asyncio.to_thread(self._ocr.close)
        await self._vision.release()


def _validated_lines(ocr_lines, visible_text: str):
    """Every OCR line judged against the model's reading (OMNI-0618).

    The model always runs (OMNI-0609), so its reading is always available
    as the second witness: a line it contains is confirmed, a line whose
    words it mostly knows but never states this way is disputed, and a line
    it never mentions stands as ocr-only — single-source, and marked as
    such. What the model read that no line corroborates is handed back as
    the remainder: the out-of-dictionary scripts and detector misses only
    the model can supply. Deliberately exact-and-token-based rather than
    fuzzy: a verdict a consumer cannot re-derive is a verdict they cannot
    audit.
    """
    model = _normalized(visible_text)
    model_tokens = set(model.split())
    lines = []
    confirmed_texts = []
    for line in ocr_lines:
        normalized = _normalized(line.text)
        if normalized and normalized in model:
            verdict = "confirmed"
            confirmed_texts.append(normalized)
        else:
            tokens = [token for token in normalized.split() if len(token) > 2]
            hits = sum(1 for token in tokens if token in model_tokens)
            verdict = "disputed" if tokens and hits * 2 >= len(tokens) else "ocr-only"
        lines.append(
            RecognisedLine(
                text=line.text,
                confidence=line.confidence,
                box_score=line.box_score,
                center_x=line.center_x,
                center_y=line.center_y,
                thickness=line.thickness,
                length=line.length,
                angle=line.angle,
                vertical=line.vertical,
                verdict=verdict,
            )
        )
    remainder = model
    for confirmed in sorted(confirmed_texts, key=len, reverse=True):
        remainder = remainder.replace(confirmed, " ", 1)
    remainder = " ".join(remainder.split())
    if remainder == model and not confirmed_texts and not lines:
        remainder = ""
    return tuple(lines), remainder


def _normalized(text: str) -> str:
    return " ".join((text or "").casefold().split())


__all__ = ["EnrichedVisualTranscriber"]
