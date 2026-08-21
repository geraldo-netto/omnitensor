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
        reading = await asyncio.to_thread(self._ocr.read, frame.path)
        transcript = await self._vision.transcribe(frame, cancellation)
        return self._merged(reading, transcript)

    def _merged(self, reading, transcript: VisualTranscript) -> VisualTranscript:
        """One answer from both lanes; ``transcript`` is not optional."""
        if isinstance(reading, OcrRefusal):
            self.last_refusal = reading
            return transcript
        self.last_refusal = None
        return VisualTranscript(
            transcript.timestamp_ms,
            transcript.visible_text,
            transcript.description,
            transcript.slide_number,
            transcript.page_number,
            tuple(
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
                )
                for line in reading.lines
            ),
        )

    async def preflight(self) -> None:
        await self._vision.preflight()

    async def release(self) -> None:
        await asyncio.to_thread(self._ocr.close)
        await self._vision.release()


__all__ = ["EnrichedVisualTranscriber"]
