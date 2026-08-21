"""The merge seam and its one rule: the vision model always runs."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

from omnitensor_media_transcription.enrichment import EnrichedVisualTranscriber
from omnitensor_media_transcription.ocr import OcrRefusal

from omnitensor.plugins.media_transcription import VisualFrame, VisualTranscript


class RecordingVision:
    def __init__(self):
        self.calls = 0
        self.released = False
        self.preflown = False

    async def transcribe(self, frame, cancellation):
        self.calls += 1
        return VisualTranscript(None, "model reading", "a page of text")

    async def preflight(self):
        self.preflown = True

    async def release(self):
        self.released = True


class FullCoverageOcr:
    """Full-coverage, high-confidence lines: the exact bait for the skip."""

    def __init__(self):
        self.reads = 0
        self.closed = False

    def read(self, _path):
        self.reads += 1
        return SimpleNamespace(
            lines=(
                SimpleNamespace(
                    text="model reading",
                    confidence=0.99,
                    box_score=0.99,
                    center_x=10.0,
                    center_y=20.0,
                    thickness=12.0,
                    length=120.0,
                    angle=90.0,
                    vertical=False,
                ),
            )
        )

    def close(self):
        self.closed = True


def test_the_vision_model_runs_even_when_ocr_found_everything():
    """The always-runs rule (OMNI-0609, owner decision 2026-08-22).

    OCR answering with complete, high-confidence coverage is the exact
    situation in which skipping the slow model is tempting — and forbidden.
    """
    vision = RecordingVision()
    ocr = FullCoverageOcr()
    subject = EnrichedVisualTranscriber(ocr, vision)

    transcript = asyncio.run(
        subject.transcribe(VisualFrame(Path("/frame.png"), None), _cancellation())
    )

    assert vision.calls == 1
    assert ocr.reads == 1
    assert transcript.visible_text == "model reading"
    assert transcript.description == "a page of text"
    assert [line.text for line in transcript.ocr_lines] == ["model reading"]
    assert (transcript.ocr_lines[0].center_x, transcript.ocr_lines[0].center_y) == (10.0, 20.0)
    assert subject.last_refusal is None


def test_an_ocr_refusal_degrades_to_model_only_and_is_remembered():
    vision = RecordingVision()

    class RefusingOcr:
        def read(self, _path):
            return OcrRefusal("ocr-artifacts-unmounted", "not installed")

        def close(self):
            return None

    subject = EnrichedVisualTranscriber(RefusingOcr(), vision)
    transcript = asyncio.run(
        subject.transcribe(VisualFrame(Path("/frame.png"), None), _cancellation())
    )

    assert vision.calls == 1
    assert transcript.ocr_lines == ()
    assert transcript.visible_text == "model reading"
    assert subject.last_refusal is not None
    assert subject.last_refusal.code == "ocr-artifacts-unmounted"


def test_release_closes_the_ocr_engine_and_the_vision_model():
    vision = RecordingVision()
    ocr = FullCoverageOcr()
    subject = EnrichedVisualTranscriber(ocr, vision)
    asyncio.run(subject.release())
    asyncio.run(subject.preflight())
    assert ocr.closed is True
    assert vision.released is True
    assert vision.preflown is True


def _cancellation():
    return SimpleNamespace(raise_if_cancelled=lambda: None, cancelled=False)
