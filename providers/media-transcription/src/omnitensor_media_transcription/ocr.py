"""Best-effort OCR enrichment: exact glyphs with coordinates, never a gate.

The VulkanOCR engine reads a rendered frame in ~100 ms and answers with the
one thing the vision model structurally cannot: pixel coordinates per line.
It is an enricher by owner decision (2026-08-22, OMNI-0609): nothing it
finds, and nothing that goes wrong with it, may gate, replace, or delay the
vision model. That contract is enforced here once — no code path in this
module raises into a job. Every failure becomes an :class:`OcrRefusal`
carrying a stable code the receipt can name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class OcrRefusal:
    """Why enrichment is not available, said once and carried in receipts."""

    code: str
    detail: str


class VulkanOcr:
    """The enrichment lane over vulkanocr's engine, built to refuse politely.

    ``det_param``/``rec_param``/``dictionary`` are the mounted artifact
    paths, or ``None`` when the artifacts are not installed — absence is the
    common state on an existing deployment and is a refusal, not an error.
    The engine itself is built lazily on the first read, so a worker whose
    machine cannot serve OCR pays nothing for the attempt after the first.
    """

    def __init__(
        self,
        det_param: Path | None,
        rec_param: Path | None,
        dictionary: Path | None,
    ) -> None:
        self._det_param = det_param
        self._rec_param = rec_param
        self._dictionary = dictionary
        self._engine: Any = None
        self._refusal: OcrRefusal | None = None
        if det_param is None or rec_param is None or dictionary is None:
            self._refusal = OcrRefusal(
                "ocr-artifacts-unmounted",
                "the PP-OCRv6 artifacts are not installed; visual answers are model-only",
            )

    def read(self, image_path: Path):
        """Every text line on one rendered frame, or why there is none.

        Returns vulkanocr's ``OcrResult`` on success and :class:`OcrRefusal`
        otherwise. A refusal is remembered: once the engine cannot be built,
        later reads answer immediately instead of retrying a broken stack.
        """
        if self._refusal is not None:
            return self._refusal
        try:
            engine = self._ensure_engine()
            image = self._loaded(image_path)
            return engine.read(image)
        except BaseException as error:  # noqa: BLE001 - the refusal is the report
            refusal = OcrRefusal(self._code_for(error), str(error))
            if self._engine is None:
                # The stack could not even be built; remember that. A read
                # that failed on one frame, though, must not condemn the
                # next frame to model-only answers.
                self._refusal = refusal
            return refusal

    def close(self) -> None:
        engine, self._engine = self._engine, None
        if engine is not None:
            engine.close()

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        from vulkanocr.engine import OcrEngine, OcrModels  # noqa: PLC0415 - optional lane

        models = OcrModels(
            det_param=self._det_param,
            rec_param=self._rec_param,
            dictionary=self._dictionary,
            blobs=("input", "output"),
            dictionary_includes_blank=True,
        )
        self._engine = OcrEngine(models, device=self._device())
        return self._engine

    def _device(self):
        """Which hardware device the engine should hold, None for the default.

        Refined by OMNI-0610; the default selection is correct everywhere it
        can run at all.
        """
        return None

    @staticmethod
    def _loaded(image_path: Path):
        import cv2  # noqa: PLC0415 - arrives with the vulkanocr dependency
        import numpy  # noqa: PLC0415

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot read an image from {image_path}")
        return numpy.ascontiguousarray(image[:, :, ::-1])

    @staticmethod
    def _code_for(error: BaseException) -> str:
        code = getattr(error, "code", "")
        if isinstance(code, str) and code:
            return f"ocr-{code}"
        if isinstance(error, ImportError):
            return "ocr-runtime-missing"
        return "ocr-failed"


__all__ = ["OcrRefusal", "VulkanOcr"]
