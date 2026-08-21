"""Verbatim OCR extraction for selected images and scanned pages.

The ask workload's grounding store must hold source text, not a model's
claim about it — so this adapter reads pixels through the VulkanOCR engine
(deterministic, verbatim, the same PP-OCRv6 pair the media lane pins) and
never through a vision model, whose reading would give hallucinated quotes
citations (OMNI-0615, owner direction 2026-08-22).

A selected image is read directly. A PDF page keeps its text layer when it
has one; a page without one — a scan — is rendered and read the same way.
The previous image path went through PyMuPDF's Tesseract binding, which the
isolated worker does not carry, so a selected screenshot failed with
"OCR is unavailable in the isolated worker"; this adapter is why it now
answers.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from .event_workload import EventWorkloadError, PageContent
from .extraction import AdapterKind
from .ingestion import IngestedFile

# Rendered resolution for a page with no text layer: matches the media
# pipeline's render bound; the engine's own detector letterboxes further.
_SCAN_RENDER_DPI = 200


class VulkanOcrExtractionAdapter:
    """Images and scanned PDF pages as verbatim, citable page text."""

    kind = AdapterKind.OCR

    def __init__(self, det_param: Path, rec_param: Path, dictionary: Path) -> None:
        self._det_param = Path(det_param)
        self._rec_param = Path(rec_param)
        self._dictionary = Path(dictionary)
        # One engine serves however many sources and pages a request
        # extracts (OMNI-0621): it was being rebuilt per page, which priced
        # a twenty-page scan at twenty model loads. The workload closes it
        # after its extraction phase, so the ~700 MiB never lingers into
        # embedding or generation.
        self._engine = None

    async def pages(self, item: IngestedFile) -> AsyncIterator[PageContent]:
        path = Path(item.path)
        if path.suffix.lower() == ".pdf":
            async for page in self._pdf_pages(path):
                yield page
            return
        text = await asyncio.to_thread(self._read_image_file, path)
        yield PageContent(1, text)

    async def _pdf_pages(self, path: Path) -> AsyncIterator[PageContent]:
        try:
            import pymupdf  # type: ignore[import-not-found]  # noqa: PLC0415 - deferred: an optional or heavy dependency
        except ImportError as error:
            raise EventWorkloadError(
                "adapter-unavailable", "install the documents extra for PDF extraction"
            ) from error
        document = await asyncio.to_thread(pymupdf.open, path)
        try:
            for index in range(document.page_count):
                page = document[index]
                text = await asyncio.to_thread(page.get_text, "text")
                if not text.strip():
                    # A scan: no text layer to quote, so the rendered page is
                    # read the same way a selected image is.
                    text = await asyncio.to_thread(self._read_rendered_page, page)
                yield PageContent(index + 1, text)
        finally:
            await asyncio.to_thread(document.close)

    def _read_rendered_page(self, page) -> str:
        import numpy  # noqa: PLC0415 - deferred: arrives with the engine

        pixmap = page.get_pixmap(dpi=_SCAN_RENDER_DPI, alpha=False)
        image = numpy.frombuffer(pixmap.samples, dtype=numpy.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )
        if pixmap.n != 3:
            image = numpy.ascontiguousarray(image[:, :, :3])
        return self._read_array(numpy.ascontiguousarray(image))

    def _read_image_file(self, path: Path) -> str:
        import cv2  # noqa: PLC0415 - deferred: arrives with the engine
        import numpy  # noqa: PLC0415

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise EventWorkloadError("source-invalid", f"cannot read an image from {path.name}")
        return self._read_array(numpy.ascontiguousarray(image[:, :, ::-1]))

    def _read_array(self, rgb) -> str:
        try:
            result = self._ensure_engine().read(rgb)
        except EventWorkloadError:
            raise
        except BaseException as error:  # noqa: BLE001 - mapped into the adapter's vocabulary
            # The engine's own refusals carry stable codes; they cross into
            # this adapter's error domain instead of escaping raw (OMNI-0621).
            code = getattr(error, "code", "") or "ocr-failed"
            raise EventWorkloadError(str(code), str(error)) from error
        return "\n".join(line.text for line in result.lines)

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        try:
            from vulkanocr import models_for_port  # noqa: PLC0415 - optional lane
            from vulkanocr.engine import OcrEngine  # noqa: PLC0415
        except ImportError as error:
            raise EventWorkloadError(
                "adapter-unavailable", "the vulkanocr engine is not installed"
            ) from error
        # The port is named, not restated (OMNI-0622/VOCR-0061).
        models = models_for_port("avafly-v6", self._det_param, self._rec_param, self._dictionary)
        self._engine = OcrEngine(models)
        return self._engine

    def close(self) -> None:
        """Give the engine back; the workload calls this after extraction."""
        engine, self._engine = self._engine, None
        if engine is not None:
            engine.close()


__all__ = ["VulkanOcrExtractionAdapter"]
