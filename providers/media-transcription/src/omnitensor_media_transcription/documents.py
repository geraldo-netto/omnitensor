"""Bounded PDF and TIFF extraction adapters."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

from omnitensor.plugins.media_transcription import (
    DocumentTranscriber,
    MediaTranscriptionError,
    VisualFrame,
    VisualTranscriber,
    VisualTranscript,
)
from omnitensor.plugins.protocol import CancellationToken

from .text import joined_visible_text

MAX_RENDER_DIMENSION = 1600


class DocumentPageTranscriber(DocumentTranscriber):
    """Transcribe bounded PDF/TIFF pages as text plus whole-page visuals."""

    def __init__(self, vision: VisualTranscriber) -> None:
        self._vision = vision

    async def transcribe(
        self, source: Path, cancellation: CancellationToken
    ) -> tuple[VisualTranscript, ...]:
        page_count = await asyncio.to_thread(_document_page_count, source)
        root = Path(tempfile.mkdtemp(prefix="omnitensor-document-pages-"))
        try:
            results = []
            for page_number in range(1, page_count + 1):
                cancellation.raise_if_cancelled()
                text, path = await asyncio.to_thread(
                    _render_document_page, source, root, page_number
                )
                visual = await self._vision.transcribe(VisualFrame(path, None), cancellation)
                results.append(
                    VisualTranscript(
                        None,
                        joined_visible_text((text, visual.visible_text)),
                        visual.description,
                        None,
                        page_number,
                    )
                )
            return tuple(results)
        finally:
            await asyncio.to_thread(shutil.rmtree, root, True)


def _document_page_count(source: Path) -> int:
    try:
        suffix = source.suffix.lower()
        if suffix == ".pdf":
            import pymupdf  # noqa: PLC0415 - deferred: an optional or heavy dependency

            with pymupdf.open(source) as document:
                count = document.page_count
                if document.needs_pass:
                    raise ValueError("encrypted PDF")
        elif suffix in {".tif", ".tiff"}:
            from PIL import Image  # noqa: PLC0415 - deferred: an optional or heavy dependency

            with Image.open(source) as image:
                count = image.n_frames
        else:  # pragma: no cover - caller suffix gate
            raise ValueError("unsupported document")
    except Exception as error:
        raise MediaTranscriptionError(
            "document-invalid", "selected document could not be decoded"
        ) from error
    if count < 1:
        raise MediaTranscriptionError("document-invalid", "document page count is invalid")
    return count


def _render_document_page(source: Path, root: Path, page_number: int) -> tuple[str, Path]:
    path = root / f"page-{page_number:02d}.png"
    try:
        if source.suffix.lower() == ".pdf":
            text = _render_pdf_page(source, path, page_number)
        else:
            text = _render_tiff_page(source, path, page_number)
    except MediaTranscriptionError:
        raise
    except Exception as error:
        raise MediaTranscriptionError(
            "document-invalid", "document page could not be rendered"
        ) from error
    return text, path


def _render_pdf_page(source: Path, path: Path, page_number: int) -> str:
    import pymupdf  # noqa: PLC0415 - deferred: an optional or heavy dependency

    with pymupdf.open(source) as document:
        page = document.load_page(page_number - 1)
        text = joined_visible_text(page.get_text("text").splitlines())
        longest = max(float(page.rect.width), float(page.rect.height), 1.0)
        scale = min(2.0, MAX_RENDER_DIMENSION / longest)
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        if pixmap.width * pixmap.height > 50_000_000:
            raise MediaTranscriptionError(
                "document-invalid", "document page dimensions are invalid"
            )
        pixmap.save(path)
    return text


def _render_tiff_page(source: Path, path: Path, page_number: int) -> str:
    from PIL import Image  # noqa: PLC0415 - deferred: an optional or heavy dependency

    with Image.open(source) as image:
        image.seek(page_number - 1)
        if image.width * image.height > 50_000_000:
            raise MediaTranscriptionError(
                "document-invalid", "document page dimensions are invalid"
            )
        rendered = image.convert("RGB")
        rendered.thumbnail((MAX_RENDER_DIMENSION, MAX_RENDER_DIMENSION))
        rendered.save(path, format="PNG", optimize=False)
    return ""


__all__ = ["DocumentPageTranscriber"]
