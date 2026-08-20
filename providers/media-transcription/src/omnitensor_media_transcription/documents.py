"""Bounded page and text document adapters.

Three kinds of source arrive here and they are not read the same way. A PDF
or a TIFF has pages that can be rendered, so each page is text *and* a picture
the vision model describes. A `.docx` has neither: without a layout engine —
which this runtime will not invoke — a Word file has no page and no picture,
only its own text in order. A `.txt` has even less.

What a text-only source gets is the answer a presentation slide with no
images already gets: its text, and a description that says factually what the
thing is rather than inventing what a model would have seen. That is why
neither needed a change to the result contract.

The PDF path also has a fallback. PyMuPDF renders and extracts; pypdf only
extracts. When PyMuPDF is missing, a PDF is still answered — as text, with no
rendered page and so no description of one, which is visible in the answer
rather than silently worse.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import zipfile
from pathlib import Path

from omnitensor.plugins.media_transcription import (
    DocumentTranscriber,
    MediaTranscriptionError,
    VisualFrame,
    VisualTranscriber,
    VisualTranscript,
)
from omnitensor.plugins.protocol import CancellationToken

from .archives import MAX_PART_XML_BYTES, archive_entries, read_archive_entry, xml_root
from .text import joined_visible_text

MAX_RENDER_DIMENSION = 1600
# What a plain text file may weigh. A bound on what somebody selected rather
# than on what they are told back: the whole file is read into memory to be
# decoded, and the result carries all of it.
MAX_TEXT_BYTES = 32 * 1024 * 1024
_WORD_BODY = "word/document.xml"
_WORD_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_KIND = "document"
_TEXT_SUFFIXES = frozenset({".docx", ".txt"})


class DocumentPageTranscriber(DocumentTranscriber):
    """Transcribe bounded PDF/TIFF pages as text plus whole-page visuals."""

    def __init__(self, vision: VisualTranscriber) -> None:
        self._vision = vision

    async def transcribe(
        self, source: Path, cancellation: CancellationToken
    ) -> tuple[VisualTranscript, ...]:
        if source.suffix.lower() in _TEXT_SUFFIXES:
            return await self._transcribe_text(source, cancellation)
        page_count = await asyncio.to_thread(_document_page_count, source)
        root = Path(tempfile.mkdtemp(prefix="omnitensor-document-pages-"))
        try:
            results = []
            for page_number in range(1, page_count + 1):
                cancellation.raise_if_cancelled()
                text, path = await asyncio.to_thread(
                    _render_document_page, source, root, page_number
                )
                if path is None:
                    # No renderer, so no picture and nothing a model could
                    # describe. The page is still answered, and the answer
                    # says what it is rather than what nobody looked at.
                    results.append(
                        VisualTranscript(None, text, _page_description(text), None, page_number)
                    )
                    continue
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

    async def _transcribe_text(
        self, source: Path, cancellation: CancellationToken
    ) -> tuple[VisualTranscript, ...]:
        """A document that has text and no pages, as one entry with no page number.

        No `pageNumber`: a Word file is paginated by whatever opens it, and
        claiming page one would be this runtime saying where the page breaks
        fall when it has not laid the document out.
        """
        cancellation.raise_if_cancelled()
        text = await asyncio.to_thread(_extract_text_document, source)
        cancellation.raise_if_cancelled()
        return (VisualTranscript(None, text, _text_description(source, text), None, None),)


def _extract_text_document(source: Path) -> str:
    """The text of a `.docx` or a `.txt`, in the order it is written."""
    suffix = source.suffix.lower()
    try:
        if suffix == ".docx":
            return _docx_text(source)
        return _plain_text(source)
    except MediaTranscriptionError:
        raise
    except Exception as error:
        raise MediaTranscriptionError(
            "document-invalid", "selected document could not be decoded"
        ) from error


def _docx_text(source: Path) -> str:
    """Every paragraph of the body, joined in document order.

    The body only. Headers, footers and comments are separate parts, and
    folding them into the same text would put a page number between two
    sentences with nothing to say which was which.
    """
    with zipfile.ZipFile(source) as archive:
        entries = archive_entries(archive, _KIND)
        body = xml_root(
            read_archive_entry(archive, entries, _WORD_BODY, MAX_PART_XML_BYTES, _KIND), _KIND
        )
    paragraphs = []
    for paragraph in body.iter(f"{_WORD_NAMESPACE}p"):
        runs = [node.text or "" for node in paragraph.iter(f"{_WORD_NAMESPACE}t")]
        paragraphs.append("".join(runs))
    return joined_visible_text(paragraphs)


def _plain_text(source: Path) -> str:
    """A text file as UTF-8, refused rather than guessed at when it is not.

    Guessing an encoding is how a file becomes mojibake that reads like a
    transcription failure: a person can convert a file they know the encoding
    of, and cannot un-guess one this had decided for them.
    """
    size = source.stat().st_size
    if size > MAX_TEXT_BYTES:
        raise MediaTranscriptionError("document-invalid", "selected text file exceeds its limit")
    raw = source.read_bytes()
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MediaTranscriptionError(
            "document-invalid", "selected text file is not UTF-8"
        ) from error
    return joined_visible_text(decoded.splitlines())


def _text_description(source: Path, text: str) -> str:
    """What the thing is, which is all that is true when nothing was rendered."""
    kind = "Word document" if source.suffix.lower() == ".docx" else "Plain text file"
    return f"{kind} containing text." if text else f"Empty {kind}."


def _page_description(text: str) -> str:
    return (
        "Document page containing text, read without rendering."
        if text
        else "Document page with no extractable text, read without rendering."
    )

def _document_page_count(source: Path) -> int:
    try:
        suffix = source.suffix.lower()
        if suffix in _TEXT_SUFFIXES:
            # One unit of answer, not one page: a text document is paginated
            # by whatever opens it.
            return 1
        if suffix == ".pdf":
            count = _pdf_page_count(source)
        elif suffix in {".tif", ".tiff"}:
            from PIL import Image  # noqa: PLC0415 - deferred: an optional or heavy dependency

            with Image.open(source) as image:
                count = image.n_frames
        else:  # pragma: no cover - caller suffix gate
            raise ValueError("unsupported document")
    except MediaTranscriptionError:
        # A reader this installation does not have is a broken install, not a
        # fact about somebody's file — and reporting it as one sends them off
        # to re-export a document that was fine.
        raise
    except Exception as error:
        raise MediaTranscriptionError(
            "document-invalid", "selected document could not be decoded"
        ) from error
    if count < 1:
        raise MediaTranscriptionError("document-invalid", "document page count is invalid")
    return count


def _pdf_page_count(source: Path) -> int:
    """How many pages, from whichever reader this installation has."""
    reader = _pdf_reader()
    if reader == "pymupdf":
        import pymupdf  # noqa: PLC0415 - deferred: an optional or heavy dependency

        with pymupdf.open(source) as document:
            if document.needs_pass:
                raise ValueError("encrypted PDF")
            return document.page_count
    from pypdf import PdfReader  # noqa: PLC0415 - deferred: an optional or heavy dependency

    document = PdfReader(source)
    if document.is_encrypted:
        raise ValueError("encrypted PDF")
    return len(document.pages)


def _pdf_reader() -> str:
    """Which reader answers a PDF here, preferring the one that can render.

    PyMuPDF extracts text *and* renders the page, so the answer carries what
    the vision model saw as well as what the page says. pypdf only extracts,
    which is a smaller answer rather than no answer — and it is the whole
    difference between a person getting their text and getting a refusal
    because an optional dependency is not installed.
    """
    try:
        import pymupdf  # noqa: F401, PLC0415 - deferred: an optional or heavy dependency
    except ImportError:
        pass
    else:
        return "pymupdf"
    try:
        import pypdf  # noqa: F401, PLC0415 - deferred: an optional or heavy dependency
    except ImportError as error:
        raise MediaTranscriptionError(
            "document-runtime-unavailable", "install the PyMuPDF or pypdf document reader"
        ) from error
    return "pypdf"


def _render_document_page(source: Path, root: Path, page_number: int) -> tuple[str, Path | None]:
    path = root / f"page-{page_number:02d}.png"
    try:
        if source.suffix.lower() != ".pdf":
            text = _render_tiff_page(source, path, page_number)
        elif _pdf_reader() == "pymupdf":
            text = _render_pdf_page(source, path, page_number)
        else:
            # Text only: pypdf cannot render, and a page nobody rendered is a
            # page nothing can describe.
            return _extract_pdf_page(source, page_number), None
    except MediaTranscriptionError:
        raise
    except Exception as error:
        raise MediaTranscriptionError(
            "document-invalid", "document page could not be rendered"
        ) from error
    return text, path


def _extract_pdf_page(source: Path, page_number: int) -> str:
    from pypdf import PdfReader  # noqa: PLC0415 - deferred: an optional or heavy dependency

    document = PdfReader(source)
    page = document.pages[page_number - 1]
    return joined_visible_text((page.extract_text() or "").splitlines())


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
