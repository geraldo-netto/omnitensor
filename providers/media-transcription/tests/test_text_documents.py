"""Word documents, plain text, and a PDF read without a renderer.

Three sources that have text and no page anybody rendered. A `.docx` is
paginated by whatever opens it, a `.txt` has no pages at all, and a PDF read
through pypdf has pages whose pictures nothing produced — so each is answered
with its text and a description of what the thing is, which is the answer a
presentation slide carrying no images has always given.
"""

from __future__ import annotations

import asyncio
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import omnitensor_media_transcription.documents as documents
import pytest

from omnitensor.plugins.media_transcription import MediaTranscriptionError
from omnitensor.sdk import CancellationController

WORD_XML = (
    '<?xml version="1.0"?>'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    "<w:body>"
    "<w:p><w:r><w:t>Release plan</w:t></w:r></w:p>"
    "<w:p><w:r><w:t>The migration ran </w:t></w:r><w:r><w:t>overnight.</w:t></w:r></w:p>"
    "<w:p/>"
    "</w:body></w:document>"
)


class _Vision:
    """A vision port that fails the test if a text document reaches it."""

    def __init__(self) -> None:
        self.calls = 0

    async def transcribe(self, frame, cancellation):
        self.calls += 1
        raise AssertionError("a text document has no page to describe")

    async def release(self):  # pragma: no cover - never reached
        return None


def _docx(path: Path, body: str = WORD_XML) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", body)
    return path


def _transcribe(source: Path):
    vision = _Vision()
    transcriber = documents.DocumentPageTranscriber(vision)
    results = asyncio.run(transcriber.transcribe(source, CancellationController()))
    return results, vision


def test_a_word_document_is_read_in_order_and_says_what_it_is(tmp_path):
    (result,), vision = _transcribe(_docx(tmp_path / "plan.docx"))

    assert result.visible_text == "Release plan\nThe migration ran overnight."
    assert result.description == "Word document containing text."
    assert result.page_number is None, "a Word file is paginated by whatever opens it"
    assert result.timestamp_ms is None
    assert vision.calls == 0


def test_an_empty_word_document_says_that_rather_than_nothing(tmp_path):
    empty = (
        '<?xml version="1.0"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p/></w:body></w:document>"
    )

    (result,), _vision = _transcribe(_docx(tmp_path / "empty.docx", empty))

    assert result.visible_text == ""
    assert result.description == "Empty Word document."


def test_a_text_file_is_read_as_utf8(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("Olá — resumo\nsegunda linha\n", encoding="utf-8")

    (result,), vision = _transcribe(source)

    assert result.visible_text == "Olá — resumo\nsegunda linha"
    assert result.description == "Plain text file containing text."
    assert vision.calls == 0


def test_a_text_file_that_is_not_utf8_is_refused_rather_than_guessed_at(tmp_path):
    source = tmp_path / "latin.txt"
    source.write_bytes("Olá".encode("latin-1"))

    with pytest.raises(MediaTranscriptionError, match="not UTF-8"):
        _transcribe(source)


def test_a_text_file_beyond_its_bound_is_refused_before_it_is_read(tmp_path, monkeypatch):
    monkeypatch.setattr(documents, "MAX_TEXT_BYTES", 8)
    source = tmp_path / "long.txt"
    source.write_text("x" * 32, encoding="utf-8")

    with pytest.raises(MediaTranscriptionError, match="exceeds its limit"):
        _transcribe(source)


def test_a_word_document_that_is_not_an_archive_is_refused(tmp_path):
    source = tmp_path / "broken.docx"
    source.write_bytes(b"not a zip")

    with pytest.raises(MediaTranscriptionError, match="could not be decoded"):
        _transcribe(source)


def test_a_word_archive_missing_its_body_is_refused_as_a_document(tmp_path):
    """And named as a document rather than as a presentation."""
    source = tmp_path / "bodyless.docx"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")

    with pytest.raises(MediaTranscriptionError) as error:
        _transcribe(source)

    assert error.value.code == "document-invalid"
    assert "document archive entry is unavailable" in error.value.detail


def test_a_text_document_counts_as_one_unit_of_answer(tmp_path):
    assert documents._document_page_count(_docx(tmp_path / "one.docx")) == 1
    text = tmp_path / "one.txt"
    text.write_text("x", encoding="utf-8")
    assert documents._document_page_count(text) == 1


def test_a_pdf_read_by_pypdf_is_text_with_no_page_to_describe(tmp_path, monkeypatch):
    """The whole point of the fallback, exercised through the reader it builds.

    pypdf extracts and cannot render, so the page is answered as text and the
    picture is `None` — which the transcriber turns into a description saying
    the page was read without rendering, rather than describing a render
    nobody made.
    """
    reader = SimpleNamespace(
        is_encrypted=False,
        pages=[
            SimpleNamespace(extract_text=lambda: " first line \n\n second line "),
            SimpleNamespace(extract_text=lambda: None),
        ],
    )
    monkeypatch.setattr(documents, "_pdf_reader", lambda: "pypdf")
    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=lambda _source: reader))

    document = documents._open_document(tmp_path / "report.pdf")

    assert document.page_count == 2
    assert document.page(tmp_path, 1) == ("first line\nsecond line", None)
    # A page pypdf could not read any text from is still a page.
    assert document.page(tmp_path, 2) == ("", None)
    document.close()


def test_an_encrypted_pdf_is_refused_by_the_reader_that_can_only_extract(tmp_path, monkeypatch):
    reader = SimpleNamespace(is_encrypted=True, pages=[])
    monkeypatch.setattr(documents, "_pdf_reader", lambda: "pypdf")
    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=lambda _source: reader))

    with pytest.raises(MediaTranscriptionError) as error:
        documents._open_document(tmp_path / "private.pdf")

    assert error.value.code == "document-invalid"
    assert str(error.value.__cause__) == "encrypted PDF"


def test_pymupdf_is_used_when_it_is_installed():
    assert documents._pdf_reader() == "pymupdf"


def test_pypdf_answers_a_pdf_when_pymupdf_is_missing(tmp_path, monkeypatch):
    """A smaller answer rather than no answer.

    Without this, an installation with pypdf and no PyMuPDF refused every PDF
    for a dependency the person cannot see from the refusal.
    """
    import builtins

    real_import = builtins.__import__

    def without_pymupdf(name, *arguments, **keywords):
        if name == "pymupdf":
            raise ImportError("no pymupdf here")
        return real_import(name, *arguments, **keywords)

    monkeypatch.setattr(builtins, "__import__", without_pymupdf)

    assert documents._pdf_reader() == "pypdf"


def test_neither_reader_is_a_broken_install_rather_than_a_broken_file(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def without_either(name, *arguments, **keywords):
        if name in {"pymupdf", "pypdf"}:
            raise ImportError("neither reader here")
        return real_import(name, *arguments, **keywords)

    monkeypatch.setattr(builtins, "__import__", without_either)

    with pytest.raises(MediaTranscriptionError) as error:
        documents._pdf_reader()

    assert error.value.code == "document-runtime-unavailable"
    assert "PyMuPDF or pypdf" in error.value.detail


def test_a_page_nobody_rendered_says_so_instead_of_describing_nothing():
    assert documents._page_description("some text") == (
        "Document page containing text, read without rendering."
    )
    assert documents._page_description("") == (
        "Document page with no extractable text, read without rendering."
    )


@pytest.mark.asyncio
async def test_every_page_is_reported_as_it_is_read(tmp_path, monkeypatch):
    """The reader's half of OMNI-0583: one report per page, with the total.

    Without it a long document is silence, and silence is what the flow
    controller and the worker budget now end a call for.
    """

    class Pages(documents._OpenDocument):
        def __init__(self):
            super().__init__(3)

        def page(self, root, page_number):
            return f"page {page_number}", None

    monkeypatch.setattr(documents, "_open_document", lambda _source: Pages())
    reported = []

    async def observe(finished, total):
        reported.append((finished, total))

    read = await documents.DocumentPageTranscriber(None).transcribe(
        tmp_path / "long.pdf", CancellationController(), observe
    )

    assert [item.page_number for item in read] == [1, 2, 3]
    assert reported == [(1, 3), (2, 3), (3, 3)]


@pytest.mark.asyncio
async def test_a_text_document_is_one_part_and_says_so(tmp_path):
    """It is quick, but a job that reports nothing at all is a job that looks
    stopped to whatever is timing it."""
    source = tmp_path / "notes.txt"
    source.write_text("a line", encoding="utf-8")
    reported = []

    async def observe(finished, total):
        reported.append((finished, total))

    read = await documents.DocumentPageTranscriber(None).transcribe(
        source, CancellationController(), observe
    )

    assert len(read) == 1
    assert reported == [(1, 1)]
