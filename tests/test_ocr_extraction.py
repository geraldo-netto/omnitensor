"""Selected pixels become verbatim, citable page text (OMNI-0615)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnitensor.plugins.event_workload import EventWorkloadError
from omnitensor.plugins.extraction import AdapterKind
from omnitensor.plugins.ocr_extraction import VulkanOcrExtractionAdapter


def _adapter(tmp_path) -> VulkanOcrExtractionAdapter:
    for name in ("det.param", "det.bin", "rec.param", "rec.bin"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "keys.txt").write_text("a\nb\n")
    return VulkanOcrExtractionAdapter(
        tmp_path / "det.param", tmp_path / "rec.param", tmp_path / "keys.txt"
    )


def _ingested(path: Path):
    return SimpleNamespace(path=str(path), suffix=path.suffix.lower(), size_bytes=1)


def _collect(adapter, item):
    async def run():
        return [page async for page in adapter.pages(item)]

    return asyncio.run(run())


def test_the_adapter_declares_the_ocr_kind(tmp_path):
    assert _adapter(tmp_path).kind is AdapterKind.OCR


def test_an_image_reads_as_one_page_of_verbatim_lines(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    import vulkanocr.engine as engine_module

    class Engine:
        def __init__(self, models, **_kwargs):
            self.models = models

        def read(self, _rgb):
            return SimpleNamespace(
                lines=(SimpleNamespace(text="first line"), SimpleNamespace(text="second line"))
            )

        def __enter__(self):
            return self

        def __exit__(self, *_exception):
            return None

    monkeypatch.setattr(engine_module, "OcrEngine", Engine)
    image = tmp_path / "shot.png"
    cv2.imwrite(str(image), np.full((8, 8, 3), 255, dtype=np.uint8))
    pages = _collect(_adapter(tmp_path), _ingested(image))
    assert [(page.page_number, page.text) for page in pages] == [(1, "first line\nsecond line")]


def test_an_unreadable_image_is_a_named_refusal(tmp_path):
    broken = tmp_path / "torn.png"
    broken.write_bytes(b"not a png")
    with pytest.raises(EventWorkloadError) as error:
        _collect(_adapter(tmp_path), _ingested(broken))
    assert "cannot read an image" in str(error.value)


def test_a_pdf_keeps_its_text_layer_and_reads_only_the_scans(tmp_path, monkeypatch):
    pymupdf = pytest.importorskip("pymupdf")

    document = pymupdf.open()
    with_text = document.new_page()
    with_text.insert_text((72, 72), "typed page")
    document.new_page()  # a scan: no text layer
    source = tmp_path / "mixed.pdf"
    document.save(str(source))
    document.close()

    read = []
    monkeypatch.setattr(
        VulkanOcrExtractionAdapter,
        "_read_array",
        lambda self, rgb: (read.append(rgb.shape), "scanned words")[1],
    )
    pages = _collect(_adapter(tmp_path), _ingested(source))
    assert pages[0].page_number == 1 and "typed page" in pages[0].text
    assert (pages[1].page_number, pages[1].text) == (2, "scanned words")
    # Only the page without a text layer was rendered and read.
    assert len(read) == 1


def test_one_engine_serves_every_page_and_close_gives_it_back(tmp_path, monkeypatch):
    """Per-request, not per-page (OMNI-0621): a twenty-page scan used to
    pay twenty model loads."""
    import vulkanocr.engine as engine_module

    built = []

    class Engine:
        def __init__(self, _models, **_kwargs):
            self.closed = False
            built.append(self)

        def read(self, _rgb):
            return SimpleNamespace(lines=(SimpleNamespace(text="line"),))

        def close(self):
            self.closed = True

    monkeypatch.setattr(engine_module, "OcrEngine", Engine)
    adapter = _adapter(tmp_path)
    assert adapter._read_array("page-one") == "line"
    assert adapter._read_array("page-two") == "line"
    assert len(built) == 1
    adapter.close()
    assert built[0].closed is True
    # Reopened on the next request's first page.
    assert adapter._read_array("next-request") == "line"
    assert len(built) == 2


def test_engine_refusals_cross_in_the_adapters_vocabulary(tmp_path, monkeypatch):
    """An OcrEngineError must not escape raw into the workload (OMNI-0621)."""

    class Breaking:
        def read(self, _rgb):
            from vulkanocr.engine import OcrEngineError

            raise OcrEngineError("image-invalid", "expected uint8 pixels")

        def close(self):
            return None

    adapter = _adapter(tmp_path)
    adapter._engine = Breaking()
    with pytest.raises(EventWorkloadError) as caught:
        adapter._read_array("page")
    assert caught.value.code == "image-invalid"


MODELS = Path("/backups/disk2/projects/cinnamon/vulkanocr/PaddleOCR-ncnn-CPP/models")


@pytest.mark.skipif(not MODELS.is_dir(), reason="model ports are not fetched here")
def test_live_image_extraction_answers_exact_text(tmp_path):
    import cv2
    import numpy as np

    adapter = VulkanOcrExtractionAdapter(
        MODELS / "PP_OCRv6_medium_det.param",
        MODELS / "PP_OCRv6_medium_rec.param",
        MODELS / "ppocr_keys_v6.txt",
    )
    image = np.full((80, 640, 3), 255, dtype=np.uint8)
    cv2.putText(image, "Vulkan 1234", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    shot = tmp_path / "shot.png"
    cv2.imwrite(str(shot), image)
    pages = _collect(adapter, _ingested(shot))
    assert [(page.page_number, page.text) for page in pages] == [(1, "Vulkan 1234")]
