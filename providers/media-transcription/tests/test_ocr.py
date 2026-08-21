"""The enrichment lane's one promise: it never raises into a job."""

from pathlib import Path

import pytest
from omnitensor_media_transcription.ocr import OcrRefusal, VulkanOcr


def test_unmounted_artifacts_are_a_remembered_refusal(tmp_path):
    reader = VulkanOcr(None, None, None)
    outcome = reader.read(tmp_path / "frame.png")
    assert isinstance(outcome, OcrRefusal)
    assert outcome.code == "ocr-artifacts-unmounted"
    # And again, without touching anything.
    assert reader.read(tmp_path / "frame.png") is not None
    reader.close()


def test_a_broken_engine_build_refuses_once_and_answers_fast(tmp_path, monkeypatch):
    for name in ("det.param", "rec.param", "keys.txt"):
        (tmp_path / name).write_text("x")
    reader = VulkanOcr(tmp_path / "det.param", tmp_path / "rec.param", tmp_path / "keys.txt")
    built = []

    def breaking(_self):
        built.append(1)
        raise RuntimeError("no vulkan device on this machine")

    monkeypatch.setattr(VulkanOcr, "_ensure_engine", breaking)
    first = reader.read(tmp_path / "frame.png")
    second = reader.read(tmp_path / "frame.png")
    assert isinstance(first, OcrRefusal) and first.code == "ocr-failed"
    assert isinstance(second, OcrRefusal)
    # The broken stack was not rebuilt per frame.
    assert built == [1]


def test_a_failed_frame_does_not_condemn_the_next(tmp_path, monkeypatch):
    for name in ("det.param", "rec.param", "keys.txt"):
        (tmp_path / name).write_text("x")
    reader = VulkanOcr(tmp_path / "det.param", tmp_path / "rec.param", tmp_path / "keys.txt")
    answers = iter([ValueError("torn frame"), "read-fine"])

    class Engine:
        def read(self, _image):
            answer = next(answers)
            if isinstance(answer, Exception):
                raise answer
            return answer

        def close(self):
            return None

    reader._engine = Engine()
    monkeypatch.setattr(VulkanOcr, "_loaded", staticmethod(lambda _p: "image"))
    first = reader.read(tmp_path / "frame.png")
    assert isinstance(first, OcrRefusal) and first.code == "ocr-failed"
    # The engine survives; the next frame is answered.
    assert reader.read(tmp_path / "frame.png") == "read-fine"
    reader.close()


def test_engine_codes_cross_with_their_prefix(tmp_path):
    reader = VulkanOcr(tmp_path / "absent.param", tmp_path / "absent.param", tmp_path / "keys")
    outcome = reader.read(tmp_path / "frame.png")
    # vulkanocr's own stable code, carried through with the lane prefix:
    # OcrModels.validated refuses model-missing before any device is touched.
    assert isinstance(outcome, OcrRefusal)
    assert outcome.code == "ocr-model-missing"


@pytest.mark.skipif(
    not Path("/backups/disk2/projects/cinnamon/vulkanocr/PaddleOCR-ncnn-CPP/models").is_dir(),
    reason="model ports are not fetched here",
)
def test_live_read_answers_lines_with_coordinates(tmp_path):
    import cv2
    import numpy as np

    models = Path("/backups/disk2/projects/cinnamon/vulkanocr/PaddleOCR-ncnn-CPP/models")
    reader = VulkanOcr(
        models / "PP_OCRv6_medium_det.param",
        models / "PP_OCRv6_medium_rec.param",
        models / "ppocr_keys_v6.txt",
    )
    frame = tmp_path / "frame.png"
    image = np.full((80, 640, 3), 255, dtype=np.uint8)
    cv2.putText(image, "Vulkan 1234", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.imwrite(str(frame), image)
    try:
        result = reader.read(frame)
        assert not isinstance(result, OcrRefusal), result
        assert [line.text for line in result.lines] == ["Vulkan 1234"]
        assert result.lines[0].center_y < 80
    finally:
        reader.close()
