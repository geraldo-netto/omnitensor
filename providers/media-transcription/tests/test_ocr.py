"""The enrichment lane\'s one promise: it never raises into a job."""

from pathlib import Path
from types import SimpleNamespace

import omnitensor_media_transcription.ocr as ocr_module
import pytest
from omnitensor_media_transcription.ocr import OcrReading, OcrRefusal, VulkanOcr


def test_unmounted_artifacts_are_a_remembered_refusal(tmp_path):
    reader = VulkanOcr(None, None, None)
    outcome = reader.read(tmp_path / "frame.png")
    assert isinstance(outcome, OcrRefusal)
    assert outcome.code == "ocr-artifacts-unmounted"
    assert reader.read(tmp_path / "frame.png") is not None
    reader.close()


def test_a_broken_placement_refuses_once_and_answers_fast(tmp_path, monkeypatch):
    for name in ("det.param", "rec.param", "keys.txt"):
        (tmp_path / name).write_text("x")
    reader = VulkanOcr(tmp_path / "det.param", tmp_path / "rec.param", tmp_path / "keys.txt")
    attempts = []

    def breaking(_self):
        attempts.append(1)
        raise RuntimeError("no vulkan device on this machine")

    monkeypatch.setattr(VulkanOcr, "_place", breaking)
    first = reader.read(tmp_path / "frame.png")
    second = reader.read(tmp_path / "frame.png")
    assert isinstance(first, OcrRefusal) and first.code == "ocr-failed"
    assert isinstance(second, OcrRefusal)
    # The broken stack was not rebuilt per frame.
    assert attempts == [1]


def test_a_failed_frame_does_not_condemn_the_next(tmp_path, monkeypatch):
    for name in ("det.param", "rec.param", "keys.txt"):
        (tmp_path / name).write_text("x")
    reader = VulkanOcr(tmp_path / "det.param", tmp_path / "rec.param", tmp_path / "keys.txt")

    def placed(self):
        self._placed = True
        self._ever_placed = True

    monkeypatch.setattr(VulkanOcr, "_place", placed)
    answers = iter([ValueError("torn frame"), OcrReading(())])

    def scripted(_self, _path):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(VulkanOcr, "_shared_read", scripted)
    first = reader.read(tmp_path / "frame.png")
    assert isinstance(first, OcrRefusal) and first.code == "ocr-failed"
    # The lane survives; the next frame is answered.
    assert isinstance(reader.read(tmp_path / "frame.png"), OcrReading)
    reader.close()


class _FakeContext:
    """A stand-in for the spawn context: scripted replies, recorded requests."""

    def __init__(self, replies):
        self.sent = []
        self.replies = list(replies)
        self.process = SimpleNamespace(
            started=False,
            is_alive=lambda: True,
            join=lambda timeout=None: None,
            terminate=lambda: None,
        )
        harness = self

        class Queue:
            def __init__(self):
                self.closed = False

            def put(self, item):
                harness.sent.append(item)

            def get(self, timeout=None):
                return harness.replies.pop(0)

            def close(self):
                self.closed = True

            def cancel_join_thread(self):
                return None

        self._queue = Queue

    def Queue(self):  # noqa: N802 - multiprocessing\'s own name
        return self._queue()

    def Process(self, *, target, args, daemon):  # noqa: N802
        def start():
            self.process.started = True

        self.process.start = start
        return self.process


def _mounted(tmp_path) -> VulkanOcr:
    for name in ("det.param", "rec.param", "keys.txt"):
        (tmp_path / name).write_text("x")
    return VulkanOcr(tmp_path / "det.param", tmp_path / "rec.param", tmp_path / "keys.txt")


def _devices(monkeypatch, count: int) -> None:
    import vulkanocr.device as device_module

    devices = [SimpleNamespace(index=i, name=f"gpu-{i}", kind=i) for i in range(count)]
    monkeypatch.setattr(device_module, "hardware_devices", lambda _runtime: devices)


def test_two_devices_run_the_lane_in_its_own_process(tmp_path, monkeypatch):
    """Not a thread (OMNI-0610/0612): ncnn holds the GIL through extract,
    and an in-process lane stalled the vision model\'s token loop ~2 s on a
    dense page even on a different device."""

    _devices(monkeypatch, 2)
    line = SimpleNamespace(text="hello")
    context = _FakeContext([("ready", "gpu-1"), ("ok", 1, (line,)), ("ok", 2, (line,))])
    monkeypatch.setattr(ocr_module.mp, "get_context", lambda _method: context)
    reader = _mounted(tmp_path)

    first = reader.read(tmp_path / "a.png")
    second = reader.read(tmp_path / "b.png")

    assert context.process.started is True
    assert isinstance(first, OcrReading) and isinstance(second, OcrReading)
    assert first.lines == (line,)
    # Two frames crossed one resident lane.
    assert context.sent == [(1, str(tmp_path / "a.png")), (2, str(tmp_path / "b.png"))]
    reader.close()
    # Shut down like the vulkanocr pool: sentinel, then the queues dropped.
    assert context.sent[-1] is None


def test_a_lane_error_reply_is_a_named_refusal_for_that_frame(tmp_path, monkeypatch):
    _devices(monkeypatch, 2)
    context = _FakeContext(
        [("ready", "gpu-1"), ("error", 1, "ocr-image-invalid", "expected uint8 pixels")]
    )
    monkeypatch.setattr(ocr_module.mp, "get_context", lambda _method: context)
    reader = _mounted(tmp_path)
    outcome = reader.read(tmp_path / "a.png")
    assert isinstance(outcome, OcrRefusal)
    assert outcome.code == "ocr-image-invalid"


def test_a_lane_that_cannot_start_refuses_with_its_own_code(tmp_path, monkeypatch):
    _devices(monkeypatch, 2)
    context = _FakeContext([("error", "ocr-model-missing", "model file is absent")])
    monkeypatch.setattr(ocr_module.mp, "get_context", lambda _method: context)
    reader = _mounted(tmp_path)
    outcome = reader.read(tmp_path / "a.png")
    assert isinstance(outcome, OcrRefusal)
    assert outcome.code == "ocr-model-missing"
    # Remembered: the lane is not respawned per frame.
    assert reader.read(tmp_path / "b.png") is outcome


def test_one_device_builds_per_frame_and_gives_the_engine_back(tmp_path, monkeypatch):
    _devices(monkeypatch, 1)
    built = []

    def fake_build(det, rec, keys, device=None):
        engine = SimpleNamespace(
            device=device,
            closed=False,
            read=lambda _image: SimpleNamespace(lines=()),
        )
        engine.close = lambda: setattr(engine, "closed", True)
        built.append(engine)
        return engine

    monkeypatch.setattr(ocr_module, "_build_engine", fake_build)
    monkeypatch.setattr(ocr_module, "_load_image", lambda _p: "image")
    reader = _mounted(tmp_path)
    reader.read(tmp_path / "a.png")
    reader.read(tmp_path / "b.png")
    # Rebuilt per frame, never held beside a loaded vision model.
    assert len(built) == 2
    assert all(engine.closed for engine in built)
    reader.close()


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
        assert isinstance(result, OcrReading), result
        assert [line.text for line in result.lines] == ["Vulkan 1234"]
        assert result.lines[0].center_y < 80
    finally:
        reader.close()


def test_a_retired_lane_is_rebuilt_not_refused_forever(tmp_path, monkeypatch):
    """A lane that served frames and then died rebuilds (OMNI-0620).

    close() resets _placed, and the refusal-memory check used to read that
    as never-placed — so one timeout or worker death condemned every later
    frame to model-only answers, against the documented rebuild.
    """

    _devices(monkeypatch, 2)
    contexts = []

    def next_context(_method):
        if not contexts:
            # First lane: ready, one answer, then silence with a dead worker.
            context = _FakeContext([("ready", "gpu-1"), ("ok", 1, ())])
            context.process.is_alive = lambda: len(context.replies) > 0
        else:
            context = _FakeContext([("ready", "gpu-1"), ("ok", 1, (SimpleNamespace(text="back"),))])
        contexts.append(context)
        return context

    monkeypatch.setattr(ocr_module.mp, "get_context", next_context)
    reader = _mounted(tmp_path)
    assert isinstance(reader.read(tmp_path / "a.png"), OcrReading)
    # The lane dies mid-wait: this frame refuses with a name...
    died = reader.read(tmp_path / "b.png")
    assert isinstance(died, OcrRefusal)
    # ...and the next frame gets a fresh lane instead of the dead one's ghost.
    revived = reader.read(tmp_path / "c.png")
    assert isinstance(revived, OcrReading)
    assert [line.text for line in revived.lines] == ["back"]
    assert len(contexts) == 2


def test_a_stale_reply_from_an_abandoned_frame_cannot_poison_the_next(tmp_path, monkeypatch):
    """The lane protocol carries sequences and drains leftovers (OMNI-0619).

    A cancelled job abandons its lane read mid-flight; the lane finishes
    anyway and its reply sits in the queue. Untagged, the next frame
    consumed it as its own answer — the previous page\'s lines on the new
    page.
    """

    _devices(monkeypatch, 2)
    stale = SimpleNamespace(text="OLD PAGE")
    fresh = SimpleNamespace(text="this page")
    context = _FakeContext([("ready", "gpu-1"), ("ok", 1, (SimpleNamespace(text="first"),))])
    monkeypatch.setattr(ocr_module.mp, "get_context", lambda _method: context)
    reader = _mounted(tmp_path)
    assert isinstance(reader.read(tmp_path / "a.png"), OcrReading)

    # Frame 2 was abandoned by a cancellation: its reply arrives late,
    # after frame 3 has already been requested.
    reader._sequence = 2
    context.replies.extend([("ok", 2, (stale,)), ("ok", 3, (fresh,))])
    outcome = reader.read(tmp_path / "c.png")

    assert isinstance(outcome, OcrReading)
    assert [line.text for line in outcome.lines] == ["this page"]
