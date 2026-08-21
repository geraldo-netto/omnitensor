"""Best-effort OCR enrichment: exact glyphs with coordinates, never a gate.

The VulkanOCR engine reads a rendered frame in ~100 ms and answers with the
one thing the vision model structurally cannot: pixel coordinates per line.
It is an enricher by owner decision (2026-08-22, OMNI-0609): nothing it
finds, and nothing that goes wrong with it, may gate, replace, or delay the
vision model. That contract is enforced here once — no code path in this
module raises into a job. Every failure becomes an :class:`OcrRefusal`
carrying a stable code the receipt can name.

Placement and lifetime (OMNI-0610, measured in OMNI-0612): with two hardware
devices the lane runs in its **own process**, resident on the second-ranked
device. Not a thread — the ncnn binding holds the GIL through ``extract``,
so an in-process lane stalled the vision model\'s token loop for ~2 s on a
dense page even when the devices were different. With one device the engine
is built per frame in this process, inside the worker that already holds
the lease, and given back so the ~700 MiB never sits beside a loaded 9B.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

_LANE_READY_TIMEOUT_S = 30.0
_LANE_READ_TIMEOUT_S = 60.0


@dataclass(frozen=True, slots=True)
class OcrRefusal:
    """Why enrichment is not available, said once and carried in receipts."""

    code: str
    detail: str


@dataclass(frozen=True, slots=True)
class OcrReading:
    """What the lane found on one frame; the merge seam reads ``lines``."""

    lines: tuple


class _LaneRefusalError(RuntimeError):
    """A lane reply that is a refusal, carried across as itself."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.refusal = OcrRefusal(code, detail)


def _code_for(error: BaseException) -> str:
    code = getattr(error, "code", "")
    if isinstance(code, str) and code:
        return f"ocr-{code}"
    if isinstance(error, ImportError):
        return "ocr-runtime-missing"
    return "ocr-failed"


def _load_image(image_path: Path):
    import cv2  # noqa: PLC0415 - arrives with the vulkanocr dependency
    import numpy  # noqa: PLC0415

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read an image from {image_path}")
    return numpy.ascontiguousarray(image[:, :, ::-1])


def _build_engine(det_param: Path, rec_param: Path, dictionary: Path, device=None):
    from vulkanocr import models_for_port  # noqa: PLC0415 - optional lane
    from vulkanocr.engine import OcrEngine  # noqa: PLC0415

    # The port is named, not restated: its blob names and blank convention
    # come from vulkanocr's own registry (OMNI-0622/VOCR-0061).
    models = models_for_port("avafly-v6", det_param, rec_param, dictionary)
    return OcrEngine(models, device=device)


def _lane_worker(det_param, rec_param, dictionary, requests, replies) -> None:
    """The resident lane: one process, the second-ranked device, one engine.

    Every failure is a reply, never a bare death: the parent turns
    ``("error", ...)`` into a refusal with a name (the same lesson
    vulkanocr\'s pool learned as VOCR-0037).
    """
    try:
        import ncnn  # noqa: PLC0415 - imported in the child on purpose
        from vulkanocr.device import hardware_devices  # noqa: PLC0415

        device = hardware_devices(ncnn)[1]
        engine = _build_engine(Path(det_param), Path(rec_param), Path(dictionary), device)
    except BaseException as error:  # noqa: BLE001 - the reply is the report
        replies.put(("error", _code_for(error), str(error)))
        return
    try:
        replies.put(("ready", engine.device_name))
        while True:
            item = requests.get()
            if item is None:
                return
            try:
                result = engine.read(_load_image(Path(item)))
                replies.put(("ok", tuple(result.lines)))
            except BaseException as error:  # noqa: BLE001 - the reply is the report
                replies.put(("error", _code_for(error), str(error)))
    finally:
        engine.close()


class VulkanOcr:
    """The enrichment lane over vulkanocr\'s engine, built to refuse politely.

    ``det_param``/``rec_param``/``dictionary`` are the mounted artifact
    paths, or ``None`` when the artifacts are not installed — absence is the
    common state on an existing deployment and is a refusal, not an error.
    Everything is built lazily on the first read, so a worker whose machine
    cannot serve OCR pays nothing for the attempt after the first.
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
        self._placed = False
        self._resident = False
        self._process: Any = None
        self._requests: Any = None
        self._replies: Any = None
        self._refusal: OcrRefusal | None = None
        if det_param is None or rec_param is None or dictionary is None:
            self._refusal = OcrRefusal(
                "ocr-artifacts-unmounted",
                "the PP-OCRv6 artifacts are not installed; visual answers are model-only",
            )

    def read(self, image_path: Path):
        """Every text line on one rendered frame, or why there is none.

        Returns :class:`OcrReading` on success and :class:`OcrRefusal`
        otherwise. A refusal to even start is remembered: once the lane
        cannot be built, later reads answer immediately instead of retrying
        a broken stack. A single failed frame refuses alone.
        """
        if self._refusal is not None:
            return self._refusal
        try:
            self._place()
            if self._resident:
                return self._lane_read(image_path)
            return self._shared_read(image_path)
        except BaseException as error:  # noqa: BLE001 - the refusal is the report
            refusal = getattr(error, "refusal", None) or OcrRefusal(_code_for(error), str(error))
            if not self._placed:
                self._refusal = refusal
            return refusal

    def close(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            # The vulkanocr pool\'s shutdown lessons apply verbatim: the
            # sentinel, the reap after terminate (VOCR-0046), and dropping
            # unsent data so exit never hangs on a feeder (VOCR-0047).
            self._requests.put(None)
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            for queue in (self._requests, self._replies):
                queue.close()
                queue.cancel_join_thread()
            self._requests = None
            self._replies = None
        self._placed = False
        self._resident = False

    def _place(self) -> None:
        if self._placed:
            return
        import ncnn  # noqa: PLC0415 - optional lane
        from vulkanocr.device import hardware_devices  # noqa: PLC0415

        self._resident = len(hardware_devices(ncnn)) > 1
        if self._resident:
            self._start_lane()
        self._placed = True

    def _start_lane(self) -> None:
        context = mp.get_context("spawn")
        self._requests = context.Queue()
        self._replies = context.Queue()
        self._process = context.Process(
            target=_lane_worker,
            args=(
                str(self._det_param),
                str(self._rec_param),
                str(self._dictionary),
                self._requests,
                self._replies,
            ),
            daemon=True,
        )
        self._process.start()
        reply = self._lane_reply(_LANE_READY_TIMEOUT_S)
        if reply[0] != "ready":
            raise _LaneRefusalError(reply[1], reply[2])

    def _lane_read(self, image_path: Path):
        self._requests.put(str(image_path))
        try:
            reply = self._lane_reply(_LANE_READ_TIMEOUT_S)
        except BaseException:
            # A dead or hung lane is retired; the next frame rebuilds it.
            self.close()
            raise
        if reply[0] == "ok":
            return OcrReading(reply[1])
        return OcrRefusal(reply[1], reply[2])

    def _lane_reply(self, timeout_s: float):
        deadline = timeout_s
        while True:
            try:
                return self._replies.get(timeout=min(0.5, deadline))
            except Empty:
                deadline -= 0.5
                if self._process is None or not self._process.is_alive():
                    raise RuntimeError("the OCR lane process died without a reply") from None
                if deadline <= 0:
                    raise TimeoutError("the OCR lane did not answer in time") from None

    def _shared_read(self, image_path: Path):
        """One device: build, read, give back — never held beside the 9B."""
        engine = _build_engine(self._det_param, self._rec_param, self._dictionary)
        try:
            return OcrReading(tuple(engine.read(_load_image(image_path)).lines))
        finally:
            engine.close()


__all__ = ["OcrReading", "OcrRefusal", "VulkanOcr"]
