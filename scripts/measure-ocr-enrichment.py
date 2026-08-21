"""Measure the OCR-enriched visual pipeline end to end (OMNI-0612).

Drives the real EnrichedVisualTranscriber — the Qwen vision model on the
primary GPU, the VulkanOCR lane wherever its placement policy puts it — over
a set of images, and reports what the enrichment costs and what it adds:

- wall clock per frame: vision model alone vs enriched (the difference is
  the lane's true cost under the placement policy);
- the always-runs rule observed live: the model's answer is byte-identical
  with and without the lane;
- what the lane adds: line count with coordinates, and its text against the
  model's reading.

Usage (paths are this machine's; every artifact is somebody's choice):
  .venv/bin/python scripts/measure-ocr-enrichment.py \
      --vision-model /backups/disk2/models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf \
      --vision-projector /backups/disk2/models/qwen3.5-9b/mmproj-F16.gguf \
      --ocr-models /backups/disk2/projects/cinnamon/vulkanocr/PaddleOCR-ncnn-CPP/models \
      IMAGE [IMAGE ...]
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-model", type=Path, required=True)
    parser.add_argument("--vision-projector", type=Path, required=True)
    parser.add_argument("--ocr-models", type=Path, required=True)
    parser.add_argument("images", nargs="+", type=Path)
    arguments = parser.parse_args()

    from omnitensor_media_transcription.enrichment import EnrichedVisualTranscriber
    from omnitensor_media_transcription.models import QwenVulkanVisualTranscriber, VulkanLease
    from omnitensor_media_transcription.ocr import OcrRefusal, VulkanOcr

    from omnitensor.plugins.media_transcription import VisualFrame

    lease_file = Path(tempfile.mkstemp(prefix="ocr-enrichment-lease-")[1])
    lease = VulkanLease(lease_file)
    qwen = QwenVulkanVisualTranscriber(arguments.vision_model, arguments.vision_projector, lease)
    ocr = VulkanOcr(
        arguments.ocr_models / "PP_OCRv6_medium_det.param",
        arguments.ocr_models / "PP_OCRv6_medium_rec.param",
        arguments.ocr_models / "ppocr_keys_v6.txt",
    )
    cancellation = SimpleNamespace(raise_if_cancelled=lambda: None, cancelled=False)

    async def run() -> None:
        enriched = EnrichedVisualTranscriber(ocr, qwen)
        model_ms: list[float] = []
        both_ms: list[float] = []
        for image in arguments.images:
            frame = VisualFrame(image, None)
            started = time.perf_counter()
            alone = await qwen.transcribe(frame, cancellation)
            model_ms.append((time.perf_counter() - started) * 1000)

            started = time.perf_counter()
            merged = await enriched.transcribe(frame, cancellation)
            both_ms.append((time.perf_counter() - started) * 1000)

            identical = (alone.visible_text, alone.description) == (
                merged.visible_text,
                merged.description,
            )
            print(f"\n== {image.name}")
            print(f"  model alone {model_ms[-1]:7.0f} ms; enriched {both_ms[-1]:7.0f} ms")
            print(f"  model answer unchanged by the lane: {identical}")
            if enriched.last_refusal is not None:
                refusal: OcrRefusal = enriched.last_refusal
                print(f"  ocr refused: {refusal.code}: {refusal.detail}")
            else:
                text = " ".join(line.text for line in merged.ocr_lines)
                print(f"  ocr lines: {len(merged.ocr_lines)}")
                print(f"  ocr text:  {text[:120]}")
            print(f"  model text: {merged.visible_text[:120]}")
        print(
            f"\nmedian: model alone {statistics.median(model_ms):.0f} ms, "
            f"enriched {statistics.median(both_ms):.0f} ms, "
            f"lane cost {statistics.median(both_ms) - statistics.median(model_ms):+.0f} ms"
        )
        await enriched.release()

    asyncio.run(run())
    lease_file.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
