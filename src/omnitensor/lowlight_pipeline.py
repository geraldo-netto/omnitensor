"""Vulkan-only Retinexformer execution and deterministic RGB reconstruction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .executors.base import (
    Availability,
    InferenceResult,
    availability_for_model,
)
from .lowlight import LowLightWorkspaceError
from .lowlight_image import LowLightGeometry, PreparedLowLightInput

MODEL_SHAPE = (1, 3, 256, 256)


class LowLightNcnnExecutor(Protocol):
    """Narrow accelerator port required by the enhancement domain."""

    backend: str
    model_formats: frozenset[str]

    def availability(self) -> Availability: ...

    def run(self, model_path: str, inputs: list) -> InferenceResult: ...


@dataclass(frozen=True, slots=True)
class ReconstructedLowLightImage:
    """Oriented sRGB pixels and evidence needed by the publication stage."""

    rgb: bytes
    width: int
    height: int
    device_duration_ms: float
    source_metadata: dict

    def evidence(self) -> dict:
        return {
            "backend": "gpu",
            "runtime": "ncnn-vulkan",
            "cpuFallback": False,
            "deviceDurationMs": round(self.device_duration_ms, 3),
            "output": {
                "colorSpace": "sRGB",
                "width": self.width,
                "height": self.height,
                "bytes": len(self.rgb),
            },
            "source": dict(self.source_metadata),
        }


def execute_low_light_pipeline(
    executor: LowLightNcnnExecutor,
    model_path: Path | str,
    prepared: PreparedLowLightInput,
) -> ReconstructedLowLightImage:
    """Run one prepared image with no backend or format fallback."""
    if not isinstance(prepared, PreparedLowLightInput):
        raise TypeError("prepared must be a PreparedLowLightInput")
    path = Path(model_path)
    _require_ncnn_vulkan(executor, path)
    try:
        result = executor.run(str(path), [prepared.tensor])
    except LowLightWorkspaceError:
        raise
    except Exception as error:  # noqa: BLE001 - injected runtimes fail arbitrarily
        raise LowLightWorkspaceError(
            "inference-failed", f"ncnn Vulkan inference failed: {type(error).__name__}"
        ) from error
    return reconstruct_low_light_output(result, prepared.geometry, prepared.metadata())


def reconstruct_low_light_output(
    result: InferenceResult,
    geometry: LowLightGeometry,
    source_metadata: dict,
) -> ReconstructedLowLightImage:
    """Validate the native tensor and resize RGB back to oriented geometry."""
    if not isinstance(result, InferenceResult):
        raise LowLightWorkspaceError("output-invalid", "executor returned no inference result")
    if not isinstance(geometry, LowLightGeometry):
        raise TypeError("geometry must be LowLightGeometry")
    if not isinstance(source_metadata, dict):
        raise TypeError("source_metadata must be a dictionary")
    duration = float(result.duration_ms)
    if not math.isfinite(duration) or duration < 0:
        raise LowLightWorkspaceError(
            "output-invalid", "executor duration must be finite and non-negative"
        )
    tensor = _validated_output_tensor(result.outputs)
    rgb = _reconstruct_rgb(tensor, geometry.oriented_width, geometry.oriented_height)
    return ReconstructedLowLightImage(
        rgb,
        geometry.oriented_width,
        geometry.oriented_height,
        duration,
        dict(source_metadata),
    )


def _require_ncnn_vulkan(executor: LowLightNcnnExecutor, model_path: Path) -> None:
    if getattr(executor, "backend", None) != "gpu":
        raise LowLightWorkspaceError(
            "backend-invalid", "low-light enhancement requires the GPU backend"
        )
    formats = getattr(executor, "model_formats", frozenset())
    if "ncnn" not in formats or model_path.suffix != ".param":
        raise LowLightWorkspaceError(
            "model-format-invalid", "low-light enhancement requires an ncnn .param artifact"
        )
    availability = availability_for_model(executor, {"format": "ncnn"})
    if not availability.available:
        detail = availability.reason or "ncnn Vulkan is unavailable"
        raise LowLightWorkspaceError("backend-unavailable", detail)


def _validated_output_tensor(outputs: list):
    try:
        import numpy  # noqa: PLC0415 - deferred: an optional or heavy dependency
    except ImportError as error:  # pragma: no cover - gpu extra owns dependency
        raise LowLightWorkspaceError(
            "runtime-unavailable", "install OmniTensor with the gpu extra for NumPy"
        ) from error
    observed = numpy.asarray(outputs)
    if observed.shape != MODEL_SHAPE:
        raise LowLightWorkspaceError(
            "output-contract-mismatch",
            f"enhanced output must have shape {list(MODEL_SHAPE)}",
        )
    if observed.dtype.kind != "f":
        raise LowLightWorkspaceError(
            "output-contract-mismatch", "enhanced output must contain floating-point RGB"
        )
    tensor = observed.astype(numpy.float32, copy=False)
    if not numpy.isfinite(tensor).all():
        raise LowLightWorkspaceError(
            "output-invalid", "enhanced output contains non-finite components"
        )
    if float(tensor.min()) < 0 or float(tensor.max()) > 1:
        raise LowLightWorkspaceError(
            "output-invalid", "enhanced output components must remain within [0,1]"
        )
    return tensor


def _reconstruct_rgb(tensor, width: int, height: int) -> bytes:
    try:
        import numpy  # noqa: PLC0415 - deferred: an optional or heavy dependency
        from PIL import Image  # noqa: PLC0415 - deferred: an optional or heavy dependency
    except ImportError as error:  # pragma: no cover - gpu extra owns dependencies
        raise LowLightWorkspaceError(
            "runtime-unavailable", "install OmniTensor with the gpu extra for Pillow and NumPy"
        ) from error
    pixels = numpy.transpose(tensor[0], (1, 2, 0))
    encoded = numpy.floor(pixels * numpy.float32(255.0) + numpy.float32(0.5)).astype(numpy.uint8)
    image = Image.fromarray(encoded, mode="RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BICUBIC)
    payload = image.tobytes()
    if len(payload) != width * height * 3:
        raise LowLightWorkspaceError(
            "output-invalid", "reconstructed RGB byte count disagrees with geometry"
        )
    return payload
