"""Bounded, deterministic image preparation for low-light enhancement."""

from __future__ import annotations

import hashlib
import io
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

from .lowlight import LowLightWorkspace, LowLightWorkspaceError, resolve_low_light_source

MODEL_IMAGE_SIZE = 256
MAX_SOURCE_IMAGE_BYTES = 64 * 1024 * 1024
MAX_SOURCE_PIXELS = 40_000_000
MAX_SOURCE_DIMENSION = 16_384
SUPPORTED_SOURCE_FORMATS = frozenset({"JPEG", "PNG", "TIFF", "WEBP"})
_READ_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class LowLightGeometry:
    """Geometry before and after applying the source orientation."""

    original_width: int
    original_height: int
    oriented_width: int
    oriented_height: int

    def document(self) -> dict[str, int]:
        return {
            "originalWidth": self.original_width,
            "originalHeight": self.original_height,
            "orientedWidth": self.oriented_width,
            "orientedHeight": self.oriented_height,
        }


@dataclass(frozen=True, slots=True)
class PreparedLowLightInput:
    """One verified source, executor-ready tensor, and reconstruction evidence."""

    source: Path
    tensor: list
    geometry: LowLightGeometry
    source_format: str
    source_sha256: str
    source_bytes: int
    orientation_applied: bool

    def metadata(self) -> dict:
        return {
            "sourcePath": str(self.source),
            "sourceFormat": self.source_format,
            "sourceSha256": self.source_sha256,
            "sourceBytes": self.source_bytes,
            "orientationApplied": self.orientation_applied,
            "colorSpace": "sRGB",
            "modelShape": [1, 3, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE],
            "resize": {"filter": "bicubic", "fit": "exact"},
            **self.geometry.document(),
        }


def prepare_low_light_input(
    workspace: LowLightWorkspace,
    source: Path | str,
    *,
    max_source_bytes: int = MAX_SOURCE_IMAGE_BYTES,
    max_source_pixels: int = MAX_SOURCE_PIXELS,
) -> PreparedLowLightInput:
    """Decode exact bounded bytes into the profile's RGB float32 NCHW tensor."""
    _positive_limit(max_source_bytes, "max_source_bytes")
    _positive_limit(max_source_pixels, "max_source_pixels")
    source_path = resolve_low_light_source(workspace, source)
    payload = _read_bounded(source_path, max_source_bytes)
    digest = hashlib.sha256(payload).hexdigest()
    image, source_format, geometry, orientation_applied = _decode_normalized(
        payload, max_source_pixels
    )
    tensor = _rgb_nchw(image)
    return PreparedLowLightInput(
        source_path,
        tensor,
        geometry,
        source_format,
        digest,
        len(payload),
        orientation_applied,
    )


def _positive_limit(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")


def _read_bounded(path: Path, maximum: int) -> bytes:
    collected = bytearray()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(min(_READ_CHUNK_BYTES, maximum + 1 - len(collected))):
                collected.extend(chunk)
                if len(collected) > maximum:
                    raise LowLightWorkspaceError(
                        "input-too-large", f"source image exceeds the {maximum} byte limit"
                    )
    except LowLightWorkspaceError:
        raise
    except OSError as error:
        raise LowLightWorkspaceError("input-unavailable", "source image cannot be read") from error
    if not collected:
        raise LowLightWorkspaceError("input-invalid", "source image is empty")
    return bytes(collected)


def _decode_normalized(payload: bytes, max_pixels: int):
    try:
        from PIL import (  # noqa: PLC0415 - deferred: an optional or heavy dependency
            Image,
            ImageCms,
            ImageOps,
        )
    except ImportError as error:  # pragma: no cover - installation-dependent
        raise LowLightWorkspaceError(
            "decoder-unavailable", "install OmniTensor with the gpu extra for Pillow"
        ) from error
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as candidate:
                source_format = str(candidate.format or "").upper()
                if source_format not in SUPPORTED_SOURCE_FORMATS:
                    raise LowLightWorkspaceError(
                        "input-format-unsupported",
                        "source must be JPEG, PNG, TIFF, or WebP",
                    )
                original = candidate.size
                _bounded_geometry(original, max_pixels)
                candidate.load()
                orientation = candidate.getexif().get(274, 1)
                oriented = ImageOps.exif_transpose(candidate)
                _bounded_geometry(oriented.size, max_pixels)
                normalized = _to_srgb(oriented, Image, ImageCms)
                normalized.load()
    except LowLightWorkspaceError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise LowLightWorkspaceError(
            "input-too-large", f"source image exceeds the {max_pixels} pixel limit"
        ) from error
    except (
        Image.UnidentifiedImageError,
        ImageCms.PyCMSError,
        OSError,
        SyntaxError,
        ValueError,
    ) as error:
        raise LowLightWorkspaceError(
            "input-decode-failed", "source image is malformed or cannot be normalized"
        ) from error
    geometry = LowLightGeometry(original[0], original[1], oriented.width, oriented.height)
    return normalized, source_format, geometry, orientation not in (None, 1)


def _bounded_geometry(size: tuple[int, int], max_pixels: int) -> None:
    width, height = size
    if (
        width < 1
        or height < 1
        or width > MAX_SOURCE_DIMENSION
        or height > MAX_SOURCE_DIMENSION
        or width * height > max_pixels
    ):
        raise LowLightWorkspaceError(
            "input-too-large",
            f"source image exceeds {MAX_SOURCE_DIMENSION}px per side or {max_pixels} pixels",
        )


def _to_srgb(image, image_module, cms_module):
    alpha = "A" in image.getbands()
    converted = image.convert("RGBA" if alpha else "RGB")
    profile = image.info.get("icc_profile")
    if profile:
        converted = cms_module.profileToProfile(
            converted,
            cms_module.ImageCmsProfile(io.BytesIO(profile)),
            cms_module.createProfile("sRGB"),
            outputMode="RGBA" if alpha else "RGB",
        )
    if not alpha:
        return converted
    background = image_module.new("RGBA", converted.size, (0, 0, 0, 255))
    return image_module.alpha_composite(background, converted).convert("RGB")


def _rgb_nchw(image) -> list:
    try:
        import numpy  # noqa: PLC0415 - deferred: an optional or heavy dependency
        from PIL import Image  # noqa: PLC0415 - deferred: an optional or heavy dependency
    except ImportError as error:  # pragma: no cover - installation-dependent
        raise LowLightWorkspaceError(
            "decoder-unavailable", "install OmniTensor with the gpu extra for Pillow and NumPy"
        ) from error
    resized = image.resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), Image.Resampling.BICUBIC)
    array = numpy.asarray(resized, dtype=numpy.float32) / numpy.float32(255.0)
    tensor = numpy.transpose(array, (2, 0, 1))[None, ...]
    if (
        tensor.shape
        != (
            1,
            3,
            MODEL_IMAGE_SIZE,
            MODEL_IMAGE_SIZE,
        )
        or not numpy.isfinite(tensor).all()
    ):
        raise LowLightWorkspaceError("input-invalid", "normalized image tensor is invalid")
    minimum = float(tensor.min())
    maximum = float(tensor.max())
    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum < 0 or maximum > 1:
        raise LowLightWorkspaceError("input-invalid", "normalized image tensor is out of range")
    return tensor.tolist()
