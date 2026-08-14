"""Low-light image preprocessing unit, regression, and property coverage."""

from __future__ import annotations

import hashlib
import tempfile
import warnings
from pathlib import Path

import numpy
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image, ImageCms

import omnitensor.lowlight_image as lowlight_image
from omnitensor.lowlight import LowLightWorkspaceError, validate_low_light_workspace
from omnitensor.lowlight_image import (
    MODEL_IMAGE_SIZE,
    LowLightGeometry,
    prepare_low_light_input,
)


def _workspace(tmp_path: Path):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    return validate_low_light_workspace(inputs, outputs)


def _save(path: Path, mode: str, size: tuple[int, int], color, **options) -> bytes:
    Image.new(mode, size, color).save(path, **options)
    return path.read_bytes()


def _array(prepared):
    return numpy.asarray(prepared.tensor, dtype=numpy.float32)


def test_png_preprocessing_is_exact_rgb_float32_nchw_with_evidence(tmp_path):
    workspace = _workspace(tmp_path)
    source = workspace.input_folder / "night.png"
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    payload = _save(source, "RGB", (5, 3), (10, 20, 30), format="PNG", icc_profile=profile)

    prepared = prepare_low_light_input(workspace, source)
    tensor = _array(prepared)

    assert tensor.shape == (1, 3, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
    assert tensor.dtype == numpy.float32
    assert numpy.isfinite(tensor).all()
    assert tensor.min() >= 0 and tensor.max() <= 1
    assert tensor[0, :, 0, 0] == pytest.approx(numpy.array([10, 20, 30]) / 255)
    assert prepared.source == source.resolve()
    assert prepared.geometry == LowLightGeometry(5, 3, 5, 3)
    assert prepared.source_format == "PNG"
    assert prepared.source_sha256 == hashlib.sha256(payload).hexdigest()
    assert prepared.source_bytes == len(payload)
    assert prepared.orientation_applied is False
    assert prepared.metadata() == {
        "sourcePath": str(source.resolve()),
        "sourceFormat": "PNG",
        "sourceSha256": hashlib.sha256(payload).hexdigest(),
        "sourceBytes": len(payload),
        "orientationApplied": False,
        "colorSpace": "sRGB",
        "modelShape": [1, 3, 256, 256],
        "resize": {"filter": "bicubic", "fit": "exact"},
        "originalWidth": 5,
        "originalHeight": 3,
        "orientedWidth": 5,
        "orientedHeight": 3,
    }


def test_jpeg_exif_orientation_is_applied_and_geometry_is_preserved(tmp_path):
    workspace = _workspace(tmp_path)
    source = workspace.input_folder / "rotated.jpg"
    image = Image.new("RGB", (2, 3), (50, 80, 120))
    exif = Image.Exif()
    exif[274] = 6
    image.save(source, format="JPEG", quality=95, exif=exif)

    prepared = prepare_low_light_input(workspace, source)

    assert prepared.source_format == "JPEG"
    assert prepared.orientation_applied is True
    assert prepared.geometry == LowLightGeometry(2, 3, 3, 2)
    assert _array(prepared).shape == (1, 3, 256, 256)


def test_real_tiff_and_webp_decoders_share_the_contract(tmp_path):
    workspace = _workspace(tmp_path)
    for suffix, image_format in (("tiff", "TIFF"), ("webp", "WEBP")):
        source = workspace.input_folder / f"scene.{suffix}"
        _save(source, "RGB", (7, 4), (1, 90, 240), format=image_format)
        prepared = prepare_low_light_input(workspace, source)
        assert prepared.source_format == image_format
        assert prepared.geometry.oriented_width == 7
        assert _array(prepared).shape == (1, 3, 256, 256)


def test_transparent_pixels_are_composited_over_black(tmp_path):
    workspace = _workspace(tmp_path)
    source = workspace.input_folder / "transparent.png"
    _save(source, "RGBA", (2, 2), (255, 200, 100, 0), format="PNG")

    tensor = _array(prepare_low_light_input(workspace, source))

    assert numpy.count_nonzero(tensor) == 0


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (b"", "input-invalid"),
        (b"not an image", "input-decode-failed"),
    ],
)
def test_empty_or_malformed_images_fail_closed(tmp_path, payload, code):
    workspace = _workspace(tmp_path)
    source = workspace.input_folder / "bad.png"
    source.write_bytes(payload)

    with pytest.raises(LowLightWorkspaceError) as caught:
        prepare_low_light_input(workspace, source)

    assert caught.value.code == code
    assert caught.value.detail == (
        "source image is empty"
        if code == "input-invalid"
        else "source image is malformed or cannot be normalized"
    )


def test_decodable_but_unsupported_bmp_is_refused(tmp_path):
    workspace = _workspace(tmp_path)
    source = workspace.input_folder / "image.bmp"
    _save(source, "RGB", (2, 2), "white", format="BMP")

    with pytest.raises(LowLightWorkspaceError) as caught:
        prepare_low_light_input(workspace, source)

    assert caught.value.code == "input-format-unsupported"
    assert caught.value.detail == "source must be JPEG, PNG, TIFF, or WebP"


def test_byte_and_pixel_limits_are_enforced_before_tensor_allocation(tmp_path):
    workspace = _workspace(tmp_path)
    source = workspace.input_folder / "image.png"
    payload = _save(source, "RGB", (4, 3), "white", format="PNG")

    with pytest.raises(LowLightWorkspaceError) as byte_limit:
        prepare_low_light_input(workspace, source, max_source_bytes=len(payload) - 1)
    assert byte_limit.value.code == "input-too-large"
    assert byte_limit.value.detail == f"source image exceeds the {len(payload) - 1} byte limit"

    with pytest.raises(LowLightWorkspaceError) as pixel_limit:
        prepare_low_light_input(workspace, source, max_source_pixels=11)
    assert (pixel_limit.value.code, pixel_limit.value.detail) == (
        "input-too-large",
        "source image exceeds 16384px per side or 11 pixels",
    )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("max_source_bytes", True),
        ("max_source_bytes", 0),
        ("max_source_pixels", -1),
        ("max_source_pixels", 1.5),
    ],
)
def test_preprocessing_limits_are_positive_integers(tmp_path, keyword, value):
    workspace = _workspace(tmp_path)
    source = workspace.input_folder / "image.png"
    _save(source, "RGB", (1, 1), "white", format="PNG")

    with pytest.raises(ValueError, match=rf"^{keyword} must be a positive integer$"):
        prepare_low_light_input(workspace, source, **{keyword: value})


def test_one_is_a_valid_resource_limit():
    assert lowlight_image._positive_limit(1, "limit") is None


def test_source_must_remain_inside_the_configured_input_root(tmp_path):
    workspace = _workspace(tmp_path)
    outside = tmp_path / "outside.png"
    _save(outside, "RGB", (1, 1), "white", format="PNG")
    link = workspace.input_folder / "escape.png"
    link.symlink_to(outside)

    with pytest.raises(LowLightWorkspaceError) as caught:
        prepare_low_light_input(workspace, link)

    assert caught.value.code == "input-invalid"


def test_image_read_failures_have_a_stable_non_path_disclosing_error(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    source = workspace.input_folder / "image.png"
    _save(source, "RGB", (1, 1), "white", format="PNG")
    real_open = Path.open

    def refused(path, *args, **kwargs):
        if path == source.resolve():
            raise OSError("private filesystem detail")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", refused)
    with pytest.raises(LowLightWorkspaceError) as caught:
        prepare_low_light_input(workspace, source)
    assert (caught.value.code, caught.value.detail) == (
        "input-unavailable",
        "source image cannot be read",
    )


def test_invalid_icc_profile_fails_instead_of_silently_changing_color(tmp_path):
    workspace = _workspace(tmp_path)
    source = workspace.input_folder / "bad-profile.png"
    _save(source, "RGB", (2, 2), "white", format="PNG", icc_profile=b"invalid")

    with pytest.raises(LowLightWorkspaceError) as caught:
        prepare_low_light_input(workspace, source)

    assert (caught.value.code, caught.value.detail) == (
        "input-decode-failed",
        "source image is malformed or cannot be normalized",
    )


@settings(max_examples=10, deadline=None)
@given(
    red=st.integers(min_value=0, max_value=255),
    green=st.integers(min_value=0, max_value=255),
    blue=st.integers(min_value=0, max_value=255),
)
def test_property_solid_rgb_channel_order_and_scaling_are_exact(red, green, blue):
    with tempfile.TemporaryDirectory() as directory:
        workspace = _workspace(Path(directory))
        source = workspace.input_folder / "solid.png"
        _save(source, "RGB", (2, 2), (red, green, blue), format="PNG")
        tensor = _array(prepare_low_light_input(workspace, source))
        expected = numpy.asarray([red, green, blue], dtype=numpy.float32) / 255
        assert tensor[0, :, 128, 128] == pytest.approx(expected)


def test_corrupt_decoder_geometry_is_refused(monkeypatch):
    with pytest.raises(LowLightWorkspaceError) as caught:
        lowlight_image._bounded_geometry((16_385, 1), 40_000_000)
    assert caught.value.code == "input-too-large"


@pytest.mark.parametrize(
    "size",
    [(0, 1), (1, 0), (16_385, 1), (1, 16_385), (4, 4)],
)
def test_geometry_rejects_each_independent_limit(size):
    maximum = 15 if size == (4, 4) else 40_000_000
    with pytest.raises(LowLightWorkspaceError) as caught:
        lowlight_image._bounded_geometry(size, maximum)
    assert caught.value.code == "input-too-large"


@pytest.mark.parametrize(
    ("size", "maximum"),
    [((1, 1), 1), ((16_384, 1), 16_384), ((1, 16_384), 16_384), ((4, 4), 16)],
)
def test_geometry_limits_are_inclusive(size, maximum):
    assert lowlight_image._bounded_geometry(size, maximum) is None


def test_decode_bomb_warning_has_an_exact_bounded_refusal(monkeypatch):
    def bomb(*_args, **_kwargs):
        warnings.warn("oversized", Image.DecompressionBombWarning, stacklevel=2)

    monkeypatch.setattr(Image, "open", bomb)
    with pytest.raises(LowLightWorkspaceError) as caught:
        lowlight_image._decode_normalized(b"image", 123)
    assert (caught.value.code, caught.value.detail) == (
        "input-too-large",
        "source image exceeds the 123 pixel limit",
    )


def test_rgb_tensor_uses_exact_resize_dtype_and_axis_contract(monkeypatch):
    calls = []
    base = Image.new("RGB", (2, 3))
    base.putdata(
        [(0, 1, 2), (10, 11, 12), (20, 21, 22), (30, 31, 32), (40, 41, 42), (255, 254, 253)]
    )

    class Source:
        def resize(self, size, resample):
            calls.append(("resize", size, resample))
            return base.resize(size, resample)

    real_asarray = numpy.asarray
    real_transpose = numpy.transpose

    def asarray(value, *, dtype):
        calls.append(("asarray", dtype))
        return real_asarray(value, dtype=dtype)

    def transpose(value, axes):
        calls.append(("transpose", axes))
        return real_transpose(value, axes)

    monkeypatch.setattr(numpy, "asarray", asarray)
    monkeypatch.setattr(numpy, "transpose", transpose)

    tensor = numpy.array(lowlight_image._rgb_nchw(Source()), dtype=numpy.float32)

    assert calls == [
        ("resize", (256, 256), Image.Resampling.BICUBIC),
        ("asarray", numpy.float32),
        ("transpose", (2, 0, 1)),
    ]
    assert tensor.shape == (1, 3, 256, 256)


@pytest.mark.parametrize(
    ("replacement", "detail"),
    [
        (numpy.zeros((3, 1, 1), dtype=numpy.float32), "normalized image tensor is invalid"),
        (
            numpy.full((3, 256, 256), numpy.nan, dtype=numpy.float32),
            "normalized image tensor is invalid",
        ),
        (
            numpy.full((3, 256, 256), -0.1, dtype=numpy.float32),
            "normalized image tensor is out of range",
        ),
        (
            numpy.full((3, 256, 256), 1.1, dtype=numpy.float32),
            "normalized image tensor is out of range",
        ),
    ],
)
def test_rgb_tensor_defensive_validation_is_exact(monkeypatch, replacement, detail):
    monkeypatch.setattr(numpy, "transpose", lambda *_args, **_kwargs: replacement)
    with pytest.raises(LowLightWorkspaceError) as caught:
        lowlight_image._rgb_nchw(Image.new("RGB", (1, 1), "black"))
    assert (caught.value.code, caught.value.detail) == ("input-invalid", detail)


def test_rgb_tensor_unit_range_is_inclusive():
    black = numpy.asarray(lowlight_image._rgb_nchw(Image.new("RGB", (1, 1), "black")))
    white = numpy.asarray(lowlight_image._rgb_nchw(Image.new("RGB", (1, 1), "white")))
    assert (float(black.min()), float(white.max())) == (0.0, 1.0)
