"""Low-light ncnn execution, output-contract, and reconstruction coverage."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

import omnitensor.lowlight_pipeline as pipeline
from omnitensor.executors.base import Availability, InferenceResult
from omnitensor.lowlight import LowLightWorkspaceError, validate_low_light_workspace
from omnitensor.lowlight_image import LowLightGeometry, prepare_low_light_input
from omnitensor.lowlight_pipeline import (
    ReconstructedLowLightImage,
    execute_low_light_pipeline,
    reconstruct_low_light_output,
)


class FakeNcnnExecutor:
    backend = "gpu"
    model_formats = frozenset({"ncnn"})

    def __init__(self, outputs, *, availability=None, duration=4.125):
        self.outputs = outputs
        self._availability = availability or Availability(True)
        self.duration = duration
        self.calls = []

    def availability(self):
        return self._availability

    def run(self, model_path, inputs):
        self.calls.append((model_path, inputs))
        return InferenceResult(self.outputs, self.duration)


def _prepared(tmp_path: Path, *, size=(3, 2), color=(20, 40, 60)):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    source = inputs / "night.png"
    Image.new("RGB", size, color).save(source)
    return prepare_low_light_input(validate_low_light_workspace(inputs, outputs), source)


def _model_output(red=0.1, green=0.2, blue=0.3):
    output = numpy.empty((3, 256, 256), dtype=numpy.float32)
    output[0].fill(red)
    output[1].fill(green)
    output[2].fill(blue)
    return [output]


def test_pipeline_runs_exact_ncnn_input_and_reconstructs_oriented_rgb(tmp_path):
    prepared = _prepared(tmp_path)
    executor = FakeNcnnExecutor(_model_output())

    result = execute_low_light_pipeline(executor, tmp_path / "model.param", prepared)

    assert isinstance(result, ReconstructedLowLightImage)
    assert (result.width, result.height, len(result.rgb)) == (3, 2, 18)
    pixels = numpy.frombuffer(result.rgb, dtype=numpy.uint8).reshape(2, 3, 3)
    assert pixels[0, 0].tolist() == [26, 51, 77]
    assert executor.calls == [(str(tmp_path / "model.param"), [prepared.tensor])]
    assert result.evidence() == {
        "backend": "gpu",
        "runtime": "ncnn-vulkan",
        "cpuFallback": False,
        "deviceDurationMs": 4.125,
        "output": {"colorSpace": "sRGB", "width": 3, "height": 2, "bytes": 18},
        "source": prepared.metadata(),
    }


@pytest.mark.parametrize(
    ("backend", "formats", "suffix", "code"),
    [
        ("cpu", frozenset({"ncnn"}), ".param", "backend-invalid"),
        ("gpu", frozenset({"onnx"}), ".param", "model-format-invalid"),
        ("gpu", frozenset({"ncnn"}), ".onnx", "model-format-invalid"),
    ],
)
def test_cpu_or_contract_fallback_is_refused(tmp_path, backend, formats, suffix, code):
    prepared = _prepared(tmp_path)
    executor = FakeNcnnExecutor(_model_output())
    executor.backend = backend
    executor.model_formats = formats

    with pytest.raises(LowLightWorkspaceError) as caught:
        execute_low_light_pipeline(executor, tmp_path / f"model{suffix}", prepared)

    assert caught.value.code == code
    expected_detail = (
        "low-light enhancement requires the GPU backend"
        if code == "backend-invalid"
        else "low-light enhancement requires an ncnn .param artifact"
    )
    assert caught.value.detail == expected_detail
    assert executor.calls == []


def test_unavailable_vulkan_is_refused_without_running_or_fallback(tmp_path):
    prepared = _prepared(tmp_path)
    executor = FakeNcnnExecutor(
        _model_output(),
        availability=Availability(False, "Only a software Vulkan device is present"),
    )

    with pytest.raises(LowLightWorkspaceError) as caught:
        execute_low_light_pipeline(executor, tmp_path / "model.param", prepared)

    assert (caught.value.code, caught.value.detail) == (
        "backend-unavailable",
        "Only a software Vulkan device is present",
    )
    assert executor.calls == []


def test_executor_failure_is_bounded_to_type_and_chained(tmp_path):
    prepared = _prepared(tmp_path)

    class Broken(FakeNcnnExecutor):
        def run(self, model_path, inputs):
            raise RuntimeError("private native detail")

    with pytest.raises(LowLightWorkspaceError) as caught:
        execute_low_light_pipeline(Broken([]), tmp_path / "model.param", prepared)
    assert (caught.value.code, caught.value.detail) == (
        "inference-failed",
        "ncnn Vulkan inference failed: RuntimeError",
    )
    assert isinstance(caught.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    ("result", "code", "detail"),
    [
        (object(), "output-invalid", "executor returned no inference result"),
        (
            InferenceResult([[[1.0]]], 1.0),
            "output-contract-mismatch",
            "enhanced output must have shape [1, 3, 256, 256]",
        ),
        (
            InferenceResult([numpy.zeros((3, 256, 256), dtype=numpy.int32)], 1.0),
            "output-contract-mismatch",
            "enhanced output must contain floating-point RGB",
        ),
        (
            InferenceResult([numpy.full((3, 256, 256), numpy.nan)], 1.0),
            "output-invalid",
            "enhanced output contains non-finite components",
        ),
        (
            InferenceResult([numpy.full((3, 256, 256), -0.01)], 1.0),
            "output-invalid",
            "enhanced output components must remain within [0,1]",
        ),
        (
            InferenceResult([numpy.full((3, 256, 256), 1.01)], 1.0),
            "output-invalid",
            "enhanced output components must remain within [0,1]",
        ),
        (
            InferenceResult(_model_output(), math.inf),
            "output-invalid",
            "executor duration must be finite and non-negative",
        ),
        (
            InferenceResult(_model_output(), -0.1),
            "output-invalid",
            "executor duration must be finite and non-negative",
        ),
    ],
)
def test_native_output_contract_drift_fails_closed(result, code, detail):
    with pytest.raises(LowLightWorkspaceError) as caught:
        reconstruct_low_light_output(
            result,
            LowLightGeometry(1, 1, 1, 1),
            {},
        )
    assert (caught.value.code, caught.value.detail) == (code, detail)


@pytest.mark.parametrize(
    ("prepared", "message"),
    [(object(), "prepared must be a PreparedLowLightInput")],
)
def test_pipeline_requires_preprocessed_contract(tmp_path, prepared, message):
    with pytest.raises(TypeError, match=f"^{message}$"):
        execute_low_light_pipeline(
            FakeNcnnExecutor(_model_output()), tmp_path / "model.param", prepared
        )


@pytest.mark.parametrize(
    ("geometry", "metadata", "message"),
    [
        (object(), {}, "geometry must be LowLightGeometry"),
        (LowLightGeometry(1, 1, 1, 1), object(), "source_metadata must be a dictionary"),
    ],
)
def test_reconstruction_requires_domain_metadata(geometry, metadata, message):
    with pytest.raises(TypeError, match=f"^{message}$"):
        reconstruct_low_light_output(InferenceResult(_model_output(), 1), geometry, metadata)


def test_exact_unit_range_and_half_up_rgb_rounding_are_accepted():
    output = _model_output(0.0, 0.5, 1.0)
    result = reconstruct_low_light_output(
        InferenceResult(output, 0.0),
        LowLightGeometry(1, 1, 256, 256),
        {"kept": True},
    )
    pixels = numpy.frombuffer(result.rgb, dtype=numpy.uint8).reshape(256, 256, 3)
    assert pixels[0, 0].tolist() == [0, 128, 255]
    assert result.source_metadata == {"kept": True}


def test_validated_native_output_is_normalized_to_float32():
    tensor = pipeline._validated_output_tensor(
        [numpy.zeros((3, 256, 256), dtype=numpy.float64)]
    )
    assert tensor.dtype == numpy.float32


def test_rgb_reconstruction_calls_exact_axis_mode_and_resize(monkeypatch):
    calls = []
    real_transpose = numpy.transpose
    real_fromarray = Image.fromarray

    def transpose(value, axes):
        calls.append(("transpose", axes))
        return real_transpose(value, axes)

    class TrackedImage:
        def __init__(self, image):
            self._image = image

        @property
        def size(self):
            return self._image.size

        def resize(self, size, resample):
            calls.append(("resize", size, resample))
            return TrackedImage(self._image.resize(size, resample))

        def tobytes(self):
            return self._image.tobytes()

    def fromarray(value, *, mode):
        calls.append(("fromarray", mode))
        return TrackedImage(real_fromarray(value, mode=mode))

    monkeypatch.setattr(numpy, "transpose", transpose)
    monkeypatch.setattr(Image, "fromarray", fromarray)

    payload = pipeline._reconstruct_rgb(
        numpy.asarray(_model_output(), dtype=numpy.float32), 3, 2
    )

    assert len(payload) == 18
    assert calls == [
        ("transpose", (1, 2, 0)),
        ("fromarray", "RGB"),
        ("resize", (3, 2), Image.Resampling.BICUBIC),
    ]


@settings(max_examples=12, deadline=None)
@given(
    red=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
    green=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
    blue=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
    width=st.integers(min_value=1, max_value=8),
    height=st.integers(min_value=1, max_value=8),
)
def test_property_reconstruction_has_exact_geometry_and_bounded_rgb(
    red, green, blue, width, height
):
    result = reconstruct_low_light_output(
        InferenceResult(_model_output(red, green, blue), 1.0),
        LowLightGeometry(width, height, width, height),
        {},
    )
    assert (result.width, result.height) == (width, height)
    assert len(result.rgb) == width * height * 3
    assert min(result.rgb) >= 0 and max(result.rgb) <= 255


def test_reconstructed_byte_count_is_guarded(monkeypatch):
    class ShortImage:
        size = (2, 2)

        def tobytes(self):
            return b"short"

    monkeypatch.setattr(Image, "fromarray", lambda *_args, **_kwargs: ShortImage())
    with pytest.raises(LowLightWorkspaceError) as caught:
        pipeline._reconstruct_rgb(numpy.zeros((1, 3, 256, 256)), 2, 2)
    assert (caught.value.code, caught.value.detail) == (
        "output-invalid",
        "reconstructed RGB byte count disagrees with geometry",
    )


def test_pipeline_does_not_modify_prepared_source(tmp_path):
    prepared = _prepared(tmp_path)
    before = (prepared.source.read_bytes(), prepared.source.stat().st_mtime_ns)
    execute_low_light_pipeline(
        FakeNcnnExecutor(_model_output()), tmp_path / "model.param", prepared
    )
    assert (prepared.source.read_bytes(), prepared.source.stat().st_mtime_ns) == before


def test_property_runner_accepts_python_float_outputs():
    with tempfile.TemporaryDirectory() as directory:
        prepared = _prepared(Path(directory))
        output = numpy.zeros((3, 256, 256), dtype=numpy.float64).tolist()
        result = execute_low_light_pipeline(
            FakeNcnnExecutor([output]), Path(directory) / "model.param", prepared
        )
        expected = prepared.geometry.oriented_width * prepared.geometry.oriented_height * 3
        assert len(result.rgb) == expected
