from __future__ import annotations

import math

import pytest
from conftest import sample_manifest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.forecastresult import (
    ForecastResultError,
    _single_finite_value,
    forecast_reading,
    parse_forecast_reading,
)
from omnitensor.plugins.orchestration import _postprocess_stage
from omnitensor.plugins.pipeline import InferenceOutput
from omnitensor.registry import Workload


def _contract(**changes):
    contract = {
        "version": 1,
        "recipe": "forecast-v1",
        "featureNames": ["load", "queue"],
        "targetFeature": "load",
        "window": 2,
        "horizon": 3,
        "observationOrder": "oldest-first",
        "flattenOrder": "observations-then-features",
    }
    contract.update(changes)
    return contract


def _model(**changes):
    model = {
        "id": "local-forecast",
        "version": "1.0.0",
        "format": "ncnn",
        "fullyQuantized": False,
        "minimumCompilerVersion": "0.0.0",
        "minimumRuntimeVersion": "0.0.0",
        "featureContract": _contract(),
        "outputContract": {"kind": "raw"},
    }
    model.update(changes)
    return model


def _workload() -> Workload:
    manifest = sample_manifest("resource-scheduler", accelerator="gpu", model=_model())
    return Workload("resource-scheduler", manifest)


@pytest.mark.parametrize(("tensors", "value"), [([0], 0.0), ([[-2.5]], -2.5), ([[[1e300]]], 1e300)])
def test_forecast_reading_accepts_supported_singleton_lane_shapes(tensors, value):
    assert forecast_reading(_model(), tensors) == {
        "kind": "forecast",
        "targetFeature": "load",
        "horizon": 3,
        "value": value,
    }


@pytest.mark.parametrize(
    ("tensors", "detail"),
    [
        (0.5, "forecast must contain exactly one output tensor"),
        ([], "forecast must produce exactly one numeric value"),
        ([1, 2], "forecast must produce exactly one numeric value"),
        ([[[[1.0]]]], "forecast output nesting exceeds supported lanes"),
        ([True], "forecast value must be numeric"),
        (["0.5"], "forecast value must be numeric"),
        ([float("nan")], "forecast value must be finite"),
        ([float("inf")], "forecast value must be finite"),
        ([float("-inf")], "forecast value must be finite"),
    ],
)
def test_forecast_reading_rejects_ambiguous_or_nonfinite_outputs(tensors, detail):
    with pytest.raises(ForecastResultError) as captured:
        forecast_reading(_model(), tensors)

    assert captured.value.code == "forecast-output-invalid"
    assert captured.value.detail == detail
    assert str(captured.value) == f"forecast-output-invalid: {detail}"


def test_forecast_reading_ignores_models_without_the_recipe():
    assert forecast_reading(None, [1.0]) is None
    assert forecast_reading({}, [1.0]) is None
    assert forecast_reading(_model(featureContract={"recipe": "other"}), [1.0]) is None


def test_public_forecast_parser_returns_one_canonical_copy():
    source = {
        "kind": "forecast",
        "targetFeature": "queueDepth",
        "horizon": 128,
        "value": 0,
    }

    parsed = parse_forecast_reading(source)
    source["targetFeature"] = "changed"

    assert parsed == {
        "kind": "forecast",
        "targetFeature": "queueDepth",
        "horizon": 128,
        "value": 0.0,
    }


@pytest.mark.parametrize(
    "reading",
    [
        None,
        "forecast",
        {},
        {"kind": "classification"},
        {"kind": "forecast", "targetFeature": "load", "horizon": 1},
        {
            "kind": "forecast",
            "targetFeature": "load",
            "horizon": 1,
            "value": 1.0,
            "unit": "%",
        },
        {"kind": "forecast", "targetFeature": "", "horizon": 1, "value": 1.0},
        {"kind": "forecast", "targetFeature": "x" * 65, "horizon": 1, "value": 1.0},
        {"kind": "forecast", "targetFeature": "load", "horizon": True, "value": 1.0},
        {"kind": "forecast", "targetFeature": "load", "horizon": 0, "value": 1.0},
        {"kind": "forecast", "targetFeature": "load", "horizon": 129, "value": 1.0},
        {"kind": "forecast", "targetFeature": "load", "horizon": 1, "value": True},
        {"kind": "forecast", "targetFeature": "load", "horizon": 1, "value": "1"},
        {"kind": "forecast", "targetFeature": "load", "horizon": 1, "value": float("nan")},
        {"kind": "forecast", "targetFeature": "load", "horizon": 1, "value": float("inf")},
    ],
)
def test_public_forecast_parser_rejects_every_malformed_field(reading):
    assert parse_forecast_reading(reading) is None


@given(
    st.text(min_size=1, max_size=64),
    st.integers(min_value=1, max_value=128),
    st.floats(allow_nan=False, allow_infinity=False),
)
def test_public_forecast_parser_property_preserves_exact_bounded_values(target, horizon, value):
    parsed = parse_forecast_reading(
        {
            "kind": "forecast",
            "targetFeature": target,
            "horizon": horizon,
            "value": value,
        }
    )

    assert parsed == {
        "kind": "forecast",
        "targetFeature": target,
        "horizon": horizon,
        "value": float(value),
    }
    assert math.isfinite(parsed["value"])


@pytest.mark.asyncio
async def test_postprocess_adds_forecast_reading_without_replacing_raw_tensors():
    stage = _postprocess_stage(_workload(), lambda: ())

    output = await stage(InferenceOutput({"outputs": [[[0.625]]], "durationMs": 2.0}))

    assert output.output == {
        "profileId": "resource-scheduler",
        "outputs": [[[0.625]]],
        "durationMs": 2.0,
        "reading": {
            "kind": "forecast",
            "targetFeature": "load",
            "horizon": 3,
            "value": 0.625,
        },
    }


@pytest.mark.asyncio
async def test_postprocess_fails_forecast_instead_of_succeeding_with_malformed_raw_output():
    stage = _postprocess_stage(_workload(), lambda: ())

    with pytest.raises(ForecastResultError) as captured:
        await stage(InferenceOutput({"outputs": [[0.5, 0.6]]}))

    assert captured.value.code == "forecast-output-invalid"


@pytest.mark.asyncio
async def test_postprocess_preserves_classification_reduction_and_labels():
    model = _model(featureContract=None, outputContract={"kind": "classification", "topK": 1})
    workload = Workload(
        "visual-library",
        sample_manifest("visual-library", accelerator="gpu", model=model),
    )
    stage = _postprocess_stage(workload, lambda: ("low", "high"))

    output = await stage(InferenceOutput({"outputs": [[0.1, 0.9]]}))

    assert output.output["outputs"] == [[0.1, 0.9]]
    assert output.output["reading"] == {
        "kind": "classification",
        "top": [{"index": 1, "score": 0.9, "label": "high"}],
    }


def test_single_value_helper_accepts_exact_nesting_boundary():
    assert _single_finite_value([[[1.0]]]) == 1.0
