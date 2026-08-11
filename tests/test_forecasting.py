from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins.forecasting import (
    MIN_TRAINING_ROWS,
    ForecastError,
    ForecastQuality,
    evaluate_forecast_predictions,
    fit_forecaster,
)
from omnitensor.plugins.recorder import TelemetryRecorder, feature_matrix


def ramp(count=40, window=3):
    """A signal a linear model should learn almost exactly."""
    inputs, targets = [], []
    for start in range(count):
        history = [0.01 * (start + step) for step in range(window)]
        inputs.append(tuple(history))
        targets.append(0.01 * (start + window))
    return inputs, targets


def test_a_forecaster_learns_a_trend_it_can_extrapolate():
    inputs, targets = ramp()

    model = fit_forecaster(inputs, targets, ("cpu",), 3)

    assert model.quality.mean_absolute_error < 1e-6
    assert model.predict((0.10, 0.11, 0.12)) == pytest.approx(0.13, abs=1e-4)


def test_a_fit_reports_error_against_data_it_did_not_see():
    inputs, targets = ramp()

    quality = fit_forecaster(inputs, targets, ("cpu",), 3).quality

    assert quality.samples > 0
    assert quality.samples < len(inputs)
    assert quality.useful


def test_a_model_no_better_than_last_time_is_reported_as_not_useful():
    """A model that cannot beat the free forecast should not be served."""
    # A random walk: the best possible prediction really is the last value.
    values = [0.5]
    state = 12345
    for _ in range(60):
        state = (1103515245 * state + 12345) % (2**31)
        values.append(min(1.0, max(0.0, values[-1] + ((state % 200) - 100) / 10000)))
    inputs = [tuple(values[index : index + 3]) for index in range(len(values) - 3)]
    targets = [values[index + 3] for index in range(len(values) - 3)]

    quality = fit_forecaster(inputs, targets, ("cpu",), 3).quality

    assert quality.skill <= 0.35


def test_skill_is_zero_when_the_baseline_is_perfect():
    assert ForecastQuality(4, 0.0, 0.0).skill == 0.0
    assert not ForecastQuality(4, 0.0, 0.0).useful


def test_every_forecaster_uses_the_same_repeat_last_quality_boundary():
    quality = evaluate_forecast_predictions(
        targets=(3.0, 5.0),
        predictions=(2.0, 7.0),
        repeat_last=(1.0, 4.0),
    )

    assert quality == ForecastQuality(
        samples=2,
        mean_absolute_error=1.5,
        baseline_error=1.5,
    )


@pytest.mark.parametrize(
    ("targets", "predictions", "baselines", "code", "detail"),
    [
        ((), (), (), "evaluation-invalid", "equal nonzero length"),
        ((1.0,), (), (1.0,), "evaluation-invalid", "equal nonzero length"),
        ((1.0,), (1.0,), (), "evaluation-invalid", "equal nonzero length"),
        ((float("nan"),), (1.0,), (1.0,), "training-invalid", "targets must be finite"),
        ((1.0,), (float("inf"),), (1.0,), "training-invalid", "predictions must be finite"),
        ((1.0,), (1.0,), (True,), "training-invalid", "baselines must be numbers"),
    ],
)
def test_forecast_evaluation_refuses_ambiguous_or_nonfinite_rows(
    targets, predictions, baselines, code, detail
):
    with pytest.raises(ForecastError) as invalid:
        evaluate_forecast_predictions(targets, predictions, baselines)

    assert invalid.value.code == code
    assert detail in invalid.value.detail


@given(
    targets=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=32,
    ),
    prediction_offset=st.floats(
        min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
    ),
    baseline_offset=st.floats(
        min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
    ),
)
def test_forecast_evaluation_preserves_mean_absolute_error(
    targets, prediction_offset, baseline_offset
):
    predictions = [value + prediction_offset for value in targets]
    baselines = [value + baseline_offset for value in targets]

    quality = evaluate_forecast_predictions(targets, predictions, baselines)

    assert quality.samples == len(targets)
    assert quality.mean_absolute_error == pytest.approx(
        sum(
            abs(prediction - target)
            for prediction, target in zip(predictions, targets, strict=True)
        )
        / len(targets)
    )
    assert quality.baseline_error == pytest.approx(
        sum(
            abs(baseline - target)
            for baseline, target in zip(baselines, targets, strict=True)
        )
        / len(targets)
    )


def test_the_holdout_is_split_by_time_not_at_random():
    """Shuffling lets a model see the future of the window it is scored on."""
    inputs, targets = ramp(count=40)
    model = fit_forecaster(inputs, targets, ("cpu",), 3, holdout=0.25)

    # The last quarter is held out, so the scored samples are the newest rows.
    assert model.quality.samples == len(inputs) - int(len(inputs) * 0.75)


def test_a_forecast_of_the_wrong_shape_is_refused():
    inputs, targets = ramp()
    model = fit_forecaster(inputs, targets, ("cpu",), 3)

    assert model.input_width == 3
    with pytest.raises(ForecastError, match="expected 3 values"):
        model.predict((0.1, 0.2))
    with pytest.raises(ForecastError, match="input-invalid"):
        model.predict("0.1,0.2,0.3")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.5", True, None])
def test_an_unusable_forecast_input_is_refused(value):
    inputs, targets = ramp()
    model = fit_forecaster(inputs, targets, ("cpu",), 3)
    with pytest.raises(ForecastError, match="input-invalid"):
        model.predict((0.1, 0.2, value))


def test_too_little_history_is_refused_rather_than_fitted():
    """An error estimate from a handful of rows is noise."""
    inputs, targets = ramp(count=MIN_TRAINING_ROWS - 1)

    with pytest.raises(ForecastError, match="insufficient-history"):
        fit_forecaster(inputs, targets, ("cpu",), 3)


def test_collinear_features_are_refused_rather_than_fitted():
    """Coefficients from a singular system are an artefact of rounding."""
    inputs = [(value, value, value) for value in (0.1 * index for index in range(20))]
    targets = [row[0] for row in inputs]

    with pytest.raises(ForecastError, match="fit-degenerate"):
        fit_forecaster(inputs, targets, ("cpu",), 3, ridge=0.0)


def test_ridge_regularisation_rescues_a_collinear_fit():
    inputs = [(value, value, value) for value in (0.1 * index for index in range(20))]
    targets = [row[0] for row in inputs]

    model = fit_forecaster(inputs, targets, ("cpu",), 3, ridge=1e-3)

    assert math.isfinite(model.predict((0.5, 0.5, 0.5)))


@pytest.mark.parametrize(
    "changes",
    [
        {"window": 0},
        {"window": True},
        {"feature_names": ()},
        {"ridge": -1},
        {"ridge": True},
        {"holdout": 0},
        {"holdout": 1},
        {"holdout": 2},
    ],
)
def test_the_fit_parameters_are_validated(changes):
    inputs, targets = ramp()
    options = {"feature_names": ("cpu",), "window": 3, **changes}
    with pytest.raises(ForecastError):
        fit_forecaster(inputs, targets, options["feature_names"], options["window"],
                       **{k: v for k, v in changes.items() if k in ("ridge", "holdout")})


def test_mismatched_training_shapes_are_refused():
    inputs, targets = ramp()
    with pytest.raises(ForecastError, match="same length"):
        fit_forecaster(inputs, targets[:-1], ("cpu",), 3)
    with pytest.raises(ForecastError, match="must have 3 values"):
        fit_forecaster([(0.1, 0.2)] * 20, targets[:20], ("cpu",), 3)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "x", True])
def test_unusable_training_data_is_refused(bad):
    inputs, targets = ramp()
    broken = [(*inputs[0][:-1], bad), *inputs[1:]]
    with pytest.raises(ForecastError, match="training-invalid"):
        fit_forecaster(broken, targets, ("cpu",), 3)
    with pytest.raises(ForecastError, match="training-invalid"):
        fit_forecaster(inputs, [bad, *targets[1:]], ("cpu",), 3)


def test_too_wide_a_fit_is_refused():
    inputs, targets = ramp()
    with pytest.raises(ForecastError, match="at most"):
        fit_forecaster(inputs, targets, tuple(f"f{i}" for i in range(300)), 3)


def test_a_model_describes_itself_for_publication():
    inputs, targets = ramp()

    described = fit_forecaster(inputs, targets, ("cpu",), 3).document()

    assert described["featureNames"] == ["cpu"]
    assert described["window"] == 3
    assert len(described["weights"]) == 3
    assert described["quality"]["useful"] is True


def test_a_forecaster_trains_end_to_end_from_recorded_history(tmp_path):
    """The whole point: recorded telemetry becomes a model with no labelling."""
    recorder = TelemetryRecorder(tmp_path / "telemetry")
    for index in range(60):
        recorder.record(
            "resource-scheduler",
            {"cpu": 0.01 * index, "queue": float(index % 4)},
            1_000 + index,
        )

    windows = recorder.windows("resource-scheduler", 3)
    inputs, targets = feature_matrix(windows, ("cpu", "queue"))
    model = fit_forecaster(inputs, targets, ("cpu", "queue"), 3)

    assert model.quality.useful
    assert model.quality.mean_absolute_error < 0.01
    next_cpu = model.predict((0.57, 1.0, 0.58, 2.0, 0.59, 3.0))
    assert 0.55 < next_cpu < 0.65


def test_a_tiny_holdout_still_scores_against_a_held_out_row():
    """A holdout that rounds to nothing must not score on the training data."""
    inputs, targets = ramp(count=MIN_TRAINING_ROWS)

    model = fit_forecaster(inputs, targets, ("cpu",), 3, holdout=0.01)

    assert model.quality.samples == 1
