

import pytest


def test_a_profile_may_declare_one_model_per_accelerator_lane():
    """An artifact is format-specific, so one entry can serve only one lane."""
    from omnitensor.registry import Workload

    gpu = {"id": "m", "version": "1.0.0", "format": "ncnn"}
    npu = {"id": "m-npu", "version": "1.0.0", "format": "openvino"}
    profile = Workload("w", {"requirements": {"models": [gpu, npu]}})

    assert [model["format"] for model in profile.models] == ["ncnn", "openvino"]
    assert profile.model_for("openvino") == npu
    assert profile.model_for("tflite") is None


def test_both_spellings_normalise_to_the_same_thing():
    from omnitensor.registry import Workload

    model = {"id": "m", "version": "1.0.0", "format": "ncnn"}

    assert Workload("w", {"requirements": {"model": model}}).models == (model,)
    assert Workload("w", {"requirements": {"models": [model]}}).models == (model,)
    assert Workload("w", {"requirements": {"model": None}}).models == ()
    assert Workload("w", {"requirements": {}}).models == ()


def test_models_that_cannot_describe_one_network_are_refused():
    from omnitensor.registry import declared_models_error

    contract = {"inputs": [{"shape": [1, 3, 8, 8], "dtype": "float32"}]}
    gpu = {"id": "m", "version": "1.0.0", "format": "ncnn", "tensorContract": contract}
    npu = {"id": "n", "version": "1.0.0", "format": "openvino", "tensorContract": contract}

    assert declared_models_error({"requirements": {"models": [gpu, npu]}}) is None
    assert declared_models_error({"requirements": {"model": gpu}}) is None
    assert declared_models_error({"requirements": {}}) is None

    twice = declared_models_error({"requirements": {"models": [gpu, dict(gpu, id="other")]}})
    assert "ncnn more than once" in twice

    other = dict(npu, tensorContract={"inputs": [{"shape": [1, 3, 9, 9], "dtype": "float32"}]})
    assert "disagree about tensorContract" in declared_models_error(
        {"requirements": {"models": [gpu, other]}}
    )

    classified = dict(gpu, outputContract={"kind": "classification"})
    embedded = dict(npu, outputContract={"kind": "embedding"})
    assert "disagree about outputContract" in declared_models_error(
        {"requirements": {"models": [classified, embedded]}}
    )


def forecast_contract(**changes):
    contract = {
        "version": 1,
        "recipe": "forecast-v1",
        "featureNames": ["load", "queue", "memory"],
        "targetFeature": "load",
        "window": 3,
        "horizon": 1,
        "observationOrder": "oldest-first",
        "flattenOrder": "observations-then-features",
    }
    contract.update(changes)
    return contract


def forecast_model(model_format="ncnn", **contract_changes):
    return {
        "id": f"forecast-{model_format}",
        "version": "1.0.0",
        "format": model_format,
        "featureContract": forecast_contract(**contract_changes),
        "tensorContract": {
            "inputs": [{"shape": [1, 9], "dtype": "float32", "layout": "NC"}]
        },
        "outputContract": {"kind": "raw"},
    }


def test_forecast_contract_refuses_same_shape_wrong_feature_order_across_lanes():
    from omnitensor.registry import declared_models_error

    gpu = forecast_model()
    npu = forecast_model("openvino", featureNames=["load", "memory", "queue"])

    assert "disagree about featureContract" in declared_models_error(
        {"requirements": {"models": [gpu, npu]}}
    )


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"targetFeature": "queue"}, "targetFeature must be the first feature"),
        ({"window": 2}, "disagrees with tensorContract"),
    ],
)
def test_a_single_forecast_model_must_agree_with_its_tensor_semantics(change, expected):
    from omnitensor.registry import declared_models_error

    assert expected in declared_models_error(
        {"requirements": {"model": forecast_model(**change)}}
    )


def test_a_forecast_contract_keeps_the_training_input_width_bound():
    from omnitensor.registry import MAX_FORECAST_INPUT_WIDTH, declared_models_error
    from omnitensor.training.contracts import MAX_INPUT_WIDTH

    assert MAX_FORECAST_INPUT_WIDTH == MAX_INPUT_WIDTH == 512

    feature_names = [f"feature-{index}" for index in range(5)]
    model = forecast_model(featureNames=feature_names, window=128, targetFeature="feature-0")
    model["tensorContract"]["inputs"][0]["shape"] = [1, 640]

    assert "exceeds 512 input values" in declared_models_error(
        {"requirements": {"model": model}}
    )


@pytest.mark.parametrize(
    "change",
    [
        {"horizon": 2},
        {"observationOrder": "newest-first"},
        {"flattenOrder": "features-then-observations"},
    ],
)
def test_forecast_variants_must_repeat_every_semantic_field(change):
    from omnitensor.registry import declared_models_error

    assert "disagree about featureContract" in declared_models_error(
        {
            "requirements": {
                "models": [forecast_model(), forecast_model("openvino", **change)]
            }
        }
    )
