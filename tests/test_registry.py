

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
