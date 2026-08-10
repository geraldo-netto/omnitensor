from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnitensor.registry import bundled_workloads_path, validate_document
from omnitensor.tensorcontract import (
    InputSpec,
    PreprocessSpec,
    contract_error,
    declared_inputs,
    measured_shape,
)


def model(**overrides) -> dict:
    declared = {
        "id": "sample-model",
        "version": "1.0.0",
        "format": "ncnn",
        "tensorContract": {
            "inputs": [{
                "shape": [1, 3, 227, 227],
                "dtype": "float32",
                "layout": "NCHW",
                "preprocess": {
                    "channelOrder": "BGR",
                    "mean": [104.0, 117.0, 123.0],
                    "scale": [1.0, 1.0, 1.0],
                },
            }],
        },
    }
    declared.update(overrides)
    return declared


def test_a_declared_contract_is_read_whole():
    specs = declared_inputs(model())

    assert specs == (
        InputSpec(
            (1, 3, 227, 227),
            "float32",
            "NCHW",
            PreprocessSpec("BGR", (104.0, 117.0, 123.0), (1.0, 1.0, 1.0)),
        ),
    )
    assert specs[0].element_count == 3 * 227 * 227


def test_a_model_declaring_no_contract_is_not_checked():
    """Every manifest written before the field still loads and still runs."""
    assert declared_inputs(model(tensorContract=None)) is None
    assert declared_inputs({"id": "sample-model"}) is None
    assert declared_inputs(None) is None
    assert declared_inputs("model") is None
    assert declared_inputs(model(tensorContract={"inputs": []})) is None


def test_a_publisher_may_state_the_shape_without_the_normalisation():
    """Knowing what the graph wants and not how the values were produced is a
    real state, and refusing to record the half that is known helps nobody."""
    specs = declared_inputs(
        model(tensorContract={"inputs": [{"shape": [1, 4], "dtype": "uint8"}]})
    )

    assert specs[0].preprocess is None
    assert specs[0].layout is None
    assert specs[0].document() == {"shape": [1, 4], "dtype": "uint8"}


def test_agreement_is_silent_and_disagreement_names_both_sides():
    specs = declared_inputs(model())

    assert contract_error(specs, [((1, 3, 227, 227), "float32")]) is None
    assert "[1, 3, 4, 4]" in contract_error(specs, [((1, 3, 4, 4), "float32")])
    assert "[1, 3, 227, 227]" in contract_error(specs, [((1, 3, 4, 4), "float32")])
    assert "float64" in contract_error(specs, [((1, 3, 227, 227), "float64")])


def test_the_number_of_inputs_is_part_of_the_contract():
    specs = declared_inputs(model())

    assert "1 input" in contract_error(specs, [])
    assert "2 were supplied" in contract_error(
        specs, [((1, 3, 227, 227), "float32")] * 2
    )


def test_what_the_caller_did_not_state_is_never_reported_as_a_disagreement():
    """An inline tensor declares no dtype, and guessing one would refuse jobs
    that run today."""
    specs = declared_inputs(model())

    assert contract_error(specs, [((1, 3, 227, 227), None)]) is None
    assert contract_error(specs, [(None, None)]) is None
    assert contract_error(None, [((9, 9), "int32")]) is None
    assert contract_error((), [((9, 9), "int32")]) is None


def test_a_nested_tensor_reports_its_shape():
    assert measured_shape([[1.0, 2.0], [3.0, 4.0]]) == (2, 2)
    assert measured_shape([1.0, 2.0, 3.0]) == (3,)
    assert measured_shape([[[1.0]]]) == (1, 1, 1)


def test_a_ragged_tensor_is_left_to_the_validator_that_can_explain_it():
    """Reporting "shape mismatch" for a ragged input would replace a precise
    message with a vaguer one."""
    assert measured_shape([[1.0, 2.0], [3.0]]) is None
    assert measured_shape([]) is None
    assert measured_shape([[]]) is None
    assert measured_shape(7.0) is None


def test_preprocessing_is_published_and_never_enforced():
    """The runtime decodes nothing, so it cannot tell a correctly normalised
    tensor from a wrong one; a contract that pretended otherwise would refuse
    correct jobs and accept incorrect ones with equal confidence."""
    import inspect

    from omnitensor import dispatch

    source = inspect.getsource(dispatch)

    for never_checked in ("channelOrder", "channel_order", "preprocess"):
        assert never_checked not in source


BUNDLED = bundled_workloads_path() / "visual-library" / "manifest.json"


def test_the_bundled_contract_matches_the_artifact_it_describes():
    """The enforceable half is checked against the model, not asserted.

    The installed ``.param`` declares its own input size, so the declared shape
    is verifiable; the normalisation is not recorded anywhere in the artifact,
    which is exactly why the publisher has to state it.
    """
    manifest = json.loads(BUNDLED.read_text())
    specs = declared_inputs(manifest["requirements"]["model"])
    param = Path.home() / ".local/share/omnitensor/artifacts/visual-library-classifier"
    param = param / manifest["requirements"]["model"]["version"] / "model.param"
    if not param.is_file():
        pytest.skip("the artifact is not installed on this host")

    declared = next(
        line for line in param.read_text().splitlines() if line.startswith("Input")
    )
    sizes = dict(
        part.split("=") for part in declared.split() if "=" in part and part[0].isdigit()
    )

    assert specs[0].shape == (1, int(sizes["2"]), int(sizes["1"]), int(sizes["0"]))


def test_the_bundled_manifest_still_validates_with_its_contract():
    manifest = json.loads(BUNDLED.read_text())

    assert validate_document("workload-manifest.schema.json", manifest) == []


def test_the_published_projection_round_trips_what_a_caller_needs():
    """The applet renders this to tell a user what to supply, so every field a
    publisher stated has to survive the projection."""
    specs = declared_inputs(model())

    assert specs[0].preprocess.document() == {
        "channelOrder": "BGR",
        "mean": [104.0, 117.0, 123.0],
        "scale": [1.0, 1.0, 1.0],
    }
    assert specs[0].document()["preprocess"]["channelOrder"] == "BGR"
    assert specs[0].document()["layout"] == "NCHW"


def test_a_publisher_may_state_how_a_picture_becomes_the_shape():
    """The resize is the one step the shape does not determine.

    Two consumers that resize the same picture differently build different
    tensors from it, and the difference leaves a model's confident answers
    intact while reordering its uncertain ones — the worst shape a defect can
    have, because it looks like nothing is wrong.
    """
    specs = declared_inputs(model())

    assert specs[0].preprocess.resize is None, "absent stays absent, as before the field"

    stated = declared_inputs(
        model(
            tensorContract={
                "inputs": [{
                    "shape": [1, 3, 8, 8],
                    "dtype": "float32",
                    "layout": "NCHW",
                    "preprocess": {
                        "channelOrder": "BGR",
                        "mean": [0.0],
                        "scale": [1.0],
                        "resize": {"filter": "bicubic", "fit": "cover"},
                    },
                }]
            }
        )
    )

    assert stated[0].preprocess.resize.resample == "bicubic"
    assert stated[0].preprocess.resize.fit == "cover"
    assert stated[0].document()["preprocess"]["resize"] == {"filter": "bicubic", "fit": "cover"}


def test_the_bundled_profile_states_its_own_resize():
    manifest = json.loads(BUNDLED.read_text())
    specs = declared_inputs(manifest["requirements"]["model"])

    assert specs[0].preprocess.resize is not None, (
        "the one profile that runs should not inherit whichever resize a consumer chose"
    )
    assert validate_document("workload-manifest.schema.json", manifest) == []
