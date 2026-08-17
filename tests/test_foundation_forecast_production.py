from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest
from hypothesis import given
from hypothesis import strategies as st
from onnx import TensorProto, helper

from omnitensor.training.foundation_forecast_production import (
    FORECAST_CONTEXT_WIDTH,
    ForecastHoldout,
    FoundationForecastEvidence,
    TorchFoundationForecastOnnxExporter,
    _mae,
    _validate_forecast_graph,
    evaluate_foundation_forecast,
    produce_foundation_forecast,
    select_median_quantile,
    select_point_forecast,
)
from omnitensor.training.recipes import FetchedModelSource, ModelRecipeError, load_model_recipe


def _context(value: float = 0.0) -> tuple[float, ...]:
    return (value,) * FORECAST_CONTEXT_WIDTH


def _holdout(*, repeat_target: bool = False) -> ForecastHoldout:
    contexts = tuple(_context(float(index)) for index in range(32))
    targets = tuple(context[-1] + (0.0 if repeat_target else 1.0) for context in contexts)
    return ForecastHoldout(contexts, targets, tuple(range(32)), "a" * 64)


class _OffsetRunner:
    def __init__(self, offset: float):
        self.offset = offset

    def predict(self, context):
        return context[-1] + self.offset


def _fetched(tmp_path: Path, identifier: str) -> FetchedModelSource:
    recipe = load_model_recipe(Path("model-recipes") / f"{identifier}.json")
    root = tmp_path / "source"
    root.mkdir()
    for source in recipe.sources:
        (root / source.filename).write_bytes(source.role.encode())
    receipt = root / "source-receipt.json"
    receipt.write_bytes(b'{"verified":true}')
    return FetchedModelSource(recipe, root, receipt)


def _onnx_model(input_shape=(1, 512), output_shape=(1, 1)):
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["context"], ["forecast"], axes=[1], keepdims=1)],
        "forecast",
        [helper.make_tensor_value_info("context", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("forecast", TensorProto.FLOAT, output_shape)],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


@given(
    values=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=64,
    )
)
def test_mae_property_is_nonnegative_and_zero_for_identical_values(values):
    expected = sum(abs(value) for value in values) / len(values)
    assert _mae(values, [0.0] * len(values)) == pytest.approx(expected)
    assert _mae(values, values) == 0.0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"contexts": (_context(),)}, "forecast holdout requires 32..512 samples"),
        ({"targets": (1.0,)}, "forecast holdout arrays must have equal lengths"),
        (
            {"observed_at_ms": tuple(range(31)) + (30,)},
            "forecast holdout timestamps must be strictly increasing",
        ),
        (
            {"contexts": ((_context()[:-1]),) + _holdout().contexts[1:]},
            "forecast context width must be 512",
        ),
        (
            {"contexts": ((_context()[:-1] + (math.inf,)),) + _holdout().contexts[1:]},
            "forecast context values must be finite",
        ),
        ({"targets": (math.nan,) + _holdout().targets[1:]}, "forecast targets must be finite"),
        (
            {"corpus_sha256": "A" * 64},
            "forecast corpus digest must be lowercase SHA-256",
        ),
    ],
)
def test_forecast_holdout_refuses_invalid_data(change, message):
    values = {
        "contexts": _holdout().contexts,
        "targets": _holdout().targets,
        "observed_at_ms": _holdout().observed_at_ms,
        "corpus_sha256": "a" * 64,
    }
    values.update(change)
    with pytest.raises(ValueError, match=f"^{message}$"):
        ForecastHoldout(**values)


def test_source_output_selectors_use_exact_recipe_coordinates():
    point = [[[float(row)] for row in range(96)]]
    quantiles = [
        [[float(quantile * 100 + horizon) for horizon in range(64)] for quantile in range(9)]
    ]
    assert select_point_forecast(point) == 0.0
    assert select_median_quantile(quantiles) == 400.0


@pytest.mark.parametrize(
    ("selector", "value", "message"),
    [
        (select_point_forecast, [], "forecast source output shape must be [1,96,1]"),
        (select_point_forecast, 1, "forecast source output shape must be [1,96,1]"),
        (
            select_median_quantile,
            [[[0.0] * 63 for _ in range(9)]],
            "forecast source output shape must be [1,9,64]",
        ),
        (
            select_point_forecast,
            [[["bad"]] + [[0.0] for _ in range(95)]],
            "TTM first-horizon output must be numeric",
        ),
        (
            select_median_quantile,
            [[[math.nan] * 64 for _ in range(9)]],
            "Chronos median first-horizon output must be finite",
        ),
    ],
)
def test_source_output_selectors_refuse_malformed_values(selector, value, message):
    with pytest.raises(ValueError) as caught:
        selector(value)
    assert str(caught.value) == message


def test_forecast_gate_accepts_only_source_parity_and_both_baselines():
    evidence = evaluate_foundation_forecast(
        _holdout(), _OffsetRunner(1.0), _OffsetRunner(1.0), _OffsetRunner(0.5)
    )
    assert evidence == FoundationForecastEvidence(0.0, 0.0, 1.0, 0.5, 1.0, 0.0, 32, True)

    parity_failure = evaluate_foundation_forecast(
        _holdout(), _OffsetRunner(1.0), _OffsetRunner(1.001), _OffsetRunner(0.5)
    )
    local_failure = evaluate_foundation_forecast(
        _holdout(), _OffsetRunner(1.0), _OffsetRunner(1.0), _OffsetRunner(1.0)
    )
    skill_failure = evaluate_foundation_forecast(
        _holdout(), _OffsetRunner(0.3), _OffsetRunner(0.3), _OffsetRunner(0.2)
    )
    assert parity_failure.portable_export_max_error == pytest.approx(0.001)
    assert not parity_failure.accepted
    assert not local_failure.accepted
    assert skill_failure.repeat_last_skill == pytest.approx(0.3)
    assert not skill_failure.accepted


def test_forecast_gate_refuses_undefined_skill_and_nonfinite_predictions():
    with pytest.raises(ValueError, match="^repeat-last MAE is zero; skill is undefined$"):
        evaluate_foundation_forecast(
            _holdout(repeat_target=True),
            _OffsetRunner(0.0),
            _OffsetRunner(0.0),
            _OffsetRunner(1.0),
        )
    with pytest.raises(ValueError, match="^forecast prediction must be finite$"):
        evaluate_foundation_forecast(
            _holdout(), _OffsetRunner(math.inf), _OffsetRunner(1.0), _OffsetRunner(0.5)
        )
    with pytest.raises(ValueError, match="^MAE arrays must be non-empty and equal length$"):
        _mae((), ())
    with pytest.raises(ValueError, match="^MAE arrays must be non-empty and equal length$"):
        _mae((1.0,), (1.0, 2.0))


@pytest.mark.parametrize(
    ("identifier", "expected_index"),
    [
        ("ibm-granite-ttm-r2", (slice(None), 0, 0)),
        ("amazon-chronos-bolt-tiny", (slice(None), 4, 0)),
    ],
)
def test_torch_exporter_uses_fixed_scalar_contract(  # noqa: C901
    tmp_path, monkeypatch, identifier, expected_index
):
    fetched = _fetched(tmp_path, identifier)
    calls = {}

    class FakeModule:
        def eval(self):
            calls["eval_count"] = calls.get("eval_count", 0) + 1
            return self

    class FakeTensor:
        def __getitem__(self, index):
            calls["index"] = index
            return self

        def reshape(self, *shape):
            calls["reshape"] = shape
            return self

    tensor = FakeTensor()

    class FakeCandidate(FakeModule):
        def __call__(self, context):
            calls["candidate_context"] = context
            return SimpleNamespace(prediction_outputs=tensor, quantile_preds=tensor)

    candidate = FakeCandidate()

    def loader(source):
        calls["loader_source"] = source
        return candidate

    def export(wrapper, dummy, destination, **kwargs):
        calls.update({"wrapper": wrapper, "dummy": dummy, **kwargs})
        onnx.save_model(_onnx_model(), destination)

    fake_torch = SimpleNamespace(
        nn=SimpleNamespace(Module=FakeModule),
        onnx=SimpleNamespace(export=export),
        zeros=lambda shape, dtype: (shape, dtype),
        float32="float32",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    real_mkstemp = tempfile.mkstemp

    def checked_mkstemp(*, prefix, suffix, dir):
        assert (prefix, suffix, dir) == (".foundation-forecast-", ".onnx", tmp_path / "out")
        return real_mkstemp(prefix=prefix, suffix=suffix, dir=dir)

    monkeypatch.setattr(tempfile, "mkstemp", checked_mkstemp)
    real_onnx_load = onnx.load

    def checked_onnx_load(path, *, load_external_data):
        calls["onnx_load"] = (path, load_external_data)
        return real_onnx_load(path, load_external_data=load_external_data)

    monkeypatch.setattr(onnx, "load", checked_onnx_load)
    destination = tmp_path / "out" / "forecast.onnx"
    TorchFoundationForecastOnnxExporter(loader).export(fetched, destination)

    assert destination.is_file()
    assert calls["loader_source"] is fetched
    assert calls["dummy"] == ((1, 512), "float32")
    assert calls["input_names"] == ["context"]
    assert calls["output_names"] == ["forecast"]
    assert calls["opset_version"] == 17
    assert calls["do_constant_folding"] is True
    assert calls["dynamo"] is False
    assert calls["onnx_load"][0].name.startswith(".foundation-forecast-")
    assert calls["onnx_load"][1] is False
    assert calls["wrapper"].forward("window") is tensor
    assert calls["candidate_context"] == "window"
    assert calls["index"] == expected_index
    assert calls["reshape"] == (1, 1)
    assert calls["eval_count"] == 2


def test_torch_exporter_refuses_wrong_recipe_or_loader(tmp_path, monkeypatch):
    fetched = _fetched(tmp_path, "ibm-granite-ttm-r2")
    wrong = replace(fetched, recipe=replace(fetched.recipe, id="wrong"))
    with pytest.raises(ModelRecipeError) as caught:
        TorchFoundationForecastOnnxExporter(lambda source: object()).export(
            wrong, tmp_path / "wrong.onnx"
        )
    assert (caught.value.code, caught.value.detail) == (
        "producer-incompatible",
        "recipe is not a pinned forecast candidate",
    )

    with pytest.raises(ModelRecipeError) as caught:
        TorchFoundationForecastOnnxExporter(lambda source: object()).export(
            fetched, tmp_path / "bad.onnx"
        )
    assert (caught.value.code, caught.value.detail) == (
        "producer-incompatible",
        "forecast loader did not return a module",
    )


def _wrong_forecast_shape(model):
    model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 2


@pytest.mark.parametrize(
    ("mutate", "detail"),
    [
        (
            lambda model: setattr(model.graph.input[0], "name", "wrong"),
            "forecast ONNX must have one context input",
        ),
        (
            lambda model: setattr(model.graph.output[0], "name", "wrong"),
            "forecast ONNX must have one scalar output",
        ),
        (_wrong_forecast_shape, "forecast ONNX shapes disagree with recipe"),
    ],
)
def test_forecast_graph_validation_rejects_contract_drift(mutate, detail):
    model = _onnx_model()
    mutate(model)
    with pytest.raises(ModelRecipeError) as caught:
        _validate_forecast_graph(model)
    assert (caught.value.code, caught.value.detail) == ("producer-invalid", detail)


def test_produce_foundation_forecast_emits_private_safe_report(tmp_path, monkeypatch):
    fetched = _fetched(tmp_path, "amazon-chronos-bolt-tiny")
    open_calls = []

    def open_source(recipe, root):
        open_calls.append((recipe, root))
        return fetched

    monkeypatch.setattr(
        "omnitensor.training.foundation_forecast_production.open_fetched_model_source",
        open_source,
    )

    class FakeExporter:
        def export(self, source, destination):
            assert source is fetched
            destination.write_bytes(b"portable-forecast")

    source_calls = []
    portable_calls = []

    def source_factory(source):
        source_calls.append(source)
        return _OffsetRunner(1.0)

    def portable_factory(path):
        portable_calls.append(path)
        return _OffsetRunner(1.0)

    produced = produce_foundation_forecast(
        "recipe",
        "sources",
        tmp_path / "nested" / "output",
        _holdout(),
        source_factory,
        portable_factory,
        _OffsetRunner(0.5),
        FakeExporter(),
    )
    report = json.loads(produced.report_path.read_text())
    assert produced.model_path.read_bytes() == b"portable-forecast"
    assert produced.report_path.read_bytes() == json.dumps(report, separators=(",", ":")).encode()
    assert tuple(report) == (
        "reportVersion",
        "kind",
        "recipeId",
        "recipeVersion",
        "recipeSha256",
        "sourceReceiptSha256",
        "sourceDigests",
        "portableModel",
        "holdout",
        "portableSourceGate",
        "nativeTargets",
        "cpuFallback",
    )
    assert tuple(report["portableModel"]) == (
        "format",
        "sha256",
        "tensorContract",
        "outputContract",
        "producer",
    )
    assert tuple(report["holdout"]) == ("corpusSha256", "sampleCount", "timeOrdered")
    assert tuple(report["portableSourceGate"]) == (
        "candidateMae",
        "sourceMae",
        "repeatLastMae",
        "localLinearMae",
        "repeatLastSkill",
        "portableExportMaxError",
        "accepted",
    )
    assert tuple(report["nativeTargets"]) == ("gpu", "npu", "tpu")
    assert all(tuple(claim) == ("status", "reason") for claim in report["nativeTargets"].values())
    assert produced.evidence == FoundationForecastEvidence(0.0, 0.0, 1.0, 0.5, 1.0, 0.0, 32, True)
    native_claim = {
        "status": "unqualified",
        "reason": "no target compiler, native parity, or named-device evidence was run",
    }
    assert report == {
        "reportVersion": 1,
        "kind": "foundation-forecast-source-production",
        "recipeId": fetched.recipe.id,
        "recipeVersion": fetched.recipe.version,
        "recipeSha256": fetched.recipe.document_sha256,
        "sourceReceiptSha256": hashlib.sha256(b'{"verified":true}').hexdigest(),
        "sourceDigests": {item.role: item.sha256 for item in fetched.recipe.sources},
        "portableModel": {
            "format": "onnx",
            "sha256": hashlib.sha256(b"portable-forecast").hexdigest(),
            "tensorContract": fetched.recipe.tensor_contract,
            "outputContract": fetched.recipe.output_contract,
            "producer": fetched.recipe.producer,
        },
        "holdout": {"corpusSha256": "a" * 64, "sampleCount": 32, "timeOrdered": True},
        "portableSourceGate": {
            "candidateMae": 0.0,
            "sourceMae": 0.0,
            "repeatLastMae": 1.0,
            "localLinearMae": 0.5,
            "repeatLastSkill": 1.0,
            "portableExportMaxError": 0.0,
            "accepted": True,
        },
        "nativeTargets": {"gpu": native_claim, "npu": native_claim, "tpu": native_claim},
        "cpuFallback": "forbidden",
    }
    assert open_calls == [("recipe", "sources")]
    assert source_calls == [fetched]
    assert portable_calls == [produced.model_path]
    serialized = json.dumps(report)
    assert "observed_at_ms" not in serialized
    assert "contexts" not in serialized
    assert "targets" not in serialized


def test_production_refuses_wrong_recipe_conflicts_and_failed_gates(tmp_path, monkeypatch):
    fetched = _fetched(tmp_path, "ibm-granite-ttm-r2")

    class FakeExporter:
        def export(self, source, destination):
            destination.write_bytes(b"portable")

    monkeypatch.setattr(
        "omnitensor.training.foundation_forecast_production.open_fetched_model_source",
        lambda recipe, root: replace(fetched, recipe=replace(fetched.recipe, id="wrong")),
    )
    with pytest.raises(ModelRecipeError) as caught:
        produce_foundation_forecast(
            "recipe",
            "sources",
            tmp_path / "wrong",
            _holdout(),
            lambda source: _OffsetRunner(1.0),
            lambda path: _OffsetRunner(1.0),
            _OffsetRunner(0.5),
            FakeExporter(),
        )
    assert (caught.value.code, caught.value.detail) == (
        "producer-incompatible",
        "recipe is not a pinned forecast candidate",
    )

    monkeypatch.setattr(
        "omnitensor.training.foundation_forecast_production.open_fetched_model_source",
        lambda recipe, root: fetched,
    )
    conflict = tmp_path / "conflict"
    conflict.mkdir()
    (conflict / f"{fetched.recipe.id}.onnx").write_bytes(b"existing")
    with pytest.raises(ModelRecipeError) as caught:
        produce_foundation_forecast(
            "recipe",
            "sources",
            conflict,
            _holdout(),
            lambda source: _OffsetRunner(1.0),
            lambda path: _OffsetRunner(1.0),
            _OffsetRunner(0.5),
            FakeExporter(),
        )
    assert (caught.value.code, caught.value.detail) == (
        "producer-conflict",
        "forecast production output already exists",
    )
    assert (conflict / f"{fetched.recipe.id}.onnx").read_bytes() == b"existing"

    failed = tmp_path / "failed"
    with pytest.raises(ModelRecipeError) as caught:
        produce_foundation_forecast(
            "recipe",
            "sources",
            failed,
            _holdout(),
            lambda source: _OffsetRunner(1.0),
            lambda path: _OffsetRunner(0.0),
            _OffsetRunner(0.5),
            FakeExporter(),
        )
    assert (caught.value.code, caught.value.detail) == (
        "quality-gate-failed",
        "foundation forecast did not pass gates",
    )
    assert list(failed.iterdir()) == []
