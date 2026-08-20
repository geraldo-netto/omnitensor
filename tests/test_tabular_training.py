from __future__ import annotations

import hashlib
import pickle

import pytest
from hypothesis import given
from hypothesis import strategies as st

import omnitensor.training.build as build
import omnitensor.training.desktop as desktop
import omnitensor.training.hardware as hardware
from omnitensor.training.contracts import TrainingError
from omnitensor.training.tabular import (
    BuildAdvisorModel,
    BuildExample,
    JsonlPolicy,
    TabularExample,
    binary_auc,
    binary_class_counts,
    chronological_split,
    fit_balanced_logistic,
    fit_output,
    floor_chronological_split,
    load_bounded_jsonl,
    normalized_features,
    numeric_training_report,
    population_normalization,
    publish_numeric_training,
    ranked_auc,
    require_binary_class_floor,
    stable_sigmoid,
    unit_interval,
)


def policy(**changes: object) -> JsonlPolicy:
    fields = {
        "max_bytes": 100,
        "max_lines": 2,
        "max_line_bytes": 20,
        "unsafe_code": "unsafe",
        "unsafe_detail": "unsafe input",
        "too_large_code": "too-large",
        "bytes_detail": "too many bytes",
        "lines_detail": "too many lines",
        "invalid_code": "invalid",
        "line_detail": "line {line_number} invalid",
        "context_detail": "line {line_number}: {detail}",
    }
    fields.update(changes)
    return JsonlPolicy(**fields)


def training_error(call) -> TrainingError:
    with pytest.raises(TrainingError) as raised:
        call()
    return raised.value


def test_legacy_build_symbols_are_canonical_shared_objects():
    assert build.BuildExample is hardware.BuildExample is desktop.BuildExample
    assert build.BuildExample is BuildExample is TabularExample
    assert build.BuildAdvisorModel is hardware.BuildAdvisorModel
    assert build.BuildAdvisorModel is desktop.BuildAdvisorModel is BuildAdvisorModel
    assert BuildExample.__name__ == "BuildExample"
    assert BuildAdvisorModel.__name__ == "BuildAdvisorModel"
    assert BuildExample.__module__ == BuildAdvisorModel.__module__ == "omnitensor.training.build"
    example = BuildExample(1, (2.0,), 0, (), ())
    assert repr(example).startswith("BuildExample(")
    assert pickle.loads(pickle.dumps(example)) == example
    assert build.fit_output is hardware.fit_output is desktop.fit_output
    assert build._normalization is hardware._normalization is desktop._normalization
    assert build._auc is hardware._auc is binary_auc
    assert BuildAdvisorModel._probability is stable_sigmoid


def test_model_prediction_uses_the_package_functions_not_a_patched_module():
    """Importing ``build`` must not rebind the shared model's own callables."""
    assert BuildAdvisorModel._normalize is normalized_features
    assert BuildAdvisorModel._probability is stable_sigmoid
    model = BuildAdvisorModel((1.0,), (2.0,), ((2.0,),), (1.0,))
    assert model.predict((9.0,)) == (stable_sigmoid(9.0),)


@pytest.mark.parametrize(
    ("size", "training", "holdout"),
    [(2, (0,), (1,)), (5, (0, 1, 2, 3), (4,)), (10, tuple(range(8)), (8, 9))],
)
def test_chronological_split_preserves_historical_boundaries(size, training, holdout):
    assert chronological_split(tuple(range(size)), insufficient_detail="need two") == (
        training,
        holdout,
    )


def test_split_policies_refuse_and_apply_floors_with_stable_detail():
    error = training_error(
        lambda: chronological_split((1,), insufficient_detail="family history short")
    )
    assert (error.code, error.detail) == ("insufficient-history", "family history short")
    error = training_error(
        lambda: floor_chronological_split(tuple(range(5)), 4, 2, insufficient_detail="floor short")
    )
    assert (error.code, error.detail) == ("insufficient-history", "floor short")
    assert floor_chronological_split(tuple(range(10)), 7, 2, insufficient_detail="unused") == (
        tuple(range(8)),
        (8, 9),
    )
    assert floor_chronological_split(tuple(range(10)), 9, 1, insufficient_detail="unused") == (
        tuple(range(9)),
        (9,),
    )


@given(st.integers(min_value=2, max_value=10_000))
def test_chronological_split_is_lossless_ordered_and_nonempty(size):
    values = tuple(range(size))
    training, holdout = chronological_split(values, insufficient_detail="unused")
    assert training + holdout == values
    assert training and holdout
    assert len(training) == min(size - 1, max(1, int(size * 0.8)))


def test_population_floors_and_balanced_fit_are_policy_driven():
    examples = (
        TabularExample(0, (1.0, 5.0), 0, (), ()),
        TabularExample(1, (1.0, 7.0), 1, (), ()),
    )
    means, storage_scales = population_normalization(
        examples, lambda item: item.features, scale_floor=1.0
    )
    _, shared_scales = population_normalization(
        examples, lambda item: item.features, scale_floor=1e-9
    )
    assert means == (1.0, 6.0)
    assert storage_scales == (1.0, 1.0)
    assert shared_scales == (1e-9, 1.0)
    weights_200, intercept_200 = fit_balanced_logistic(
        examples,
        lambda item: item.build_failed,
        lambda item: item.features,
        means,
        storage_scales,
        iterations=200,
    )
    weights_240, intercept_240 = fit_balanced_logistic(
        examples,
        lambda item: item.build_failed,
        lambda item: item.features,
        means,
        storage_scales,
        iterations=240,
    )
    assert weights_200 == (0.0, 1.5733816527894948)
    assert intercept_200 == pytest.approx(0.0, abs=1e-15)
    assert weights_240 == (0.0, 1.6625980395819149)
    assert intercept_240 == pytest.approx(0.0, abs=1e-15)
    assert fit_output(examples, lambda item: item.build_failed, means, storage_scales) == (
        weights_240,
        intercept_240,
    )


def test_normalized_model_and_validation_match_legacy_contract():
    model = BuildAdvisorModel(
        means=(1.0, 2.0),
        scales=(2.0, 4.0),
        weights=((1.0, -1.0), (2.0, -2.0)),
        intercepts=(0.0, 1.0),
    )
    assert model.predict((3.0, 6.0)) == (
        stable_sigmoid(3.0),
        stable_sigmoid(-2.0),
    )
    assert stable_sigmoid(1_000.0) == pytest.approx(1.0)
    assert stable_sigmoid(-1_000.0) > 0.0
    for values, detail in (
        ((1.0,), "width disagrees"),
        ((True, 1.0), "must be numbers"),
        ((float("inf"), 1.0), "must be finite"),
    ):
        error = training_error(lambda values=values: normalized_features(values, (0, 0), (1, 1)))
        assert error.code == "features-invalid"
        assert detail in error.detail


def test_metrics_class_floors_and_unit_bounds_are_shared_without_drift():
    scored = ((0.1, 0), (0.5, 1), (0.5, 0), (0.9, 1))
    assert ranked_auc(scored, empty_class_detail=None) == 0.875
    assert binary_auc(scored) == 0.875
    examples = (
        TabularExample(0, (), 0, (), ()),
        TabularExample(1, (), 1, (), ()),
    )
    assert binary_class_counts(examples, lambda item: item.build_failed) == (1, 1)
    require_binary_class_floor(examples, lambda item: item.build_failed, 1, "unused")
    error = training_error(
        lambda: require_binary_class_floor(
            examples, lambda item: item.build_failed, 2, "exact class detail"
        )
    )
    assert (error.code, error.detail) == ("class-imbalance", "exact class detail")
    error = training_error(lambda: binary_auc(((0.1, 1), (0.2, 1))))
    assert (error.code, error.detail) == (
        "class-imbalance",
        "AUC requires both build outcomes",
    )
    assert unit_interval(0, "gate") == 0.0
    assert unit_interval(1.0, "gate") == 1.0
    for value in (True, -0.1, 1.1, float("nan"), "0.5"):
        with pytest.raises(ValueError, match=r"gate must be in \[0, 1\]"):
            unit_interval(value, "gate")


def test_bounded_jsonl_hashes_exact_bytes_and_calls_hook_in_order(tmp_path):
    source = tmp_path / "rows.jsonl"
    raw = b'{"n":1}\r\n {"n":2}'
    source.write_bytes(raw)
    seen = []
    items, digest = load_bounded_jsonl(
        source,
        policy(),
        lambda document: document["n"],
        on_item=seen.append,
    )
    assert items == (1, 2)
    assert seen == [1, 2]
    assert digest == hashlib.sha256(raw).hexdigest()


def test_bounded_jsonl_enforces_exact_limits_and_context(tmp_path):
    source = tmp_path / "rows.jsonl"
    source.write_bytes(b"{}\n{}\n")
    assert load_bounded_jsonl(source, policy(max_bytes=6), lambda item: item)[0] == ({}, {})
    error = training_error(
        lambda: load_bounded_jsonl(source, policy(max_bytes=5), lambda item: item)
    )
    assert (error.code, error.detail) == ("too-large", "too many bytes")
    error = training_error(
        lambda: load_bounded_jsonl(source, policy(max_lines=1), lambda item: item)
    )
    assert (error.code, error.detail) == ("too-large", "too many lines")
    source.write_bytes(b"{bad}\n")
    error = training_error(lambda: load_bounded_jsonl(source, policy(), lambda item: item))
    assert (error.code, error.detail) == ("invalid", "line 1: invalid JSON")
    source.write_bytes(b"{}\n")
    error = training_error(
        lambda: load_bounded_jsonl(
            source,
            policy(),
            lambda _item: (_ for _ in ()).throw(TrainingError("nested", "typed detail")),
        )
    )
    assert (error.code, error.detail) == ("invalid", "line 1: typed detail")


def test_bounded_jsonl_rejects_unsafe_blank_and_oversized_lines(tmp_path):
    missing = training_error(
        lambda: load_bounded_jsonl(tmp_path / "missing", policy(), lambda item: item)
    )
    assert (missing.code, missing.detail) == ("unsafe", "unsafe input")
    source = tmp_path / "rows.jsonl"
    source.write_bytes(b" \n")
    blank = training_error(lambda: load_bounded_jsonl(source, policy(), lambda item: item))
    assert (blank.code, blank.detail) == ("invalid", "line 1 invalid")
    source.write_bytes(b"{}\n")
    long = training_error(
        lambda: load_bounded_jsonl(source, policy(max_line_bytes=2), lambda item: item)
    )
    assert (long.code, long.detail) == ("invalid", "line 1 invalid")


def test_numeric_report_and_publication_keep_order_and_precedence(tmp_path):
    model_path = tmp_path / "standalone.onnx"
    model_path.write_bytes(b"model")
    report = numeric_training_report(
        {"version": 1, "family": "fixture"},
        model_path,
        3,
        digest=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    assert tuple(report) == (
        "version",
        "family",
        "model",
        "tensorContract",
        "outputContract",
        "targets",
    )
    assert tuple(report["model"]) == ("format", "filename", "sha256")
    assert tuple(report["tensorContract"]["inputs"][0]) == ("shape", "dtype", "layout")
    assert tuple(report["targets"]) == ("tpu", "npu", "gpu")

    events = []

    class Exporter:
        def export(self, _model, destination):
            events.append("export")
            destination.write_bytes(b"portable")

    def make_report(path):
        events.append("report")
        return {"digest": hashlib.sha256(path.read_bytes()).hexdigest()}

    def writer(path, payload, *, prefix, numeric):
        events.append(("writer", path.name, payload, prefix, numeric))

    payload = publish_numeric_training(
        tmp_path / "fit",
        object(),
        Exporter(),
        report_name="report.json",
        report_prefix=".fixture-",
        report=make_report,
        writer=writer,
    )
    assert payload == {"digest": hashlib.sha256(b"portable").hexdigest()}
    assert events == [
        "export",
        "report",
        ("writer", "report.json", payload, ".fixture-", True),
    ]


@pytest.mark.parametrize("content", [None, b""])
def test_numeric_publication_refuses_missing_or_empty_export(tmp_path, content):
    class Exporter:
        def export(self, _model, destination):
            if content is not None:
                destination.write_bytes(content)

    error = training_error(
        lambda: publish_numeric_training(
            tmp_path / "fit",
            object(),
            Exporter(),
            report_name="report.json",
            report_prefix=".fixture-",
            report=lambda _path: {},
            writer=lambda *_args, **_kwargs: None,
        )
    )
    assert (error.code, error.detail) == (
        "export-failed",
        "exporter produced no portable model",
    )
