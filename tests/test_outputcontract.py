from __future__ import annotations

import pytest

from omnitensor.outputcontract import (
    DEFAULT_TOP_K,
    MAX_LABEL_CHARS,
    MAX_LABELS,
    MAX_TOP_K,
    OutputSpec,
    _scores,
    declared_output,
    parse_labels,
    reduce_output,
)

SCORES = [0.1, 0.7, 0.05, 0.15]


def model(**contract) -> dict:
    return {"id": "sample-model", "outputContract": {"kind": "classification", **contract}}


def test_a_declared_contract_is_read_with_its_defaults():
    assert declared_output(model()) == OutputSpec("classification", DEFAULT_TOP_K, None)
    assert declared_output(model(topK=2, labels="labels.txt")) == OutputSpec(
        "classification", 2, "labels.txt"
    )


def test_the_top_k_bound_is_enforced_without_clamping():
    assert declared_output(model(topK=1)).top_k == 1
    assert declared_output(model(topK=MAX_TOP_K)).top_k == MAX_TOP_K
    for invalid in (0, -1, MAX_TOP_K + 1):
        with pytest.raises(ValueError) as raised:
            declared_output(model(topK=invalid))
        assert str(raised.value) == (f"output contract topK must be in [1, {MAX_TOP_K}]")


def test_the_top_k_bound_matches_the_canonical_schema():
    from omnitensor.registry import workload_model_contract_schemas

    schema = workload_model_contract_schemas()["outputContract"]
    top_k = schema["properties"]["topK"]
    assert top_k["type"] == "integer"
    assert top_k["minimum"] == 1
    assert top_k["maximum"] == MAX_TOP_K


def test_a_model_declaring_no_contract_returns_its_tensors_unchanged():
    """Every profile written before the field behaves exactly as it did."""
    assert declared_output({"id": "sample-model"}) is None
    assert declared_output(None) is None
    assert declared_output("model") is None
    assert declared_output({"outputContract": {}}) is None
    assert reduce_output(None, [SCORES]) is None


def test_a_classification_reduces_to_the_highest_scoring_entries():
    reading = reduce_output(OutputSpec("classification", 2), [SCORES])

    assert reading["kind"] == "classification"
    assert [entry["index"] for entry in reading["top"]] == [1, 3]
    assert reading["top"][0]["score"] == 0.7
    assert "label" not in reading["top"][0]


def test_direct_output_specs_obey_the_same_top_k_bound():
    scores = [float(index) for index in range(MAX_TOP_K + 1)]
    reading = reduce_output(OutputSpec("classification", MAX_TOP_K), [scores])
    assert len(reading["top"]) == MAX_TOP_K
    assert [entry["index"] for entry in reading["top"][:2]] == [MAX_TOP_K, MAX_TOP_K - 1]
    assert reading["top"][-1]["index"] == 1

    for invalid in (False, True, "5", 0, MAX_TOP_K + 1):
        with pytest.raises(ValueError) as raised:
            reduce_output(OutputSpec("classification", invalid), [scores])
        assert str(raised.value) == (f"output contract topK must be in [1, {MAX_TOP_K}]")

    with pytest.raises(ValueError):
        reduce_output(OutputSpec("classification", MAX_TOP_K + 1), [])
    with pytest.raises(ValueError):
        reduce_output(OutputSpec("raw", MAX_TOP_K + 1), [scores])


def test_labels_are_attached_where_they_exist_and_never_invented():
    """A guessed label is worse than an honest index: it is wrong in a form
    that reads as right."""
    labels = ("alpha", "beta")

    reading = reduce_output(OutputSpec("classification", 3), [SCORES], labels)

    assert reading["top"][0]["label"] == "beta"
    assert "label" not in reading["top"][1], "index 3 has no label and must not gain one"


def test_ties_are_broken_by_index_so_two_hosts_answer_identically():
    reading = reduce_output(OutputSpec("classification", 2), [[0.5, 0.5, 0.5]])

    assert [entry["index"] for entry in reading["top"]] == [0, 1]


def test_a_nested_output_is_reduced_because_that_is_how_one_arrives():
    """The live ncnn classifier returns its 1000 scores nested one deep."""
    reading = reduce_output(OutputSpec("classification", 1), [[[0.2, 0.9]]])

    assert reading["top"] == [{"index": 1, "score": 0.9}]


def test_an_output_the_reduction_cannot_describe_is_left_alone():
    """Reporting a confident answer about an arbitrary slice would be worse
    than reporting nothing."""
    assert reduce_output(OutputSpec("classification"), [[[0.1], [0.2]]]) is None
    assert reduce_output(OutputSpec("classification"), []) is None
    assert reduce_output(OutputSpec("classification"), [[]]) is None
    assert reduce_output(OutputSpec("classification"), ["scores"]) is None
    assert reduce_output(OutputSpec("classification"), [[True, False]]) is None


@pytest.mark.parametrize(
    "tensor",
    [None, [], [[]], [[0.1], [0.2]], "scores", [True], [1.0, "bad"]],
)
def test_score_extraction_rejects_every_ambiguous_shape_or_value(tensor):
    assert _scores(tensor) is None


def test_score_extraction_accepts_one_numeric_row_and_normalizes_to_float():
    assert _scores([1, 2.5]) == [1.0, 2.5]
    assert _scores([[1, 2.5]]) == [1.0, 2.5]


def test_a_kind_that_does_not_reduce_is_returned_whole():
    for kind in ("embedding", "raw"):
        assert reduce_output(OutputSpec(kind), [SCORES]) is None
        assert OutputSpec(kind).reduces is False


def test_a_non_finite_score_is_dropped_rather_than_ranked():
    reading = reduce_output(OutputSpec("classification", 3), [[float("nan"), 0.5, 0.2]])

    assert [entry["index"] for entry in reading["top"]] == [1, 2]


def test_a_malformed_top_k_falls_back_rather_than_failing():
    """The schema bounds it, so this only matters for a manifest that reached
    here another way — and a refusal would cost the whole result."""
    assert declared_output(model(topK="many")).top_k == DEFAULT_TOP_K
    assert declared_output(model(topK=True)).top_k == DEFAULT_TOP_K


def test_labels_are_one_per_line_bounded_and_blank_tolerant():
    labels = parse_labels("alpha\n\n  beta  \n" + "x" * (MAX_LABEL_CHARS + 10) + "\n")

    assert labels[0] == "alpha"
    assert labels[1] == "beta"
    assert len(labels[2]) == MAX_LABEL_CHARS


def test_a_label_list_is_bounded_like_any_other_artifact():
    labels = parse_labels("\n".join(str(index) for index in range(MAX_LABELS + 100)))

    assert len(labels) == MAX_LABELS


def test_the_reduction_never_replaces_the_tensors_it_reads():
    """A consumer that wants raw scores must not lose them to a reading."""
    import inspect

    from omnitensor.plugins import orchestration

    source = inspect.getsource(orchestration._postprocess_stage)

    assert '"reading"' in source
    assert "pop" not in source and "del " not in source


def _resolution(path, ready=True):
    class Resolution:
        def __init__(self):
            self.ready = ready
            self.path = path

    return Resolution()


def test_labels_are_read_from_the_directory_the_resolver_verified(tmp_path):
    from omnitensor.plugins.orchestration import _label_reader

    (tmp_path / "labels.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    reads = []

    def resolve(artifact_id):
        reads.append(artifact_id)
        return _resolution(tmp_path / "model.param")

    read = _label_reader(model(labels="labels.txt"), resolve)

    assert read() == ("alpha", "beta")
    assert read() == ("alpha", "beta")
    assert reads == ["sample-model"], "the labels file is read once, not per job"


def test_an_artifact_installed_later_is_still_picked_up(tmp_path):
    """Caching an empty list would report indices forever on a host where the
    artifact arrives a moment after startup."""
    from omnitensor.plugins.orchestration import _label_reader

    state = {"ready": False}

    def resolve(artifact_id):
        return _resolution(tmp_path / "model.param", ready=state["ready"])

    read = _label_reader(model(labels="labels.txt"), resolve)

    assert read() == ()
    state["ready"] = True
    (tmp_path / "labels.txt").write_text("alpha\n", encoding="utf-8")
    assert read() == ("alpha",)


def test_a_missing_or_unreadable_labels_file_reports_indices(tmp_path):
    from omnitensor.plugins.orchestration import _label_reader

    read = _label_reader(
        model(labels="absent.txt"), lambda _id: _resolution(tmp_path / "model.param")
    )

    assert read() == ()


def test_a_model_declaring_no_labels_never_consults_the_store():
    from omnitensor.plugins.orchestration import _label_reader

    def resolve(artifact_id):
        raise AssertionError("the artifact store must not be touched")

    assert _label_reader(model(), resolve)() == ()
    assert _label_reader({"id": "sample-model"}, resolve)() == ()


def test_a_labels_file_that_is_not_a_file_is_refused_at_preparation(tmp_path):
    import pytest

    from omnitensor.preparation import PreparationError, prepare_artifact

    source = tmp_path / "model.onnx"
    source.write_bytes(b"a model")

    with pytest.raises(PreparationError, match="labels-invalid"):
        prepare_artifact(
            source,
            artifact_id="sample-model",
            version="1.0.0",
            model_format="onnx",
            labels=tmp_path / "absent.txt",
        )


def test_a_labels_file_is_staged_and_re_verified_like_any_other_companion(tmp_path):
    """The same digest machinery, so a label list swapped after installation
    makes the artifact unresolvable instead of renaming every result."""
    from omnitensor.plugins import ArtifactInstaller
    from omnitensor.preparation import install_prepared, prepare_artifact

    source = tmp_path / "model.onnx"
    source.write_bytes(b"a model")
    labels = tmp_path / "names.txt"
    labels.write_text("alpha\nbeta\n", encoding="utf-8")
    store = tmp_path / "store"

    prepared = prepare_artifact(
        source,
        artifact_id="sample-model",
        version="1.0.0",
        model_format="onnx",
        labels=labels,
    )
    install_prepared(prepared, store)
    installer = ArtifactInstaller(store)
    resolved = installer.resolve_active("sample-model")
    installed = resolved.path.parent / "labels.txt"

    assert resolved.ready is True
    assert installed.read_text(encoding="utf-8") == "alpha\nbeta\n"

    installed.write_text("substituted\n", encoding="utf-8")
    assert installer.resolve_active("sample-model").ready is False
