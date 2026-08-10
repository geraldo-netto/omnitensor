from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from omnitensor.conversion import (
    MAX_RANK,
    ConversionError,
    ConversionOutcome,
    PnnxConverter,
    convert_to_ncnn,
    main,
    parse_input_shape,
)


def onnx_file(tmp_path, name="model.onnx"):
    path = tmp_path / name
    path.write_bytes(b"ONNX-ish bytes")
    return path


class FakeConverter:
    """A stand-in for the port, so the policy is testable with no binary."""

    def __init__(self, *, succeeded=True, produce=("model.ncnn.param", "model.ncnn.bin")):
        self._succeeded = succeeded
        self._produce = produce
        self.calls = []

    def convert(self, source, shape, workdir, timeout_seconds):
        self.calls.append((Path(source), tuple(shape), Path(workdir), timeout_seconds))
        for name in self._produce:
            (Path(workdir) / name).write_bytes(b"converted")
        return ConversionOutcome(self._succeeded, "converter said no")


def test_a_model_converts_into_an_ncnn_pair(tmp_path):
    source = onnx_file(tmp_path)
    converter = FakeConverter()

    converted = convert_to_ncnn(source, "[1,3,224,224]", converter=converter)

    assert converted.param.name == "model.ncnn.param"
    assert converted.binary.name == "model.ncnn.bin"
    assert converted.input_shape == (1, 3, 224, 224)


def test_the_traced_input_shape_is_passed_to_the_converter(tmp_path):
    """A converted graph is specialised to the shape it was traced with."""
    source = onnx_file(tmp_path)
    converter = FakeConverter()

    convert_to_ncnn(source, "[1, 3, 224, 224]", converter=converter)

    _source, shape, _workdir, _timeout = converter.calls[0]
    assert shape == (1, 3, 224, 224)


@pytest.mark.parametrize(
    "text",
    ["", "1,3,224", "[]", "[0,3,224,224]", "[1,3,224,99999]", "[1,2,3,4,5,6]", "not a shape", 7],
)
def test_an_unusable_input_shape_is_refused(text):
    with pytest.raises(ConversionError, match="shape-invalid"):
        parse_input_shape(text)


def test_the_rank_bound_is_stated():
    assert parse_input_shape("[1]") == (1,)
    with pytest.raises(ConversionError, match="rank"):
        parse_input_shape("[" + ",".join(["1"] * (MAX_RANK + 1)) + "]")


def test_a_source_the_converter_cannot_read_is_refused(tmp_path):
    with pytest.raises(ConversionError, match="source-unsupported"):
        convert_to_ncnn(tmp_path / "weights.bin", "[1,3,224,224]")
    with pytest.raises(ConversionError, match="source-invalid"):
        convert_to_ncnn(tmp_path / "absent.onnx", "[1,3,224,224]")


def test_a_failing_converter_reports_its_own_output(tmp_path):
    source = onnx_file(tmp_path)
    converter = FakeConverter(succeeded=False, produce=())

    with pytest.raises(ConversionError) as failure:
        convert_to_ncnn(source, "[1,3,224,224]", converter=converter)

    assert failure.value.code == "conversion-failed"
    assert "converter said no" in failure.value.detail


def test_a_converter_that_exits_zero_producing_nothing_is_caught(tmp_path):
    """Checking the files beats trusting the status code."""
    source = onnx_file(tmp_path)
    converter = FakeConverter(produce=("model.ncnn.param",))

    with pytest.raises(ConversionError) as failure:
        convert_to_ncnn(source, "[1,3,224,224]", converter=converter)

    assert failure.value.code == "conversion-incomplete"
    assert "model.ncnn.bin" in failure.value.detail


def test_output_can_be_directed_elsewhere(tmp_path):
    source = onnx_file(tmp_path)
    destination = tmp_path / "build"
    converter = FakeConverter()

    converted = convert_to_ncnn(
        source, "[1,3,224,224]", output_dir=destination, converter=converter
    )

    assert converted.param.parent == destination
    assert destination.is_dir()


def test_the_service_never_imports_the_converter():
    """Conversion is a packaging step: converting on demand would install
    unverified weights while answering a request."""
    import ast
    import inspect

    from omnitensor import service

    tree = ast.parse(inspect.getsource(service))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)

    assert not any("conversion" in name for name in imported)


def test_the_cli_reports_the_pair_and_the_next_command(tmp_path, capsys, monkeypatch):
    import json

    source = onnx_file(tmp_path)
    converter = FakeConverter()
    monkeypatch.setattr(
        "omnitensor.conversion.convert_to_ncnn",
        lambda *a, **k: convert_to_ncnn(source, "[1,3,224,224]", converter=converter),
    )

    code = main([str(source), "--input-shape", "[1,3,224,224]"])

    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["inputShape"] == [1, 3, 224, 224]
    assert "omnitensor-prepare-artifact" in document["next"]


def test_the_cli_reports_a_failure_without_a_traceback(tmp_path, capsys):
    code = main([str(tmp_path / "absent.onnx"), "--input-shape", "[1,3,224,224]"])

    assert code == 1
    assert "conversion failed" in capsys.readouterr().err


def test_a_missing_converter_says_how_to_get_one(tmp_path, monkeypatch):
    """The adapter owns binary discovery, so this is the adapter's failure."""
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(ConversionError) as failure:
        PnnxConverter().convert(onnx_file(tmp_path), (1, 3, 224, 224), tmp_path, 10.0)

    assert failure.value.code == "converter-missing"
    assert "[convert]" in failure.value.detail


@pytest.mark.skipif(shutil.which("pnnx") is None, reason="the [convert] extra is not installed")
def test_a_real_onnx_model_converts_and_the_pair_is_usable(tmp_path):
    """End to end against the real converter when it is available."""
    fixture = Path("/tmp/claude-1000/sq.onnx")
    if not fixture.is_file():
        pytest.skip("no ONNX fixture on this host")
    source = tmp_path / "sq.onnx"
    source.write_bytes(fixture.read_bytes())

    converted = convert_to_ncnn(source, "[1,3,224,224]", timeout_seconds=600)

    assert converted.param.is_file() and converted.binary.is_file()
    assert converted.binary.stat().st_size > converted.param.stat().st_size


def test_the_adapter_reports_a_converter_that_failed(tmp_path, monkeypatch):
    import subprocess as sp

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/true")
    monkeypatch.setattr(
        sp, "run", lambda *a, **k: sp.CompletedProcess(a, 3, "", "bad graph")
    )

    outcome = PnnxConverter().convert(onnx_file(tmp_path), (1, 3, 4, 4), tmp_path, 10.0)

    assert outcome.succeeded is False
    assert "bad graph" in outcome.output


def test_the_adapter_reports_a_converter_that_hangs(tmp_path, monkeypatch):
    import subprocess as sp

    def hang(*args, **kwargs):
        raise sp.TimeoutExpired(cmd="pnnx", timeout=1)

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/true")
    monkeypatch.setattr(sp, "run", hang)

    with pytest.raises(ConversionError, match="conversion-timeout"):
        PnnxConverter().convert(onnx_file(tmp_path), (1, 3, 4, 4), tmp_path, 1.0)


def test_the_adapter_reports_a_converter_it_cannot_execute(tmp_path, monkeypatch):
    import subprocess as sp

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/true")
    monkeypatch.setattr(sp, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("denied")))

    with pytest.raises(ConversionError, match="converter-missing"):
        PnnxConverter().convert(onnx_file(tmp_path), (1, 3, 4, 4), tmp_path, 10.0)


def test_no_output_at_all_still_reports_something(tmp_path, monkeypatch):
    import subprocess as sp

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/true")
    monkeypatch.setattr(sp, "run", lambda *a, **k: sp.CompletedProcess(a, 1, "", ""))
    converter = PnnxConverter()

    outcome = converter.convert(onnx_file(tmp_path), (1, 3, 4, 4), tmp_path, 10.0)

    with pytest.raises(ConversionError, match="no output"):
        convert_to_ncnn(
            onnx_file(tmp_path),
            "[1,3,4,4]",
            converter=type("C", (), {"convert": lambda self, *a: outcome})(),
        )
