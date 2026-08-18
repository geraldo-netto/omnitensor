from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.training.compilers import (
    CompilationRequest,
    CompilerCapability,
    CompilerError,
    EdgeTpuSubprocess,
    EdgeTpuTargetCompiler,
    NcnnTargetCompiler,
    OpenVinoTargetCompiler,
    _bounded_tail,
    compatible_available_targets,
    compiler_catalog,
    default_target_compilers,
)
from omnitensor.training.contracts import TrainingError
from omnitensor.training.installation import available_targets


def request(tmp_path, **changes):
    values = {
        "source": tmp_path / "model.onnx",
        "source_format": "onnx",
        "input_shape": (1, 6),
        "fully_quantized": False,
    }
    values.update(changes)
    values["source"].write_bytes(b"portable")
    return CompilationRequest(**values)


class FakeCompiler:
    def __init__(self, target, *, available=True, incompatible=None):
        self.accelerator = target
        self._available = available
        self._incompatible = incompatible
        self.observed = []

    def capability(self):
        return CompilerCapability(self.accelerator, self._available, "test capability")

    def incompatibility(self, source_format, fully_quantized):
        self.observed.append((source_format, fully_quantized))
        return self._incompatible


def test_default_catalog_resolves_every_accelerator_once():
    observed = []

    def resolve(name):
        observed.append(name)
        return Path("/tools") / name

    compilers = default_target_compilers(resolve)

    assert [compiler.accelerator for compiler in compilers] == ["gpu", "npu", "tpu"]
    assert observed == ["pnnx", "ovc", "edgetpu_compiler"]
    assert [compiler.capability().available for compiler in compilers] == [True, True, True]


@pytest.mark.parametrize(
    ("compilers", "detail"),
    [
        ((FakeCompiler("cpu"),), "unsupported compiler target: cpu"),
        ((FakeCompiler("gpu"), FakeCompiler("gpu")), "duplicate compiler target: gpu"),
    ],
)
def test_catalog_refuses_ambiguous_or_unknown_providers(compilers, detail):
    with pytest.raises(CompilerError) as captured:
        compiler_catalog(compilers)

    assert captured.value.code == "compiler-invalid"
    assert captured.value.detail == detail


def test_available_targets_require_both_tool_and_source_compatibility(tmp_path):
    compilers = (
        FakeCompiler("tpu", incompatible="wrong source"),
        FakeCompiler("npu", available=False),
        FakeCompiler("gpu"),
    )

    assert compatible_available_targets("onnx", False, compilers) == ("gpu",)


def test_public_available_targets_translates_an_invalid_provider_catalog():
    with pytest.raises(TrainingError) as captured:
        available_targets(compilers=(FakeCompiler("gpu"), FakeCompiler("gpu")))

    assert captured.value.code == "compiler-invalid"
    assert captured.value.detail == "duplicate compiler target: gpu"


def test_public_available_targets_passes_exact_source_capability():
    compiler = FakeCompiler("tpu")

    assert available_targets(
        source_format="tflite", fully_quantized=True, compilers=(compiler,)
    ) == ("tpu",)
    assert compiler.observed == [("tflite", True)]


def test_public_available_targets_injects_the_real_tool_resolver(monkeypatch):
    observed = []

    def factory(resolver):
        observed.append(resolver)
        return (FakeCompiler("gpu"),)

    monkeypatch.setattr("omnitensor.training.installation.default_target_compilers", factory)

    assert available_targets() == ("gpu",)
    assert len(observed) == 1
    assert callable(observed[0])


@pytest.mark.parametrize(
    ("compiler", "source_format", "fully_quantized", "detail"),
    [
        (
            EdgeTpuTargetCompiler(Path("/tools/edgetpu_compiler")),
            "onnx",
            True,
            "TPU compilation needs a fully-int8 portable TFLite source",
        ),
        (
            EdgeTpuTargetCompiler(Path("/tools/edgetpu_compiler")),
            "tflite",
            False,
            "TPU compilation needs a fully-int8 TFLite graph",
        ),
        (
            OpenVinoTargetCompiler(Path("/tools/ovc")),
            "tflite",
            False,
            "NPU compilation needs portable ONNX input",
        ),
        (
            NcnnTargetCompiler(Path("/tools/pnnx")),
            "tflite",
            False,
            "GPU compilation needs portable ONNX or TorchScript input",
        ),
    ],
)
def test_providers_refuse_incompatible_portable_sources(
    tmp_path, compiler, source_format, fully_quantized, detail
):
    model = request(
        tmp_path,
        source=tmp_path / f"model.{source_format}",
        source_format=source_format,
        fully_quantized=fully_quantized,
    )

    with pytest.raises(CompilerError) as captured:
        compiler.compile(model, tmp_path / "out")

    assert captured.value.code == "source-incompatible"
    assert captured.value.detail == detail


@pytest.mark.parametrize(
    ("compiler", "tool"),
    [
        (EdgeTpuTargetCompiler(None), "edgetpu_compiler is not installed"),
        (OpenVinoTargetCompiler(None), "ovc is not installed"),
        (NcnnTargetCompiler(None), "pnnx is not installed"),
    ],
)
def test_compatible_provider_refuses_a_missing_compiler(tmp_path, compiler, tool):
    source_format = "tflite" if compiler.accelerator == "tpu" else "onnx"
    model = request(
        tmp_path,
        source=tmp_path / f"model.{source_format}",
        source_format=source_format,
        fully_quantized=compiler.accelerator == "tpu",
    )

    with pytest.raises(CompilerError) as captured:
        compiler.compile(model, tmp_path / "out")

    assert captured.value.code == "compiler-missing"
    assert captured.value.detail == tool


def test_ncnn_provider_preserves_shape_and_compiled_contract(tmp_path, monkeypatch):
    model = request(tmp_path)
    observed = []
    primary = tmp_path / "out/model.ncnn.param"

    def convert(source, shape, **kwargs):
        observed.append((source, shape, kwargs))
        return SimpleNamespace(param=primary)

    monkeypatch.setattr("omnitensor.training.compilers.convert_to_ncnn", convert)
    compiler = NcnnTargetCompiler(Path("/tools/pnnx"))

    compiled = compiler.compile(model, tmp_path / "out")

    assert compiled.primary == primary
    assert compiled.model_format == "ncnn"
    assert compiled.fully_quantized is False
    assert observed[0][0:2] == (model.source, "[1,6]")
    assert observed[0][2]["output_dir"] == tmp_path / "out"
    assert observed[0][2]["converter"]._executable == "/tools/pnnx"


def test_ncnn_provider_declares_both_portable_source_formats():
    compiler = NcnnTargetCompiler(Path("/tools/pnnx"))

    assert compiler.incompatibility("onnx", False) is None
    assert compiler.incompatibility("torchscript", False) is None
    assert "ONNX or TorchScript" in compiler.incompatibility("tflite", True)


def test_openvino_provider_preserves_shape_and_compiled_contract(tmp_path, monkeypatch):
    model = request(tmp_path)
    observed = []
    primary = tmp_path / "out/model.xml"

    def convert(source, shape, **kwargs):
        observed.append((source, shape, kwargs))
        return SimpleNamespace(xml=primary)

    monkeypatch.setattr("omnitensor.training.compilers.convert_to_openvino", convert)
    compiler = OpenVinoTargetCompiler(Path("/tools/ovc"))

    compiled = compiler.compile(model, tmp_path / "out")

    assert compiled.primary == primary
    assert compiled.model_format == "openvino"
    assert compiled.fully_quantized is False
    assert observed[0][0:2] == (model.source, "[1,6]")
    assert observed[0][2]["output_dir"] == tmp_path / "out"
    assert observed[0][2]["converter"]._executable == "/tools/ovc"


class FakeEdgeCommand:
    def __init__(self, report, artifact=b"compiled"):
        self.report = report
        self.artifact = artifact
        self.observed = []

    def compile(self, source, output_dir, report_path, timeout_seconds):
        self.observed.append((source, output_dir, report_path, timeout_seconds))
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(self.report, encoding="utf-8")
        compiled = output_dir / f"{source.stem}_edgetpu.tflite"
        compiled.write_bytes(self.artifact)
        return compiled


def test_edge_tpu_provider_requires_and_retains_complete_mapping_report(tmp_path):
    source = tmp_path / "model.tflite"
    model = request(
        tmp_path,
        source=source,
        source_format="tflite",
        fully_quantized=True,
    )
    command = FakeEdgeCommand(
        "Number of operations that will run on Edge TPU: 17\n"
        "Number of operations that will run on CPU: 0\n"
    )
    compiler = EdgeTpuTargetCompiler(Path("/tools/edgetpu_compiler"), command)

    compiled = compiler.compile(model, tmp_path / "out")

    assert compiled.model_format == "tflite-edgetpu"
    assert compiled.fully_quantized is True
    assert compiled.compiler_report == tmp_path / "out/edgetpu-compiler-report.txt"
    assert compiled.primary.read_bytes() == b"compiled"
    assert command.observed[0][0:3] == (
        source,
        tmp_path / "out",
        tmp_path / "out/edgetpu-compiler-report.txt",
    )


@pytest.mark.parametrize(
    ("report", "code", "detail"),
    [
        ("unstructured output", "compiler-report-invalid", "no complete TPU/CPU"),
        (
            "Number of operations that will run on Edge TPU: 2\n"
            "Number of operations that will run on CPU: 1\n",
            "mapping-incomplete",
            "mapped 1 operations to CPU",
        ),
        (
            "Number of operations that will run on Edge TPU: 0\n"
            "Number of operations that will run on CPU: 0\n",
            "compiler-report-invalid",
            "no complete TPU/CPU",
        ),
    ],
)
def test_edge_tpu_provider_rejects_unproven_full_mapping(tmp_path, report, code, detail):
    model = request(
        tmp_path,
        source=tmp_path / "model.tflite",
        source_format="tflite",
        fully_quantized=True,
    )
    compiler = EdgeTpuTargetCompiler(Path("/tools/edgetpu_compiler"), FakeEdgeCommand(report))

    with pytest.raises(CompilerError) as captured:
        compiler.compile(model, tmp_path / "out")

    assert captured.value.code == code
    assert detail in captured.value.detail


def test_edge_tpu_provider_refuses_a_missing_report(tmp_path):
    model = request(
        tmp_path,
        source=tmp_path / "model.tflite",
        source_format="tflite",
        fully_quantized=True,
    )

    class MissingReportCommand:
        def compile(self, source, output_dir, _report_path, _timeout_seconds):
            output_dir.mkdir(parents=True)
            compiled = output_dir / f"{source.stem}_edgetpu.tflite"
            compiled.write_bytes(b"compiled")
            return compiled

    compiler = EdgeTpuTargetCompiler(Path("/tools/edgetpu_compiler"), MissingReportCommand())

    with pytest.raises(CompilerError) as captured:
        compiler.compile(model, tmp_path / "out")

    assert captured.value.code == "compiler-report-invalid"
    assert "cannot read compiler report" in captured.value.detail


@given(
    tpu_operations=st.integers(min_value=1, max_value=100_000),
    cpu_operations=st.integers(min_value=0, max_value=100_000),
)
def test_edge_mapping_gate_accepts_exactly_zero_cpu_operations(tpu_operations, cpu_operations):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model = request(
            root,
            source=root / "model.tflite",
            source_format="tflite",
            fully_quantized=True,
        )
        report = (
            f"Number of operations that will run on Edge TPU: {tpu_operations}\n"
            f"Number of operations that will run on CPU: {cpu_operations}\n"
        )
        compiler = EdgeTpuTargetCompiler(Path("/tools/edgetpu_compiler"), FakeEdgeCommand(report))

        if cpu_operations == 0:
            assert compiler.compile(model, root / "out").fully_quantized is True
        else:
            with pytest.raises(CompilerError, match="mapping-incomplete"):
                compiler.compile(model, root / "out")


def test_edge_tpu_subprocess_uses_fixed_cli_and_writes_combined_report(tmp_path, monkeypatch):
    source = tmp_path / "portable.tflite"
    source.write_bytes(b"portable")
    observed = []

    def run(argv, **kwargs):
        observed.append((argv, kwargs))
        output = tmp_path / "out/portable_edgetpu.tflite"
        output.write_bytes(b"compiled")
        return SimpleNamespace(returncode=0, stdout="mapping", stderr="warning")

    monkeypatch.setattr("omnitensor.training.compilers.subprocess.run", run)
    report = tmp_path / "out/report.txt"

    compiled = EdgeTpuSubprocess(Path("/tools/edgetpu_compiler")).compile(
        source, tmp_path / "out", report, 12.5
    )

    assert compiled == tmp_path / "out/portable_edgetpu.tflite"
    assert report.read_text() == "mapping\nwarning"
    assert observed == [
        (
            [
                "/tools/edgetpu_compiler",
                "--show_operations",
                "--out_dir",
                str(tmp_path / "out"),
                str(source.resolve()),
            ],
            {
                "cwd": str(tmp_path / "out"),
                "capture_output": True,
                "text": True,
                "timeout": 12.5,
                "check": False,
            },
        )
    ]


@pytest.mark.parametrize(
    ("effect", "code", "detail"),
    [
        (
            SimpleNamespace(returncode=2, stdout="bad graph", stderr=""),
            "compilation-failed",
            "bad graph",
        ),
        (
            SimpleNamespace(returncode=0, stdout="ok", stderr=""),
            "compilation-incomplete",
            "no non-empty model",
        ),
        (subprocess.TimeoutExpired("compiler", 9), "compilation-timeout", "exceeded 9s"),
        (OSError("not found"), "compiler-missing", "not found"),
    ],
)
def test_edge_tpu_subprocess_has_stable_failures(tmp_path, monkeypatch, effect, code, detail):
    source = tmp_path / "model.tflite"
    source.write_bytes(b"portable")

    def run(*_args, **_kwargs):
        if isinstance(effect, BaseException):
            raise effect
        return effect

    monkeypatch.setattr("omnitensor.training.compilers.subprocess.run", run)

    with pytest.raises(CompilerError) as captured:
        EdgeTpuSubprocess(Path("/tools/edgetpu_compiler")).compile(
            source, tmp_path / "out", tmp_path / "out/report.txt", 9
        )

    assert captured.value.code == code
    assert detail in captured.value.detail


def test_compiler_diagnostic_tail_is_exactly_bounded_and_never_empty():
    assert _bounded_tail(" \n ") == "compiler returned no diagnostics"
    assert _bounded_tail("prefix-" + "x" * 400) == "x" * 400
