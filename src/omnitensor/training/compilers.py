"""Offline compiler ports for accelerator-native model variants.

Portable model bytes are the shareable source of truth.  A provider turns
those bytes into one device-family format, but availability never implies
that the resulting artifact has passed hardware acceptance on this host.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..conversion import (
    DEFAULT_TIMEOUT_SECONDS,
    OvcConverter,
    PnnxConverter,
    convert_to_ncnn,
    convert_to_openvino,
)

TARGET_ORDER = ("tpu", "npu", "gpu")
_EDGE_TPU_COUNT = re.compile(
    r"Number of operations that will run on (?P<lane>Edge TPU|CPU):\s*(?P<count>\d+)",
    re.IGNORECASE,
)


class CompilerError(ValueError):
    """Stable producer-side compilation refusal."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class CompilationRequest:
    """Portable input and the exact contract a native compiler must preserve."""

    source: Path
    source_format: str
    input_shape: tuple[int, ...]
    fully_quantized: bool


@dataclass(frozen=True, slots=True)
class CompiledTarget:
    """One compiler output, before artifact preparation and installation."""

    accelerator: str
    primary: Path
    model_format: str
    fully_quantized: bool
    compiler_report: Path | None = None


@dataclass(frozen=True, slots=True)
class CompilerCapability:
    """Whether the producer tool exists, independent of model compatibility."""

    target: str
    available: bool
    detail: str


class TargetCompiler(Protocol):
    """Port implemented once per accelerator-native format."""

    accelerator: str

    def capability(self) -> CompilerCapability: ...

    def incompatibility(self, source_format: str, fully_quantized: bool) -> str | None: ...

    def compile(self, request: CompilationRequest, output_dir: Path) -> CompiledTarget: ...


class EdgeTpuCommand(Protocol):
    """Subprocess boundary for the separately installed Coral compiler."""

    def compile(
        self, source: Path, output_dir: Path, report_path: Path, timeout_seconds: float
    ) -> Path: ...


class EdgeTpuSubprocess:
    """Invoke ``edgetpu_compiler`` and retain its bounded mapping report."""

    def __init__(self, executable: Path) -> None:
        self._executable = executable

    def compile(
        self, source: Path, output_dir: Path, report_path: Path, timeout_seconds: float
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            str(self._executable),
            "--show_operations",
            "--out_dir",
            str(output_dir),
            str(source.resolve()),
        ]
        try:
            result = subprocess.run(  # noqa: S603 - argv has no shell or caller options
                argv,
                cwd=str(output_dir),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise CompilerError(
                "compilation-timeout",
                f"Edge TPU compilation exceeded {timeout_seconds:g}s",
            ) from error
        except OSError as error:
            raise CompilerError(
                "compiler-missing", f"cannot run edgetpu_compiler: {error}"
            ) from error
        report = "\n".join(part for part in (result.stdout, result.stderr) if part)
        report_path.write_text(report, encoding="utf-8")
        if result.returncode != 0:
            raise CompilerError("compilation-failed", _bounded_tail(report))
        compiled = output_dir / f"{source.stem}_edgetpu.tflite"
        if not compiled.is_file() or compiled.stat().st_size == 0:
            raise CompilerError(
                "compilation-incomplete", "edgetpu_compiler produced no non-empty model"
            )
        return compiled


class NcnnTargetCompiler:
    accelerator = "gpu"

    def __init__(self, executable: Path | None) -> None:
        self.executable = executable

    def capability(self) -> CompilerCapability:
        return _capability(self.accelerator, self.executable, "pnnx")

    def incompatibility(self, source_format: str, fully_quantized: bool) -> str | None:
        del fully_quantized
        return (
            None
            if source_format in {"onnx", "torchscript"}
            else ("GPU compilation needs portable ONNX or TorchScript input")
        )

    def compile(self, request: CompilationRequest, output_dir: Path) -> CompiledTarget:
        _require_compatible(self, request)
        executable = _require_executable(self.capability(), self.executable)
        converted = convert_to_ncnn(
            request.source,
            _shape_text(request.input_shape),
            output_dir=output_dir,
            converter=PnnxConverter(str(executable)),
        )
        return CompiledTarget(self.accelerator, converted.param, "ncnn", False)


class OpenVinoTargetCompiler:
    accelerator = "npu"

    def __init__(self, executable: Path | None) -> None:
        self.executable = executable

    def capability(self) -> CompilerCapability:
        return _capability(self.accelerator, self.executable, "ovc")

    def incompatibility(self, source_format: str, fully_quantized: bool) -> str | None:
        del fully_quantized
        return None if source_format == "onnx" else ("NPU compilation needs portable ONNX input")

    def compile(self, request: CompilationRequest, output_dir: Path) -> CompiledTarget:
        _require_compatible(self, request)
        executable = _require_executable(self.capability(), self.executable)
        converted = convert_to_openvino(
            request.source,
            _shape_text(request.input_shape),
            output_dir=output_dir,
            converter=OvcConverter(str(executable)),
        )
        return CompiledTarget(self.accelerator, converted.xml, "openvino", False)


class EdgeTpuTargetCompiler:
    accelerator = "tpu"

    def __init__(self, executable: Path | None, command: EdgeTpuCommand | None = None) -> None:
        self.executable = executable
        self.command = command

    def capability(self) -> CompilerCapability:
        return _capability(self.accelerator, self.executable, "edgetpu_compiler")

    def incompatibility(self, source_format: str, fully_quantized: bool) -> str | None:
        if source_format != "tflite":
            return "TPU compilation needs a fully-int8 portable TFLite source"
        if not fully_quantized:
            return "TPU compilation needs a fully-int8 TFLite graph"
        return None

    def compile(self, request: CompilationRequest, output_dir: Path) -> CompiledTarget:
        _require_compatible(self, request)
        executable = _require_executable(self.capability(), self.executable)
        report = output_dir / "edgetpu-compiler-report.txt"
        command = self.command or EdgeTpuSubprocess(executable)
        compiled = command.compile(request.source, output_dir, report, DEFAULT_TIMEOUT_SECONDS)
        _require_full_edge_tpu_mapping(report)
        return CompiledTarget(
            self.accelerator,
            compiled,
            "tflite-edgetpu",
            True,
            report,
        )


ToolResolver = Callable[[str], Path | None]


def default_target_compilers(resolve: ToolResolver) -> tuple[TargetCompiler, ...]:
    """Build the fixed target catalog while injecting host-tool discovery."""
    return (
        EdgeTpuTargetCompiler(resolve("edgetpu_compiler")),
        OpenVinoTargetCompiler(resolve("ovc")),
        NcnnTargetCompiler(resolve("pnnx")),
    )


def compiler_catalog(compilers: Sequence[TargetCompiler]) -> dict[str, TargetCompiler]:
    """Validate one unambiguous provider per supported accelerator."""
    catalog: dict[str, TargetCompiler] = {}
    for compiler in compilers:
        if compiler.accelerator not in TARGET_ORDER:
            raise CompilerError(
                "compiler-invalid", f"unsupported compiler target: {compiler.accelerator}"
            )
        if compiler.accelerator in catalog:
            raise CompilerError(
                "compiler-invalid", f"duplicate compiler target: {compiler.accelerator}"
            )
        catalog[compiler.accelerator] = compiler
    return catalog


def compatible_available_targets(
    source_format: str,
    fully_quantized: bool,
    compilers: Sequence[TargetCompiler],
) -> tuple[str, ...]:
    catalog = compiler_catalog(compilers)
    return tuple(
        target
        for target in TARGET_ORDER
        if target in catalog
        and catalog[target].incompatibility(source_format, fully_quantized) is None
        and catalog[target].capability().available
    )


def _capability(target: str, executable: Path | None, name: str) -> CompilerCapability:
    detail = f"{name} is available" if executable is not None else f"{name} is not installed"
    return CompilerCapability(target, executable is not None, detail)


def _require_compatible(compiler: TargetCompiler, request: CompilationRequest) -> None:
    detail = compiler.incompatibility(request.source_format, request.fully_quantized)
    if detail is not None:
        raise CompilerError("source-incompatible", detail)


def _require_executable(capability: CompilerCapability, executable: Path | None) -> Path:
    if not capability.available or executable is None:
        raise CompilerError("compiler-missing", capability.detail)
    return executable


def _shape_text(shape: Sequence[int]) -> str:
    return f"[{','.join(str(dimension) for dimension in shape)}]"


def _require_full_edge_tpu_mapping(report_path: Path) -> None:
    try:
        report = report_path.read_text(encoding="utf-8")
    except OSError as error:
        raise CompilerError(
            "compiler-report-invalid", f"cannot read compiler report: {error}"
        ) from error
    counts = {
        match.group("lane").lower(): int(match.group("count"))
        for match in _EDGE_TPU_COUNT.finditer(report)
    }
    if counts.get("edge tpu", 0) < 1 or "cpu" not in counts:
        raise CompilerError(
            "compiler-report-invalid", "compiler report has no complete TPU/CPU mapping counts"
        )
    if counts["cpu"] != 0:
        raise CompilerError(
            "mapping-incomplete", f"Edge TPU compiler mapped {counts['cpu']} operations to CPU"
        )


def _bounded_tail(output: str, limit: int = 400) -> str:
    cleaned = output.strip()
    return cleaned[-limit:] if cleaned else "compiler returned no diagnostics"
