"""Convert a pretrained model into an accelerator-native artifact.

Most pretrained models are published as ONNX or TorchScript, and the only GPU
runtime available on an AMD host is ncnn over Vulkan — onnxruntime's GPU
providers are CUDA and ROCm, and the CPU-only wheel is refused by design.  So
without a conversion step, "download a pretrained model" and "run it on this
machine" are unrelated activities.  This closes that gap.

Conversion is a packaging step and deliberately not a runtime one.  It needs a
toolchain measured in tens of megabytes, it takes seconds to minutes, and its
output must be digested and declared before anything will dispatch it — a
service that converted on demand would be installing unverified weights while
answering a request.  The extra is therefore separate (``[convert]``) and this
module is a command, never imported by the service.

Converters are invoked as subprocesses rather than in-process.  Native model
conversion can abort on a malformed graph, and taking the caller down with it
would lose the diagnosis along with the process.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

SUPPORTED_SOURCES = (".onnx", ".pt")
OPENVINO_SOURCES = (".onnx",)
DEFAULT_TIMEOUT_SECONDS = 900.0
MAX_DIMENSION = 65_536
MAX_RANK = 5
_SHAPE = re.compile(r"^\[(\d+(?:,\d+)*)\]$")


class ConversionError(ValueError):
    """Stable conversion failure, safe to show to whoever ran the command."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class ConversionOutcome:
    """What a converter reported, in terms this module can reason about."""

    succeeded: bool
    output: str


class ModelConverter(Protocol):
    """Turn a source model into a native pair inside ``workdir``.

    The port owns everything about *how* conversion happens — which binary,
    where it lives, how it is invoked — so the policy below can be exercised
    without one installed, and a different converter can be substituted
    without touching the rules about shapes and outputs.
    """

    def convert(
        self,
        source: Path,
        shape: Sequence[int],
        workdir: Path,
        timeout_seconds: float,
    ) -> ConversionOutcome: ...


@dataclass(frozen=True, slots=True)
class ConvertedModel:
    """The ncnn pair a conversion produced."""

    param: Path
    binary: Path
    source: Path
    input_shape: tuple[int, ...]

    def document(self) -> dict:
        return {
            "param": str(self.param),
            "bin": str(self.binary),
            "source": str(self.source),
            "inputShape": list(self.input_shape),
            "paramBytes": self.param.stat().st_size,
            "binBytes": self.binary.stat().st_size,
        }


@dataclass(frozen=True, slots=True)
class ConvertedOpenVinoModel:
    """The OpenVINO IR pair a conversion produced."""

    xml: Path
    binary: Path
    source: Path
    input_shape: tuple[int, ...]

    def document(self) -> dict:
        return {
            "xml": str(self.xml),
            "bin": str(self.binary),
            "source": str(self.source),
            "inputShape": list(self.input_shape),
            "xmlBytes": self.xml.stat().st_size,
            "binBytes": self.binary.stat().st_size,
        }


def parse_input_shape(text: str) -> tuple[int, ...]:
    """Parse ``[1,3,224,224]`` into a bounded, positive shape.

    Required rather than inferred: a converted graph is specialised to the
    shape it was traced with, and guessing one silently produces a model that
    runs and is wrong for the caller's input.
    """
    if not isinstance(text, str):
        raise ConversionError("shape-invalid", "input shape must be text like [1,3,224,224]")
    match = _SHAPE.match(text.replace(" ", ""))
    if not match:
        raise ConversionError("shape-invalid", f"cannot read an input shape from {text!r}")
    dimensions = tuple(int(part) for part in match.group(1).split(","))
    if not 1 <= len(dimensions) <= MAX_RANK:
        raise ConversionError("shape-invalid", f"rank must be between 1 and {MAX_RANK}")
    for dimension in dimensions:
        if not 1 <= dimension <= MAX_DIMENSION:
            raise ConversionError("shape-invalid", f"dimension out of range: {dimension}")
    return dimensions


def convert_to_ncnn(
    source: Path | str,
    input_shape: str,
    *,
    output_dir: Path | str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    converter: ModelConverter | None = None,
) -> ConvertedModel:
    """Convert ``source`` to an ncnn param/bin pair beside it."""
    path = Path(source)
    if path.suffix.lower() not in SUPPORTED_SOURCES:
        raise ConversionError(
            "source-unsupported",
            f"{path.suffix or 'that file'} is not one of {', '.join(SUPPORTED_SOURCES)}",
        )
    if not path.is_file():
        raise ConversionError("source-invalid", f"not a regular file: {path}")
    shape = parse_input_shape(input_shape)
    destination = Path(output_dir) if output_dir else path.parent
    destination.mkdir(parents=True, exist_ok=True)

    workdir = destination if output_dir else path.parent
    outcome = (converter or PnnxConverter()).convert(path, shape, workdir, timeout_seconds)
    if not outcome.succeeded:
        raise ConversionError("conversion-failed", f"the converter failed: {_tail(outcome.output)}")
    stem = path.stem
    param = workdir / f"{stem}.ncnn.param"
    binary = workdir / f"{stem}.ncnn.bin"
    missing = [item.name for item in (param, binary) if not item.is_file()]
    if missing:
        # The converter can exit zero having produced nothing useful, so the
        # files are checked rather than the status code trusted.
        raise ConversionError(
            "conversion-incomplete",
            f"the converter produced no {', '.join(missing)}",
        )
    return ConvertedModel(param, binary, path, shape)


def convert_to_openvino(
    source: Path | str,
    input_shape: str,
    *,
    output_dir: Path | str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    converter: ModelConverter | None = None,
) -> ConvertedOpenVinoModel:
    """Convert an ONNX model to an OpenVINO IR XML/bin pair."""
    path = Path(source)
    if path.suffix.lower() not in OPENVINO_SOURCES:
        raise ConversionError(
            "source-unsupported",
            f"{path.suffix or 'that file'} is not one of {', '.join(OPENVINO_SOURCES)}",
        )
    if not path.is_file():
        raise ConversionError("source-invalid", f"not a regular file: {path}")
    shape = parse_input_shape(input_shape)
    destination = Path(output_dir) if output_dir else path.parent
    destination.mkdir(parents=True, exist_ok=True)

    outcome = (converter or OvcConverter()).convert(path, shape, destination, timeout_seconds)
    if not outcome.succeeded:
        raise ConversionError("conversion-failed", f"the converter failed: {_tail(outcome.output)}")
    xml = destination / f"{path.stem}.xml"
    binary = destination / f"{path.stem}.bin"
    missing = [item.name for item in (xml, binary) if not item.is_file()]
    if missing:
        raise ConversionError(
            "conversion-incomplete",
            f"the converter produced no {', '.join(missing)}",
        )
    return ConvertedOpenVinoModel(xml, binary, path, shape)


class PnnxConverter:
    """The pnnx adapter: the only place that knows a binary is involved.

    Invoked as a subprocess rather than in-process because pnnx is native code
    that can abort on a malformed graph, and taking the caller down with it
    would lose the diagnosis along with the process.
    """

    def __init__(self, executable: str = "pnnx") -> None:
        self._executable = executable

    def convert(
        self,
        source: Path,
        shape: Sequence[int],
        workdir: Path,
        timeout_seconds: float,
    ) -> ConversionOutcome:
        found = shutil.which(self._executable)
        if found is None:
            raise ConversionError(
                "converter-missing",
                f"{self._executable} is not installed; install the [convert] extra"
                " to convert models",
            )
        argv = [
            found,
            str(source.resolve()),
            f"inputshape=[{','.join(str(d) for d in shape)}]",
            f"pnnxparam={source.stem}.pnnx.param",
            f"pnnxbin={source.stem}.pnnx.bin",
            f"pnnxpy={source.stem}_pnnx.py",
            f"pnnxonnx={source.stem}.pnnx.onnx",
            f"ncnnparam={source.stem}.ncnn.param",
            f"ncnnbin={source.stem}.ncnn.bin",
            f"ncnnpy={source.stem}_ncnn.py",
        ]
        return _run_converter(argv, workdir, timeout_seconds)


class OvcConverter:
    """Adapter for OpenVINO's offline Model Converter command."""

    def __init__(self, executable: str = "ovc") -> None:
        self._executable = executable

    def convert(
        self,
        source: Path,
        shape: Sequence[int],
        workdir: Path,
        timeout_seconds: float,
    ) -> ConversionOutcome:
        found = shutil.which(self._executable)
        if found is None:
            raise ConversionError(
                "converter-missing",
                f"{self._executable} is not installed; install the [convert-npu] extra"
                " to convert models",
            )
        output = workdir / f"{source.stem}.xml"
        argv = [
            found,
            str(source.resolve()),
            "--input",
            f"[{','.join(str(dimension) for dimension in shape)}]",
            "--output_model",
            str(output),
        ]
        return _run_converter(argv, workdir, timeout_seconds)


def _run_converter(argv: list[str], workdir: Path, timeout_seconds: float) -> ConversionOutcome:
    try:
        result = subprocess.run(  # noqa: S603 - argv is built here, never from input
            argv,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ConversionError(
            "conversion-timeout",
            f"the converter did not finish within {timeout_seconds:g}s",
        ) from error
    except OSError as error:
        raise ConversionError(
            "converter-missing", f"could not run the converter: {error}"
        ) from error
    return ConversionOutcome(result.returncode == 0, result.stderr or result.stdout or "")


def _tail(output: str | None, limit: int = 400) -> str:
    if not output:
        return "no output"
    return output.strip()[-limit:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omnitensor-convert-model",
        description="Convert a model into an artifact for an accelerator lane",
    )
    parser.add_argument("source", help="the .onnx or .pt model to convert")
    parser.add_argument(
        "--format",
        choices=("ncnn", "openvino"),
        default="ncnn",
        dest="model_format",
    )
    parser.add_argument(
        "--input-shape",
        required=True,
        help="the traced input shape, e.g. [1,3,224,224]",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    arguments = parser.parse_args(argv)

    try:
        convert = convert_to_openvino if arguments.model_format == "openvino" else convert_to_ncnn
        converted = convert(
            arguments.source,
            arguments.input_shape,
            output_dir=arguments.output_dir,
            timeout_seconds=arguments.timeout,
        )
    except ConversionError as error:
        print(f"conversion failed: {error}", file=sys.stderr)
        return 1
    document = converted.document()
    primary = converted.xml if isinstance(converted, ConvertedOpenVinoModel) else converted.param
    document["next"] = (
        "omnitensor-prepare-artifact "
        f"{primary} --id <artifact-id> --version <version> --format {arguments.model_format} "
        "--install-root ~/.local/share/omnitensor/artifacts"
    )
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
