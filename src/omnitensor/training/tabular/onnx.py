"""Lazy ONNX construction and atomic publication for tabular models."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ..contracts import TrainingError
from .model import NormalizedLogisticModel


def load_onnx_dependency() -> tuple[object, object, object]:
    """Load the producer-only ONNX dependency or expose the stable refusal."""
    try:
        import onnx  # noqa: PLC0415 - optional producer dependency
        from onnx import TensorProto, helper  # noqa: PLC0415
    except ImportError as error:
        raise TrainingError(
            "exporter-missing", "onnx is not installed; install the [train] extra"
        ) from error
    return onnx, TensorProto, helper


def checked_model(graph: object, onnx: object, helper: object) -> object:
    """Create and validate the fixed opset/IR portable model envelope."""
    portable = helper.make_model(
        graph,
        producer_name="omnitensor",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    portable.ir_version = min(portable.ir_version, 8)
    try:
        onnx.checker.check_model(portable)
    except Exception as error:  # noqa: BLE001 - exporter diagnostics vary
        raise TrainingError("export-failed", f"ONNX validation failed: {error}") from error
    return portable


def normalized_logistic_model(
    model: NormalizedLogisticModel,
    *,
    graph_name: str,
    intercept_name: str,
    logit_name: str,
    output_name: str,
    onnx: object,
    tensor_proto: object,
    helper: object,
    input_width: int | None = None,
) -> object:
    """Build the exact normalized multi-output logistic ONNX topology."""
    width = len(model.means) if input_width is None else input_width
    outputs = len(model.intercepts)
    graph = helper.make_graph(
        [
            helper.make_node("Sub", ["input", "means"], ["centered"]),
            helper.make_node("Div", ["centered", "scales"], ["normalized"]),
            helper.make_node("MatMul", ["normalized", "weights"], ["linear"]),
            helper.make_node("Add", ["linear", intercept_name], [logit_name]),
            helper.make_node("Sigmoid", [logit_name], [output_name]),
        ],
        graph_name,
        [helper.make_tensor_value_info("input", tensor_proto.FLOAT, [1, width])],
        [helper.make_tensor_value_info(output_name, tensor_proto.FLOAT, [1, outputs])],
        [
            helper.make_tensor("means", tensor_proto.FLOAT, [width], model.means),
            helper.make_tensor("scales", tensor_proto.FLOAT, [width], model.scales),
            helper.make_tensor(
                "weights",
                tensor_proto.FLOAT,
                [width, outputs],
                [value for row in model.weights for value in row],
            ),
            helper.make_tensor(intercept_name, tensor_proto.FLOAT, [outputs], model.intercepts),
        ],
    )
    return checked_model(graph, onnx, helper)


def save_onnx_atomic(
    portable: object,
    destination: Path,
    onnx: object,
    *,
    prefix: str,
) -> None:
    """Publish one checked ONNX model through an exact-prefix sibling stage."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=prefix, dir=destination.parent)
    os.close(handle)
    try:
        onnx.save_model(portable, temporary)
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
