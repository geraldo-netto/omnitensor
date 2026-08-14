from __future__ import annotations

from types import SimpleNamespace

from conftest import sample_manifest

from omnitensor.executors.base import (
    DEVICE_ABSENT,
    FORMAT_UNSUPPORTED,
    RUNTIME_MISSING,
    RUNTIME_UNUSABLE,
    Availability,
    most_actionable,
)
from omnitensor.executors.gpu import CompositeGpuExecutor, GpuExecutor
from omnitensor.registry import Workload, validate_workload_document
from omnitensor.scheduler import (
    NO_PREFERENCE,
    BackendChoice,
    runnable_model,
    select_backend,
)


class _Lane:
    def __init__(
        self,
        backend: str,
        formats: set[str],
        availability: Availability,
    ) -> None:
        self.backend = backend
        self.model_formats = frozenset(formats)
        self._availability = availability
        self.availability_calls = 0

    def availability(self) -> Availability:
        self.availability_calls += 1
        return self._availability

    def run(self, _model_path: str, _inputs: list) -> None:
        raise AssertionError("routing tests must not execute a model")


def _model(model_format: str) -> dict:
    return {"id": f"model-{model_format}", "format": model_format}


def _workload(*models: dict, preference: list[str] | None = None) -> Workload:
    requirements = {"models": list(models)}
    if preference is not None:
        requirements["acceleratorPreference"] = preference
    return Workload("routing-regression", {"requirements": requirements})


def test_default_preference_chooses_tpu_then_npu_independent_of_model_order():
    workload = _workload(_model("ncnn"), _model("openvino"), _model("tflite-edgetpu"))
    tpu = _Lane("tpu", {"tflite-edgetpu"}, Availability(True))
    npu = _Lane("npu", {"openvino"}, Availability(True))
    gpu = _Lane("gpu", {"ncnn"}, Availability(True))
    executors = {"gpu": gpu, "npu": npu, "tpu": tpu}

    assert workload.preference == ("tpu", "npu", "gpu")
    assert select_backend(workload, executors) == BackendChoice("tpu", "", "")
    assert runnable_model(workload, "tpu", executors)["format"] == "tflite-edgetpu"
    assert (tpu.availability_calls, npu.availability_calls, gpu.availability_calls) == (1, 0, 0)

    tpu._availability = Availability(False, "No Coral device", DEVICE_ABSENT)
    assert select_backend(workload, executors) == BackendChoice("npu", "", "")
    assert runnable_model(workload, "npu", executors)["format"] == "openvino"
    assert (tpu.availability_calls, npu.availability_calls, gpu.availability_calls) == (2, 1, 0)


def test_refusal_order_actionability_and_tie_breaks_are_stable():
    workload = _workload(_model("ncnn"), _model("openvino"), _model("tflite-edgetpu"))
    executors = {
        "gpu": _Lane(
            "gpu",
            {"ncnn"},
            Availability(False, "ONNX Runtime is not installed", RUNTIME_MISSING),
        ),
        "npu": _Lane(
            "npu",
            {"openvino"},
            Availability(False, "OpenVINO has no NPU plugin", RUNTIME_UNUSABLE),
        ),
        "tpu": _Lane(
            "tpu",
            {"tflite-edgetpu"},
            Availability(False, "No Coral device", DEVICE_ABSENT),
        ),
    }

    assert select_backend(workload, executors) == BackendChoice(
        None,
        "tpu: No Coral device; npu: OpenVINO has no NPU plugin; "
        "gpu: ONNX Runtime is not installed",
        RUNTIME_MISSING,
    )

    ordered = (RUNTIME_MISSING, RUNTIME_UNUSABLE, FORMAT_UNSUPPORTED, DEVICE_ABSENT)
    assert tuple(most_actionable(ordered[index:]) for index in range(len(ordered))) == ordered
    assert most_actionable(("zeta-remedy", "alpha-remedy")) == "alpha-remedy"
    assert select_backend(SimpleNamespace(preference=()), {}) == BackendChoice(
        None, "No backend in preference list", NO_PREFERENCE
    )


def test_mixed_gpu_availability_is_scoped_to_the_declared_model_format():
    ncnn = _Lane(
        "gpu", {"ncnn"}, Availability(False, "ncnn is not installed", RUNTIME_MISSING)
    )
    unrelated_onnx = _Lane("gpu", {"onnx"}, Availability(True))
    gpu = CompositeGpuExecutor([ncnn, unrelated_onnx])
    npu = _Lane("npu", {"openvino"}, Availability(True))
    workload = _workload(
        _model("openvino"),
        _model("ncnn"),
        preference=["gpu", "npu"],
    )
    executors = {"gpu": gpu, "npu": npu}

    assert select_backend(workload, executors) == BackendChoice("npu", "", "")
    assert runnable_model(workload, "npu", executors)["format"] == "openvino"
    assert (ncnn.availability_calls, unrelated_onnx.availability_calls) == (1, 0)

    ncnn._availability = Availability(True)
    assert select_backend(workload, executors) == BackendChoice("gpu", "", "")
    assert runnable_model(workload, "gpu", executors)["format"] == "ncnn"
    assert (ncnn.availability_calls, unrelated_onnx.availability_calls) == (2, 0)


def test_multi_model_routing_refuses_cpu_only_onnx_and_ignores_cpu_backend():
    class _CpuOnlyOrt:
        @staticmethod
        def get_available_providers() -> list[str]:
            return ["CPUExecutionProvider"]

    gpu = GpuExecutor(device_present=True, runtime=_CpuOnlyOrt())
    npu = _Lane("npu", {"openvino"}, Availability(True))
    cpu = _Lane("cpu", {"onnx", "openvino"}, Availability(True))
    workload = _workload(
        _model("onnx"),
        _model("openvino"),
        preference=["gpu", "npu"],
    )

    executors = {"gpu": gpu, "npu": npu, "cpu": cpu}
    assert select_backend(workload, executors) == BackendChoice("npu", "", "")
    assert runnable_model(workload, "npu", executors)["format"] == "openvino"
    assert cpu.availability_calls == 0

    assert select_backend(workload, {"gpu": gpu, "cpu": cpu}) == BackendChoice(
        None,
        "gpu: onnxruntime has no CUDA or ROCm execution provider; "
        "the CPU provider is not used; npu: no executor",
        RUNTIME_UNUSABLE,
    )
    assert cpu.availability_calls == 0


def test_manifest_schema_does_not_admit_a_cpu_preference():
    manifest = sample_manifest(acceleratorPreference=["tpu", "cpu"])

    assert validate_workload_document(manifest) == [
        "requirements/acceleratorPreference/1: 'cpu' is not one of ['tpu', 'npu', 'gpu']"
    ]
