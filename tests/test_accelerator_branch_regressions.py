from __future__ import annotations

from omnitensor.discovery import Device
from omnitensor.execution import build_executors
from omnitensor.executors.base import (
    DEVICE_ABSENT,
    RUNTIME_MISSING,
    Availability,
    InferenceResult,
    most_actionable,
)
from omnitensor.executors.gpu import CompositeGpuExecutor, GpuExecutor
from omnitensor.registry import Workload
from omnitensor.scheduler import NO_EXECUTOR, BackendChoice, runnable_model, select_backend


class _Lane:
    def __init__(
        self,
        backend: str,
        formats: set[str],
        availability: Availability,
        marker: str = "",
    ) -> None:
        self.backend = backend
        self.model_formats = frozenset(formats)
        self._availability = availability
        self._marker = marker
        self.availability_calls = 0
        self.run_calls: list[tuple[str, list]] = []

    def availability(self) -> Availability:
        self.availability_calls += 1
        return self._availability

    def run(self, model_path: str, inputs: list) -> InferenceResult:
        self.run_calls.append((model_path, inputs))
        return InferenceResult([self._marker], 0.0)


def _model(model_format: str) -> dict:
    return {"id": f"model-{model_format}", "format": model_format}


def _workload(*models: dict, preference: list[str] | None = None) -> Workload:
    requirements = {"models": list(models)} if models else {}
    if preference is not None:
        requirements["acceleratorPreference"] = preference
    return Workload("accelerator-branches", {"requirements": requirements})


def test_public_runnable_model_covers_missing_empty_unsupported_and_supported():
    ncnn = _model("ncnn")
    openvino = _model("openvino")
    npu = _Lane("npu", {"openvino", "onnx"}, Availability(True))
    executors = {"npu": npu}

    assert runnable_model(_workload(ncnn), "missing", executors) is None
    assert runnable_model(_workload(), "npu", executors) is None
    assert runnable_model(_workload(ncnn), "npu", executors) is None
    assert runnable_model(_workload(ncnn, openvino), "npu", executors) is openvino


def test_empty_refusal_code_falls_back_to_stable_no_executor_code():
    gpu = _Lane("gpu", {"ncnn"}, Availability(False, "lane broken"))
    workload = _workload(_model("ncnn"), preference=["gpu"])

    assert most_actionable(()) == ""
    assert select_backend(workload, {"gpu": gpu}) == BackendChoice(
        None, "gpu: lane broken", NO_EXECUTOR
    )


def test_gpu_runtime_filters_cpu_and_unknown_providers_without_reordering():
    class _Session:
        @staticmethod
        def get_inputs() -> list:
            return [type("Input", (), {"name": "input"})()]

        @staticmethod
        def run(_outputs: None, feed: dict) -> list:
            return [feed["input"]]

    class _Runtime:
        def __init__(self) -> None:
            self.session_providers: tuple[str, ...] | None = None

        @staticmethod
        def get_available_providers() -> list[str]:
            return [
                "CPUExecutionProvider",
                "ROCMExecutionProvider",
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
            ]

        def InferenceSession(self, _model_path: str, providers: list[str]) -> _Session:  # noqa: N802
            self.session_providers = tuple(providers)
            return _Session()

    runtime = _Runtime()
    executor = GpuExecutor(device_present=True, runtime=runtime)

    assert executor.availability() == Availability(True)
    assert executor.run("model.onnx", [[1, 2]]).outputs == [[1, 2]]
    assert runtime.session_providers == (
        "ROCMExecutionProvider",
        "CUDAExecutionProvider",
    )


def test_composite_falls_through_only_between_matching_format_lanes():
    unavailable = _Lane(
        "gpu",
        {"onnx"},
        Availability(False, "first ONNX lane missing", RUNTIME_MISSING),
        "unavailable",
    )
    available = _Lane("gpu", {"onnx"}, Availability(True), "available")
    unrelated = _Lane("gpu", {"ncnn"}, Availability(True), "unrelated")
    composite = CompositeGpuExecutor([unavailable, available, unrelated])
    model = {"format": "onnx"}

    assert composite.availability_for(model) == Availability(True)
    result = composite.run_for_format("onnx", "misleading.param", [7])

    assert result.outputs == ["available"]
    assert unavailable.run_calls == []
    assert available.run_calls == [("misleading.param", [7])]
    assert unrelated.run_calls == []
    assert (
        unavailable.availability_calls,
        available.availability_calls,
        unrelated.availability_calls,
    ) == (2, 2, 0)


def test_executor_reconciliation_replaces_only_removed_and_readded_lane():
    old_tpu = Device("tpu-old", "tpu", "Old Coral", "pcie")
    initial = build_executors([old_tpu])

    removed = build_executors(
        [],
        previous_devices=[old_tpu],
        previous_executors=initial,
    )
    assert removed["tpu"] is not initial["tpu"]
    assert removed["tpu"]._device_present is False
    assert removed["npu"] is initial["npu"]
    assert removed["gpu"] is initial["gpu"]

    new_tpu = Device("tpu-new", "tpu", "New Coral", "usb")
    readded = build_executors(
        [new_tpu],
        previous_devices=[],
        previous_executors=removed,
    )
    assert readded["tpu"] is not removed["tpu"]
    assert readded["tpu"]._device_present is True
    assert readded["npu"] is removed["npu"]
    assert readded["gpu"] is removed["gpu"]


def test_absent_selected_gpu_placeholder_mirrors_the_gpu_lane_formats():
    executors = build_executors([Device("gpu-0", "gpu", "Some GPU", "pci")])
    placeholder = executors.for_device("gpu-vanished")["gpu"]

    assert placeholder.model_formats == executors["gpu"].model_formats
    assert placeholder.availability().code == DEVICE_ABSENT
