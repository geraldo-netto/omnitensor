from __future__ import annotations

import pytest

from omnitensor.executors.base import Availability, InferenceResult
from omnitensor.executors.gpu import CompositeGpuExecutor, GpuExecutor
from omnitensor.executors.vulkan import VulkanGpuExecutor


class FakeGpuInfo:
    def __init__(self, device_type):
        self._type = device_type

    def type(self):
        return self._type


class FakeMat(list):
    pass


class FakeExtractor:
    def __init__(self):
        self._inputs = {}

    def input(self, name, mat):
        self._inputs[name] = mat

    def extract(self, name):
        return 0, FakeMat(abs(value) for value in self._inputs["in0"])


class FakeOptions:
    use_vulkan_compute = False


class FakeNet:
    def __init__(self, runtime):
        self.opt = FakeOptions()
        self._runtime = runtime

    def set_vulkan_device(self, device):
        self._runtime.selected_device = device

    def load_param(self, path):
        return 0 if path.endswith(".param") else -1

    def load_model(self, path):
        return 0 if path.endswith(".bin") else -1

    def input_names(self):
        return ["in0"]

    def output_names(self):
        return ["out0"]

    def create_extractor(self):
        return FakeExtractor()


class FakeNcnn:
    """Enough of the pyncnn surface for the executor's full run path."""

    Mat = FakeMat

    def __init__(self, device_types):
        self._device_types = device_types
        self.selected_device = None

    def get_gpu_count(self):
        return len(self._device_types)

    def get_gpu_info(self, index):
        return FakeGpuInfo(self._device_types[index])

    def Net(self):  # noqa: N802 - mirrors the ncnn API
        return FakeNet(self)


DISCRETE, INTEGRATED, VIRTUAL, CPU = 0, 1, 2, 3


def test_vulkan_prefers_discrete_and_never_selects_software_devices():
    runtime = FakeNcnn([INTEGRATED, DISCRETE, CPU])
    executor = VulkanGpuExecutor(device_present=True, runtime=runtime)
    assert executor.availability().available is True
    result = executor.run("model.param", [[-1, 2, -3]])
    assert runtime.selected_device == 1
    assert result.outputs == [[1, 2, 3]]


def test_vulkan_software_only_reports_no_cpu_rule():
    executor = VulkanGpuExecutor(device_present=True, runtime=FakeNcnn([CPU]))
    availability = executor.availability()
    assert availability.available is False
    assert "CPU execution is not used" in availability.reason


def test_vulkan_degrades_without_device_runtime_or_gpus():
    assert "render node" in VulkanGpuExecutor(device_present=False).availability().reason
    missing = VulkanGpuExecutor(device_present=True, runtime=None)
    missing._runtime = None
    assert "ncnn is not installed" in missing.availability().reason
    empty = VulkanGpuExecutor(device_present=True, runtime=FakeNcnn([]))
    assert "No Vulkan device" in empty.availability().reason


def test_vulkan_enumeration_failure_is_a_reason_not_a_crash():
    class ExplodingRuntime:
        @staticmethod
        def get_gpu_count():
            raise OSError("loader broken")

    executor = VulkanGpuExecutor(device_present=True, runtime=ExplodingRuntime())
    availability = executor.availability()
    assert availability.available is False
    assert "enumeration failed" in availability.reason


class StubExecutor:
    backend = "gpu"

    def __init__(self, formats, available, reason="", marker=None):
        self.model_formats = frozenset(formats)
        self._availability = Availability(available, reason)
        self._marker = marker

    def availability(self):
        return self._availability

    def run(self, model_path, inputs):
        return InferenceResult(outputs=[self._marker], duration_ms=0.0)


def test_composite_dispatches_by_extension_and_prefers_vulkan():
    vulkan = StubExecutor({"ncnn"}, True, marker="vulkan")
    onnx = StubExecutor({"onnx"}, True, marker="onnx")
    composite = CompositeGpuExecutor([vulkan, onnx])
    assert composite.availability().available is True
    assert composite.model_formats == {"ncnn", "onnx"}
    assert composite.run("model.param", []).outputs == ["vulkan"]
    assert composite.run("model.onnx", []).outputs == ["onnx"]


def test_composite_reports_joined_reasons_and_missing_runtime():
    composite = CompositeGpuExecutor([
        StubExecutor({"ncnn"}, False, "no vulkan"),
        StubExecutor({"onnx"}, False, "no provider"),
    ])
    availability = composite.availability()
    assert availability.available is False
    assert availability.reason == "no vulkan; no provider"
    with pytest.raises(RuntimeError, match="No available GPU runtime"):
        composite.run("model.param", [])

    empty = CompositeGpuExecutor([])
    assert empty.availability().reason == "No GPU runtime configured"
    assert empty.model_formats == frozenset()


def test_composite_with_real_onnx_fallback_shape():
    class FakeOrt:
        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

    composite = CompositeGpuExecutor([
        VulkanGpuExecutor(device_present=True, runtime=FakeNcnn([CPU])),
        GpuExecutor(device_present=True, runtime=FakeOrt()),
    ])
    availability = composite.availability()
    assert availability.available is False
    assert "CPU execution is not used" in availability.reason
    assert "CPU provider is not used" in availability.reason
