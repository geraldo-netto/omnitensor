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
    def clone(self):
        # The real Mat borrows the numpy buffer it was built from and does not
        # keep the owner alive, so the executor clones every Mat before it
        # reaches ncnn. A fake that could not be cloned would let that
        # requirement be removed without a test noticing.
        return FakeMat(self)


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

    def clear(self):
        # The real Net releases the Vulkan allocators its extractor borrows;
        # the executor now tears them down in order rather than leaving it to
        # garbage collection, so the fake has to answer for it.
        self._runtime.cleared += 1

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
        self.cleared = 0

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


def test_vulkan_run_enumerates_devices_once():
    class CountingNcnn(FakeNcnn):
        def __init__(self):
            super().__init__([INTEGRATED, DISCRETE])
            self.enumerations = 0

        def get_gpu_count(self):
            self.enumerations += 1
            return super().get_gpu_count()

    runtime = CountingNcnn()
    result = VulkanGpuExecutor(device_present=True, runtime=runtime).run("model.param", [[1]])
    assert runtime.enumerations == 1
    assert runtime.selected_device == 1
    assert result.outputs == [[1]]


def test_vulkan_software_only_reports_no_cpu_rule():
    executor = VulkanGpuExecutor(device_present=True, runtime=FakeNcnn([CPU]))
    availability = executor.availability()
    assert availability.available is False
    assert "CPU execution is not used" in availability.reason


def test_vulkan_degrades_without_device_runtime_or_gpus():
    absent = VulkanGpuExecutor(device_present=False).availability()
    assert absent == Availability(False, "No GPU render node detected", "device-absent")
    missing = VulkanGpuExecutor(device_present=True, runtime=None)
    missing._runtime = None
    assert missing.availability() == Availability(
        False, "ncnn is not installed", "runtime-missing",
    )
    with pytest.raises(RuntimeError) as raised:
        missing.run("model.param", [])
    assert str(raised.value) == "gpu executor unavailable: ncnn is not installed"
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


def test_composite_availability_is_scoped_to_requested_format():
    composite = CompositeGpuExecutor([
        StubExecutor({"ncnn"}, True),
        StubExecutor({"onnx"}, False, "no GPU provider"),
    ])
    assert composite.availability().available is True
    assert composite.availability_for({"format": "ncnn"}) == Availability(True)
    assert composite.availability_for({"format": "onnx"}) == Availability(
        False, "no GPU provider",
    )
    assert composite.availability_for({"format": "openvino"}) == Availability(
        False, "No GPU runtime supports the model format", "format-unsupported",
    )


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


def test_the_device_is_enumerated_once_and_reused():
    """Enumerating Vulkan is not a read-only question.

    It initialises the loader and queries every driver, and the snapshot
    publisher asks whether this backend is available twice a second from a
    different thread from the one running inference. Doing that under a
    running job corrupted its results.
    """
    class CountingNcnn(FakeNcnn):
        def __init__(self):
            super().__init__([INTEGRATED, DISCRETE])
            self.enumerations = 0

        def get_gpu_count(self):
            self.enumerations += 1
            return super().get_gpu_count()

    runtime = CountingNcnn()
    executor = VulkanGpuExecutor(device_present=True, runtime=runtime)

    for _ in range(5):
        executor.availability()
    executor.run("model.param", [[1]])

    assert runtime.enumerations == 1, "the answer changes only when hardware does"


def test_the_net_is_released_in_order_after_every_run():
    """The extractor borrows Vulkan memory the Net's allocators own, so the
    Net cannot simply fall out of scope while the extractor still holds it."""
    runtime = FakeNcnn([DISCRETE])
    executor = VulkanGpuExecutor(device_present=True, runtime=runtime)

    executor.run("model.param", [[1]])
    executor.run("model.param", [[1]])

    assert runtime.cleared == 2, "each run releases the Net it created"
