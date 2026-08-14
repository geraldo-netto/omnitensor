from __future__ import annotations

import pytest

from omnitensor.executors.base import Availability, InferenceResult
from omnitensor.executors.gpu import CompositeGpuExecutor, GpuExecutor
from omnitensor.executors.vulkan import (
    VulkanDevice,
    VulkanGpuExecutor,
    VulkanSelectionError,
    select_vulkan_device,
)


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
    use_fp16_packed = True
    use_fp16_storage = True
    use_fp16_arithmetic = True


class FakeNet:
    def __init__(self, runtime):
        self.opt = FakeOptions()
        self._runtime = runtime
        runtime.last_net = self

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
        self.nets_created = 0
        self.last_net = None

    def get_gpu_count(self):
        return len(self._device_types)

    def get_gpu_info(self, index):
        return FakeGpuInfo(self._device_types[index])

    def Net(self):  # noqa: N802 - mirrors the ncnn API
        self.nets_created += 1
        return FakeNet(self)


DISCRETE, INTEGRATED, VIRTUAL, CPU = 0, 1, 2, 3


def test_public_vulkan_selector_returns_named_hardware_and_refuses_exactly():
    class NamedInfo(FakeGpuInfo):
        def __init__(self, device_type, name):
            super().__init__(device_type)
            self._name = name

        def device_name(self):
            return self._name

    runtime = FakeNcnn([INTEGRATED, DISCRETE, CPU])
    runtime.get_gpu_info = lambda index: NamedInfo(
        runtime._device_types[index], ("integrated", "discrete", "cpu")[index]
    )
    assert select_vulkan_device(runtime) == VulkanDevice(1, "discrete", DISCRETE)
    assert select_vulkan_device(runtime, 0) == VulkanDevice(0, "integrated", INTEGRATED)

    with pytest.raises(VulkanSelectionError) as requested:
        select_vulkan_device(runtime, 2)
    assert (requested.value.kind, requested.value.reason) == (
        "requested",
        "Vulkan device 2 is absent or software-only",
    )

    software = FakeNcnn([CPU])
    with pytest.raises(VulkanSelectionError) as refused:
        select_vulkan_device(software)
    assert (refused.value.kind, refused.value.reason) == (
        "software-only",
        "Only a software (CPU) Vulkan device is present; CPU execution is not used",
    )


def test_public_vulkan_selector_contains_enumeration_and_empty_inventory():
    class BrokenRuntime:
        @staticmethod
        def get_gpu_count():
            raise OSError("loader failed")

    with pytest.raises(VulkanSelectionError) as broken:
        select_vulkan_device(BrokenRuntime())
    assert (broken.value.kind, broken.value.reason) == ("enumeration", "loader failed")

    with pytest.raises(VulkanSelectionError) as absent:
        select_vulkan_device(FakeNcnn([]))
    assert (absent.value.kind, absent.value.reason) == (
        "absent",
        "No Vulkan device is available",
    )


def test_vulkan_prefers_discrete_and_never_selects_software_devices():
    runtime = FakeNcnn([INTEGRATED, DISCRETE, CPU])
    executor = VulkanGpuExecutor(device_present=True, runtime=runtime)
    assert executor.availability().available is True
    result = executor.run("model.param", [[-1, 2, -3]])
    assert runtime.selected_device == 1
    assert result.outputs == [[1, 2, 3]]
    assert runtime.last_net.opt.use_vulkan_compute is True
    assert runtime.last_net.opt.use_fp16_packed is False
    assert runtime.last_net.opt.use_fp16_storage is False
    assert runtime.last_net.opt.use_fp16_arithmetic is False


def test_vulkan_preserves_bounded_integer_inputs_and_floats_everything_else():
    class TypedMat:
        def __init__(self, value):
            self.value = value

        def clone(self):
            return self

    runtime = FakeNcnn([DISCRETE])
    runtime.Mat = TypedMat
    executor = VulkanGpuExecutor(device_present=True, runtime=runtime)

    integers = executor._to_mat([[1, 2, 3]])  # noqa: SLF001 - dtype boundary
    booleans = executor._to_mat([[True, False]])  # noqa: SLF001 - dtype boundary
    floats = executor._to_mat([[1.25, -2.5]])  # noqa: SLF001 - dtype boundary
    boundaries = executor._to_mat(  # noqa: SLF001 - dtype boundary
        [[-(2**31), 2**31 - 1]]
    )

    assert integers.value.dtype.name == "int32"
    assert booleans.value.dtype.name == "float32"
    assert floats.value.dtype.name == "float32"
    assert floats.value.tolist() == [[1.25, -2.5]]
    assert boundaries.value.dtype.name == "int32"
    assert boundaries.value.tolist() == [[-(2**31), 2**31 - 1]]
    with pytest.raises(RuntimeError) as overflow:
        executor._to_mat([[2**31]])  # noqa: SLF001 - dtype boundary
    assert str(overflow.value) == "ncnn integer input exceeds int32 range"


def test_vulkan_preserves_three_axes_unwraps_one_batch_and_rejects_many():
    class ShapedMat:
        def __init__(self, value):
            self.value = value

        def clone(self):
            return self

    runtime = FakeNcnn([DISCRETE])
    runtime.Mat = ShapedMat
    executor = VulkanGpuExecutor(device_present=True, runtime=runtime)

    three_axes = executor._to_mat([[[1.0, 2.0]]])  # noqa: SLF001
    one_batch = executor._to_mat([[[[1.0, 2.0]]]])  # noqa: SLF001
    assert three_axes.value.shape == (1, 1, 2)
    assert one_batch.value.shape == (1, 1, 2)
    assert one_batch.value.tolist() == [[[1.0, 2.0]]]

    with pytest.raises(RuntimeError) as batch:
        executor._to_mat(  # noqa: SLF001
            [[[[1.0, 2.0]]], [[[3.0, 4.0]]]]
        )
    assert str(batch.value) == "ncnn runs one image at a time; received a batch of 2"


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
        False,
        "ncnn is not installed",
        "runtime-missing",
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
        self.availability_calls = 0
        self.run_calls = []

    def availability(self):
        self.availability_calls += 1
        return self._availability

    def run(self, model_path, inputs):
        self.run_calls.append((model_path, inputs))
        return InferenceResult(outputs=[self._marker], duration_ms=0.0)


def test_composite_dispatches_by_extension_and_prefers_vulkan():
    vulkan = StubExecutor({"ncnn"}, True, marker="vulkan")
    onnx = StubExecutor({"onnx"}, True, marker="onnx")
    composite = CompositeGpuExecutor([vulkan, onnx])
    assert composite.availability().available is True
    assert composite.model_formats == {"ncnn", "onnx"}
    assert composite.run("model.param", []).outputs == ["vulkan"]
    assert composite.run("model.onnx", []).outputs == ["onnx"]


@pytest.mark.parametrize(
    ("model_format", "misleading_path", "expected"),
    [
        ("ncnn", "model.onnx", "vulkan"),
        ("onnx", "model.param", "onnx"),
    ],
)
def test_composite_dispatches_by_declared_format_not_path(
    model_format,
    misleading_path,
    expected,
):
    vulkan = StubExecutor({"ncnn"}, True, marker="vulkan")
    onnx = StubExecutor({"onnx"}, True, marker="onnx")
    composite = CompositeGpuExecutor([vulkan, onnx])

    result = composite.run_for_format(model_format, misleading_path, [1])

    assert result.outputs == [expected]
    assert vulkan.run_calls == ([(misleading_path, [1])] if expected == "vulkan" else [])
    assert onnx.run_calls == ([(misleading_path, [1])] if expected == "onnx" else [])


def test_composite_refuses_unknown_format_without_probing_or_running_a_lane():
    vulkan = StubExecutor({"ncnn"}, True)
    onnx = StubExecutor({"onnx"}, True)
    composite = CompositeGpuExecutor([vulkan, onnx])

    with pytest.raises(RuntimeError) as raised:
        composite.run_for_format("openvino", "model.param", [])

    assert str(raised.value) == "No available GPU runtime for model: model.param"
    assert vulkan.availability_calls == 0
    assert onnx.availability_calls == 0
    assert vulkan.run_calls == []
    assert onnx.run_calls == []


def test_composite_does_not_fall_through_to_another_format_when_lane_is_down():
    vulkan = StubExecutor({"ncnn"}, False, "no Vulkan", marker="vulkan")
    onnx = StubExecutor({"onnx"}, True, marker="onnx")
    composite = CompositeGpuExecutor([vulkan, onnx])

    with pytest.raises(RuntimeError) as raised:
        composite.run_for_format("ncnn", "misleading.onnx", [])

    assert str(raised.value) == ("No available GPU runtime for model: misleading.onnx")
    assert vulkan.availability_calls == 1
    assert onnx.availability_calls == 0
    assert vulkan.run_calls == []
    assert onnx.run_calls == []


def test_composite_availability_is_scoped_to_requested_format():
    composite = CompositeGpuExecutor(
        [
            StubExecutor({"ncnn"}, True),
            StubExecutor({"onnx"}, False, "no GPU provider"),
        ]
    )
    assert composite.availability().available is True
    assert composite.availability_for({"format": "ncnn"}) == Availability(True)
    assert composite.availability_for({"format": "onnx"}) == Availability(
        False,
        "no GPU provider",
    )
    assert composite.availability_for({"format": "openvino"}) == Availability(
        False,
        "No GPU runtime supports the model format",
        "format-unsupported",
    )


def test_composite_reports_joined_reasons_and_missing_runtime():
    composite = CompositeGpuExecutor(
        [
            StubExecutor({"ncnn"}, False, "no vulkan"),
            StubExecutor({"onnx"}, False, "no provider"),
        ]
    )
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

    composite = CompositeGpuExecutor(
        [
            VulkanGpuExecutor(device_present=True, runtime=FakeNcnn([CPU])),
            GpuExecutor(device_present=True, runtime=FakeOrt()),
        ]
    )
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


def test_the_loaded_net_is_reused_between_jobs(tmp_path):
    """Loading and compiling the same model on every job defeats warm inference."""
    runtime = FakeNcnn([DISCRETE])
    executor = VulkanGpuExecutor(device_present=True, runtime=runtime)
    model = tmp_path / "model.param"
    model.write_text("parameter graph")
    model.with_suffix(".bin").write_bytes(b"weights")

    executor.run(str(model), [[1]])
    executor.run(str(model), [[1]])

    assert runtime.nets_created == 1
    assert runtime.cleared == 0, "the cached Net must outlive each extractor"


def test_replacing_a_vulkan_model_reloads_its_net(tmp_path):
    runtime = FakeNcnn([DISCRETE])
    executor = VulkanGpuExecutor(device_present=True, runtime=runtime)
    model = tmp_path / "model.param"
    model.write_text("first graph")
    model.with_suffix(".bin").write_bytes(b"weights")

    executor.run(str(model), [[1]])
    model.write_text("a different graph with a different size")
    executor.run(str(model), [[1]])

    assert runtime.nets_created == 2, "the retired graph was served from cache"


def test_replacing_vulkan_weights_reloads_their_net(tmp_path):
    runtime = FakeNcnn([DISCRETE])
    executor = VulkanGpuExecutor(device_present=True, runtime=runtime)
    model = tmp_path / "model.param"
    model.write_text("unchanged graph")
    weights = model.with_suffix(".bin")
    weights.write_bytes(b"first weights")

    executor.run(str(model), [[1]])
    weights.write_bytes(b"different replacement weights")
    executor.run(str(model), [[1]])

    assert runtime.nets_created == 2, "retired weights were served from cache"


def test_vulkan_model_cache_is_bounded_and_least_recently_used(tmp_path):
    runtime = FakeNcnn([DISCRETE])
    executor = VulkanGpuExecutor(
        device_present=True,
        runtime=runtime,
        max_cached_models=2,
    )
    models = []
    for index in range(3):
        model = tmp_path / f"model-{index}.param"
        model.write_text(f"graph {index}")
        model.with_suffix(".bin").write_bytes(f"weights {index}".encode())
        models.append(model)

    for model in models[:2]:
        executor.run(str(model), [[1]])
    executor.run(str(models[0]), [[1]])
    executor.run(str(models[2]), [[1]])
    executor.run(str(models[1]), [[1]])

    assert runtime.nets_created == 4, "least-recently-used model was not evicted"


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "2"])
def test_vulkan_model_cache_bound_must_be_a_positive_integer(bad):
    with pytest.raises(ValueError, match="max_entries"):
        VulkanGpuExecutor(True, runtime=FakeNcnn([DISCRETE]), max_cached_models=bad)
