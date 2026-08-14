"""Exact-behavior assertions added to kill surviving mutants.

Each block pins observable behavior (exact messages, exact documents, exact
call arguments, exact math) that broader tests only checked loosely.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from conftest import sample_manifest, write_workload

from omnitensor.control import ControlService, build_control_service
from omnitensor.discovery import (
    DiscoveryPaths,
    detect_gpu,
    detect_npu,
    detect_tpu,
    device_utilization,
)
from omnitensor.executors.base import Availability, InferenceResult, run_executor
from omnitensor.executors.gpu import CompositeGpuExecutor, GpuExecutor
from omnitensor.executors.npu import NpuExecutor
from omnitensor.executors.tpu import TpuExecutor
from omnitensor.executors.vulkan import VulkanGpuExecutor
from omnitensor.registry import (
    MAX_MANIFEST_BYTES,
    ManifestError,
    Workload,
    load_workloads,
    validate_document,
)
from omnitensor.scheduler import Scheduler, pick_backend
from omnitensor.snapshot import build_snapshot, write_snapshot
from omnitensor.state import PolicyState, PolicyStore, ProfilePolicy

# --------------------------------------------------------------------------
# control: exact acknowledgement contents


class MemoryStorage:
    def __init__(self):
        self.saved: list[PolicyState] = []

    def load(self) -> PolicyState:
        return PolicyState(profiles={"wl": ProfilePolicy(enabled=True, weight=2)})

    def save(self, state: PolicyState) -> None:
        self.saved.append(state)


def apply(control: ControlService, document) -> dict:
    text = asyncio.run(
        control.apply_command_text(
            document if isinstance(document, str) else json.dumps(document),
        )
    )
    # The wire form is compact JSON.
    parsed = json.loads(text)
    assert text == json.dumps(parsed, separators=(",", ":"))
    return parsed


def valid_command(**overrides) -> dict:
    command = {
        "version": 1,
        "id": "cmd-1",
        "issuedAt": 1,
        "expectedRevision": 0,
        "operation": "set-paused",
        "profileId": None,
        "value": True,
    }
    command.update(overrides)
    return command


def test_control_exact_rejection_and_acknowledgement_messages():
    control = ControlService(MemoryStorage())
    assert apply(control, "{nope")["message"] == "Command is not valid JSON"
    assert apply(control, "[]")["message"] == "Command is not an object"
    bad = apply(control, {"version": 2})
    assert bad["message"] == "Command does not match the version 1 contract"
    assert bad["commandId"] == "invalid"
    unknown = apply(
        control,
        valid_command(operation="set-profile-enabled", profileId="missing", value=True),
    )
    assert unknown["message"] == "Unknown workload profile: missing"
    applied = apply(control, valid_command())
    assert applied["message"] == "Policy applied"
    assert applied["status"] == "applied"
    assert applied["revision"] == 1


def test_control_applied_at_is_current_epoch_milliseconds():
    control = ControlService(MemoryStorage())
    acknowledgement = apply(control, valid_command())
    assert abs(acknowledgement["appliedAt"] - time.time() * 1000) < 60_000


def test_control_saves_exactly_the_published_state():
    storage = MemoryStorage()
    control = ControlService(storage)
    apply(control, valid_command(operation="set-profile-weight", profileId="wl", value=4))
    assert len(storage.saved) == 1
    assert storage.saved[0] is control.state
    assert control.state.profiles["wl"].weight == 4
    assert control.state.revision == 1


def test_build_control_service_wires_a_store_at_the_given_path(tmp_path):
    control = build_control_service(
        tmp_path / "policy.json", {"wl": ProfilePolicy(enabled=True, weight=2)},
    )
    apply(control, valid_command())
    persisted = json.loads((tmp_path / "policy.json").read_text())
    assert persisted["revision"] == 1
    assert persisted["paused"] is True


# --------------------------------------------------------------------------
# state: exact persisted document


def test_policy_save_writes_exact_compact_sorted_document(tmp_path):
    path = tmp_path / "policy.json"
    PolicyStore(path, {}).save(PolicyState(
        paused=False,
        profiles={
            "zeta": ProfilePolicy(enabled=True, weight=5),
            "alpha": ProfilePolicy(enabled=False, weight=1),
        },
        revision=7,
    ))
    expected = (
        '{"paused":false,'
        '"profiles":{"alpha":{"enabled":false,"weight":1},"zeta":{"enabled":true,"weight":5}},'
        '"revision":7}'
    )
    assert path.read_text() == expected


def test_policy_load_revision_boundaries(tmp_path):
    path = tmp_path / "policy.json"
    for value, expected in ((0, 0), (1, 1), (-1, 0)):
        path.write_text(json.dumps({"revision": value}))
        assert PolicyStore(path, {}).load().revision == expected


# --------------------------------------------------------------------------
# snapshot: exact document shape


def device_kwargs(index: int = 0) -> dict:
    from omnitensor.discovery import Device

    return Device(
        id=f"gpu-renderD{128 + index}", backend="gpu", name="AMD GPU", kind="dri",
    )


def test_snapshot_truncates_devices_to_the_contract_maximum():
    devices = [device_kwargs(index) for index in range(17)]
    snapshot = build_snapshot(devices=devices, metrics={}, profiles={})
    assert len(snapshot["devices"]) == 16
    assert snapshot["devices"][0]["id"] == "gpu-renderD128"


def test_snapshot_defaults_are_exact():
    snapshot = build_snapshot(
        devices=[device_kwargs()], metrics={}, profiles={}, generated_at_ms=1234,
    )
    assert snapshot["version"] == 1
    assert snapshot["generatedAt"] == 1234
    assert snapshot["metrics"] == {"queueDepth": 0, "runningProfiles": 0}
    assert snapshot["alerts"] == []
    assert snapshot["profiles"] == {}


def test_snapshot_generated_at_defaults_to_current_epoch_milliseconds():
    snapshot = build_snapshot(devices=[device_kwargs()], metrics={}, profiles={})
    assert abs(snapshot["generatedAt"] - time.time() * 1000) < 60_000


def test_snapshot_metric_conversion_and_error_message():
    snapshot = build_snapshot(
        devices=[device_kwargs()],
        metrics={"queueDepth": 7.9, "runningProfiles": "3"},
        profiles={},
        generated_at_ms=1,
    )
    assert snapshot["metrics"] == {"queueDepth": 7, "runningProfiles": 3}
    with pytest.raises(ValueError, match="metrics queueDepth is not an integer: None"):
        build_snapshot(devices=[device_kwargs()], metrics={"queueDepth": None}, profiles={})


def test_write_snapshot_leaves_only_the_target_file(tmp_path):
    target = tmp_path / "runtime-snapshot.json"
    payload = {"version": 1, "text": "é"}
    write_snapshot(target, payload)
    assert target.read_text(encoding="utf-8") == json.dumps(payload, separators=(",", ":"))
    assert [entry.name for entry in tmp_path.iterdir()] == ["runtime-snapshot.json"]


# --------------------------------------------------------------------------
# registry: exact violation strings and bounds


def test_validate_document_formats_paths_exactly():
    root_violations = validate_document("runtime-command.schema.json", 5)
    assert root_violations == ["/: 5 is not of type 'object'"]
    manifest = sample_manifest()
    manifest["requirements"]["accelerator"] = "bogus"
    nested = validate_document("workload-manifest.schema.json", manifest)
    assert len(nested) == 1
    assert nested[0].startswith("requirements/accelerator: ")
    assert "bogus" in nested[0]


def manifest_of_exact_size(size: int) -> str:
    text = json.dumps(sample_manifest())
    assert len(text) < size
    return text + " " * (size - len(text))


def test_registry_manifest_size_boundary_is_exact(tmp_path):
    directory = tmp_path / "sample-workload"
    directory.mkdir()
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(manifest_of_exact_size(MAX_MANIFEST_BYTES))
    assert sorted(load_workloads(tmp_path)) == ["sample-workload"]
    manifest_path.write_text(manifest_of_exact_size(MAX_MANIFEST_BYTES + 1))
    with pytest.raises(ManifestError, match=r"manifest exceeds 65536 bytes"):
        load_workloads(tmp_path)


def test_registry_enforces_the_workload_count_bound(tmp_path):
    for index in range(129):
        write_workload(tmp_path, sample_manifest(f"workload-{index:03d}"))
    with pytest.raises(ManifestError, match="more than 128 workloads"):
        load_workloads(tmp_path)
    (tmp_path / "workload-128/manifest.json").unlink()
    (tmp_path / "workload-128").rmdir()
    assert len(load_workloads(tmp_path)) == 128


def test_registry_ignores_files_and_manifestless_directories(tmp_path):
    (tmp_path / "stray-file").write_text("ignored")
    (tmp_path / "empty-dir").mkdir()
    write_workload(tmp_path, sample_manifest())
    assert sorted(load_workloads(tmp_path)) == ["sample-workload"]


# --------------------------------------------------------------------------
# discovery: exact identities, truncation, clamping


def paths(tmp_path) -> DiscoveryPaths:
    dev = tmp_path / "dev"
    sys = tmp_path / "sys"
    dev.mkdir(exist_ok=True)
    sys.mkdir(exist_ok=True)
    return DiscoveryPaths(dev=dev, sys=sys)


def test_tpu_pcie_index_and_naming_are_exact(tmp_path):
    nodes = paths(tmp_path)
    (nodes.dev / "apex_3").touch()
    device = detect_tpu(nodes)
    assert device.id == "tpu-pcie-3"
    assert device.name == "Coral PCIe Edge TPU 4"
    assert device.kind == "pcie"
    assert device.backend == "tpu"
    assert device.vendor == ""


def test_tpu_pcie_candidate_count_is_bounded_but_ids_are_not(tmp_path):
    nodes = paths(tmp_path)
    (nodes.dev / "apex_80").touch()
    assert detect_tpu(nodes).id == "tpu-pcie-80"
    for index in range(8):
        (nodes.dev / f"apex_{index}").touch()
    assert detect_tpu(nodes, "tpu-pcie-80") is None


def test_tpu_usb_identities_are_exact(tmp_path):
    nodes = paths(tmp_path)
    device_dir = nodes.sys / "bus/usb/devices/1-1"
    device_dir.mkdir(parents=True)
    (device_dir / "idVendor").write_text("1A6E\n")
    (device_dir / "idProduct").write_text("089A\n")
    device = detect_tpu(nodes)
    assert device.id == "tpu-usb"
    assert device.name == "Coral USB Accelerator (DFU)"
    assert device.kind == "usb"
    assert device.vendor == "1a6e:089a"


def test_vendor_read_truncates_to_32_characters_and_replaces_bad_bytes(tmp_path):
    nodes = paths(tmp_path)
    (nodes.dev / "accel").mkdir()
    (nodes.dev / "accel/accel0").touch()
    vendor_dir = nodes.sys / "class/accel/accel0/device"
    vendor_dir.mkdir(parents=True)
    (vendor_dir / "vendor").write_text("x" * 40)
    assert detect_npu(nodes).vendor == "x" * 32
    (vendor_dir / "vendor").write_bytes(b"\xff\xfe")
    assert detect_npu(nodes).vendor == "��"


def test_gpu_candidate_count_is_bounded_but_render_ids_are_not(tmp_path):
    nodes = paths(tmp_path)
    (nodes.dev / "dri").mkdir()
    (nodes.dev / "dri/renderD913").touch()
    device = detect_gpu(nodes)
    assert device.id == "gpu-renderD913"
    assert device.kind == "dri"
    for node in range(128, 136):
        (nodes.dev / f"dri/renderD{node}").touch()
    assert detect_gpu(nodes, "gpu-renderD913") is None


def test_utilization_clamps_negative_and_overflow_values(tmp_path):
    nodes = paths(tmp_path)
    (nodes.dev / "dri").mkdir()
    (nodes.dev / "dri/renderD128").touch()
    gpu_dir = nodes.sys / "class/drm/renderD128/device"
    gpu_dir.mkdir(parents=True)
    device = detect_gpu(nodes)
    (gpu_dir / "gpu_busy_percent").write_text("-5")
    assert device_utilization(nodes, device) == 0.0
    (gpu_dir / "gpu_busy_percent").write_text("250")
    assert device_utilization(nodes, device) == 100.0
    (gpu_dir / "gpu_busy_percent").write_text("62.5")
    assert device_utilization(nodes, device) == 62.5


# --------------------------------------------------------------------------
# scheduler: exact stats, EMA math, and worker lifecycle


class RecordingExecutor:
    backend = "tpu"
    model_formats = frozenset({"tflite-edgetpu"})

    def __init__(self):
        self.calls: list[tuple[str, list]] = []

    def availability(self):
        return Availability(True)

    def run(self, model_path, inputs):
        self.calls.append((model_path, inputs))
        return InferenceResult(outputs=[inputs], duration_ms=1.0)


def test_scheduler_initial_stats_are_exact():
    async def scenario():
        scheduler = Scheduler({"tpu": RecordingExecutor()}, weight_of=lambda _p: 1)
        assert scheduler.stats() == {
            "queueDepth": 0,
            "runningProfiles": 0,
            "loads": {"tpu": None},
        }

    asyncio.run(scenario())


def test_scheduler_tick_ema_math_is_exact():
    async def scenario():
        scheduler = Scheduler({"tpu": RecordingExecutor()}, weight_of=lambda _p: 1)
        queue = scheduler._queues["tpu"]
        queue.busy_ms = 1e12  # clamps to 100%
        scheduler.tick()
        assert queue.load == 100.0
        assert queue.busy_ms == 0.0
        scheduler.tick()  # no busy time: EMA halves
        assert queue.load == 50.0
        scheduler.tick()
        assert queue.load == 25.0

    asyncio.run(scenario())


def test_scheduler_passes_exact_job_arguments_and_result():
    async def scenario():
        executor = RecordingExecutor()
        scheduler = Scheduler({"tpu": executor}, weight_of=lambda _p: 1)
        scheduler.start()
        result = await scheduler.submit("tpu", "profile-a", "model.tflite", [1, 2])
        assert executor.calls == [("model.tflite", [1, 2])]
        assert result.outputs == [[1, 2]]
        assert scheduler._queues["tpu"].busy_ms > 0.0
        assert scheduler.stats()["runningProfiles"] == 0
        await scheduler.stop()

    asyncio.run(scenario())


def test_scheduler_worker_keeps_serving_after_going_idle():
    async def scenario():
        executor = RecordingExecutor()
        scheduler = Scheduler({"tpu": executor}, weight_of=lambda _p: 1)
        scheduler.start()
        await scheduler.submit("tpu", "profile-a", "first", [])
        await asyncio.sleep(0.01)  # worker drains and goes back to waiting
        await asyncio.wait_for(
            scheduler.submit("tpu", "profile-a", "second", []), timeout=2,
        )
        assert [call[0] for call in executor.calls] == ["first", "second"]
        await scheduler.stop()

    asyncio.run(scenario())


def test_scheduler_stop_terminates_all_workers():
    async def scenario():
        scheduler = Scheduler(
            {"tpu": RecordingExecutor(), "npu": RecordingExecutor()},
            weight_of=lambda _p: 1,
        )
        scheduler.start()
        await asyncio.sleep(0)
        workers = list(scheduler._workers.values())
        await scheduler.stop()
        assert scheduler._stopped is True
        assert workers
        assert all(worker.done() for worker in workers)
        assert scheduler._workers == {}

    asyncio.run(scenario())


def test_update_executors_before_start_never_spawns_workers():
    async def scenario():
        tpu = RecordingExecutor()
        scheduler = Scheduler({"tpu": tpu}, weight_of=lambda _p: 1)
        assert scheduler._started is False
        assert scheduler._stopped is False
        scheduler.update_executors({"tpu": tpu, "npu": RecordingExecutor()})
        assert scheduler._workers == {}
        scheduler.start()
        assert scheduler._workers.keys() == {"tpu", "npu"}
        await scheduler.stop()

    asyncio.run(scenario())


def test_pick_backend_reason_composition_is_exact():
    executors = {
        "tpu": TpuExecutor(device_present=False),
        "npu": NpuExecutor(device_present=False),
    }
    manifest = sample_manifest(acceleratorPreference=["tpu", "npu", "gpu"])
    workload = Workload(id=manifest["id"], manifest=manifest)
    backend, reason = pick_backend(workload, executors)
    assert backend is None
    assert reason == (
        "tpu: No Coral Edge TPU device detected; "
        "npu: No /dev/accel NPU device detected; "
        "gpu: no executor"
    )


# --------------------------------------------------------------------------
# executors: exact runtime interactions


class RecordingTflite:
    class Interpreter:
        last = None

        def __init__(self, model_path, experimental_delegates):
            self.model_path = model_path
            self.delegates = experimental_delegates
            self.tensors = {}
            RecordingTflite.Interpreter.last = self

        def allocate_tensors(self):
            pass

        def get_input_details(self):
            return [{"index": 0}]

        def get_output_details(self):
            return [{"index": 1}]

        def set_tensor(self, index, value):
            self.tensors[index] = value

        def invoke(self):
            self.tensors[1] = self.tensors[0]

        def get_tensor(self, index):
            return self.tensors[index]

    def __init__(self):
        self.delegate_names: list[str] = []

    def load_delegate(self, name):
        self.delegate_names.append(name)
        return "delegate-token"


def test_tpu_run_wires_the_edgetpu_delegate_exactly():
    runtime = RecordingTflite()
    executor = TpuExecutor(device_present=True, runtime=runtime)
    result = executor.run("model.tflite", [[9]])
    assert runtime.delegate_names == ["libedgetpu.so.1"]
    interpreter = RecordingTflite.Interpreter.last
    assert interpreter.model_path == "model.tflite"
    assert interpreter.delegates == ["delegate-token"]
    assert result.outputs == [[9]]
    assert isinstance(result.duration_ms, float) and result.duration_ms >= 0.0


def test_tpu_delegate_failure_is_cached_as_unavailability():
    class BrokenDelegateRuntime(RecordingTflite):
        def load_delegate(self, name):
            raise OSError("libedgetpu.so.1 missing")

    executor = TpuExecutor(device_present=True, runtime=BrokenDelegateRuntime())
    assert executor.availability() == Availability(True)
    with pytest.raises(RuntimeError, match=r"Could not load libedgetpu\.so\.1"):
        executor.run("model.tflite", [[1]])
    availability = executor.availability()
    assert availability.available is False
    assert availability.reason.startswith("Could not load libedgetpu.so.1: ")


def test_tpu_availability_reasons_are_exact():
    assert TpuExecutor(device_present=False).availability() == Availability(
        False, "No Coral Edge TPU device detected", "device-absent",
    )
    missing = TpuExecutor(device_present=True, runtime=None)
    missing._runtime = None
    assert missing.availability() == Availability(
        False, "tflite-runtime is not installed", "runtime-missing",
    )


class RecordingOrt:
    class InferenceSession:
        last = None

        def __init__(self, model_path, providers):
            self.model_path = model_path
            self.providers = providers
            RecordingOrt.InferenceSession.last = self

        def get_inputs(self):
            class _Input:
                name = "input"

            return [_Input()]

        def run(self, outputs, feed):
            assert outputs is None
            return [feed["input"]]

    def __init__(self, providers):
        self._providers = providers

    def get_available_providers(self):
        return self._providers


def test_gpu_run_uses_only_gpu_providers_in_order():
    runtime = RecordingOrt(["CPUExecutionProvider", "ROCMExecutionProvider"])
    executor = GpuExecutor(device_present=True, runtime=runtime)
    assert executor.availability() == Availability(True)
    result = executor.run("model.onnx", [[5, 6]])
    session = RecordingOrt.InferenceSession.last
    assert session.model_path == "model.onnx"
    assert session.providers == ["ROCMExecutionProvider"]
    assert result.outputs == [[5, 6]]
    assert isinstance(result.duration_ms, float) and result.duration_ms >= 0.0


def test_gpu_availability_reasons_are_exact():
    assert GpuExecutor(device_present=False).availability() == Availability(
        False, "No GPU render node detected", "device-absent",
    )
    missing = GpuExecutor(device_present=True, runtime=None)
    missing._runtime = None
    assert missing.availability() == Availability(
        False, "onnxruntime is not installed", "runtime-missing",
    )
    cpu_only = GpuExecutor(device_present=True, runtime=RecordingOrt(["CPUExecutionProvider"]))
    assert cpu_only.availability() == Availability(
        False,
        "onnxruntime has no CUDA or ROCm execution provider; the CPU provider is not used",
        "runtime-unusable",
    )


def test_composite_forwards_exact_arguments():
    class Recording:
        backend = "gpu"
        model_formats = frozenset({"ncnn"})

        def __init__(self):
            self.calls = []

        def availability(self):
            return Availability(True)

        def run(self, model_path, inputs):
            self.calls.append((model_path, inputs))
            return InferenceResult(outputs=["done"], duration_ms=0.0)

    inner = Recording()
    composite = CompositeGpuExecutor([inner])
    composite.run("model.param", [7, 8])
    assert inner.calls == [("model.param", [7, 8])]


def test_optional_format_execution_preserves_the_legacy_executor_contract():
    class FormatAware:
        backend = "gpu"
        model_formats = frozenset({"ncnn"})

        def __init__(self):
            self.calls = []

        def run_for_format(self, model_format, model_path, inputs):
            self.calls.append(("format", model_format, model_path, inputs))
            return InferenceResult(outputs=[model_format], duration_ms=0.0)

        def run(self, model_path, inputs):
            self.calls.append(("legacy", model_path, inputs))
            return InferenceResult(outputs=["legacy"], duration_ms=0.0)

    class Legacy:
        backend = "gpu"
        model_formats = frozenset({"ncnn"})

        def __init__(self):
            self.calls = []

        def run(self, model_path, inputs):
            self.calls.append((model_path, inputs))
            return InferenceResult(outputs=["done"], duration_ms=0.0)

    aware = FormatAware()
    assert run_executor(
        aware, "misleading.onnx", [1], model_format="ncnn"
    ).outputs == ["ncnn"]
    assert run_executor(aware, "legacy.param", [2]).outputs == ["legacy"]
    assert aware.calls == [
        ("format", "ncnn", "misleading.onnx", [1]),
        ("legacy", "legacy.param", [2]),
    ]

    legacy = Legacy()
    result = run_executor(legacy, "model.param", [3], model_format="ncnn")
    assert result.outputs == ["done"]
    assert legacy.calls == [("model.param", [3])]


class RecordingOpenVino:
    class Core:
        last = None

        available_devices = ["CPU", "NPU"]

        def __init__(self):
            RecordingOpenVino.Core.last = self
            self.compiled_args = None

        def compile_model(self, model_path, device):
            self.compiled_args = (model_path, device)

            def compiled(inputs):
                return {"out": inputs[0]}

            return compiled


def test_npu_run_compiles_for_the_npu_device_exactly():
    executor = NpuExecutor(device_present=True, runtime=RecordingOpenVino())
    result = executor.run("model.xml", [[4, 5]])
    assert RecordingOpenVino.Core.last.compiled_args == ("model.xml", "NPU")
    assert result.outputs == [[4, 5]]
    assert isinstance(result.duration_ms, float) and result.duration_ms >= 0.0


def test_npu_availability_reasons_are_exact():
    assert NpuExecutor(device_present=False).availability() == Availability(
        False, "No /dev/accel NPU device detected", "device-absent",
    )
    missing = NpuExecutor(device_present=True, runtime=None)
    missing._runtime = None
    assert missing.availability() == Availability(
        False, "openvino is not installed", "runtime-missing",
    )


def test_vulkan_availability_reasons_are_exact():
    assert VulkanGpuExecutor(device_present=False).availability() == Availability(
        False, "No GPU render node detected", "device-absent",
    )


class FakeMat(list):
    def clone(self):
        # The real Mat borrows the buffer it was built from, so the executor
        # clones every Mat before ncnn sees it.
        return FakeMat(self)


class FakeExtractor:
    def __init__(self, net):
        self._net = net
        self.inputs = {}
        self.extracted: list[str] = []

    def input(self, name, mat):
        self.inputs[name] = mat

    def extract(self, name):
        self.extracted.append(name)
        assert name in self._net.output_list
        return self._net.extract_code, FakeMat(self.inputs["in0"])


class FakeNet:
    def __init__(self, runtime):
        self.opt = type("Opt", (), {"use_vulkan_compute": False})()
        self._runtime = runtime
        self.loaded: list[str] = []
        self.output_list = ["out0"]
        self.extract_code = 0
        self.extractor = None
        runtime.net = self

    def clear(self):
        """The executor releases the Net's allocators explicitly now."""

    def set_vulkan_device(self, device):
        self._runtime.selected_device = device

    def load_param(self, path):
        self.loaded.append(path)
        return 0 if path.endswith(".param") else -1

    def load_model(self, path):
        self.loaded.append(path)
        return 0 if path.endswith(".bin") else self._runtime.model_load_code

    def input_names(self):
        return ["in0"]

    def output_names(self):
        return self.output_list

    def create_extractor(self):
        self.extractor = FakeExtractor(self)
        return self.extractor


class FakeNcnn:
    Mat = FakeMat

    def __init__(self, device_types):
        self._device_types = device_types
        self.selected_device = None
        self.net = None
        self.model_load_code = 0

    def get_gpu_count(self):
        return len(self._device_types)

    def get_gpu_info(self, index):
        return type("Info", (), {"type": lambda self_: self._device_types[index]})()

    def Net(self):  # noqa: N802 - mirrors the ncnn API
        return FakeNet(self)


DISCRETE, INTEGRATED, VIRTUAL, CPU = 0, 1, 2, 3


def test_vulkan_run_configures_the_net_exactly():
    runtime = FakeNcnn([DISCRETE])
    executor = VulkanGpuExecutor(device_present=True, runtime=runtime)
    result = executor.run("model.param", [FakeMat([1, 2])])
    net = runtime.net
    assert net.opt.use_vulkan_compute is True
    assert runtime.selected_device == 0
    assert net.loaded == ["model.param", "model.bin"]
    assert net.extractor.extracted == ["out0"]
    assert result.outputs == [[1, 2]]
    assert isinstance(result.duration_ms, float) and result.duration_ms >= 0.0


def test_vulkan_run_error_messages_are_exact(tmp_path):
    runtime = FakeNcnn([DISCRETE])
    executor = VulkanGpuExecutor(device_present=True, runtime=runtime)
    with pytest.raises(RuntimeError, match=r"Could not load ncnn param: model\.txt"):
        executor.run("model.txt", [])

    runtime = FakeNcnn([DISCRETE])
    runtime.model_load_code = -1
    executor = VulkanGpuExecutor(device_present=True, runtime=runtime)

    class NoBinNet(FakeNet):
        def load_model(self, path):
            return -1

    runtime.Net = lambda: NoBinNet(runtime)  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match=r"Could not load ncnn model: model\.bin"):
        executor.run("model.param", [])


def test_vulkan_run_rejects_mismatched_inputs_and_failed_extraction():
    runtime = FakeNcnn([DISCRETE])
    executor = VulkanGpuExecutor(device_present=True, runtime=runtime)
    with pytest.raises(ValueError):
        executor.run("model.param", [FakeMat([1]), FakeMat([2])])

    class BadExtractNet(FakeNet):
        def create_extractor(self):
            extractor = super().create_extractor()
            self.extract_code = -100
            return extractor

    runtime = FakeNcnn([DISCRETE])
    runtime.Net = lambda: BadExtractNet(runtime)  # type: ignore[method-assign]
    executor = VulkanGpuExecutor(device_present=True, runtime=runtime)
    with pytest.raises(RuntimeError, match="ncnn extraction failed for output out0"):
        executor.run("model.param", [FakeMat([1])])


def test_vulkan_device_preference_is_exact():
    cases = [
        ([VIRTUAL, INTEGRATED], 1),
        ([INTEGRATED, DISCRETE], 1),
        ([VIRTUAL], 0),
        ([CPU, DISCRETE], 1),
    ]
    for device_types, expected in cases:
        runtime = FakeNcnn(device_types)
        executor = VulkanGpuExecutor(device_present=True, runtime=runtime)
        executor.run("model.param", [FakeMat([0])])
        assert runtime.selected_device == expected, device_types

    software_only = VulkanGpuExecutor(device_present=True, runtime=FakeNcnn([CPU]))
    assert software_only.availability() == Availability(
        False,
        "Only a software (CPU) Vulkan device is present; CPU execution is not used",
        "runtime-unusable",
    )
    none_at_all = VulkanGpuExecutor(device_present=True, runtime=FakeNcnn([]))
    assert none_at_all.availability() == Availability(
        False, "No Vulkan device is available", "runtime-unusable",
    )


def test_vulkan_mat_conversions_are_exact():
    numpy = pytest.importorskip("numpy")
    runtime = FakeNcnn([DISCRETE])
    executor = VulkanGpuExecutor(device_present=True, runtime=runtime)
    # A Mat handed in is cloned like any other: it was built the same way and
    # borrows its memory the same way, so passing it through untouched would
    # keep the use-after-free alive on exactly the path that skips conversion.
    existing = FakeMat([1.0])
    passed = executor._to_mat(existing)
    assert passed is not existing
    assert passed == existing

    class NumpyMat:
        def numpy(self):
            return numpy.array([[1.5, 2.5]], dtype=numpy.float32)

    assert executor._from_mat(NumpyMat()) == [[1.5, 2.5]]
    assert executor._from_mat((3, 4)) == [3, 4]


# --------------------------------------------------------------------------
# service: exact wiring


def test_build_executors_wiring_is_exact():
    from omnitensor.discovery import Device
    from omnitensor.service import build_executors

    devices = [
        Device(id="tpu-pcie-0", backend="tpu", name="t", kind="pcie"),
        Device(id="gpu-renderD128", backend="gpu", name="g", kind="dri"),
    ]
    executors = build_executors(devices)
    assert sorted(executors) == ["gpu", "npu", "tpu"]
    assert executors["tpu"].backend == "tpu"
    assert executors["tpu"]._device_present is True
    assert executors["npu"]._device_present is False
    composite = executors["gpu"]
    assert composite.backend == "gpu"
    assert composite.model_formats == {"ncnn", "onnx"}
    assert [type(sub) for sub in composite._executors] == [VulkanGpuExecutor, GpuExecutor]
    assert all(sub._device_present is True for sub in composite._executors)

    absent = build_executors([])
    assert absent["tpu"]._device_present is False
    assert all(sub._device_present is False for sub in absent["gpu"]._executors)


def test_profile_statuses_documents_are_exact():
    from omnitensor.service import profile_statuses
    from omnitensor.state import PolicyState

    class StatsScheduler:
        def profile_stats(self):
            return {}

    policy = PolicyState()
    manifest = sample_manifest(acceleratorPreference=["tpu"])
    workloads = {manifest["id"]: Workload(id=manifest["id"], manifest=manifest)}
    executors = {"tpu": TpuExecutor(device_present=False)}
    statuses = profile_statuses(workloads, executors, StatsScheduler(), policy)
    assert statuses == {
        "sample-workload": {
            "status": "unavailable",
            "queued": 0,
            "detail": "tpu: No Coral Edge TPU device detected",
            "reason": "device-absent",
        },
    }

    class LongReasonExecutor:
        backend = "tpu"
        model_formats = frozenset({"tflite-edgetpu"})

        def availability(self):
            return Availability(False, "x" * 400)

        def run(self, model_path, inputs):
            raise AssertionError("not executed")

    truncated = profile_statuses(
        workloads, {"tpu": LongReasonExecutor()}, StatsScheduler(), policy,
    )
    assert len(truncated["sample-workload"]["detail"]) == 240

    class ReadyExecutor(LongReasonExecutor):
        def availability(self):
            return Availability(True)

    idle = profile_statuses(workloads, {"tpu": ReadyExecutor()}, StatsScheduler(), policy)
    assert idle == {
        "sample-workload": {
            "status": "idle",
            "queued": 0,
            "detail": "Ready on tpu; no model bundled",
            "reason": "no-model",
        },
    }


def test_device_load_prefers_utilization_then_scheduler_load(tmp_path):
    from omnitensor.service import OmniTensorService

    class NoUtilization:
        def detect(self):
            return []

        def utilization(self, device):
            return None

    workloads_root = tmp_path / "workloads"
    workloads_root.mkdir()
    service = OmniTensorService(
        snapshot_path=tmp_path / "snap.json",
        policy_path=tmp_path / "policy.json",
        workloads_path=workloads_root,
        discovery=NoUtilization(),
    )
    device = device_kwargs()
    assert service._device_load(device, {"loads": {"gpu": 41.0}}) == 41.0
    assert service._device_load(device, {"loads": {}}) is None


def test_build_service_from_env_honours_environment_paths(tmp_path, monkeypatch):
    from omnitensor.service import build_service_from_env

    workloads_root = tmp_path / "workloads"
    write_workload(workloads_root, sample_manifest())
    monkeypatch.setenv("OMNITENSOR_STATE_PATH", str(tmp_path / "snap.json"))
    monkeypatch.setenv("OMNITENSOR_POLICY_PATH", str(tmp_path / "policy.json"))
    monkeypatch.setenv("OMNITENSOR_WORKLOADS", str(workloads_root))
    service = build_service_from_env()
    assert service._publisher_port._path == tmp_path / "snap.json"
    assert service.control._store._path == tmp_path / "policy.json"
    assert "sample-workload" in service._workloads
    assert "hardware-health" in service._workloads


def test_build_service_from_env_defaults_expand_home(monkeypatch):
    from omnitensor.service import build_service_from_env

    for name in ("OMNITENSOR_STATE_PATH", "OMNITENSOR_POLICY_PATH", "OMNITENSOR_WORKLOADS"):
        monkeypatch.delenv(name, raising=False)
    service = build_service_from_env()
    assert service._publisher_port._path == (
        Path.home() / ".local/state/xpu-workload-manager/state.json"
    )
    assert service.control._store._path == Path.home() / ".local/state/omnitensor/policy.json"


def test_env_path_defaults_expand_the_home_directory(monkeypatch):
    from omnitensor.composition import (
        DEFAULT_POLICY_PATH,
        DEFAULT_STATE_PATH,
        DEFAULT_WORKLOADS_PATH,
        _env_path,
    )

    for name, default in (
        ("OMNITENSOR_STATE_PATH", DEFAULT_STATE_PATH),
        ("OMNITENSOR_POLICY_PATH", DEFAULT_POLICY_PATH),
        ("OMNITENSOR_WORKLOADS", DEFAULT_WORKLOADS_PATH),
    ):
        monkeypatch.delenv(name, raising=False)
        resolved = _env_path(name, default)
        assert not str(resolved).startswith("~")
        assert str(resolved).startswith(str(Path.home()))


# --------------------------------------------------------------------------
# atomicio: exact bytes on disk


def test_atomic_write_produces_exact_compact_utf8_bytes(tmp_path):
    from omnitensor.atomicio import write_json_atomic

    target = tmp_path / "doc.json"
    payload = {"key": "väl", "n": 3}
    write_json_atomic(target, payload, prefix=".doc-")
    assert target.read_bytes() == json.dumps(payload, separators=(",", ":")).encode("utf-8")
