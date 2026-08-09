from __future__ import annotations

import asyncio

import pytest
from conftest import sample_manifest

from omnitensor.executors.base import supports_model
from omnitensor.executors.gpu import GpuExecutor
from omnitensor.executors.npu import NpuExecutor
from omnitensor.executors.tpu import TpuExecutor
from omnitensor.registry import Workload
from omnitensor.scheduler import Scheduler, pick_backend


class FakeTfliteInterpreter:
    def __init__(self, model_path, experimental_delegates):
        self.model_path = model_path
        self.delegates = experimental_delegates
        self.tensors = {}

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [{"index": 0}]

    def get_output_details(self):
        return [{"index": 1}]

    def set_tensor(self, index, value):
        self.tensors[index] = value

    def invoke(self):
        self.tensors[1] = [value * 2 for value in self.tensors[0]]

    def get_tensor(self, index):
        return self.tensors[index]


class FakeTfliteRuntime:
    Interpreter = FakeTfliteInterpreter

    @staticmethod
    def load_delegate(name):
        assert name == "libedgetpu.so.1"
        return object()


class FakeOrtSession:
    def __init__(self, model_path, providers):
        self.providers = providers

    def get_inputs(self):
        class _Input:
            name = "input"
        return [_Input()]

    def run(self, _outputs, feed):
        return [[value + 1 for value in feed["input"]]]


class FakeOrtRuntime:
    InferenceSession = FakeOrtSession

    def __init__(self, providers):
        self._providers = providers

    def get_available_providers(self):
        return self._providers


def workload(**requirement_overrides) -> Workload:
    manifest = sample_manifest(**requirement_overrides)
    return Workload(id=manifest["id"], manifest=manifest)


def test_tpu_executor_runs_real_delegate_path_with_injected_runtime():
    executor = TpuExecutor(device_present=True, runtime=FakeTfliteRuntime())
    result = executor.run("model.tflite", [[1, 2, 3]])
    assert result.outputs == [[2, 4, 6]]
    assert result.duration_ms >= 0


def test_tpu_executor_degrades_without_device_or_runtime():
    assert "No Coral" in TpuExecutor(device_present=False).availability().reason
    missing = TpuExecutor(device_present=True, runtime=None)
    missing._runtime = None
    assert "not installed" in missing.availability().reason
    with pytest.raises(RuntimeError, match="unavailable"):
        missing.run("model.tflite", [[1]])


def test_gpu_executor_requires_gpu_provider_and_never_uses_cpu():
    cpu_only = GpuExecutor(device_present=True, runtime=FakeOrtRuntime(["CPUExecutionProvider"]))
    availability = cpu_only.availability()
    assert availability.available is False
    assert "CPU provider is not used" in availability.reason

    cuda = GpuExecutor(device_present=True, runtime=FakeOrtRuntime(
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
    ))
    assert cuda.availability().available is True
    result = cuda.run("model.onnx", [[1, 2]])
    assert result.outputs == [[2, 3]]


def test_npu_executor_reports_missing_plugin():
    class FakeCore:
        available_devices = ["CPU", "GPU"]

    class FakeOpenVino:
        Core = FakeCore

    executor = NpuExecutor(device_present=True, runtime=FakeOpenVino())
    availability = executor.availability()
    assert availability.available is False
    assert "no NPU plugin" in availability.reason


def test_model_format_compatibility():
    tpu = TpuExecutor(device_present=True, runtime=FakeTfliteRuntime())
    assert supports_model(tpu, None) is True
    assert supports_model(tpu, {"format": "tflite-edgetpu"}) is True
    assert supports_model(tpu, {"format": "onnx"}) is False


def test_pick_backend_follows_preference_and_reports_reasons():
    executors = {
        "tpu": TpuExecutor(device_present=False),
        "gpu": GpuExecutor(device_present=True, runtime=FakeOrtRuntime(["CUDAExecutionProvider"])),
    }
    gpu_capable = workload(
        accelerator="gpu",
        acceleratorPreference=["tpu", "gpu"],
        model={
            "id": "sample-model",
            "version": "1.0.0",
            "format": "onnx",
            "fullyQuantized": False,
            "minimumCompilerVersion": "1",
            "minimumRuntimeVersion": "1",
        },
    )
    backend, reason = pick_backend(gpu_capable, executors)
    assert backend == "gpu"
    assert reason == ""

    tpu_only = workload(acceleratorPreference=["tpu"])
    backend, reason = pick_backend(tpu_only, executors)
    assert backend is None
    assert "No Coral" in reason

    missing_backend = workload(acceleratorPreference=["npu"])
    backend, reason = pick_backend(missing_backend, executors)
    assert backend is None
    assert "npu: no executor" in reason


class SlowExecutor:
    backend = "tpu"
    model_formats = frozenset({"tflite-edgetpu"})

    def __init__(self):
        self.served = []

    def availability(self):
        from omnitensor.executors.base import Availability
        return Availability(True)

    def run(self, model_path, inputs):
        from omnitensor.executors.base import InferenceResult
        self.served.append(model_path)
        return InferenceResult(outputs=[inputs], duration_ms=1.0)


def test_scheduler_serializes_per_device_and_reports_stats():
    async def scenario():
        executor = SlowExecutor()
        scheduler = Scheduler({"tpu": executor}, weight_of=lambda _profile: 1)
        scheduler.start()
        futures = [
            scheduler.submit("tpu", "profile-a", f"job-{index}", [index])
            for index in range(5)
        ]
        results = await asyncio.gather(*futures)
        assert len(results) == 5
        assert sorted(executor.served) == sorted(f"job-{index}" for index in range(5))
        scheduler.tick()
        stats = scheduler.stats()
        assert stats["queueDepth"] == 0
        assert stats["runningProfiles"] == 0
        assert stats["loads"]["tpu"] is not None
        await scheduler.stop()

    asyncio.run(scenario())


def test_scheduler_weighted_shares_favour_heavier_profiles():
    async def scenario():
        executor = SlowExecutor()
        weights = {"heavy": 5, "light": 1}
        scheduler = Scheduler({"tpu": executor}, weight_of=lambda profile: weights[profile])
        futures = []
        for index in range(6):
            futures.append(scheduler.submit("tpu", "light", f"light-{index}", []))
            futures.append(scheduler.submit("tpu", "heavy", f"heavy-{index}", []))
        scheduler.start()
        await asyncio.gather(*futures)
        first_six = executor.served[:6]
        heavy_share = sum(1 for name in first_six if name.startswith("heavy"))
        assert heavy_share >= 4
        await scheduler.stop()

    asyncio.run(scenario())


def test_scheduler_propagates_executor_failures():
    class FailingExecutor(SlowExecutor):
        def run(self, model_path, inputs):
            raise RuntimeError("device wedged")

    async def scenario():
        scheduler = Scheduler({"tpu": FailingExecutor()}, weight_of=lambda _profile: 1)
        scheduler.start()
        with pytest.raises(RuntimeError, match="device wedged"):
            await scheduler.submit("tpu", "profile-a", "job", [])
        await scheduler.stop()

    asyncio.run(scenario())


def test_stop_cancels_in_flight_and_queued_futures_deterministically():
    import threading

    gate = threading.Event()
    started = threading.Event()

    class BlockingExecutor(SlowExecutor):
        def run(self, model_path, inputs):
            started.set()
            gate.wait(timeout=5)
            return super().run(model_path, inputs)

    async def scenario():
        scheduler = Scheduler({"tpu": BlockingExecutor()}, weight_of=lambda _profile: 1)
        scheduler.start()
        f1 = scheduler.submit("tpu", "profile-a", "job-1", [])
        f2 = scheduler.submit("tpu", "profile-a", "job-2", [])
        f3 = scheduler.submit("tpu", "profile-b", "job-3", [])
        await asyncio.to_thread(started.wait, 5)
        await scheduler.stop()
        gate.set()
        # The in-flight job is cancelled, not failed with RuntimeError('').
        assert f1.cancelled()
        # Queued jobs no longer hang forever: their futures are cancelled too.
        assert f2.cancelled()
        assert f3.cancelled()
        for future in (f1, f2, f3):
            with pytest.raises(asyncio.CancelledError):
                await future
        assert scheduler.stats()["queueDepth"] == 0

    asyncio.run(scenario())


def test_submit_after_stop_raises_instead_of_queueing_forever():
    async def scenario():
        scheduler = Scheduler({"tpu": SlowExecutor()}, weight_of=lambda _profile: 1)
        scheduler.start()
        await scheduler.stop()
        with pytest.raises(RuntimeError, match="stopped"):
            scheduler.submit("tpu", "profile-a", "job", [])

    asyncio.run(scenario())


def test_submit_after_stop_error_message_is_exact():
    async def scenario():
        scheduler = Scheduler({"tpu": SlowExecutor()}, weight_of=lambda _profile: 1)
        await scheduler.stop()
        with pytest.raises(RuntimeError) as excinfo:
            scheduler.submit("tpu", "profile-a", "job", [])
        assert str(excinfo.value) == "scheduler is stopped"

    asyncio.run(scenario())


def test_worker_survives_executor_failure_and_serves_next_job():
    class FlakyExecutor(SlowExecutor):
        def run(self, model_path, inputs):
            if model_path == "bad":
                raise RuntimeError("device wedged")
            return super().run(model_path, inputs)

    async def scenario():
        scheduler = Scheduler({"tpu": FlakyExecutor()}, weight_of=lambda _profile: 1)
        scheduler.start()
        with pytest.raises(RuntimeError, match="device wedged"):
            await scheduler.submit("tpu", "profile-a", "bad", [])
        # A plain executor failure must not kill the worker.
        result = await asyncio.wait_for(
            scheduler.submit("tpu", "profile-a", "good", []), timeout=2,
        )
        assert result.outputs == [[]]
        await scheduler.stop()

    asyncio.run(scenario())


def test_non_exception_base_exception_fails_future_with_exact_repr():
    class Abort(BaseException):
        def __repr__(self):
            return "Abort(sentinel)"

    class AbortingExecutor(SlowExecutor):
        def run(self, model_path, inputs):
            raise Abort()

    async def scenario():
        scheduler = Scheduler({"tpu": AbortingExecutor()}, weight_of=lambda _profile: 1)
        scheduler.start()
        future = scheduler.submit("tpu", "profile-a", "job", [])
        with pytest.raises(RuntimeError) as excinfo:
            await future
        assert str(excinfo.value) == "Abort(sentinel)"
        await scheduler.stop()

    asyncio.run(scenario())


def test_stop_cancels_queued_futures_even_when_never_started():
    async def scenario():
        scheduler = Scheduler({"tpu": SlowExecutor()}, weight_of=lambda _profile: 1)
        future = scheduler.submit("tpu", "profile-a", "job", [])
        await scheduler.stop()
        assert future.cancelled()

    asyncio.run(scenario())


def test_update_executors_dispatches_on_the_replacement_executor():
    async def scenario():
        old = SlowExecutor()
        new = SlowExecutor()
        scheduler = Scheduler({"tpu": old}, weight_of=lambda _profile: 1)
        scheduler.start()
        await scheduler.submit("tpu", "profile-a", "job-old", [])
        scheduler.update_executors({"tpu": new})
        await scheduler.submit("tpu", "profile-a", "job-new", [])
        assert old.served == ["job-old"]
        assert new.served == ["job-new"]
        await scheduler.stop()

    asyncio.run(scenario())


def test_update_executors_spawns_a_worker_for_a_new_backend():
    async def scenario():
        tpu = SlowExecutor()
        npu = SlowExecutor()
        scheduler = Scheduler({"tpu": tpu}, weight_of=lambda _profile: 1)
        scheduler.start()
        scheduler.update_executors({"tpu": tpu, "npu": npu})
        await scheduler.submit("npu", "profile-a", "job-npu", [])
        assert npu.served == ["job-npu"]
        assert scheduler.stats()["loads"].keys() == {"tpu", "npu"}
        await scheduler.stop()

    asyncio.run(scenario())


def test_update_executors_cancels_queued_jobs_of_removed_backends():
    async def scenario():
        tpu = SlowExecutor()
        npu = SlowExecutor()
        scheduler = Scheduler({"tpu": tpu, "npu": npu}, weight_of=lambda _profile: 1)
        future = scheduler.submit("npu", "profile-a", "job-npu", [])
        scheduler.update_executors({"tpu": tpu})
        assert future.cancelled()
        assert "npu" not in scheduler.stats()["loads"]
        scheduler.start()
        await scheduler.submit("tpu", "profile-a", "job-tpu", [])
        assert tpu.served == ["job-tpu"]
        await scheduler.stop()
        assert scheduler._retired == []

    asyncio.run(scenario())


def test_scheduler_holds_non_admitted_jobs_and_resumes_on_kick():
    async def scenario():
        executor = SlowExecutor()
        held = {"profile-a"}
        scheduler = Scheduler(
            {"tpu": executor},
            weight_of=lambda _profile: 1,
            admits=lambda profile_id: profile_id not in held,
        )
        scheduler.start()
        held_future = scheduler.submit("tpu", "profile-a", "held-job", [])
        await scheduler.submit("tpu", "profile-b", "free-job", [])
        await asyncio.sleep(0.01)
        # The held profile's job stays queued, only the admitted one ran.
        assert executor.served == ["free-job"]
        assert scheduler.stats()["queueDepth"] == 1
        assert not held_future.done()
        held.clear()
        scheduler.kick()
        await asyncio.wait_for(held_future, timeout=2)
        assert executor.served == ["free-job", "held-job"]
        assert scheduler.stats()["queueDepth"] == 0
        await scheduler.stop()

    asyncio.run(scenario())


def test_update_executors_retires_the_worker_of_a_removed_backend():
    async def scenario():
        tpu = SlowExecutor()
        npu = SlowExecutor()
        scheduler = Scheduler({"tpu": tpu, "npu": npu}, weight_of=lambda _profile: 1)
        scheduler.start()
        npu_worker = scheduler._workers["npu"]
        scheduler.update_executors({"tpu": tpu})
        assert scheduler._workers.keys() == {"tpu"}
        assert scheduler._wakeups.keys() == {"tpu"}
        assert scheduler._retired == [npu_worker]
        await scheduler.stop()
        assert npu_worker.cancelled()
        assert scheduler._retired == []

    asyncio.run(scenario())


def test_npu_executor_runs_via_openvino_when_plugin_present():
    class FakeCompiled:
        def __call__(self, inputs):
            return {"output": [value * 3 for value in inputs[0]]}

    class FakeCore:
        available_devices = ["CPU", "NPU"]

        def compile_model(self, model_path, device):
            assert device == "NPU"
            return FakeCompiled()

    class FakeOpenVino:
        Core = FakeCore

    executor = NpuExecutor(device_present=True, runtime=FakeOpenVino())
    assert executor.availability().available is True
    result = executor.run("model.xml", [[1, 2]])
    assert result.outputs == [[3, 6]]


def test_npu_executor_degrades_without_device_or_runtime():
    assert "No /dev/accel" in NpuExecutor(device_present=False).availability().reason
    missing = NpuExecutor(device_present=True, runtime=None)
    missing._runtime = None
    assert "not installed" in missing.availability().reason


def test_npu_executor_reports_discovery_failure():
    class ExplodingCore:
        def __init__(self):
            raise OSError("plugin registry corrupt")

    class FakeOpenVino:
        Core = ExplodingCore

    executor = NpuExecutor(device_present=True, runtime=FakeOpenVino())
    availability = executor.availability()
    assert availability.available is False
    assert "discovery failed" in availability.reason
