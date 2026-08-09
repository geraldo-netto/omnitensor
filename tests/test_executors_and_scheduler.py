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


def test_profile_stats_report_per_profile_queued_and_running():
    import threading

    gate = threading.Event()
    started = threading.Event()

    class BlockingExecutor(SlowExecutor):
        def run(self, model_path, inputs):
            started.set()
            gate.wait(timeout=5)
            return super().run(model_path, inputs)

    async def scenario():
        scheduler = Scheduler(
            {"tpu": BlockingExecutor(), "npu": SlowExecutor()},
            weight_of=lambda _profile: 1,
        )
        scheduler.start()
        in_flight = scheduler.submit("tpu", "profile-a", "job-1", [])
        await asyncio.to_thread(started.wait, 5)
        queued = scheduler.submit("tpu", "profile-a", "job-2", [])
        await scheduler.submit("npu", "profile-b", "job-3", [])
        assert scheduler.profile_stats() == {
            "profile-a": {"queued": 1, "running": 1},
        }
        gate.set()
        await asyncio.gather(in_flight, queued)
        assert scheduler.profile_stats() == {}
        await scheduler.stop()

    asyncio.run(scenario())


def test_profile_stats_sum_queued_jobs_across_backends():
    async def scenario():
        scheduler = Scheduler(
            {"tpu": SlowExecutor(), "npu": SlowExecutor()},
            weight_of=lambda _profile: 1,
        )
        scheduler.submit("tpu", "profile-a", "job-1", [])
        scheduler.submit("npu", "profile-a", "job-2", [])
        scheduler.submit("npu", "profile-b", "job-3", [])
        assert scheduler.profile_stats() == {
            "profile-a": {"queued": 2, "running": 0},
            "profile-b": {"queued": 1, "running": 0},
        }
        await scheduler.stop()

    asyncio.run(scenario())


def test_profile_stats_report_running_with_no_backlog():
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
        future = scheduler.submit("tpu", "profile-a", "job-1", [])
        await asyncio.to_thread(started.wait, 5)
        assert scheduler.profile_stats() == {"profile-a": {"queued": 0, "running": 1}}
        gate.set()
        await future
        await scheduler.stop()

    asyncio.run(scenario())


def test_same_profile_on_two_backends_counts_until_both_finish():
    import threading

    gates = {"tpu": threading.Event(), "npu": threading.Event()}
    started = {"tpu": threading.Event(), "npu": threading.Event()}

    def blocking_executor(backend_name):
        class BlockingExecutor(SlowExecutor):
            backend = backend_name

            def run(self, model_path, inputs):
                started[backend_name].set()
                gates[backend_name].wait(timeout=5)
                return super().run(model_path, inputs)

        return BlockingExecutor()

    async def scenario():
        scheduler = Scheduler(
            {"tpu": blocking_executor("tpu"), "npu": blocking_executor("npu")},
            weight_of=lambda _profile: 1,
        )
        scheduler.start()
        tpu_future = scheduler.submit("tpu", "profile-a", "job-tpu", [])
        npu_future = scheduler.submit("npu", "profile-a", "job-npu", [])
        await asyncio.to_thread(started["tpu"].wait, 5)
        await asyncio.to_thread(started["npu"].wait, 5)
        # One profile running on two backends is one running profile with two
        # in-flight jobs.
        assert scheduler.stats()["runningProfiles"] == 1
        assert scheduler.profile_stats() == {"profile-a": {"queued": 0, "running": 2}}
        gates["tpu"].set()
        await tpu_future
        # Regression (OMNI-0013): finishing the first job used to discard the
        # profile from the running set while the second was still in flight.
        assert scheduler.stats()["runningProfiles"] == 1
        assert scheduler.profile_stats() == {"profile-a": {"queued": 0, "running": 1}}
        gates["npu"].set()
        await npu_future
        assert scheduler.stats()["runningProfiles"] == 0
        assert scheduler.profile_stats() == {}
        await scheduler.stop()

    asyncio.run(scenario())


def test_stride_late_joiner_alternates_instead_of_bursting():
    from omnitensor.scheduler import _BackendQueue, _Job

    queue = _BackendQueue()
    for index in range(20):
        queue.push(_Job("a", f"a-{index}", [], future=None))
    for _ in range(10):
        assert queue.pop_weighted(lambda _p: 1).workload_id == "a"
    for index in range(10):
        queue.push(_Job("b", f"b-{index}", [], future=None))
    served = [queue.pop_weighted(lambda _p: 1).workload_id for _ in range(20)]
    # Regression (OMNI-0008): with equal weights the late joiner used to be
    # served ten times in a row ("catching up" from pass 0).
    assert served == ["a", "b"] * 10


def test_queue_prunes_drained_profiles_and_rejoins_at_virtual_time():
    from omnitensor.scheduler import _BackendQueue, _Job

    queue = _BackendQueue()
    queue.push(_Job("a", "a-0", [], future=None))
    queue.push(_Job("b", "b-0", [], future=None))
    assert queue.pop_weighted(lambda _p: 1) is not None
    assert queue.pop_weighted(lambda _p: 1) is not None
    assert queue.profiles == {}
    assert queue.order == []
    assert queue.passes == {}
    queue.push(_Job("a", "a-1", [], future=None))
    assert queue.passes == {"a": 0.0}


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


def test_npu_ensure_core_constructs_exactly_one_core_across_threads():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    constructed = []
    release = threading.Event()

    class SlowCore:
        available_devices = ["NPU"]

        def __init__(self):
            constructed.append(self)
            release.wait(timeout=2)

    class FakeOpenVino:
        Core = SlowCore

    executor = NpuExecutor(device_present=True, runtime=FakeOpenVino())
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(executor._ensure_core) for _ in range(8)]
        release.set()
        cores = {id(future.result()) for future in futures}
    # Regression (OMNI-0014): concurrent lazy init used to be able to build
    # two openvino.Core objects; the lock serializes construction.
    assert len(constructed) == 1
    assert len(cores) == 1
    assert executor._ensure_core() is constructed[0]


def test_tpu_delegate_error_written_in_worker_thread_is_read_consistently():
    from concurrent.futures import ThreadPoolExecutor

    class FailingDelegateRuntime(FakeTfliteRuntime):
        @staticmethod
        def load_delegate(name):
            raise ValueError("no edgetpu")

    executor = TpuExecutor(device_present=True, runtime=FailingDelegateRuntime())
    assert executor.availability().available is True
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(executor.run, "model.tflite", [[1]])
        with pytest.raises(RuntimeError, match="Could not load libedgetpu.so.1"):
            future.result()
    availability = executor.availability()
    assert availability.available is False
    assert availability.reason == "Could not load libedgetpu.so.1: no edgetpu"


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


def test_update_executors_reaps_finished_retired_workers():
    """Regression (OMNI-0027): retired worker tasks must be pruned once they
    finish cancelling, not accumulate until stop() under backend churn."""
    async def scenario():
        tpu = SlowExecutor()
        npu = SlowExecutor()
        scheduler = Scheduler({"tpu": tpu, "npu": npu}, weight_of=lambda _profile: 1)
        scheduler.start()
        for _round in range(5):
            scheduler.update_executors({"tpu": tpu})
            # Let the event loop process the cancellation so the retired
            # worker completes before the next churn round.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            scheduler.update_executors({"tpu": tpu, "npu": npu})
        assert len(scheduler._retired) <= 1
        await scheduler.submit("npu", "profile-a", "job-npu", [])
        assert npu.served == ["job-npu"]
        await scheduler.stop()
        assert scheduler._retired == []

    asyncio.run(scenario())
