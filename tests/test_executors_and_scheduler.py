from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from conftest import sample_manifest

from omnitensor.executors.base import (
    DEVICE_ABSENT,
    FORMAT_UNSUPPORTED,
    Availability,
    InferenceResult,
    ModelCache,
    ModelStore,
    release_runtime_value,
    supports_model,
)
from omnitensor.executors.gpu import CompositeGpuExecutor, GpuExecutor
from omnitensor.executors.npu import NpuExecutor
from omnitensor.executors.tpu import TpuExecutor
from omnitensor.registry import Workload
from omnitensor.scheduler import (
    NO_EXECUTOR,
    NO_PREFERENCE,
    QueueFullError,
    Scheduler,
    _BackendQueue,
    _Job,
    select_backend,
)


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


def test_tpu_executor_reuses_one_interpreter_per_model(tmp_path):
    class CountingRuntime:
        def __init__(self):
            self.delegate_loads = 0
            self.interpreters = []

        def load_delegate(self, name):
            assert name == "libedgetpu.so.1"
            self.delegate_loads += 1
            return object()

        def Interpreter(self, model_path, experimental_delegates):  # noqa: N802
            interpreter = FakeTfliteInterpreter(model_path, experimental_delegates)
            self.interpreters.append(interpreter)
            return interpreter

    runtime = CountingRuntime()
    executor = TpuExecutor(device_present=True, runtime=runtime)
    alpha = _model_file(tmp_path, "alpha.tflite")
    beta = _model_file(tmp_path, "beta.tflite")
    assert executor.run(alpha, [[1]]).outputs == [[2]]
    assert executor.run(alpha, [[3]]).outputs == [[6]]
    assert executor.run(beta, [[4]]).outputs == [[8]]
    assert runtime.delegate_loads == 2
    assert [item.model_path for item in runtime.interpreters] == [alpha, beta]


def test_tpu_executor_rejects_wrong_input_count():
    executor = TpuExecutor(device_present=True, runtime=FakeTfliteRuntime())
    with pytest.raises(ValueError, match=r"zip\(\) argument 2 is longer"):
        executor.run("model.tflite", [[1], [2]])


def test_tpu_executor_reports_inference_duration_in_milliseconds():
    ticks = iter([10.0, 10.25])
    result = TpuExecutor(
        device_present=True, runtime=FakeTfliteRuntime(), clock=lambda: next(ticks)
    ).run("model.tflite", [[1]])
    assert result.duration_ms == 250.0


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

    cuda = GpuExecutor(
        device_present=True,
        runtime=FakeOrtRuntime(
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
    )
    assert cuda.availability().available is True
    result = cuda.run("model.onnx", [[1, 2]])
    assert result.outputs == [[2, 3]]


class CountingOrtRuntime:
    def __init__(self):
        self.sessions = []

    @staticmethod
    def get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def InferenceSession(self, model_path, providers):  # noqa: N802
        self.sessions.append((model_path, tuple(providers)))
        return FakeOrtSession(model_path, providers)


def test_gpu_executor_reuses_one_session_per_model_revision(tmp_path):
    runtime = CountingOrtRuntime()
    executor = GpuExecutor(device_present=True, runtime=runtime)
    model = _model_file(tmp_path, "model.onnx")

    assert executor.run(model, [[1]]).outputs == [[2]]
    assert executor.run(model, [[2]]).outputs == [[3]]

    assert runtime.sessions == [(model, ("CUDAExecutionProvider",))]


def test_gpu_executor_rebuilds_a_replaced_model_session(tmp_path):
    runtime = CountingOrtRuntime()
    executor = GpuExecutor(device_present=True, runtime=runtime)
    model = _model_file(tmp_path, "model.onnx")

    executor.run(model, [[1]])
    Path(model).write_bytes(b"replacement model with another revision")
    executor.run(model, [[1]])

    assert runtime.sessions == [
        (model, ("CUDAExecutionProvider",)),
        (model, ("CUDAExecutionProvider",)),
    ]


def test_gpu_executor_evicts_the_least_recently_used_session(tmp_path):
    runtime = CountingOrtRuntime()
    executor = GpuExecutor(device_present=True, runtime=runtime, max_cached_models=2)
    alpha = _model_file(tmp_path, "alpha.onnx")
    beta = _model_file(tmp_path, "beta.onnx")
    gamma = _model_file(tmp_path, "gamma.onnx")

    for model in (alpha, beta, alpha, gamma, beta):
        executor.run(model, [[1]])

    assert [model for model, _providers in runtime.sessions] == [
        alpha,
        beta,
        gamma,
        beta,
    ]
    assert executor._models.paths() == (gamma, beta)


def test_gpu_executor_rejects_wrong_input_count():
    executor = GpuExecutor(device_present=True, runtime=CountingOrtRuntime())

    with pytest.raises(ValueError, match=r"zip\(\) argument 2 is longer"):
        executor.run("model.onnx", [[1], [2]])


def test_gpu_executor_reports_inference_duration_in_milliseconds():
    ticks = iter([10.0, 10.25])

    result = GpuExecutor(
        device_present=True, runtime=CountingOrtRuntime(), clock=lambda: next(ticks)
    ).run("model.onnx", [[1]])

    assert result.duration_ms == 250.0


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


def test_select_backend_follows_preference_and_reports_reasons_and_codes():
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
    choice = select_backend(gpu_capable, executors)
    assert (choice.backend, choice.reason, choice.code) == ("gpu", "", "")

    tpu_only = workload(acceleratorPreference=["tpu"])
    choice = select_backend(tpu_only, executors)
    assert choice.backend is None
    assert "No Coral" in choice.reason
    assert choice.code == DEVICE_ABSENT

    missing_backend = workload(acceleratorPreference=["npu"])
    choice = select_backend(missing_backend, executors)
    assert choice.backend is None
    assert "npu: no executor" in choice.reason
    assert choice.code == NO_EXECUTOR


def test_select_backend_skips_gpu_when_matching_runtime_lane_is_unavailable():
    class Lane:
        backend = "gpu"

        def __init__(self, formats, available, reason=""):
            self.model_formats = frozenset(formats)
            self._availability = Availability(available, reason)

        def availability(self):
            return self._availability

    gpu = CompositeGpuExecutor(
        [
            Lane({"ncnn"}, True),
            Lane({"onnx"}, False, "No CUDA or ROCm provider"),
        ]
    )
    npu = Lane({"onnx"}, True)
    manifest = sample_manifest(
        accelerator="gpu",
        acceleratorPreference=["gpu", "npu"],
        model={
            "id": "sample-model",
            "version": "1.0.0",
            "format": "onnx",
            "fullyQuantized": False,
            "minimumCompilerVersion": "1",
            "minimumRuntimeVersion": "1",
        },
    )
    choice = select_backend(
        Workload(id=manifest["id"], manifest=manifest),
        {
            "gpu": gpu,
            "npu": npu,
        },
    )
    assert (choice.backend, choice.reason, choice.code) == ("npu", "", "")

    choice = select_backend(
        Workload(id=manifest["id"], manifest=manifest),
        {"npu": npu},
    )
    assert (choice.backend, choice.reason, choice.code) == ("npu", "", "")

    incompatible_gpu = Lane({"ncnn"}, True)
    choice = select_backend(
        Workload(id=manifest["id"], manifest=manifest),
        {
            "gpu": incompatible_gpu,
            "npu": npu,
        },
    )
    assert (choice.backend, choice.reason, choice.code) == ("npu", "", "")
    choice = select_backend(
        Workload(id=manifest["id"], manifest=manifest),
        {"gpu": incompatible_gpu},
    )
    assert choice.backend is None
    assert choice.reason == "gpu: model format not supported; npu: no executor"
    assert choice.code == FORMAT_UNSUPPORTED

    class NoPreference:
        preference = ()

    choice = select_backend(NoPreference(), {})
    assert (choice.backend, choice.reason, choice.code) == (
        None,
        "No backend in preference list",
        NO_PREFERENCE,
    )


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


async def wait_for_thread_event(event, timeout=5.0):
    async with asyncio.timeout(timeout):
        while not event.is_set():
            await asyncio.sleep(0.001)


def test_scheduler_serializes_per_device_and_reports_stats():
    async def scenario():
        executor = SlowExecutor()
        scheduler = Scheduler({"tpu": executor}, weight_of=lambda _profile: 1)
        scheduler.start()
        futures = [
            scheduler.submit("tpu", "profile-a", f"job-{index}", [index]) for index in range(5)
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


def test_scheduler_executor_uses_dedicated_off_loop_thread(monkeypatch):
    import threading

    observed = []

    class ThreadRecordingExecutor(SlowExecutor):
        def run(self, model_path, inputs):
            observed.append(threading.current_thread())
            return super().run(model_path, inputs)

    async def forbidden_shared_executor(*_args, **_kwargs):
        pytest.fail("scheduler used asyncio's shared executor")

    monkeypatch.setattr(asyncio, "to_thread", forbidden_shared_executor)

    async def scenario():
        loop_thread = threading.get_ident()
        scheduler = Scheduler({"tpu": ThreadRecordingExecutor()}, weight_of=lambda _profile: 1)
        scheduler.start()
        result = await asyncio.wait_for(scheduler.submit("tpu", "profile-a", "job", [1]), timeout=2)
        await scheduler.stop()
        return loop_thread, result

    loop_thread, result = asyncio.run(scenario())

    assert result.outputs == [[1]]
    assert len(observed) == 1
    assert observed[0].ident != loop_thread
    assert observed[0].daemon is True
    assert observed[0].name == "omnitensor-plugin-io"


def test_scheduler_forwards_the_declared_format_through_the_queued_job():
    class FormatAwareRecording(SlowExecutor):
        backend = "gpu"
        model_formats = frozenset({"ncnn"})

        def __init__(self):
            super().__init__()
            self.format_calls = []

        def run_for_format(self, model_format, model_path, inputs):
            self.format_calls.append((model_format, model_path, inputs))
            return InferenceResult(outputs=[inputs], duration_ms=1.0)

        def run(self, model_path, inputs):  # pragma: no cover - must not dispatch here
            raise AssertionError("declared format was discarded")

    async def scenario():
        executor = FormatAwareRecording()
        scheduler = Scheduler({"gpu": executor}, weight_of=lambda _profile: 1)
        future = scheduler.submit(
            "gpu",
            "profile-a",
            "misleading.onnx",
            [1, 2],
            model_format="ncnn",
        )
        queued = scheduler._queues["gpu"].profiles["profile-a"][0]
        assert queued.model_format == "ncnn"
        scheduler.start()
        result = await future
        await scheduler.stop()
        return executor, result

    executor, result = asyncio.run(scenario())
    assert executor.format_calls == [("ncnn", "misleading.onnx", [1, 2])]
    assert result.outputs == [[1, 2]]


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
        await wait_for_thread_event(started)
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


def test_submit_rejects_a_full_backend_queue_without_allocating_more_work():
    async def scenario():
        scheduler = Scheduler(
            {"tpu": SlowExecutor()},
            weight_of=lambda _profile: 1,
            max_backend_queue_depth=2,
        )
        accepted = [scheduler.submit("tpu", "profile-a", f"job-{index}", []) for index in range(2)]
        with pytest.raises(QueueFullError) as excinfo:
            scheduler.submit("tpu", "profile-a", "rejected", [])

        assert str(excinfo.value) == "tpu queue is full (2 jobs)"
        assert scheduler.stats()["queueDepth"] == 2
        assert scheduler.profile_stats() == {
            "profile-a": {"queued": 2, "running": 0},
        }
        await scheduler.stop()
        assert all(future.cancelled() for future in accepted)

    asyncio.run(scenario())


def test_scheduler_rejects_non_positive_backend_queue_limits():
    with pytest.raises(ValueError) as excinfo:
        Scheduler(
            {"tpu": SlowExecutor()},
            weight_of=lambda _profile: 1,
            max_backend_queue_depth=0,
        )
    assert str(excinfo.value) == "max_backend_queue_depth must be positive"

    def weight_of(_profile):
        return 1

    scheduler = Scheduler(
        {"tpu": SlowExecutor()},
        weight_of=weight_of,
        max_backend_queue_depth=1,
    )
    assert scheduler._weight_of is weight_of
    assert scheduler._max_backend_queue_depth == 1


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
            scheduler.submit("tpu", "profile-a", "good", []),
            timeout=2,
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
        await wait_for_thread_event(started)
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
        await wait_for_thread_event(started)
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
        await wait_for_thread_event(started["tpu"])
        await wait_for_thread_event(started["npu"])
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


def test_npu_executor_reuses_one_compiled_model_per_path(tmp_path):
    class FakeCompiled:
        def __call__(self, inputs):
            return {"output": inputs[0]}

    class CountingCore:
        available_devices = ["NPU"]

        def __init__(self):
            self.compiled_paths = []

        def compile_model(self, model_path, device):
            assert device == "NPU"
            self.compiled_paths.append(model_path)
            return FakeCompiled()

    class FakeOpenVino:
        Core = CountingCore

    executor = NpuExecutor(device_present=True, runtime=FakeOpenVino())
    alpha = _model_file(tmp_path, "alpha.xml")
    beta = _model_file(tmp_path, "beta.xml")
    assert executor.run(alpha, [[1]]).outputs == [[1]]
    assert executor.run(alpha, [[2]]).outputs == [[2]]
    assert executor.run(beta, [[3]]).outputs == [[3]]
    assert executor._core.compiled_paths == [alpha, beta]


def test_npu_executor_reports_inference_duration_in_milliseconds():
    class Compiled:
        def __call__(self, inputs):
            return {"output": inputs[0]}

    class Core:
        available_devices = ["NPU"]

        def compile_model(self, model_path, device):
            return Compiled()

    class Runtime:
        pass

    Runtime.Core = Core

    ticks = iter([10.0, 10.25])
    result = NpuExecutor(device_present=True, runtime=Runtime(), clock=lambda: next(ticks)).run(
        "model.xml", [[1]]
    )
    assert result.duration_ms == 250.0


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


def test_backend_queue_initial_state_and_exact_bounded_stride():
    queue = _BackendQueue()
    assert queue.busy_ms == 0.0
    assert queue.load is None
    queue.push(_Job("alpha", "first", [], future=None))
    queue.push(_Job("alpha", "second", [], future=None))

    def oversized_weight(profile_id):
        assert profile_id == "alpha"
        return 6

    assert queue.pop_weighted(oversized_weight).model_path == "first"
    assert queue.passes == {"alpha": 0.2}


def test_a_profile_held_out_by_policy_does_not_burst_when_re_enabled():
    """Held out, a profile used to bank credit for service it was never
    eligible for, then monopolize the device catching up."""
    queue = _BackendQueue()
    for index in range(10):
        queue.push(_Job("held", f"held-{index}", [], future=None))
        queue.push(_Job("served", f"served-{index}", [], future=None))

    for _ in range(8):
        assert queue.pop_weighted(lambda _p: 1, lambda p: p == "served").workload_id == ("served")

    served = [queue.pop_weighted(lambda _p: 1).workload_id for _ in range(4)]

    assert served == ["held", "served", "held", "served"]


def test_holding_out_a_profile_does_not_starve_it_either():
    """Clamping raises a stale pass to virtual time and never past it, so a
    re-admitted profile is served next rather than pushed to the back."""
    queue = _BackendQueue()
    queue.push(_Job("held", "held-0", [], future=None))
    queue.push(_Job("served", "served-0", [], future=None))
    queue.push(_Job("served", "served-1", [], future=None))

    queue.pop_weighted(lambda _p: 1, lambda p: p == "served")

    assert queue.passes["held"] == queue.passes["served"]
    assert queue.pop_weighted(lambda _p: 1).workload_id == "held"


def test_clamping_leaves_an_ordinary_unserved_profile_alone():
    """Not yet served is not the same as held out; stride must still apply."""
    queue = _BackendQueue()
    for index in range(4):
        queue.push(_Job("a", f"a-{index}", [], future=None))
        queue.push(_Job("b", f"b-{index}", [], future=None))

    served = [queue.pop_weighted(lambda _p: 1).workload_id for _ in range(4)]

    assert served == ["a", "b", "a", "b"]


def test_a_backend_failing_outside_execution_fails_its_queued_jobs():
    """The worker task used to die silently, leaving futures pending forever."""

    async def scenario():
        scheduler = Scheduler({"tpu": SlowExecutor()}, _exploding_weight_of())
        scheduler.start()
        first = scheduler.submit("tpu", "sample-workload", "job-1", [])
        second = scheduler.submit("tpu", "sample-workload", "job-2", [])

        with pytest.raises(RuntimeError, match="tpu scheduling failed"):
            await asyncio.wait_for(first, timeout=2)
        with pytest.raises(RuntimeError, match="tpu scheduling failed"):
            await asyncio.wait_for(second, timeout=2)

        assert scheduler.degraded_backends() == {"tpu": "ZeroDivisionError"}
        assert not scheduler._workers["tpu"].done(), "the worker task died"
        await scheduler.stop()

    asyncio.run(scenario())


def _exploding_weight_of():
    def weight_of(_profile_id):
        raise ZeroDivisionError("private")

    return weight_of


def test_a_healthy_backend_reports_no_degradation():
    async def scenario():
        scheduler = Scheduler({"tpu": SlowExecutor()}, lambda _p: 1)
        scheduler.start()
        await asyncio.wait_for(scheduler.submit("tpu", "sample-workload", "job-1", []), 2)
        assert scheduler.degraded_backends() == {}
        await scheduler.stop()

    asyncio.run(scenario())


def test_a_successful_job_clears_a_previous_scheduler_degradation():
    async def scenario():
        explode = [True]

        def weight_of(_profile_id):
            if explode[0]:
                raise ZeroDivisionError("private")
            return 1

        scheduler = Scheduler({"tpu": SlowExecutor()}, weight_of)
        scheduler.start()
        failed = scheduler.submit("tpu", "sample-workload", "job-1", [])
        with pytest.raises(RuntimeError, match="tpu scheduling failed"):
            await asyncio.wait_for(failed, timeout=2)
        assert scheduler.degraded_backends() == {"tpu": "ZeroDivisionError"}

        explode[0] = False
        recovered = scheduler.submit("tpu", "sample-workload", "job-2", [])
        await asyncio.wait_for(recovered, timeout=2)
        assert scheduler.degraded_backends() == {}
        await scheduler.stop()

    asyncio.run(scenario())


def test_removing_a_backend_discards_its_stale_degradation_before_readd():
    scheduler = Scheduler({"tpu": SlowExecutor()}, lambda _profile: 1)
    scheduler._degraded["tpu"] = "RuntimeError"

    scheduler.update_executors({})
    assert scheduler.degraded_backends() == {}

    scheduler.update_executors({"tpu": SlowExecutor()})
    assert scheduler.degraded_backends() == {}


def test_clearing_an_unknown_backend_is_idempotent():
    scheduler = Scheduler({}, lambda _profile: 1)

    scheduler._clear_degradation("missing")

    assert scheduler.degraded_backends() == {}


class _FlakyDelegateRuntime:
    """A delegate that fails a fixed number of times, then loads."""

    def __init__(self, failures):
        self.remaining_failures = failures
        self.delegate_loads = 0

    def load_delegate(self, name):
        assert name == "libedgetpu.so.1"
        if self.remaining_failures:
            self.remaining_failures -= 1
            raise OSError("device is busy")
        self.delegate_loads += 1
        return object()

    def Interpreter(self, model_path, experimental_delegates):  # noqa: N802
        return FakeTfliteInterpreter(model_path, experimental_delegates)


def test_a_transient_delegate_failure_does_not_disable_the_backend_forever():
    """The failure used to latch until the device set changed."""
    now = [100.0]
    runtime = _FlakyDelegateRuntime(failures=1)
    executor = TpuExecutor(
        device_present=True,
        runtime=runtime,
        delegate_retry_seconds=30.0,
        clock=lambda: now[0],
    )

    with pytest.raises(RuntimeError, match="Could not load"):
        executor.run("model.tflite", [[1]])
    assert executor.availability().available is False

    now[0] += 30.0
    assert executor.availability().available is True
    assert executor.run("model.tflite", [[1]]).outputs == [[2]]
    assert runtime.delegate_loads == 1


def test_a_delegate_failure_is_reported_within_its_retry_window():
    now = [100.0]
    executor = TpuExecutor(
        device_present=True,
        runtime=_FlakyDelegateRuntime(failures=5),
        delegate_retry_seconds=30.0,
        clock=lambda: now[0],
    )

    with pytest.raises(RuntimeError):
        executor.run("model.tflite", [[1]])

    now[0] += 29.0
    availability = executor.availability()
    assert availability.available is False
    assert "Could not load libedgetpu.so.1" in availability.reason


def test_a_successful_delegate_load_clears_an_earlier_failure():
    now = [100.0]
    executor = TpuExecutor(
        device_present=True,
        runtime=_FlakyDelegateRuntime(failures=1),
        delegate_retry_seconds=0.0,
        clock=lambda: now[0],
    )

    with pytest.raises(RuntimeError):
        executor.run("model.tflite", [[1]])
    executor.run("model.tflite", [[1]])

    assert executor._delegate_error is None
    assert executor.availability().available is True


def _model_file(directory, name, content=b"model"):
    path = directory / name
    path.write_bytes(content)
    return str(path)


def test_a_replaced_model_file_is_not_served_from_cache(tmp_path):
    """Keyed on the path alone, the cache kept serving a retired model."""

    class CountingRuntime:
        def __init__(self):
            self.builds = 0

        def load_delegate(self, name):
            return object()

        def Interpreter(self, model_path, experimental_delegates):  # noqa: N802
            self.builds += 1
            return FakeTfliteInterpreter(model_path, experimental_delegates)

    runtime = CountingRuntime()
    executor = TpuExecutor(device_present=True, runtime=runtime)
    model = _model_file(tmp_path, "model.tflite")

    executor.run(model, [[1]])
    executor.run(model, [[1]])
    assert runtime.builds == 1

    os.utime(model, ns=(0, 0))
    executor.run(model, [[1]])
    assert runtime.builds == 2, "the replaced model was served from cache"


def test_the_model_cache_is_bounded_and_evicts_least_recently_used(tmp_path):
    cache = ModelCache(max_entries=2)
    paths = [_model_file(tmp_path, f"m{index}.tflite") for index in range(3)]

    for path in paths[:2]:
        cache.get_or_build(path, lambda: object())
    cache.get_or_build(paths[0], lambda: pytest.fail("cached entry was rebuilt"))
    cache.get_or_build(paths[2], lambda: object())

    assert len(cache) == 2
    assert cache.paths() == (paths[0], paths[2])


def test_a_model_that_cannot_be_stated_is_never_cached(tmp_path):
    cache = ModelCache()
    builds = []

    for _ in range(3):
        cache.get_or_build(str(tmp_path / "absent.tflite"), lambda: builds.append(1))

    assert len(builds) == 3
    assert len(cache) == 0


def test_a_changed_companion_invalidates_its_cached_model(tmp_path):
    cache = ModelCache()
    model = _model_file(tmp_path, "model.param")
    companion = _model_file(tmp_path, "model.bin")
    builds = []

    for _ in range(2):
        cache.get_or_build(
            model,
            lambda: builds.append(1),
            companion_paths=(companion,),
        )
    Path(companion).write_bytes(b"replacement with a different size")
    cache.get_or_build(
        model,
        lambda: builds.append(1),
        companion_paths=(companion,),
    )

    assert len(builds) == 2


def test_an_evicted_entry_is_released_rather_than_left_to_the_collector(tmp_path):
    released = []
    cache = ModelCache(max_entries=1, release=released.append)
    first = _model_file(tmp_path, "first.tflite")
    second = _model_file(tmp_path, "second.tflite")

    cache.get_or_build(first, lambda: "first-session")
    cache.get_or_build(second, lambda: "second-session")

    assert released == ["first-session"]


def test_a_rebuilt_entry_releases_the_runtime_state_it_replaces(tmp_path):
    released = []
    cache = ModelCache(release=released.append)
    model = _model_file(tmp_path, "model.tflite")

    cache.get_or_build(model, lambda: "old-session")
    os.utime(model, ns=(0, 0))
    cache.get_or_build(model, lambda: "new-session")

    assert released == ["old-session"]


def test_clearing_the_cache_releases_every_entry(tmp_path):
    released = []
    cache = ModelCache(release=released.append)
    for index in range(2):
        cache.get_or_build(_model_file(tmp_path, f"m{index}.tflite"), lambda: "session")

    cache.clear()

    assert len(released) == 2
    assert len(cache) == 0


def test_a_runtime_object_is_released_by_whichever_method_it_offers():
    class Ncnn:
        def __init__(self):
            self.cleared = False

        def clear(self):
            self.cleared = True

    class Exploding:
        def close(self):
            raise RuntimeError("driver is unhappy")

    net = Ncnn()
    release_runtime_value(net)
    assert net.cleared
    release_runtime_value(object())  # nothing to call is not an error
    release_runtime_value(Exploding())  # nor is a library that fails to tear down


def test_a_store_releases_at_once_when_no_job_is_running(tmp_path):
    released = []
    store = ModelStore(max_entries=1, release=released.append)
    store.get_or_build(_model_file(tmp_path, "a.tflite"), lambda: "a")
    store.get_or_build(_model_file(tmp_path, "b.tflite"), lambda: "b")

    assert released == ["a"]


def test_a_store_never_releases_under_a_running_job(tmp_path):
    released = []
    store = ModelStore(max_entries=1, release=released.append)
    first = _model_file(tmp_path, "a.tflite")

    with store.running():
        store.get_or_build(first, lambda: "a")
        store.get_or_build(_model_file(tmp_path, "b.tflite"), lambda: "b")
        store.close()
        assert released == [], "runtime state was torn down under a running job"

    assert sorted(released) == ["a", "b"]


def test_closing_a_store_twice_releases_once(tmp_path):
    released = []
    store = ModelStore(release=released.append)
    store.get_or_build(_model_file(tmp_path, "a.tflite"), lambda: "a")

    store.close()
    store.close()

    assert released == ["a"]


def test_closing_an_executor_releases_its_cached_runtime_state(tmp_path):
    executor = TpuExecutor(device_present=True, runtime=FakeTfliteRuntime())
    model = _model_file(tmp_path, "model.tflite")
    executor.run(model, [[1]])

    executor.close()
    executor.close()

    assert executor._models.paths() == ()


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "2"])
def test_the_model_cache_bound_is_a_positive_integer(bad):
    with pytest.raises(ValueError, match="max_entries"):
        ModelCache(bad)


def test_every_lane_measures_the_same_thing_as_duration_ms():
    """The four lanes each timed something different and called it all
    duration_ms, so the number was not comparable across backends.  One
    definition now: the window opens at the runtime call that executes the
    graph and closes once its outputs are in hand -- never around the model
    build, and never around marshalling the inputs."""
    log = []

    class LoggingInterpreter(FakeTfliteInterpreter):
        def __init__(self, model_path, experimental_delegates):
            log.append("build")
            super().__init__(model_path, experimental_delegates)

        def set_tensor(self, index, value):
            log.append("marshal-input")
            super().set_tensor(index, value)

        def invoke(self):
            log.append("execute")
            super().invoke()

        def get_tensor(self, index):
            log.append("read-output")
            return super().get_tensor(index)

    class LoggingTflite(FakeTfliteRuntime):
        Interpreter = LoggingInterpreter

    class LoggingSession(FakeOrtSession):
        def __init__(self, model_path, providers):
            log.append("build")
            super().__init__(model_path, providers)

        def run(self, outputs, feed):
            log.append("execute")
            log.append("read-output")
            return super().run(outputs, feed)

    class LoggingOrt(FakeOrtRuntime):
        InferenceSession = LoggingSession

    lanes = (
        (TpuExecutor(device_present=True, runtime=LoggingTflite()), "model.tflite"),
        (
            GpuExecutor(device_present=True, runtime=LoggingOrt(["CUDAExecutionProvider"])),
            "model.onnx",
        ),
    )
    for executor, model_path in lanes:
        executor._clock = lambda: (log.append("clock"), 0.0)[1]
        log.clear()

        executor.run(model_path, [[1]])

        # Building the model falls before the first clock reading, as does
        # any marshalling of the inputs; the execute and the output read fall
        # between the two readings.
        opened = log.index("clock")
        assert "build" in log[:opened]
        assert "execute" not in log[:opened]
        assert log[opened:] == ["clock", "execute", "read-output", "clock"]
