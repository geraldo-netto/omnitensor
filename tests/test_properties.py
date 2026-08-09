"""Property-based (fuzz) coverage for the input boundaries and invariants:

- control acknowledgements stay contract-valid for arbitrary input,
- stride scheduling stays proportionally fair and always drains,
- policy loading never raises and always yields bounded state,
- snapshot building returns schema-valid or raises ValueError,
- discovery never raises on arbitrary sysfs file contents.
"""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
from pathlib import Path

from conftest import sample_plugin_manifest
from hypothesis import given, settings
from hypothesis import strategies as st

from omnitensor.control import ControlService
from omnitensor.discovery import (
    Device,
    DiscoveryPaths,
    detect_devices,
    device_utilization,
)
from omnitensor.executors.base import Availability, availability_for_model
from omnitensor.executors.gpu import CompositeGpuExecutor
from omnitensor.executors.npu import NpuExecutor
from omnitensor.executors.tpu import TpuExecutor
from omnitensor.registry import merge_workloads, validate_document
from omnitensor.scheduler import QueueFullError, Scheduler, _BackendQueue, _Job
from omnitensor.service import build_executors
from omnitensor.snapshot import build_snapshot
from omnitensor.state import (
    MAX_WEIGHT,
    MIN_WEIGHT,
    PolicyState,
    PolicyStore,
    ProfilePolicy,
)

json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(10**9), max_value=10**9)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=40),
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(st.text(max_size=8), children, max_size=4),
    max_leaves=25,
)

command_like = st.dictionaries(
    st.sampled_from(
        ["version", "id", "issuedAt", "expectedRevision", "operation", "profileId", "value", "x"],
    ),
    json_values
    | st.sampled_from(["set-profile-enabled", "set-profile-weight", "set-paused"])
    | st.sampled_from(["sample-workload", "missing-profile"])
    | st.integers(min_value=-3, max_value=7)
    | st.booleans(),
    max_size=8,
)

workload_ids = st.from_regex(r"[a-z][a-z0-9]{0,7}", fullmatch=True)
permission_pattern = re.compile(r"^[a-z][a-z0-9-]*:[a-zA-Z0-9*._/-]+$")


class MemoryStorage:
    """In-memory PolicyStorage for fast property runs."""

    def load(self) -> PolicyState:
        return PolicyState(
            profiles={"sample-workload": ProfilePolicy(enabled=True, weight=2)},
        )

    def save(self, state: PolicyState) -> None:
        pass


@given(permission=st.text(max_size=180))
def test_plugin_permission_schema_matches_the_documented_wire_grammar(permission):
    manifest = sample_plugin_manifest()
    manifest["plugin"]["permissions"] = [permission]
    violations = validate_document("workload-manifest.schema.json", manifest)
    expected_valid = 3 <= len(permission) <= 160 and permission_pattern.fullmatch(permission)
    assert (violations == []) is bool(expected_valid)


class CacheInterpreter:
    def __init__(self, model_path, experimental_delegates):
        self.tensors = {}

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [{"index": 0}]

    def get_output_details(self):
        return [{"index": 0}]

    def set_tensor(self, index, value):
        self.tensors[index] = value

    def invoke(self):
        pass

    def get_tensor(self, index):
        return self.tensors[index]


class CacheTfliteRuntime:
    Interpreter = CacheInterpreter

    def __init__(self):
        self.delegate_loads = 0

    def load_delegate(self, name):
        self.delegate_loads += 1
        return object()


class CacheCore:
    available_devices = ["NPU"]

    def __init__(self):
        self.compile_calls = []

    def compile_model(self, model_path, device):
        self.compile_calls.append(model_path)
        return lambda inputs: {"output": inputs[0]}


class CacheOpenVinoRuntime:
    Core = CacheCore


class FormatLane:
    backend = "gpu"

    def __init__(self, model_format: str, available: bool):
        self.model_formats = frozenset({model_format})
        self._availability = Availability(available, f"{model_format} unavailable")

    def availability(self):
        return self._availability


@given(
    bundled=st.dictionaries(workload_ids, st.integers(), max_size=30),
    user=st.dictionaries(workload_ids, st.integers(), max_size=30),
)
def test_merged_workloads_preserve_all_ids_and_bundled_precedence(bundled, user):
    merged = merge_workloads(bundled, user)
    assert set(merged) == set(bundled) | set(user)
    for workload_id, workload in bundled.items():
        assert merged[workload_id] == workload
    for workload_id, workload in user.items():
        if workload_id not in bundled:
            assert merged[workload_id] == workload


def assert_valid_acknowledgement(text: str) -> None:
    acknowledgement = json.loads(text)
    assert validate_document("runtime-acknowledgement.schema.json", acknowledgement) == []


@given(text=st.text(max_size=300))
def test_control_never_raises_on_arbitrary_text(text):
    control = ControlService(MemoryStorage())
    assert_valid_acknowledgement(control.apply_command_text(text))


@given(raw=st.binary(max_size=300))
def test_control_never_raises_on_arbitrary_bytes(raw):
    control = ControlService(MemoryStorage())
    assert_valid_acknowledgement(control.apply_command_text(raw))


@given(command=command_like)
def test_control_never_raises_on_arbitrary_command_documents(command):
    control = ControlService(MemoryStorage())
    assert_valid_acknowledgement(control.apply_command_text(json.dumps(command)))
    state = control.state
    for policy in state.profiles.values():
        assert MIN_WEIGHT <= policy.weight <= MAX_WEIGHT
    assert state.revision >= 0


@given(document=json_values)
def test_policy_load_never_raises_on_arbitrary_json_documents(document):
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "policy.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        state = PolicyStore(
            path, {"sample-workload": ProfilePolicy(enabled=True, weight=2)},
        ).load()
    assert isinstance(state.paused, bool)
    assert isinstance(state.revision, int) and not isinstance(state.revision, bool)
    assert state.revision >= 0
    for policy in state.profiles.values():
        assert MIN_WEIGHT <= policy.weight <= MAX_WEIGHT
        assert isinstance(policy.enabled, bool)


@given(raw=st.binary(max_size=200))
def test_policy_load_never_raises_on_arbitrary_bytes(raw):
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "policy.json"
        path.write_bytes(raw)
        state = PolicyStore(path, {"w": ProfilePolicy(enabled=False, weight=4)}).load()
    assert MIN_WEIGHT <= state.profiles["w"].weight <= MAX_WEIGHT


profile_plan = st.dictionaries(
    st.sampled_from(["alpha", "beta", "gamma", "delta"]),
    st.tuples(st.integers(min_value=1, max_value=5), st.integers(min_value=1, max_value=25)),
    min_size=1,
    max_size=4,
)


@given(plan=profile_plan, data=st.data())
def test_stride_scheduling_is_proportionally_fair_and_drains(plan, data):
    submissions = [
        profile_id for profile_id, (_weight, jobs) in plan.items() for _ in range(jobs)
    ]
    order = data.draw(st.permutations(submissions))
    queue = _BackendQueue()
    for index, profile_id in enumerate(order):
        queue.push(_Job(profile_id, f"{profile_id}-{index}", [], future=None))

    weights = {profile_id: weight for profile_id, (weight, _jobs) in plan.items()}
    served = dict.fromkeys(plan, 0)
    popped = 0
    while True:
        all_pending = all(queue.profiles.get(profile_id) for profile_id in plan)
        job = queue.pop_weighted(lambda profile_id: weights[profile_id])
        if job is None:
            break
        popped += 1
        served[job.workload_id] += 1
        if all_pending:
            # Stride invariant: while every profile is pending, normalized
            # service (served / weight) never diverges by more than one stride.
            normalized = [served[p] / weights[p] for p in plan]
            assert max(normalized) - min(normalized) <= 1.0 + 1e-9

    assert popped == len(submissions)
    assert queue.depth() == 0
    assert queue.pop_weighted(lambda profile_id: weights[profile_id]) is None
    assert served == {profile_id: jobs for profile_id, (_w, jobs) in plan.items()}
    # Drained profiles are pruned: no unbounded passes/order/profiles growth.
    assert queue.profiles == {}
    assert queue.order == []
    assert queue.passes == {}


@given(
    weight_a=st.integers(min_value=1, max_value=5),
    weight_b=st.integers(min_value=1, max_value=5),
    pre_served=st.integers(min_value=1, max_value=20),
    a_jobs=st.integers(min_value=1, max_value=15),
    b_jobs=st.integers(min_value=1, max_value=15),
)
def test_stride_late_joiner_gets_no_catchup_burst(
    weight_a, weight_b, pre_served, a_jobs, b_jobs,
):
    weights = {"a": weight_a, "b": weight_b}
    queue = _BackendQueue()
    for index in range(pre_served + a_jobs):
        queue.push(_Job("a", f"a-{index}", [], future=None))
    for _ in range(pre_served):
        assert queue.pop_weighted(lambda p: weights[p]).workload_id == "a"
    for index in range(b_jobs):
        queue.push(_Job("b", f"b-{index}", [], future=None))

    served_since_join = {"a": 0, "b": 0}
    while True:
        both_pending = bool(queue.profiles.get("a")) and bool(queue.profiles.get("b"))
        job = queue.pop_weighted(lambda p: weights[p])
        if job is None:
            break
        if both_pending:
            served_since_join[job.workload_id] += 1
            # From the join point, while both stay pending, normalized service
            # never diverges by more than one stride — no newcomer burst.
            normalized = [served_since_join[p] / weights[p] for p in weights]
            assert max(normalized) - min(normalized) <= 1.0 + 1e-9


@given(plan=profile_plan, data=st.data())
def test_stride_scheduling_serves_only_admitted_profiles_and_holds_the_rest(plan, data):
    admitted = data.draw(
        st.sets(st.sampled_from(sorted(plan)), min_size=0, max_size=len(plan)),
    )
    queue = _BackendQueue()
    submissions = [
        profile_id for profile_id, (_weight, jobs) in plan.items() for _ in range(jobs)
    ]
    for index, profile_id in enumerate(data.draw(st.permutations(submissions))):
        queue.push(_Job(profile_id, f"{profile_id}-{index}", [], future=None))

    weights = {profile_id: weight for profile_id, (weight, _jobs) in plan.items()}
    served = dict.fromkeys(plan, 0)
    while True:
        job = queue.pop_weighted(
            lambda profile_id: weights[profile_id],
            admits=lambda profile_id: profile_id in admitted,
        )
        if job is None:
            break
        assert job.workload_id in admitted
        served[job.workload_id] += 1

    for profile_id, (_weight, jobs) in plan.items():
        if profile_id in admitted:
            assert served[profile_id] == jobs
        else:
            # Held jobs stay queued, never dropped.
            assert served[profile_id] == 0
            assert len(queue.profiles[profile_id]) == jobs
    assert queue.depth() == sum(
        jobs for profile_id, (_w, jobs) in plan.items() if profile_id not in admitted
    )


@given(rounds=st.integers(min_value=1, max_value=30))
def test_retired_worker_tracking_stays_bounded_under_backend_churn(rounds):
    async def scenario():
        executors = {"tpu": object(), "npu": object()}
        scheduler = Scheduler(executors, weight_of=lambda _profile: 1)
        scheduler.start()
        for _round in range(rounds):
            scheduler.update_executors({"tpu": executors["tpu"]})
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            scheduler.update_executors(executors)
            assert len(scheduler._retired) <= 1
        await scheduler.stop()
        assert scheduler._retired == []

    asyncio.run(scenario())


@given(
    capacity=st.integers(min_value=1, max_value=12),
    attempts=st.integers(min_value=0, max_value=30),
)
def test_scheduler_backend_queue_never_exceeds_its_capacity(capacity, attempts):
    async def scenario():
        scheduler = Scheduler(
            {"tpu": object()},
            weight_of=lambda _profile: 1,
            max_backend_queue_depth=capacity,
        )
        accepted = []
        rejected = 0
        for index in range(attempts):
            try:
                accepted.append(
                    scheduler.submit("tpu", f"profile-{index % 3}", str(index), []),
                )
            except QueueFullError:
                rejected += 1

        assert len(accepted) == min(capacity, attempts)
        assert rejected == max(0, attempts - capacity)
        assert scheduler.stats()["queueDepth"] == min(capacity, attempts)
        await scheduler.stop()

    asyncio.run(scenario())


@settings(max_examples=30)
@given(paths=st.lists(st.sampled_from(["alpha", "beta", "gamma"]), min_size=1, max_size=20))
def test_executor_model_caches_create_once_per_distinct_path(paths):
    tflite = CacheTfliteRuntime()
    tpu = TpuExecutor(device_present=True, runtime=tflite)
    npu = NpuExecutor(device_present=True, runtime=CacheOpenVinoRuntime())
    for path in paths:
        assert tpu.run(path, [[path]]).outputs == [[path]]
        assert npu.run(path, [[path]]).outputs == [[path]]

    assert tflite.delegate_loads == len(set(paths))
    assert npu._core.compile_calls == list(dict.fromkeys(paths))


@settings(max_examples=30)
@given(states=st.lists(st.none() | st.sampled_from(["tpu-a", "tpu-b"]), min_size=1, max_size=20))
def test_executor_reconciliation_reuses_exactly_unchanged_device_states(states):
    previous_devices = None
    previous_executors = None
    for device_id in states:
        devices = [] if device_id is None else [
            Device(id=device_id, backend="tpu", name=device_id, kind="pcie"),
        ]
        executors = build_executors(
            devices,
            previous_devices=previous_devices,
            previous_executors=previous_executors,
        )
        if previous_executors is not None:
            unchanged = devices == previous_devices
            assert (executors["tpu"] is previous_executors["tpu"]) is unchanged
            assert executors["npu"] is previous_executors["npu"]
            assert executors["gpu"] is previous_executors["gpu"]
        previous_devices = devices
        previous_executors = executors


@given(
    ncnn_available=st.booleans(),
    onnx_available=st.booleans(),
    requested=st.sampled_from(["ncnn", "onnx", "openvino"]),
)
def test_composite_gpu_availability_matches_requested_runtime_lane(
    ncnn_available, onnx_available, requested,
):
    composite = CompositeGpuExecutor([
        FormatLane("ncnn", ncnn_available),
        FormatLane("onnx", onnx_available),
    ])
    availability = availability_for_model(composite, {"format": requested})
    expected = {"ncnn": ncnn_available, "onnx": onnx_available, "openvino": False}
    assert availability.available is expected[requested]


device_entries = st.builds(
    Device,
    id=st.text(max_size=90),
    backend=st.sampled_from(["tpu", "npu", "gpu", "cpu", ""]),
    name=st.text(max_size=130),
    kind=st.sampled_from(["usb", "pcie", "accel", "dri", "unknown", "weird"]),
    vendor=st.text(max_size=90),
    load=st.none() | st.floats(min_value=-50, max_value=150, allow_nan=False),
    available=st.booleans(),
    reason=st.text(max_size=250),
)


@given(
    devices=st.lists(device_entries, max_size=20),
    metrics=st.dictionaries(
        st.sampled_from(["queueDepth", "runningProfiles", "loads", "junk"]),
        json_values,
        max_size=4,
    ),
    profiles=st.dictionaries(st.text(max_size=12), json_values, max_size=4),
    generated_at=st.none() | st.integers(min_value=-10, max_value=2**53),
)
def test_build_snapshot_is_schema_valid_or_value_error(devices, metrics, profiles, generated_at):
    try:
        snapshot = build_snapshot(
            devices=devices,
            metrics=metrics,
            profiles=profiles,
            generated_at_ms=generated_at,
        )
    except ValueError:
        return
    assert validate_document("runtime-snapshot.schema.json", snapshot) == []
    json.dumps(snapshot)


@settings(max_examples=40)
@given(
    id_vendor=st.binary(max_size=64),
    id_product=st.binary(max_size=64),
    npu_vendor=st.binary(max_size=64),
    gpu_vendor=st.binary(max_size=64),
    busy=st.binary(max_size=64),
)
def test_discovery_never_raises_on_arbitrary_file_contents(
    id_vendor, id_product, npu_vendor, gpu_vendor, busy,
):
    with tempfile.TemporaryDirectory() as root:
        paths = DiscoveryPaths(dev=Path(root) / "dev", sys=Path(root) / "sys")
        usb = paths.sys / "bus/usb/devices/1-1"
        usb.mkdir(parents=True)
        (usb / "idVendor").write_bytes(id_vendor)
        (usb / "idProduct").write_bytes(id_product)
        (paths.dev / "accel").mkdir(parents=True)
        (paths.dev / "accel/accel0").touch()
        npu_dir = paths.sys / "class/accel/accel0/device"
        npu_dir.mkdir(parents=True)
        (npu_dir / "vendor").write_bytes(npu_vendor)
        (paths.dev / "dri").mkdir(parents=True)
        (paths.dev / "dri/renderD128").touch()
        gpu_dir = paths.sys / "class/drm/renderD128/device"
        gpu_dir.mkdir(parents=True)
        (gpu_dir / "vendor").write_bytes(gpu_vendor)
        (gpu_dir / "gpu_busy_percent").write_bytes(busy)

        devices = detect_devices(paths)
        assert {device.backend for device in devices} <= {"tpu", "npu", "gpu"}
        for device in devices:
            utilization = device_utilization(paths, device)
            assert utilization is None or 0.0 <= utilization <= 100.0
