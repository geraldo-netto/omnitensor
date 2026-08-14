from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from conftest import add_npu, add_pcie_tpu, sample_manifest, write_workload

from omnitensor import service as service_module
from omnitensor.discovery import Device, detect_devices
from omnitensor.executors.base import Availability
from omnitensor.executors.tpu import TpuExecutor
from omnitensor.jobs import UnavailableJobDispatcher
from omnitensor.plugins import ArtifactInstaller, ArtifactReference, ArtifactResolution
from omnitensor.plugins.kernel_telemetry import parse_aggregate
from omnitensor.profile_selection import profile_status
from omnitensor.registry import _schema_path, validate_document
from omnitensor.scheduler import select_backend
from omnitensor.service import (
    OmniTensorService,
    build_executors,
    profile_statuses,
)


def build_service(fake_nodes, tmp_path, manifests=()):
    workloads_root = tmp_path / "workloads"
    workloads_root.mkdir(exist_ok=True)
    for manifest in manifests:
        write_workload(workloads_root, manifest)
    return OmniTensorService(
        snapshot_path=tmp_path / "state/runtime-snapshot.json",
        policy_path=tmp_path / "state/policy.json",
        workloads_path=workloads_root,
        discovery_paths=fake_nodes,
    )


class ReadyKernelTelemetry:
    def read(self):
        return parse_aggregate({
            "version": 1,
            "collectedAtMs": 1_700_000_000_000,
            "histograms": [
                {"name": "runq_latency_us", "unit": "us", "buckets": [1, 2]},
                {"name": "block_latency_us", "unit": "us", "buckets": [3, 4]},
            ],
            "counters": [],
        })


def test_publish_once_emits_contract_valid_snapshot(fake_nodes, tmp_path):
    async def scenario():
        add_pcie_tpu(fake_nodes)
        add_npu(fake_nodes)
        service = build_service(fake_nodes, tmp_path, [sample_manifest()])
        service._scheduler.start()
        snapshot = service.publish_once()
        await service._scheduler.stop()
        return snapshot

    snapshot = asyncio.run(scenario())
    assert validate_document("runtime-snapshot.schema.json", snapshot) == []
    assert [device["backend"] for device in snapshot["devices"]] == ["tpu", "npu"]
    written = json.loads((tmp_path / "state/runtime-snapshot.json").read_text())
    assert written == snapshot


def test_publish_once_reads_kernel_telemetry_into_the_applet_contract(fake_nodes, tmp_path):
    add_pcie_tpu(fake_nodes)
    workloads_root = tmp_path / "workloads"
    workloads_root.mkdir()
    service = OmniTensorService(
        snapshot_path=tmp_path / "state/runtime-snapshot.json",
        policy_path=tmp_path / "state/policy.json",
        workloads_path=workloads_root,
        discovery_paths=fake_nodes,
        kernel_telemetry_source=ReadyKernelTelemetry(),
    )

    snapshot = service.publish_once()

    assert snapshot["kernelTelemetry"]["state"] == "ready"
    assert [item["name"] for item in snapshot["kernelTelemetry"]["histograms"]] == [
        "runq_latency_us",
        "block_latency_us",
    ]


def test_plugin_artifact_resolution_is_fail_closed_without_a_store(fake_nodes, tmp_path):
    service = build_service(fake_nodes, tmp_path)
    reference = ArtifactReference("provider-model", "1.0.0", "gguf", "a" * 64)

    resolution = service._resolve_plugin_artifact(reference)

    assert resolution == ArtifactResolution(
        False,
        None,
        "no artifact store is configured for this service",
        0,
    )


def test_plugin_artifact_resolution_delegates_the_exact_reference(fake_nodes, tmp_path):
    service = build_service(fake_nodes, tmp_path)
    reference = ArtifactReference("provider-model", "1.0.0", "gguf", "a" * 64)
    expected = ArtifactResolution(True, tmp_path / "model.gguf", "ready", 17)
    calls = []

    class Store:
        def resolve(self, candidate):
            calls.append(candidate)
            return expected

    service._artifact_store = Store()

    assert service._resolve_plugin_artifact(reference) is expected
    assert calls == [reference]


def test_plugin_accelerator_devices_exposes_only_supported_device_nodes(fake_nodes, tmp_path):
    service = build_service(fake_nodes, tmp_path)
    service._devices = [
        Device("gpu-renderD128", "gpu", "GPU 0", "dri"),
        Device("gpu-renderD129", "gpu", "GPU 1", "dri"),
        Device("npu-accel0", "npu", "NPU 0", "accel"),
        Device("tpu-apex_0", "tpu", "TPU", "apex"),
        Device("gpu-card0", "gpu", "GPU card", "card"),
    ]

    assert service._plugin_accelerator_devices() == {
        "gpu": Path("/dev/dri/renderD128"),
        "npu": Path("/dev/accel/accel0"),
    }


def test_default_plugin_runtime_receives_exact_host_owned_resources(
    fake_nodes, tmp_path, monkeypatch
):
    observed = {}

    class Runtime:
        snapshot = SimpleNamespace(catalog=SimpleNamespace(plugins=()), workers=())

        def plugin_ids(self):
            return frozenset()

    runtime = Runtime()

    def make_runtime(path, **options):
        observed.update(path=path, **options)
        return runtime

    monkeypatch.setattr(service_module, "InstalledPluginRuntime", make_runtime)
    snapshot = tmp_path / "state/runtime-snapshot.json"
    service = OmniTensorService(
        snapshot_path=snapshot,
        policy_path=tmp_path / "state/policy.json",
        workloads_path=tmp_path / "workloads",
        discovery_paths=fake_nodes,
        artifact_root=tmp_path / "artifacts",
    )

    assert service._plugin_runtime is runtime
    assert observed["path"] == service_module.bundled_workloads_path()
    assert observed["grant_source"] is service._grants
    assert observed["selected_files_root"] == snapshot.parent / "plugin-inputs"
    assert observed["worker_state_root"] == snapshot.parent / "plugin-state"
    assert observed["resolve_artifact"] == service._resolve_plugin_artifact
    assert observed["accelerator_devices"] == service._plugin_accelerator_devices
    progress = SimpleNamespace(job_id="job", stage="load", fraction=0.5, detail="working")
    calls = []
    service._note_job_progress = lambda *arguments: calls.append(arguments)
    observed["progress_sink"](progress)
    assert calls == [("job", "load", 0.5, "working")]


def test_profile_statuses_report_backend_or_reason(fake_nodes, tmp_path):
    add_pcie_tpu(fake_nodes)
    devices = detect_devices(fake_nodes)
    executors = build_executors(devices)

    async def scenario():
        service = build_service(fake_nodes, tmp_path, [
            sample_manifest(),
            sample_manifest("gpu-only", accelerator="gpu", acceleratorPreference=["gpu"]),
        ])
        service._scheduler.start()
        statuses = profile_statuses(
            service._workloads, executors, service._scheduler, service.control.state,
        )
        await service._scheduler.stop()
        return statuses

    statuses = asyncio.run(scenario())
    gpu_only = statuses["gpu-only"]
    assert gpu_only["status"] == "unavailable"
    assert "gpu" in gpu_only["detail"]
    assert statuses["sample-workload"]["status"] in {"idle", "unavailable"}


def test_control_service_defaults_come_from_manifests(fake_nodes, tmp_path):
    service = build_service(fake_nodes, tmp_path, [sample_manifest()])
    portfolio = service.control.state.portfolio()
    assert portfolio["profiles"]["sample-workload"] == {"enabled": True, "weight": 2}


def test_weight_of_falls_back_for_unknown_profiles(fake_nodes, tmp_path):
    service = build_service(fake_nodes, tmp_path, [sample_manifest()])
    assert service._weight_of("sample-workload") == 2
    assert service._weight_of("missing") == 1


def test_build_executors_marks_absent_backends(fake_nodes):
    executors = build_executors([])
    for executor in executors.values():
        assert executor.availability().available is False


def test_build_executors_reuses_only_unchanged_device_adapters():
    tpu = Device(id="tpu-a", backend="tpu", name="TPU A", kind="pcie")
    first = build_executors([tpu])
    interpreters = first["tpu"]._interpreters

    same = build_executors(
        [tpu], previous_devices=[tpu], previous_executors=first,
    )
    assert same["tpu"] is first["tpu"]
    assert same["tpu"]._interpreters is interpreters
    assert same["npu"] is first["npu"]
    assert same["gpu"] is first["gpu"]

    incomplete_history = build_executors([tpu], previous_executors=first)
    assert incomplete_history["tpu"] is not first["tpu"]
    assert incomplete_history["npu"] is not first["npu"]
    assert incomplete_history["gpu"] is not first["gpu"]

    replacement = Device(id="tpu-b", backend="tpu", name="TPU B", kind="usb")
    changed = build_executors(
        [replacement], previous_devices=[tpu], previous_executors=same,
    )
    assert changed["tpu"] is not same["tpu"]
    assert changed["npu"] is same["npu"]
    assert changed["gpu"] is same["gpu"]


def test_publish_prefers_kernel_gpu_utilization(fake_nodes, tmp_path):
    from conftest import add_gpu

    add_gpu(fake_nodes, node=128, vendor="0x1002")
    (fake_nodes.sys / "class/drm/renderD128/device/gpu_busy_percent").write_text("42\n")

    async def scenario():
        service = build_service(fake_nodes, tmp_path)
        service._scheduler.start()
        snapshot = service.publish_once()
        await service._scheduler.stop()
        return snapshot

    snapshot = asyncio.run(scenario())
    assert snapshot["devices"][0]["load"] == 42.0


MODEL_PAYLOAD = b"a tflite model would live here"
MODEL_DIGEST = hashlib.sha256(MODEL_PAYLOAD).hexdigest()


def sample_manifest_with_model(digest=MODEL_DIGEST):
    """A profile that can actually execute, so a runner is built for it.

    It pins its model, because an unpinned one no longer runs: the store can
    prove a file has not changed since installation and cannot prove it is the
    file the publisher meant.
    """
    manifest = sample_manifest()
    manifest["id"] = "runnable-workload"
    manifest["requirements"]["model"] = {
        "id": "runnable-model",
        "version": "1.0.0",
        "format": "tflite-edgetpu",
        "fullyQuantized": True,
        "minimumCompilerVersion": "validated-release",
        "minimumRuntimeVersion": "validated-release",
        "sha256": digest,
    }
    return manifest


def test_the_service_builds_a_runner_for_each_executable_profile(fake_nodes, tmp_path):
    add_pcie_tpu(fake_nodes)
    service = build_service(
        fake_nodes, tmp_path, [sample_manifest(), sample_manifest_with_model()]
    )

    assert "runnable-workload" in service.runners.runners
    assert "declares no model" in service.runners.skipped["sample-workload"]


def test_a_profile_without_a_model_says_so_at_startup_not_at_first_job(fake_nodes, tmp_path):
    """One place, at startup, beats a stage failure at the first submission."""
    add_pcie_tpu(fake_nodes)
    service = build_service(fake_nodes, tmp_path, [sample_manifest()])

    assert service.runners.get("sample-workload") is None
    assert service.runners.document()["skipped"]["sample-workload"]


def test_interrupted_jobs_are_reconciled_when_the_service_starts(fake_nodes, tmp_path):
    journal = tmp_path / "state/cancellations.json"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(
        json.dumps({"version": 1, "jobs": {"job-1": "runnable-workload"}})
    )
    add_pcie_tpu(fake_nodes)
    service = build_service(fake_nodes, tmp_path, [sample_manifest_with_model()])

    reconciled = service.reconcile_interrupted_jobs()

    assert [item.job_id for item in reconciled.interrupted] == ["job-1"]
    assert service.reconcile_interrupted_jobs().interrupted == ()


def test_a_clean_start_reconciles_nothing(fake_nodes, tmp_path):
    add_pcie_tpu(fake_nodes)
    service = build_service(fake_nodes, tmp_path, [sample_manifest_with_model()])
    assert service.reconcile_interrupted_jobs().interrupted == ()


def test_a_declared_model_that_is_not_installed_is_not_reported_as_serving(
    fake_nodes, tmp_path
):
    """Declaring a model is not having one. Every fresh checkout ships the
    manifest without the artifact, and this claimed the profile was working."""
    add_pcie_tpu(fake_nodes)
    service = build_service(fake_nodes, tmp_path, [sample_manifest_with_model()])

    statuses = service._build_runtime_snapshot()["profiles"]

    entry = statuses["runnable-workload"]
    assert entry["status"] == "unavailable"
    assert "Serving" not in entry["detail"]


def test_a_profile_with_no_model_is_never_told_its_artifact_is_missing(fake_nodes, tmp_path):
    """A profile that declares nothing cannot be missing anything."""
    add_pcie_tpu(fake_nodes)
    service = build_service(fake_nodes, tmp_path, [sample_manifest()])

    entry = service._build_runtime_snapshot()["profiles"]["sample-workload"]

    assert "artifact" not in entry["detail"]
    assert entry["status"] != "watching"
    ready, reason = service._profile_artifact_ready(service._workloads["sample-workload"])
    assert (ready, reason) == (True, "")


def test_the_snapshot_and_the_dispatcher_consult_the_same_resolver(fake_nodes, tmp_path):
    """Otherwise the snapshot can promise what dispatch would refuse."""
    add_pcie_tpu(fake_nodes)
    service = build_service(fake_nodes, tmp_path, [sample_manifest_with_model()])

    ready, reason = service._profile_artifact_ready(service._workloads["runnable-workload"])

    assert ready is False
    assert reason == service._resolve_artifact("runnable-model").reason


SERVING_STATUSES = frozenset({"watching", "running"})


class _AvailableExecutor:
    """The narrowest thing ``select_backend`` accepts as a working backend."""

    model_formats = ("tflite-edgetpu",)

    def availability(self) -> Availability:
        return Availability(True, "")


class _OnnxOnlyExecutor(_AvailableExecutor):
    """Present and healthy, but not for the format the workload declares."""

    model_formats = ("onnx",)


def _serving_profiles(service) -> set[str]:
    return {
        workload_id
        for workload_id, entry in service._build_runtime_snapshot()["profiles"].items()
        if entry["status"] in SERVING_STATUSES
    }


def test_no_profile_serves_where_the_dispatcher_refuses_every_job(fake_nodes, tmp_path):
    """The status a user reads and the answer SubmitJob gives must agree.

    With no artifact root the dispatcher is fail-closed for every workload, so
    any serving profile in the snapshot is a promise the service cannot keep.
    Asserted over the whole snapshot rather than one profile, because the two
    halves are computed by different code and only their agreement matters.
    """
    add_pcie_tpu(fake_nodes)
    service = build_service(
        fake_nodes, tmp_path, [sample_manifest(), sample_manifest_with_model()]
    )

    assert isinstance(service._job_dispatcher._inference, UnavailableJobDispatcher)
    assert _serving_profiles(service) == set()


def test_a_profile_serves_once_its_artifact_is_installed_and_the_job_is_admitted(
    fake_nodes, tmp_path
):
    """The other direction, so the test above cannot pass by never serving."""
    add_pcie_tpu(fake_nodes)
    store = tmp_path / "artifacts"
    source = tmp_path / "model.tflite"
    source.write_bytes(MODEL_PAYLOAD)
    reference = ArtifactReference("runnable-model", "1.0.0", "tflite-edgetpu", MODEL_DIGEST)
    ArtifactInstaller(store).install(reference, source)

    workloads_root = tmp_path / "workloads"
    workloads_root.mkdir(exist_ok=True)
    write_workload(workloads_root, sample_manifest_with_model())
    service = OmniTensorService(
        snapshot_path=tmp_path / "state/runtime-snapshot.json",
        policy_path=tmp_path / "state/policy.json",
        workloads_path=workloads_root,
        discovery_paths=fake_nodes,
        artifact_root=store,
    )

    # This host has no Edge TPU runtime, so the backend that would serve the
    # profile is stubbed. Everything the assertion is about — the resolver, the
    # dispatcher, and the status they agree on — is still the real thing.
    service._executors = {"tpu": _AvailableExecutor()}

    assert not isinstance(service._job_dispatcher, UnavailableJobDispatcher)
    assert _serving_profiles(service) == {"runnable-workload"}


def _snapshot_reason_codes() -> set[str]:
    schema = json.loads(_schema_path("runtime-snapshot.schema.json").read_text())
    entry = schema["properties"]["profiles"]["additionalProperties"]
    return set(entry["properties"]["reason"]["enum"])


def test_every_profile_carries_a_reason_code_the_schema_allows(fake_nodes, tmp_path):
    """The applet chooses a remedy from this code.

    It used to choose by matching substrings of ``detail``, so rewording a
    sentence here cost a user their remedy silently. A branch that forgets the
    code has to fail here rather than in a popup.
    """
    add_pcie_tpu(fake_nodes)
    service = build_service(
        fake_nodes, tmp_path, [sample_manifest(), sample_manifest_with_model()]
    )
    allowed = _snapshot_reason_codes()

    profiles = service._build_runtime_snapshot()["profiles"]

    assert profiles
    for workload_id, entry in profiles.items():
        assert entry["reason"] in allowed, workload_id


def test_the_reason_names_the_state_each_branch_reached(fake_nodes, tmp_path):
    add_pcie_tpu(fake_nodes)
    service = build_service(
        fake_nodes, tmp_path, [sample_manifest(), sample_manifest_with_model()]
    )
    service._executors = {"tpu": _AvailableExecutor()}

    def reason_for(workload_id: str) -> str:
        return service._build_runtime_snapshot()["profiles"][workload_id]["reason"]

    # No model declared, versus one declared with nothing installed.
    assert reason_for("sample-workload") == "no-model"
    assert reason_for("runnable-workload") == "artifact-unavailable"

    service.control.state.profiles["sample-workload"].enabled = False
    assert reason_for("sample-workload") == "profile-disabled"
    service.control.state.paused = True
    assert reason_for("sample-workload") == "paused-by-policy"


def test_a_backend_reason_survives_as_a_code_not_a_sentence(fake_nodes, tmp_path):
    """The three refusals a user must tell apart come from the executors."""
    service = build_service(fake_nodes, tmp_path, [sample_manifest_with_model()])
    workload = service._workloads["runnable-workload"]

    absent = select_backend(workload, {"tpu": TpuExecutor(device_present=False)})
    unimportable = TpuExecutor(device_present=True, runtime=None)
    # Set explicitly rather than relying on the import failing: this must read
    # the same on a host that does have tflite-runtime.
    unimportable._runtime = None
    missing = select_backend(workload, {"tpu": unimportable})
    # Healthy and present, but not for the format this workload declares.
    unsupported = select_backend(workload, {"tpu": _OnnxOnlyExecutor()})

    assert (absent.backend, absent.code) == (None, "device-absent")
    assert (missing.backend, missing.code) == (None, "runtime-missing")
    assert (unsupported.backend, unsupported.code) == (None, "format-unsupported")
    assert select_backend(workload, {}).code == "no-executor"


def test_a_profile_that_needs_consent_says_so_before_a_job_is_submitted(fake_nodes, tmp_path):
    """The gate refuses these jobs, so reporting 'serving' would promise
    something the first job is refused for — and the remedy is a command
    nobody would guess."""
    from omnitensor.registry import Workload

    add_pcie_tpu(fake_nodes)
    service = build_service(fake_nodes, tmp_path, [sample_manifest()])
    # Only a version 2 plugin manifest can declare permissions; the workload
    # schema forbids them on a version 1 document, which is why no bundled
    # profile has ever exercised this gate.
    declaring = Workload(
        id="declaring-workload",
        manifest={
            **sample_manifest_with_model(),
            "plugin": {"permissions": ["files:read", "net:connect"]},
        },
    )

    missing = service._profile_permissions_missing(declaring)

    assert missing == ("files:read", "net:connect")

    # A backend that resolves, so the status reaches the consent branch rather
    # than stopping at a lane this host does not have.
    entry = profile_status(
        declaring,
        {"tpu": _AvailableExecutor()},
        {"queued": 0, "running": 0},
        service.control.state,
        lambda _workload: (True, ""),
        lambda _workload: ("files:read",),
    )

    assert entry["reason"] == "consent-missing"
    assert entry["status"] == "unavailable"
    assert "files:read" in entry["detail"]


def test_a_profile_declaring_nothing_is_never_told_it_needs_consent(fake_nodes, tmp_path):
    add_pcie_tpu(fake_nodes)
    service = build_service(fake_nodes, tmp_path, [sample_manifest()])

    assert service._profile_permissions_missing(service._workloads["sample-workload"]) == ()


def test_a_granted_permission_stops_being_reported_as_missing(fake_nodes, tmp_path):
    from omnitensor.plugins.grants import GrantLedger, GrantOrigin, GrantProvenance
    from omnitensor.registry import Workload

    add_pcie_tpu(fake_nodes)
    service = build_service(fake_nodes, tmp_path, [sample_manifest()])
    ledger = GrantLedger(tmp_path / "grants.json")
    service._grants = ledger
    workload = Workload(
        id="declaring-workload",
        manifest={**sample_manifest_with_model(), "plugin": {"permissions": ["files:read"]}},
    )

    assert service._profile_permissions_missing(workload) == ("files:read",)

    ledger.grant(
        "declaring-workload",
        "files:read",
        {"files:read"},
        GrantProvenance("tester", GrantOrigin.USER, "test", 1, "req-1"),
        expected_revision=ledger.revision,
    )

    assert service._profile_permissions_missing(workload) == ()


def test_readiness_follows_the_lane_dispatch_would_choose(fake_nodes, tmp_path):
    """A profile declaring one model per accelerator has one answer per lane.

    Reading only the first declared entry reported unavailable for a profile
    the runtime would serve, and serving for one whose chosen lane refuses —
    the snapshot-versus-dispatch disagreement the resolver exists to prevent.
    """
    from omnitensor.registry import Workload

    add_pcie_tpu(fake_nodes)
    service = build_service(fake_nodes, tmp_path, [sample_manifest()])
    resolved: list[str] = []
    service._resolve_artifact = lambda artifact_id: (
        resolved.append(artifact_id) or ArtifactResolution(True, None, "", 1)
    )

    manifest = sample_manifest_with_model()
    requirements = manifest["requirements"]
    requirements.pop("model")
    requirements["acceleratorPreference"] = ["tpu", "gpu"]
    requirements["models"] = [
        {
            "id": "gpu-model",
            "version": "1.0.0",
            "format": "ncnn",
            "fullyQuantized": False,
            "minimumCompilerVersion": "v",
            "minimumRuntimeVersion": "v",
            "sha256": "a" * 64,
        },
        {
            "id": "tpu-model",
            "version": "1.0.0",
            "format": "tflite-edgetpu",
            "fullyQuantized": True,
            "minimumCompilerVersion": "v",
            "minimumRuntimeVersion": "v",
            "sha256": "b" * 64,
        },
    ]
    workload = Workload(id="multi-lane", manifest=manifest)
    service._executors = {"tpu": _AvailableExecutor()}

    ready, _reason = service._profile_artifact_ready(workload)

    assert ready is True
    assert resolved == ["tpu-model"], "the TPU lane won, so its own artifact was asked about"


def test_a_profile_no_lane_can_run_leaves_readiness_quiet(fake_nodes, tmp_path):
    """`select_backend` already reports that; a second vaguer reason helps nobody."""
    from omnitensor.registry import Workload

    add_pcie_tpu(fake_nodes)
    service = build_service(fake_nodes, tmp_path, [sample_manifest()])
    def _never(_artifact_id):
        raise AssertionError("the store must not be consulted for an unrunnable profile")

    service._resolve_artifact = _never
    service._executors = {"gpu": _OnnxOnlyExecutor()}
    workload = Workload(id="unrunnable", manifest=sample_manifest_with_model())

    assert service._profile_artifact_ready(workload) == (True, "")


def test_a_resolution_is_not_recomputed_until_the_artifact_moves(fake_nodes, tmp_path):
    """Resolution re-digests the model and its companions, and the publisher
    asks for every profile on every tick — so a large catalogue was hashed end
    to end twice a second for an answer that changes only when a file does."""
    import hashlib

    add_pcie_tpu(fake_nodes)
    store = tmp_path / "artifacts"
    source = tmp_path / "model.tflite"
    source.write_bytes(MODEL_PAYLOAD)
    reference = ArtifactReference("runnable-model", "1.0.0", "tflite-edgetpu", MODEL_DIGEST)
    ArtifactInstaller(store).install(reference, source)

    workloads_root = tmp_path / "workloads"
    workloads_root.mkdir(exist_ok=True)
    write_workload(workloads_root, sample_manifest_with_model())
    service = OmniTensorService(
        snapshot_path=tmp_path / "state/runtime-snapshot.json",
        policy_path=tmp_path / "state/policy.json",
        workloads_path=workloads_root,
        discovery_paths=fake_nodes,
        artifact_root=store,
    )
    calls = []
    real = service._artifact_store.resolve
    service._artifact_store.resolve = lambda ref: (calls.append(ref) or real(ref))

    first = service._resolve_artifact("runnable-model")
    for _ in range(5):
        service._resolve_artifact("runnable-model")

    assert first.ready is True
    assert len(calls) == 1, "asked once, then answered from what was already proven"

    # A file replaced in place moves its mtime, so the cache cannot hold a
    # verdict the store would now refuse.
    installed = store / "runnable-model" / "1.0.0" / "model.tflite"
    installed.write_bytes(b"substituted after installation")
    service._resolve_artifact("runnable-model")

    assert len(calls) == 2
    assert service._resolve_artifact("runnable-model").ready is False
    assert hashlib.sha256(installed.read_bytes()).hexdigest() != MODEL_DIGEST
