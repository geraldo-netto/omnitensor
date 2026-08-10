from __future__ import annotations

import asyncio
import json

from conftest import add_npu, add_pcie_tpu, sample_manifest, write_workload

from omnitensor.discovery import Device, detect_devices
from omnitensor.registry import validate_document
from omnitensor.service import OmniTensorService, build_executors, profile_statuses


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


def sample_manifest_with_model():
    """A profile that can actually execute, so a runner is built for it."""
    manifest = sample_manifest()
    manifest["id"] = "runnable-workload"
    manifest["requirements"]["model"] = {
        "id": "runnable-model",
        "version": "1.0.0",
        "format": "tflite-edgetpu",
        "fullyQuantized": True,
        "minimumCompilerVersion": "validated-release",
        "minimumRuntimeVersion": "validated-release",
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
