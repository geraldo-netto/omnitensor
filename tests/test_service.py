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
