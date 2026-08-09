from __future__ import annotations

import json

import pytest
from conftest import add_pcie_tpu, sample_manifest, write_workload

from omnitensor.discovery import detect_devices
from omnitensor.registry import ManifestError, load_workloads
from omnitensor.snapshot import build_snapshot, write_snapshot


def test_snapshot_is_contract_valid_and_written_atomically(fake_nodes, tmp_path):
    add_pcie_tpu(fake_nodes)
    devices = detect_devices(fake_nodes)
    snapshot = build_snapshot(
        devices=devices,
        metrics={"queueDepth": 3, "runningProfiles": 1},
        profiles={"hardware-health": {"status": "running", "queued": 3, "detail": "sampling"}},
        generated_at_ms=1_700_000_000_000,
    )
    assert snapshot["version"] == 1
    assert snapshot["devices"][0]["backend"] == "tpu"
    assert "load" not in snapshot["metrics"]

    target = tmp_path / "state/runtime-snapshot.json"
    write_snapshot(target, snapshot)
    assert json.loads(target.read_text()) == snapshot
    leftovers = [entry for entry in target.parent.iterdir() if entry.name.startswith(".snapshot-")]
    assert leftovers == []


def test_snapshot_with_no_devices_is_rejected_by_contract(tmp_path):
    with pytest.raises(ValueError, match="devices"):
        build_snapshot(devices=[], metrics={}, profiles={})


def test_registry_loads_valid_manifests_and_preference(tmp_path):
    write_workload(tmp_path, sample_manifest())
    write_workload(tmp_path, sample_manifest("gpu-workload", accelerator="gpu",
                                             acceleratorPreference=["gpu"]))
    workloads = load_workloads(tmp_path)
    assert sorted(workloads) == ["gpu-workload", "sample-workload"]
    assert workloads["sample-workload"].preference == ("tpu", "npu", "gpu")
    assert workloads["gpu-workload"].preference == ("gpu",)
    assert workloads["gpu-workload"].accelerator == "gpu"


def test_registry_default_preference_when_omitted(tmp_path):
    manifest = sample_manifest()
    del manifest["requirements"]["acceleratorPreference"]
    write_workload(tmp_path, manifest)
    assert load_workloads(tmp_path)["sample-workload"].preference == ("tpu", "npu", "gpu")


def test_registry_rejects_invalid_manifest(tmp_path):
    bad = sample_manifest()
    bad["requirements"]["accelerator"] = "edge-tpu"
    write_workload(tmp_path, bad)
    with pytest.raises(ManifestError, match="accelerator"):
        load_workloads(tmp_path)


def test_registry_rejects_id_directory_mismatch(tmp_path):
    manifest = sample_manifest()
    directory = tmp_path / "other-name"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ManifestError, match="directory"):
        load_workloads(tmp_path)


def test_registry_missing_root_is_empty(tmp_path):
    assert load_workloads(tmp_path / "missing") == {}
