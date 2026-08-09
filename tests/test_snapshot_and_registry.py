from __future__ import annotations

import json

import pytest
from conftest import add_pcie_tpu, sample_manifest, write_workload

import omnitensor.registry as registry
from omnitensor.discovery import detect_devices
from omnitensor.registry import (
    MAX_WORKLOADS,
    ManifestError,
    bundled_workloads_path,
    load_workload_catalog,
    load_workloads,
    merge_workloads,
    validate_document,
)
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


@pytest.mark.parametrize("bad", [None, "garbage", 3 + 4j, float("nan"), [1]])
def test_snapshot_raises_value_error_for_non_integer_metrics(fake_nodes, bad):
    add_pcie_tpu(fake_nodes)
    devices = detect_devices(fake_nodes)
    with pytest.raises(ValueError):
        build_snapshot(devices=devices, metrics={"queueDepth": bad}, profiles={})


def test_registry_wraps_malformed_json_in_manifest_error(tmp_path):
    directory = tmp_path / "broken-workload"
    directory.mkdir()
    (directory / "manifest.json").write_text("{nope")
    with pytest.raises(ManifestError, match="invalid JSON"):
        load_workloads(tmp_path)


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


def test_bundled_catalog_matches_the_cinnamon_profiles():
    workloads = load_workloads(bundled_workloads_path())
    assert sorted(workloads) == [
        "build-advisor",
        "desktop-context",
        "document-intelligence",
        "hardware-health",
        "low-light-enhancement",
        "network-peripherals",
        "resource-scheduler",
        "storage-intelligence",
        "visual-library",
    ]


def test_packaged_workload_catalog_wins_over_source_fallback(monkeypatch, tmp_path):
    packaged = tmp_path / "packaged"
    source = tmp_path / "source"
    packaged.mkdir()
    source.mkdir()
    monkeypatch.setattr(registry, "_PACKAGED_WORKLOADS", packaged)
    monkeypatch.setattr(registry, "_SOURCE_WORKLOADS", source)
    assert bundled_workloads_path() == packaged


def test_source_workload_catalog_is_used_in_editable_checkout(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(registry, "_PACKAGED_WORKLOADS", tmp_path / "missing")
    monkeypatch.setattr(registry, "_SOURCE_WORKLOADS", source)
    assert bundled_workloads_path() == source


def test_missing_bundled_workload_catalog_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "_PACKAGED_WORKLOADS", tmp_path / "missing")
    monkeypatch.setattr(registry, "_SOURCE_WORKLOADS", None)
    with pytest.raises(FileNotFoundError) as excinfo:
        bundled_workloads_path()
    assert str(excinfo.value) == "bundled workload catalog is not installed"


def test_workload_catalog_adds_user_profiles_but_bundled_ids_win(tmp_path):
    bundled_root = tmp_path / "bundled"
    user_root = tmp_path / "user"
    bundled = sample_manifest("shared-profile")
    user_collision = sample_manifest("shared-profile")
    user_collision["defaults"]["weight"] = 5
    user_only = sample_manifest("user-profile")
    write_workload(bundled_root, bundled)
    write_workload(user_root, user_collision)
    write_workload(user_root, user_only)

    catalog = load_workload_catalog(user_root, bundled_root=bundled_root)

    assert sorted(catalog) == ["shared-profile", "user-profile"]
    assert catalog["shared-profile"].default_policy().weight == 2


def test_combined_catalog_enforces_the_global_workload_bound():
    bundled = {f"bundled-{index}": object() for index in range(MAX_WORKLOADS)}
    assert len(merge_workloads(bundled, {})) == MAX_WORKLOADS
    with pytest.raises(ManifestError, match="combined catalog"):
        merge_workloads(bundled, {"user-extra": object()})


def test_packaged_schema_wins_over_source_fallback(monkeypatch, tmp_path):
    packaged = tmp_path / "package-schemas"
    source = tmp_path / "source-schemas"
    packaged.mkdir()
    source.mkdir()
    name = "contract.schema.json"
    (packaged / name).write_text('{"type":"integer"}')
    (source / name).write_text('{"type":"string"}')
    monkeypatch.setattr(registry, "_PACKAGED_SCHEMAS", packaged)
    monkeypatch.setattr(registry, "_SOURCE_SCHEMAS", source)

    assert validate_document(name, 7) == []
    assert validate_document(name, "wrong")


def test_source_schema_is_used_in_editable_checkout(monkeypatch, tmp_path):
    source = tmp_path / "source-schemas"
    source.mkdir()
    name = "contract.schema.json"
    (source / name).write_text('{"type":"integer"}')
    monkeypatch.setattr(registry, "_PACKAGED_SCHEMAS", tmp_path / "missing")
    monkeypatch.setattr(registry, "_SOURCE_SCHEMAS", source)

    assert validate_document(name, 7) == []


def test_schema_updates_apply_without_restart(monkeypatch, tmp_path):
    packaged = tmp_path / "package-schemas"
    packaged.mkdir()
    name = "contract.schema.json"
    path = packaged / name
    monkeypatch.setattr(registry, "_PACKAGED_SCHEMAS", packaged)
    monkeypatch.setattr(registry, "_SOURCE_SCHEMAS", None)

    path.write_text('{"type":"integer"}')
    assert validate_document(name, 7) == []
    path.write_text('{"type":"string"}')
    assert validate_document(name, 7)


def test_missing_canonical_schema_never_uses_unrelated_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "_PACKAGED_SCHEMAS", tmp_path / "missing")
    monkeypatch.setattr(registry, "_SOURCE_SCHEMAS", None)
    with pytest.raises(FileNotFoundError, match="canonical schema is not installed"):
        registry.load_schema("contract.schema.json")
