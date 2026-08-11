from __future__ import annotations

import copy
import json

from conftest import sample_manifest, write_workload

from omnitensor.registry import (
    apply_model_bindings,
    bundled_workloads_path,
    load_workload_catalog,
    load_workloads,
)


def model(model_format="ncnn", artifact_id="local-model-gpu"):
    return {
        "id": artifact_id,
        "version": "1.0.0",
        "format": model_format,
        "fullyQuantized": False,
        "minimumCompilerVersion": "0.0.0",
        "minimumRuntimeVersion": "0.0.0",
        "sha256": "0" * 64,
        "tensorContract": {"inputs": [{"shape": [1, 3], "dtype": "float32", "layout": "NC"}]},
        "outputContract": {"kind": "raw"},
    }


def bound_manifest(base):
    bound = copy.deepcopy(base)
    bound["requirements"].update(
        {
            "accelerator": "gpu",
            "acceleratorPreference": ["gpu"],
            "model": model(),
        }
    )
    return bound


def test_a_local_binding_can_supply_only_model_and_routing_fields(tmp_path):
    bundled_root = tmp_path / "bundled"
    bindings_root = tmp_path / "bindings"
    users_root = tmp_path / "users"
    base = sample_manifest("hardware-health")
    write_workload(bundled_root, base)
    write_workload(bindings_root, bound_manifest(base))

    catalog = load_workload_catalog(
        users_root,
        bundled_root=bundled_root,
        model_bindings_root=bindings_root,
    )

    assert catalog["hardware-health"].accelerator == "gpu"
    assert catalog["hardware-health"].model["id"] == "local-model-gpu"
    assert catalog["hardware-health"].default_policy().weight == 2


def test_a_binding_cannot_replace_policy_ui_or_pipeline(tmp_path, caplog):
    base = sample_manifest("hardware-health")
    changed = bound_manifest(base)
    changed["defaults"]["weight"] = 5
    bundled_root = tmp_path / "bundled"
    bindings_root = tmp_path / "bindings"
    write_workload(bundled_root, base)
    write_workload(bindings_root, changed)

    merged = apply_model_bindings(
        load_workloads(bundled_root),
        load_workloads(bindings_root),
    )

    assert merged["hardware-health"].model is None
    assert "non-model fields" in caplog.text


def test_a_binding_for_an_unknown_profile_is_skipped(tmp_path, caplog):
    bundled_root = tmp_path / "bundled"
    bindings_root = tmp_path / "bindings"
    write_workload(bundled_root, sample_manifest("known"))
    write_workload(bindings_root, bound_manifest(sample_manifest("unknown")))

    merged = apply_model_bindings(
        load_workloads(bundled_root),
        load_workloads(bindings_root),
    )

    assert set(merged) == {"known"}
    assert "unknown profile unknown" in caplog.text


def test_a_malformed_binding_does_not_stop_catalog_loading(tmp_path, caplog):
    bundled_root = tmp_path / "bundled"
    bindings_root = tmp_path / "bindings"
    write_workload(bundled_root, sample_manifest("known"))
    directory = bindings_root / "known"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text("not json")

    catalog = load_workload_catalog(
        tmp_path / "users",
        bundled_root=bundled_root,
        model_bindings_root=bindings_root,
    )

    assert catalog["known"].model is None
    assert "Skipping unloadable workload manifest" in caplog.text


def test_binding_is_a_full_canonical_manifest_not_an_unvalidated_patch(tmp_path):
    base = sample_manifest("hardware-health")
    binding = bound_manifest(base)
    path = tmp_path / "bindings" / "hardware-health" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(binding))

    loaded = load_workloads(tmp_path / "bindings")

    assert loaded["hardware-health"].manifest == binding


def test_production_service_reads_the_configured_binding_root(tmp_path, monkeypatch):
    from omnitensor.service import build_service_from_env

    base = load_workloads(bundled_workloads_path())["hardware-health"].manifest
    bindings = tmp_path / "bindings"
    write_workload(bindings, bound_manifest(base))
    monkeypatch.setenv("OMNITENSOR_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("OMNITENSOR_POLICY_PATH", str(tmp_path / "policy.json"))
    monkeypatch.setenv("OMNITENSOR_WORKLOADS", str(tmp_path / "workloads"))
    monkeypatch.setenv("OMNITENSOR_MODEL_BINDINGS", str(bindings))
    monkeypatch.setenv("OMNITENSOR_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    service = build_service_from_env()

    assert service._workloads["hardware-health"].model["id"] == "local-model-gpu"
