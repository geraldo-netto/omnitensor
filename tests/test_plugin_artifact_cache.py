from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins import (
    ArtifactActivation,
    ArtifactCache,
    ArtifactCacheAccounting,
    ArtifactCacheCollection,
    ArtifactCacheError,
    ArtifactInstaller,
    ArtifactReference,
)


def reference(content: bytes, *, artifact_id: str = "sample-model", version: str):
    return ArtifactReference(
        artifact_id,
        version,
        "onnx",
        hashlib.sha256(content).hexdigest(),
    )


def install_version(
    tmp_path: Path,
    store: Path,
    content: bytes,
    *,
    artifact_id: str = "sample-model",
    version: str,
) -> ArtifactReference:
    artifact = reference(content, artifact_id=artifact_id, version=version)
    source = tmp_path / f"{artifact_id}-{version}.onnx"
    source.write_bytes(content)
    ArtifactInstaller(store).install(artifact, source)
    return artifact


def install_three_versions(tmp_path: Path, store: Path):
    v1 = install_version(tmp_path, store, b"version-one", version="1.0.0")
    v2 = install_version(tmp_path, store, b"version-two", version="2.0.0")
    v3 = install_version(tmp_path, store, b"version-three", version="3.0.0")
    return v1, v2, v3


def version_bytes(store: Path, artifact: ArtifactReference) -> int:
    return sum(path.stat().st_size for path in (store / artifact.id / artifact.version).iterdir())


def test_cache_reports_protected_and_unreferenced_accounting(tmp_path):
    store = tmp_path / "store"
    v1, v2, v3 = install_three_versions(tmp_path, store)

    accounting = ArtifactCache(store, max_bytes=10_000, max_items=10).accounting()

    sizes = {item.version: version_bytes(store, item) for item in (v1, v2, v3)}
    assert accounting == ArtifactCacheAccounting(
        total_items=3,
        total_bytes=sum(sizes.values()),
        protected_items=2,
        protected_bytes=sizes["2.0.0"] + sizes["3.0.0"],
        unreferenced_items=1,
        unreferenced_bytes=sizes["1.0.0"],
    )


def test_item_quota_collects_only_unreferenced_version_and_preserves_rollback(tmp_path):
    store = tmp_path / "store"
    v1, v2, v3 = install_three_versions(tmp_path, store)
    cache = ArtifactCache(store, max_bytes=10_000, max_items=2)

    collection = cache.enforce()

    assert collection.removed == (v1,)
    assert collection.before.total_items == 3
    assert collection.after.total_items == 2
    assert not (store / v1.id / v1.version).exists()
    assert (store / v2.id / v2.version).is_dir()
    assert (store / v3.id / v3.version).is_dir()
    installer = ArtifactInstaller(store)
    assert installer.activation(v1.id) == ArtifactActivation(v3, v2)
    installer.rollback(v1.id)
    assert installer.activation(v1.id) == ArtifactActivation(v2, v3)


def test_byte_quota_removes_unreferenced_bytes_exactly(tmp_path):
    store = tmp_path / "store"
    v1, _v2, _v3 = install_three_versions(tmp_path, store)
    initial = ArtifactCache(store, max_bytes=10_000, max_items=10).accounting()
    cache = ArtifactCache(
        store,
        max_bytes=initial.protected_bytes,
        max_items=10,
    )

    collection = cache.enforce()

    assert collection.removed == (v1,)
    assert collection.after.total_bytes == initial.protected_bytes
    assert collection.after.unreferenced_bytes == 0


def test_collection_is_noop_when_cache_is_within_both_quotas(tmp_path):
    store = tmp_path / "store"
    install_three_versions(tmp_path, store)
    cache = ArtifactCache(store, max_bytes=10_000, max_items=3)
    before = cache.accounting()
    assert cache.enforce() == ArtifactCacheCollection(before, before, ())


def test_exact_byte_and_item_boundaries_do_not_collect(tmp_path):
    store = tmp_path / "store"
    install_three_versions(tmp_path, store)
    measured = ArtifactCache(store, max_bytes=100_000, max_items=10).accounting()
    cache = ArtifactCache(
        store,
        max_bytes=measured.total_bytes,
        max_items=measured.total_items,
    )
    assert cache.enforce() == ArtifactCacheCollection(measured, measured, ())


def test_unsatisfiable_protected_quota_fails_before_any_collection(tmp_path):
    store = tmp_path / "store"
    v1, v2, v3 = install_three_versions(tmp_path, store)
    cache = ArtifactCache(store, max_bytes=10_000, max_items=1)

    with pytest.raises(ArtifactCacheError) as excinfo:
        cache.enforce()

    assert excinfo.value.code == "quota-unsatisfiable"
    assert excinfo.value.detail == "active and rollback artifacts exceed cache quotas"
    assert str(excinfo.value) == (
        "quota-unsatisfiable: active and rollback artifacts exceed cache quotas"
    )
    assert all((store / item.id / item.version).is_dir() for item in (v1, v2, v3))


def test_collection_order_is_deterministic_across_artifact_identities(tmp_path):
    store = tmp_path / "store"
    alpha = [
        install_version(
            tmp_path,
            store,
            f"alpha-{version}".encode(),
            artifact_id="alpha",
            version=f"{version}.0.0",
        )
        for version in (1, 2, 3)
    ]
    beta = [
        install_version(
            tmp_path,
            store,
            f"beta-{version}".encode(),
            artifact_id="beta",
            version=f"{version}.0.0",
        )
        for version in (1, 2, 3)
    ]

    collection = ArtifactCache(store, max_bytes=100_000, max_items=5).enforce()

    assert collection.removed == (alpha[0],)
    assert not (store / "alpha" / "1.0.0").exists()
    assert (store / "beta" / "1.0.0").is_dir()
    assert ArtifactInstaller(store).activation("beta") == ArtifactActivation(beta[2], beta[1])


@pytest.mark.parametrize(
    ("max_bytes", "max_items", "reason"),
    [
        (-1, 1, "max_bytes must be a non-negative integer"),
        (True, 1, "max_bytes must be a non-negative integer"),
        (1, -1, "max_items must be a non-negative integer"),
        (1, False, "max_items must be a non-negative integer"),
    ],
)
def test_cache_configuration_is_strict(tmp_path, max_bytes, max_items, reason):
    with pytest.raises(ValueError) as excinfo:
        ArtifactCache(tmp_path, max_bytes=max_bytes, max_items=max_items)
    assert str(excinfo.value) == reason


def test_empty_cache_accepts_zero_quotas(tmp_path):
    cache = ArtifactCache(tmp_path / "store", max_bytes=0, max_items=0)
    empty = ArtifactCacheAccounting(0, 0, 0, 0, 0, 0)
    assert cache.accounting() == empty
    assert cache.enforce() == ArtifactCacheCollection(empty, empty, ())


def test_cache_creates_missing_parent_directories(tmp_path):
    cache = ArtifactCache(tmp_path / "nested" / "store", max_bytes=0, max_items=0)
    assert cache.accounting() == ArtifactCacheAccounting(0, 0, 0, 0, 0, 0)


def test_invalid_artifact_identity_directory_is_rejected(tmp_path):
    store = tmp_path / "store"
    (store / "Bad Directory").mkdir(parents=True)
    with pytest.raises(ArtifactCacheError) as excinfo:
        ArtifactCache(store, max_bytes=0, max_items=0).accounting()
    assert excinfo.value.code == "cache-state-invalid"
    assert excinfo.value.detail == "invalid artifact cache directory: Bad Directory"


def test_hidden_cache_entries_cannot_bypass_accounting(tmp_path):
    store = tmp_path / "store"
    (store / ".hidden-artifact").mkdir(parents=True)
    with pytest.raises(ArtifactCacheError) as excinfo:
        ArtifactCache(store, max_bytes=0, max_items=0).accounting()
    assert excinfo.value.code == "cache-state-invalid"
    assert excinfo.value.detail == "invalid artifact cache directory: .hidden-artifact"


@pytest.mark.parametrize(
    ("corrupt", "reason"),
    [
        (
            lambda document: document.update(extra=True),
            "artifact metadata has invalid fields",
        ),
        (
            lambda document: document.update(version=2),
            "artifact metadata version is invalid",
        ),
        (
            lambda document: document["artifact"].update(id="other"),
            "artifact cache identity does not match",
        ),
    ],
)
def test_corrupt_metadata_stops_collection_without_deleting_versions(tmp_path, corrupt, reason):
    store = tmp_path / "store"
    v1, v2, v3 = install_three_versions(tmp_path, store)
    metadata = store / v1.id / v1.version / "artifact.json"
    document = json.loads(metadata.read_text())
    corrupt(document)
    metadata.write_text(json.dumps(document))

    with pytest.raises(ArtifactCacheError) as excinfo:
        ArtifactCache(store, max_bytes=0, max_items=0).enforce()

    assert excinfo.value.code == "cache-state-invalid"
    assert excinfo.value.detail == reason
    assert all((store / item.id / item.version).is_dir() for item in (v1, v2, v3))


def test_corrupt_activation_stops_collection_fail_closed(tmp_path):
    store = tmp_path / "store"
    artifact = install_version(tmp_path, store, b"model", version="1.0.0")
    (store / artifact.id / "activation.json").write_text("not-json")
    with pytest.raises(ArtifactCacheError) as excinfo:
        ArtifactCache(store, max_bytes=0, max_items=0).enforce()
    assert excinfo.value.code == "cache-state-invalid"
    assert excinfo.value.detail.startswith("cannot read ")
    assert (store / artifact.id / artifact.version).is_dir()


def test_symlinked_activation_is_rejected_before_protection_is_computed(tmp_path):
    store = tmp_path / "store"
    artifact = install_version(tmp_path, store, b"model", version="1.0.0")
    activation = store / artifact.id / "activation.json"
    outside = tmp_path / "activation.json"
    activation.replace(outside)
    activation.symlink_to(outside)

    with pytest.raises(ArtifactCacheError) as excinfo:
        ArtifactCache(store, max_bytes=0, max_items=0).enforce()

    assert excinfo.value.code == "cache-state-invalid"
    assert excinfo.value.detail == "artifact activation path is a symlink"
    assert (store / artifact.id / artifact.version).is_dir()


def test_hidden_version_entries_cannot_bypass_accounting(tmp_path):
    store = tmp_path / "store"
    artifact = install_version(tmp_path, store, b"model", version="1.0.0")
    (store / artifact.id / ".stale").mkdir()

    with pytest.raises(ArtifactCacheError) as excinfo:
        ArtifactCache(store, max_bytes=100_000, max_items=10).accounting()

    assert excinfo.value.code == "cache-state-invalid"
    assert excinfo.value.detail == ("invalid artifact version directory: sample-model/.stale")


def test_invalid_stored_reference_preserves_cache_error_code(tmp_path):
    store = tmp_path / "store"
    artifact = install_version(tmp_path, store, b"model", version="1.0.0")
    metadata = store / artifact.id / artifact.version / "artifact.json"
    document = json.loads(metadata.read_text())
    document["artifact"]["sha256"] = "invalid"
    metadata.write_text(json.dumps(document))

    with pytest.raises(ArtifactCacheError) as excinfo:
        ArtifactCache(store, max_bytes=0, max_items=0).accounting()

    assert excinfo.value.code == "cache-state-invalid"
    assert excinfo.value.detail == "invalid artifact sha256"


def test_symlinked_identity_directory_is_never_followed_or_deleted(tmp_path):
    store = tmp_path / "store"
    outside = tmp_path / "outside"
    outside.mkdir()
    store.mkdir()
    (store / "linked-model").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactCacheError) as excinfo:
        ArtifactCache(store, max_bytes=0, max_items=0).enforce()

    assert excinfo.value.code == "cache-state-invalid"
    assert outside.is_dir()


def test_nested_or_symlinked_version_entries_are_never_collected(tmp_path):
    store = tmp_path / "store"
    v1, _v2, _v3 = install_three_versions(tmp_path, store)
    nested = store / v1.id / v1.version / "nested"
    nested.mkdir()

    with pytest.raises(ArtifactCacheError) as excinfo:
        ArtifactCache(store, max_bytes=0, max_items=0).enforce()

    assert excinfo.value.code == "cache-state-invalid"
    assert excinfo.value.detail == "artifact version contains unsafe entry: nested"
    assert nested.is_dir()


def test_primary_artifact_must_remain_a_regular_file(tmp_path):
    store = tmp_path / "store"
    v1, _v2, _v3 = install_three_versions(tmp_path, store)
    primary = store / v1.id / v1.version / "model.onnx"
    primary.unlink()
    primary.symlink_to(tmp_path / "missing")

    with pytest.raises(ArtifactCacheError) as excinfo:
        ArtifactCache(store, max_bytes=0, max_items=0).enforce()

    assert excinfo.value.code == "cache-state-invalid"
    assert excinfo.value.detail == "artifact cache primary file is invalid"


def test_remove_entry_preflight_rejects_nested_content_without_partial_delete(tmp_path):
    import omnitensor.plugins.artifact_cache as cache_module

    version_root = tmp_path / "sample-model" / "1.0.0"
    version_root.mkdir(parents=True)
    primary = version_root / "model.onnx"
    primary.write_bytes(b"model")
    nested = version_root / "nested"
    nested.mkdir()
    entry = cache_module._CacheEntry(reference(b"model", version="1.0.0"), version_root, 5)

    with pytest.raises(ArtifactCacheError) as excinfo:
        ArtifactCache(tmp_path, max_bytes=0, max_items=0)._remove_entry(entry)

    assert excinfo.value.code == "cache-delete-failed"
    assert excinfo.value.detail == "artifact version contains unsafe entry: nested"
    assert primary.is_file()


def test_remove_entry_rejects_path_beneath_symlinked_parent(tmp_path):
    import omnitensor.plugins.artifact_cache as cache_module

    outside = tmp_path / "outside" / "1.0.0"
    outside.mkdir(parents=True)
    primary = outside / "model.onnx"
    primary.write_bytes(b"model")
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(outside.parent, target_is_directory=True)
    linked_version = linked_parent / "1.0.0"
    entry = cache_module._CacheEntry(reference(b"model", version="1.0.0"), linked_version, 5)

    with pytest.raises(ArtifactCacheError) as excinfo:
        ArtifactCache(tmp_path, max_bytes=0, max_items=0)._remove_entry(entry)

    assert excinfo.value.code == "cache-delete-failed"
    assert excinfo.value.detail == "artifact version path changed"
    assert primary.is_file()


def test_remove_entry_wraps_filesystem_failures(tmp_path, monkeypatch):
    import omnitensor.plugins.artifact_cache as cache_module

    version_root = tmp_path / "sample-model" / "1.0.0"
    version_root.mkdir(parents=True)
    primary = version_root / "model.onnx"
    primary.write_bytes(b"model")
    entry = cache_module._CacheEntry(reference(b"model", version="1.0.0"), version_root, 5)
    monkeypatch.setattr(Path, "unlink", lambda _path: (_ for _ in ()).throw(OSError("busy")))

    with pytest.raises(ArtifactCacheError) as excinfo:
        ArtifactCache(tmp_path, max_bytes=0, max_items=0)._remove_entry(entry)

    assert excinfo.value.code == "cache-delete-failed"
    assert excinfo.value.detail == "cannot remove sample-model@1.0.0: busy"
    assert primary.is_file()


def test_version_accounting_wraps_filesystem_failures(tmp_path, monkeypatch):
    import omnitensor.plugins.artifact_cache as cache_module

    version_root = tmp_path / "version"
    version_root.mkdir()

    def denied(path):
        if path == version_root:
            raise OSError("denied")
        return original_iterdir(path)

    original_iterdir = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", denied)
    with pytest.raises(ArtifactCacheError) as excinfo:
        cache_module._version_size(version_root)
    assert excinfo.value.code == "cache-state-invalid"
    assert excinfo.value.detail == "cannot account artifact version: denied"


@given(max_items=st.integers(min_value=0, max_value=5))
def test_protected_versions_survive_arbitrary_item_quotas(max_items):
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        store = tmp_path / "store"
        _v1, v2, v3 = install_three_versions(tmp_path, store)
        cache = ArtifactCache(store, max_bytes=100_000, max_items=max_items)
        if max_items < 2:
            with pytest.raises(ArtifactCacheError):
                cache.enforce()
        else:
            cache.enforce()
        assert (store / v2.id / v2.version).is_dir()
        assert (store / v3.id / v3.version).is_dir()
