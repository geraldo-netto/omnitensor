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
    ArtifactInstallationError,
    ArtifactInstaller,
    ArtifactReference,
    ArtifactResolution,
    ArtifactResolver,
    artifact_filename,
    artifact_reference_error,
    artifact_reference_from_document,
)


def reference(content: bytes = b"model", **overrides) -> ArtifactReference:
    values = {
        "id": "sample-model",
        "version": "1.2.3",
        "format": "onnx",
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    values.update(overrides)
    return ArtifactReference(**values)


def install(root: Path, artifact: ArtifactReference, content: bytes = b"model") -> Path:
    path = root / artifact.id / artifact.version / artifact_filename(artifact.format)
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    return path.resolve()


@pytest.mark.parametrize(
    ("model_format", "filename"),
    [
        ("tflite-edgetpu", "model.tflite"),
        ("tflite", "model.tflite"),
        ("onnx", "model.onnx"),
        ("openvino", "model.xml"),
        ("ncnn", "model.param"),
    ],
)
def test_artifact_filename_is_stable_for_each_executor_format(model_format, filename):
    assert artifact_filename(model_format) == filename


def test_artifact_filename_rejects_unknown_formats_exactly():
    with pytest.raises(ValueError) as excinfo:
        artifact_filename("pickle")
    assert str(excinfo.value) == "unsupported artifact format: pickle"


def test_resolver_returns_only_a_verified_regular_file(tmp_path):
    artifact = reference()
    path = install(tmp_path, artifact)

    resolution = ArtifactResolver([tmp_path]).resolve(artifact)

    assert resolution == ArtifactResolution(True, path, "", 5)


def test_resolver_reports_missing_and_corrupt_artifacts_without_a_path(tmp_path):
    artifact = reference()
    resolver = ArtifactResolver([tmp_path])
    assert resolver.resolve(artifact) == ArtifactResolution(
        False,
        None,
        "artifact not installed: sample-model@1.2.3",
        0,
    )

    install(tmp_path, artifact, b"corrupt")
    assert resolver.resolve(artifact) == ArtifactResolution(
        False,
        None,
        "artifact sha256 mismatch",
        7,
    )


def test_resolver_uses_ordered_precedence_without_bypassing_a_corrupt_first_root(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    artifact = reference()
    install(first, artifact, b"bad")
    install(second, artifact)

    resolution = ArtifactResolver([first, second]).resolve(artifact)

    assert resolution.reason == "artifact sha256 mismatch"
    assert resolution.path is None


def test_resolver_continues_from_a_missing_root_to_the_next_root(tmp_path):
    missing = tmp_path / "missing"
    installed = tmp_path / "installed"
    artifact = reference()
    path = install(installed, artifact)
    assert ArtifactResolver([missing, installed]).resolve(artifact) == ArtifactResolution(
        True,
        path,
        "",
        5,
    )


def test_resolver_enforces_the_exact_size_boundary(tmp_path):
    accepted = reference(b"12345")
    install(tmp_path, accepted, b"12345")
    assert ArtifactResolver([tmp_path], max_artifact_bytes=5).resolve(accepted).ready is True

    rejected = reference(b"123456", id="large-model")
    install(tmp_path, rejected, b"123456")
    resolution = ArtifactResolver([tmp_path], max_artifact_bytes=5).resolve(rejected)
    assert resolution == ArtifactResolution(False, None, "artifact exceeds 5 bytes", 6)


def test_resolver_rejects_symlink_escape_from_an_allowlisted_root(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    artifact = reference()
    outside_path = install(outside, artifact)
    link = root / artifact.id / artifact.version / artifact_filename(artifact.format)
    link.parent.mkdir(parents=True)
    link.symlink_to(outside_path)

    resolution = ArtifactResolver([root]).resolve(artifact)

    assert resolution == ArtifactResolution(
        False,
        None,
        f"artifact path escapes allowlisted root: {root.resolve()}",
        0,
    )


def test_resolver_rejects_a_directory_at_the_artifact_path(tmp_path):
    artifact = reference()
    path = tmp_path / artifact.id / artifact.version / artifact_filename(artifact.format)
    path.mkdir(parents=True)
    assert ArtifactResolver([tmp_path]).resolve(artifact) == ArtifactResolution(
        False,
        None,
        f"artifact is not a regular file: {path.resolve()}",
        0,
    )


def test_resolver_reports_read_errors_without_exposing_the_path(tmp_path, monkeypatch):
    artifact = reference()
    install(tmp_path, artifact)
    resolver = ArtifactResolver([tmp_path])

    def denied(_path):
        raise OSError("denied")

    monkeypatch.setattr(resolver, "_digest", denied)
    assert resolver.resolve(artifact) == ArtifactResolution(
        False,
        None,
        "artifact cannot be read: denied",
        0,
    )


def test_resolver_rechecks_the_bytes_read_when_a_file_grows(tmp_path, monkeypatch):
    artifact = reference()
    install(tmp_path, artifact)
    resolver = ArtifactResolver([tmp_path], max_artifact_bytes=5)
    monkeypatch.setattr(resolver, "_digest", lambda _path: (artifact.sha256, 6))
    assert resolver.resolve(artifact) == ArtifactResolution(
        False,
        None,
        "artifact exceeds 5 bytes",
        6,
    )


def test_resolver_rejects_invalid_configuration():
    with pytest.raises(ValueError) as empty:
        ArtifactResolver([])
    assert str(empty.value) == "at least one artifact root is required"

    with pytest.raises(ValueError) as non_positive:
        ArtifactResolver([Path("models")], max_artifact_bytes=0)
    assert str(non_positive.value) == "max_artifact_bytes must be positive"
    assert ArtifactResolver([Path("models")], max_artifact_bytes=1)._max_artifact_bytes == 1


@pytest.mark.parametrize(
    ("artifact", "reason"),
    [
        (reference(id="../escape"), "invalid artifact id: '../escape'"),
        (reference(version="v1"), "invalid artifact version: 'v1'"),
        (reference(format="pickle"), "unsupported artifact format: pickle"),
        (reference(sha256="A" * 64), "invalid artifact sha256"),
    ],
)
def test_reference_validation_returns_one_stable_reason(artifact, reason):
    assert artifact_reference_error(artifact) == reason
    assert ArtifactResolver([Path("models")]).resolve(artifact) == ArtifactResolution(
        False,
        None,
        reason,
        0,
    )


def test_resolver_rejects_an_invalid_reference_before_path_lookup(tmp_path):
    artifact = reference(id="../escape")

    assert ArtifactResolver([tmp_path]).resolve(artifact) == ArtifactResolution(
        False,
        None,
        "invalid artifact id: '../escape'",
        0,
    )


def test_digest_reads_bounded_chunks_and_accumulates_the_total(monkeypatch):
    chunks = [b"ab", b"cd", b""]
    requested = []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, size):
            requested.append(size)
            return chunks.pop(0)

    monkeypatch.setattr(Path, "open", lambda _path, _mode: Stream())
    resolver = ArtifactResolver([Path("models")], max_artifact_bytes=3)
    assert resolver._digest(Path("model")) == (hashlib.sha256(b"ab").hexdigest(), 4)
    assert requested == [1024 * 1024, 1024 * 1024]


@given(
    artifact_id=st.text(max_size=100),
    version=st.text(max_size=60),
    digest=st.text(max_size=80),
)
def test_arbitrary_reference_metadata_never_escapes_or_raises(
    artifact_id,
    version,
    digest,
):
    artifact = ArtifactReference(artifact_id, version, "onnx", digest)
    with tempfile.TemporaryDirectory() as root:
        resolution = ArtifactResolver([Path(root)]).resolve(artifact)
    assert resolution.path is None
    assert resolution.ready is False


def test_installer_activates_verified_versions_and_rolls_back(tmp_path):
    store = tmp_path / "store"
    source_v1 = tmp_path / "v1.onnx"
    source_v2 = tmp_path / "v2.onnx"
    source_v1.write_bytes(b"version-one")
    source_v2.write_bytes(b"version-two")
    v1 = reference(b"version-one", version="1.0.0")
    v2 = reference(b"version-two", version="2.0.0")
    installer = ArtifactInstaller(store)

    first = installer.install(v1, source_v1)
    second = installer.install(v2, source_v2)

    assert first == first.__class__(v1, first.path, None)
    assert first.path.read_bytes() == b"version-one"
    assert second == second.__class__(v2, second.path, v1)
    assert installer.activation(v1.id) == ArtifactActivation(v2, v1)
    assert installer.resolve_active(v1.id).path == second.path
    rolled_back = installer.rollback(v1.id)
    assert rolled_back == rolled_back.__class__(v1, first.path, v2)
    assert installer.activation(v1.id) == ArtifactActivation(v1, v2)


def test_installer_rejects_invalid_references_with_stable_error(tmp_path):
    source = tmp_path / "model.onnx"
    source.write_bytes(b"model")
    installer = ArtifactInstaller(tmp_path / "store")

    with pytest.raises(ArtifactInstallationError) as excinfo:
        installer.install(reference(id="../escape"), source)

    assert excinfo.value.code == "reference-invalid"
    assert excinfo.value.detail == "invalid artifact id: '../escape'"
    assert str(excinfo.value) == "reference-invalid: invalid artifact id: '../escape'"


def test_installer_configuration_and_staging_are_exact(tmp_path, monkeypatch):
    import omnitensor.plugins.artifact_installation as installation_module

    with pytest.raises(ValueError) as excinfo:
        ArtifactInstaller(tmp_path, max_artifact_bytes=0)
    assert str(excinfo.value) == "max_artifact_bytes must be positive"

    source = tmp_path / "model.onnx"
    source.write_bytes(b"model")
    store = tmp_path / "store"
    calls = []
    real_mkdtemp = tempfile.mkdtemp

    def recording_mkdtemp(*, prefix, dir):
        calls.append((prefix, dir))
        return real_mkdtemp(prefix=prefix, dir=dir)

    monkeypatch.setattr(installation_module.tempfile, "mkdtemp", recording_mkdtemp)
    installer = ArtifactInstaller(store, max_artifact_bytes=5)
    installer.install(reference(), source)

    assert installer._root == store.resolve()
    assert installer._max_artifact_bytes == 5
    assert calls == [(".install-", store.resolve() / "sample-model")]


def test_installer_persists_exact_artifact_and_activation_metadata(tmp_path):
    source = tmp_path / "model.onnx"
    source.write_bytes(b"model")
    artifact = reference()
    store = tmp_path / "store"
    installer = ArtifactInstaller(store)

    installed = installer.install(artifact, source)

    assert json.loads((installed.path.parent / "artifact.json").read_text()) == {
        "version": 1,
        "artifact": {
            "id": "sample-model",
            "version": "1.2.3",
            "format": "onnx",
            "sha256": artifact.sha256,
        },
        # onnx is a single file, so it declares no companions; the key is
        # always written so a reader never has to distinguish "none" from
        # "written by a version that did not record them".
        "companions": {},
    }
    assert json.loads((store / artifact.id / "activation.json").read_text()) == {
        "version": 1,
        "artifactId": "sample-model",
        "active": {
            "id": "sample-model",
            "version": "1.2.3",
            "format": "onnx",
            "sha256": artifact.sha256,
        },
        "rollback": None,
    }


def test_idempotent_install_keeps_activation_and_returns_active_reference(tmp_path):
    source = tmp_path / "model.onnx"
    source.write_bytes(b"model")
    artifact = reference()
    installer = ArtifactInstaller(tmp_path / "store")
    first = installer.install(artifact, source)

    repeated = installer.install(artifact, source)

    assert repeated == repeated.__class__(artifact, first.path, artifact)
    assert installer.activation(artifact.id) == ArtifactActivation(artifact, None)


def test_invalid_install_never_changes_active_version_or_leaves_staging(tmp_path):
    store = tmp_path / "store"
    valid_source = tmp_path / "valid.onnx"
    invalid_source = tmp_path / "invalid.onnx"
    valid_source.write_bytes(b"valid")
    invalid_source.write_bytes(b"tampered")
    v1 = reference(b"valid", version="1.0.0")
    invalid_v2 = reference(b"expected", version="2.0.0")
    installer = ArtifactInstaller(store)
    installer.install(v1, valid_source)

    with pytest.raises(ArtifactInstallationError) as excinfo:
        installer.install(invalid_v2, invalid_source)

    assert excinfo.value.code == "digest-mismatch"
    assert installer.activation(v1.id) == ArtifactActivation(v1, None)
    assert not (store / v1.id / invalid_v2.version).exists()
    assert not list((store / v1.id).glob(".install-*"))


def test_interrupted_activation_preserves_previous_pointer(tmp_path, monkeypatch):
    import omnitensor.plugins.artifact_installation as installation_module

    store = tmp_path / "store"
    first_source = tmp_path / "first.onnx"
    second_source = tmp_path / "second.onnx"
    first_source.write_bytes(b"first")
    second_source.write_bytes(b"second")
    first = reference(b"first", version="1.0.0")
    second = reference(b"second", version="2.0.0")
    installer = ArtifactInstaller(store)
    installer.install(first, first_source)
    real_write = installation_module.write_json_atomic

    def interrupted(path, payload, prefix):
        if path.name == "activation.json":
            raise OSError("interrupted")
        return real_write(path, payload, prefix)

    monkeypatch.setattr(installation_module, "write_json_atomic", interrupted)
    with pytest.raises(ArtifactInstallationError) as excinfo:
        installer.install(second, second_source)

    assert excinfo.value.code == "activation-failed"
    assert installer.activation(first.id) == ArtifactActivation(first, None)
    assert installer.resolve_active(first.id).path.read_bytes() == b"first"
    assert (store / second.id / second.version / "model.onnx").read_bytes() == b"second"


def test_installer_rejects_symlink_sources_and_immutable_version_conflicts(tmp_path):
    store = tmp_path / "store"
    real_source = tmp_path / "real.onnx"
    source_link = tmp_path / "link.onnx"
    real_source.write_bytes(b"one")
    source_link.symlink_to(real_source)
    installer = ArtifactInstaller(store)
    first = reference(b"one", version="1.0.0")

    with pytest.raises(ArtifactInstallationError) as symlink_error:
        installer.install(first, source_link)
    assert symlink_error.value.code == "source-invalid"
    assert "Too many levels of symbolic links" in symlink_error.value.detail

    with pytest.raises(ArtifactInstallationError) as directory_error:
        installer.install(first, tmp_path)
    assert directory_error.value.code == "source-invalid"
    assert (
        directory_error.value.detail
        == "cannot open artifact: artifact source is not a regular file"
    )

    installer.install(first, real_source)
    (store / first.id / first.version / "model.onnx").write_bytes(b"changed")
    with pytest.raises(ArtifactInstallationError) as conflict:
        installer.install(first, real_source)
    assert conflict.value.code == "version-conflict"
    assert conflict.value.detail == (
        "installed version is not the requested immutable artifact: 1.0.0"
    )


def test_installer_enforces_size_and_rollback_boundaries(tmp_path):
    source = tmp_path / "model.onnx"
    source.write_bytes(b"12345")
    artifact = reference(b"12345")
    installer = ArtifactInstaller(tmp_path / "store", max_artifact_bytes=4)

    with pytest.raises(ArtifactInstallationError) as too_large:
        installer.install(artifact, source)
    assert too_large.value.code == "artifact-too-large"
    with pytest.raises(ArtifactInstallationError) as no_rollback:
        installer.rollback(artifact.id)
    assert no_rollback.value.code == "rollback-unavailable"
    assert no_rollback.value.detail == "artifact has no rollback version: sample-model"
    assert installer.resolve_active(artifact.id) == ArtifactResolution(
        False,
        None,
        "artifact has no active version: sample-model",
        0,
    )


def test_rollback_reverifies_the_preserved_version(tmp_path):
    first_source = tmp_path / "first.onnx"
    second_source = tmp_path / "second.onnx"
    first_source.write_bytes(b"first")
    second_source.write_bytes(b"second")
    first = reference(b"first", version="1.0.0")
    second = reference(b"second", version="2.0.0")
    store = tmp_path / "store"
    installer = ArtifactInstaller(store)
    installer.install(first, first_source)
    installer.install(second, second_source)
    (store / first.id / first.version / "model.onnx").write_bytes(b"corrupt")

    with pytest.raises(ArtifactInstallationError) as excinfo:
        installer.rollback(first.id)

    assert excinfo.value.code == "rollback-invalid"
    assert excinfo.value.detail == "artifact sha256 mismatch"
    assert installer.activation(first.id) == ArtifactActivation(second, first)


def test_artifact_store_rejects_symlinked_identity_directories(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    store = tmp_path / "store"
    store.mkdir()
    (store / "sample-model").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactInstallationError) as excinfo:
        ArtifactInstaller(store).activation("sample-model")

    assert excinfo.value.code == "store-path-invalid"
    assert excinfo.value.detail == "artifact directory escapes store root: sample-model"


@pytest.mark.parametrize(
    ("document", "detail"),
    [
        ({}, "activation has invalid fields"),
        (
            {"version": 2, "artifactId": "sample-model", "active": None, "rollback": None},
            "activation identity is invalid",
        ),
        (
            {
                "version": 1,
                "artifactId": "sample-model",
                "active": {
                    "id": "other-model",
                    "version": "1.0.0",
                    "format": "onnx",
                    "sha256": "0" * 64,
                },
                "rollback": None,
            },
            "active artifact id does not match",
        ),
        (
            {
                "version": 1,
                "artifactId": "sample-model",
                "active": None,
                "rollback": {
                    "id": "other-model",
                    "version": "1.0.0",
                    "format": "onnx",
                    "sha256": "0" * 64,
                },
            },
            "rollback artifact id does not match",
        ),
    ],
)
def test_activation_metadata_fails_closed(tmp_path, document, detail):
    state = tmp_path / "store/sample-model/activation.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps(document))

    with pytest.raises(ArtifactInstallationError) as excinfo:
        ArtifactInstaller(tmp_path / "store").activation("sample-model")

    assert excinfo.value.code == "activation-state-invalid"
    assert excinfo.value.detail == detail


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"id": 3, "version": "1.0.0", "format": "onnx", "sha256": "0" * 64},
        {"id": "model", "version": "1.0.0", "format": "onnx", "sha256": "bad"},
    ],
)
def test_stored_artifact_references_fail_closed(document):
    with pytest.raises(ArtifactInstallationError) as excinfo:
        artifact_reference_from_document(document)
    assert excinfo.value.code == "metadata-invalid"


def ncnn_pair(tmp_path, graph=b"7767517\n2 2\n", weights=b"WEIGHTS" * 100):
    param = tmp_path / "src.param"
    param.write_bytes(graph)
    binary = tmp_path / "src.bin"
    binary.write_bytes(weights)
    return param, binary


def ncnn_reference(param, version="1.0.0"):
    from omnitensor.plugins.artifacts import ArtifactReference

    return ArtifactReference(
        "demo-ncnn", version, "ncnn", hashlib.sha256(param.read_bytes()).hexdigest()
    )


def test_an_ncnn_artifact_installs_its_weights_beside_the_graph(tmp_path):
    """The param is only the graph; without the bin nothing can execute."""
    param, binary = ncnn_pair(tmp_path)
    installer = ArtifactInstaller(tmp_path / "store")

    installed = installer.install(
        ncnn_reference(param), param, companions={"model.bin": binary}
    )

    assert (installed.path.parent / "model.param").is_file()
    assert (installed.path.parent / "model.bin").read_bytes() == binary.read_bytes()


def test_an_ncnn_artifact_without_its_weights_is_refused(tmp_path):
    """It would resolve ready and fail at load; refusing is the honest answer."""
    param, _binary = ncnn_pair(tmp_path)
    installer = ArtifactInstaller(tmp_path / "store")

    with pytest.raises(ArtifactInstallationError) as failure:
        installer.install(ncnn_reference(param), param)

    assert failure.value.code == "companion-missing"
    assert "model.bin" in failure.value.detail


def test_a_companion_the_format_does_not_declare_is_refused(tmp_path):
    """An unread file would be installed and digested but never load-bearing."""
    param, binary = ncnn_pair(tmp_path)
    installer = ArtifactInstaller(tmp_path / "store")

    with pytest.raises(ArtifactInstallationError) as failure:
        installer.install(
            ncnn_reference(param), param, companions={"model.bin": binary, "extra.dat": binary}
        )

    assert failure.value.code == "companion-unexpected"


def test_installed_weights_are_digested_and_recorded(tmp_path):
    param, binary = ncnn_pair(tmp_path)
    installer = ArtifactInstaller(tmp_path / "store")

    installed = installer.install(
        ncnn_reference(param), param, companions={"model.bin": binary}
    )

    metadata = json.loads((installed.path.parent / "artifact.json").read_text())
    assert metadata["companions"] == {
        "model.bin": hashlib.sha256(binary.read_bytes()).hexdigest()
    }


def test_replaced_weights_are_detected_after_installation(tmp_path):
    """The manifest pins only the graph, so this is what notices tampering."""
    param, binary = ncnn_pair(tmp_path)
    installer = ArtifactInstaller(tmp_path / "store")
    installed = installer.install(
        ncnn_reference(param), param, companions={"model.bin": binary}
    )

    assert installer.resolve_active("demo-ncnn").ready is True

    (installed.path.parent / "model.bin").write_bytes(b"different weights entirely")

    resolution = installer.resolve_active("demo-ncnn")
    assert resolution.ready is False
    assert "does not match its recorded digest" in resolution.reason


def test_removed_weights_are_detected_after_installation(tmp_path):
    param, binary = ncnn_pair(tmp_path)
    installer = ArtifactInstaller(tmp_path / "store")
    installed = installer.install(
        ncnn_reference(param), param, companions={"model.bin": binary}
    )

    (installed.path.parent / "model.bin").unlink()

    resolution = installer.resolve_active("demo-ncnn")
    assert resolution.ready is False
    assert "missing" in resolution.reason


def test_a_single_file_format_needs_no_companions(tmp_path):
    source = tmp_path / "model.onnx"
    source.write_bytes(b"onnx bytes")
    from omnitensor.plugins.artifacts import ArtifactReference

    reference = ArtifactReference(
        "demo-onnx", "1.0.0", "onnx", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    installer = ArtifactInstaller(tmp_path / "store")

    installer.install(reference, source)

    assert installer.resolve_active("demo-onnx").ready is True


def test_an_unreadable_companion_is_reported(tmp_path):
    param, _binary = ncnn_pair(tmp_path)
    installer = ArtifactInstaller(tmp_path / "store")

    with pytest.raises(ArtifactInstallationError) as failure:
        installer.install(
            ncnn_reference(param), param, companions={"model.bin": tmp_path / "absent.bin"}
        )

    assert failure.value.code == "companion-unreadable"


def test_a_store_written_before_companions_existed_still_resolves(tmp_path):
    """An older store has no companions key and must not be read as corrupt."""
    source = tmp_path / "model.onnx"
    source.write_bytes(b"onnx bytes")
    from omnitensor.plugins.artifacts import ArtifactReference

    reference = ArtifactReference(
        "legacy", "1.0.0", "onnx", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    installer = ArtifactInstaller(tmp_path / "store")
    installed = installer.install(reference, source)

    metadata_path = installed.path.parent / "artifact.json"
    document = json.loads(metadata_path.read_text())
    del document["companions"]
    metadata_path.write_text(json.dumps(document))

    assert installer.resolve_active("legacy").ready is True


def test_a_manifest_declared_reference_resolves_through_the_installer(tmp_path):
    """The path the service takes when a profile actually declares a model.

    It went unexercised until one did, and was broken the whole time.
    """
    param, binary = ncnn_pair(tmp_path)
    installer = ArtifactInstaller(tmp_path / "store")
    reference = ncnn_reference(param)
    installer.install(reference, param, companions={"model.bin": binary})

    resolution = installer.resolve(reference)

    assert resolution.ready is True, resolution.reason
    assert Path(resolution.path).name == "model.param"


def test_a_declared_reference_whose_weights_changed_does_not_resolve(tmp_path):
    param, binary = ncnn_pair(tmp_path)
    installer = ArtifactInstaller(tmp_path / "store")
    reference = ncnn_reference(param)
    installed = installer.install(reference, param, companions={"model.bin": binary})

    (installed.path.parent / "model.bin").write_bytes(b"swapped")

    assert installer.resolve(reference).ready is False


def test_a_declared_reference_with_the_wrong_digest_does_not_resolve(tmp_path):
    from omnitensor.plugins.artifacts import ArtifactReference

    param, binary = ncnn_pair(tmp_path)
    installer = ArtifactInstaller(tmp_path / "store")
    installer.install(ncnn_reference(param), param, companions={"model.bin": binary})

    forged = ArtifactReference("demo-ncnn", "1.0.0", "ncnn", "0" * 64)
    assert installer.resolve(forged).ready is False


def openvino_pair(tmp_path):
    graph = tmp_path / "ir.xml"
    graph.write_bytes(b"<net name='m'/>")
    weights = tmp_path / "ir.bin"
    weights.write_bytes(b"WEIGHTS" * 40)
    return graph, weights


def test_an_openvino_artifact_also_carries_its_weights(tmp_path):
    """IR is a graph in .xml and weights in .bin — the same shape as ncnn."""
    from omnitensor.plugins.artifacts import ArtifactReference, companion_filenames

    assert companion_filenames("openvino") == ("model.bin",)
    graph, weights = openvino_pair(tmp_path)
    reference = ArtifactReference(
        "demo-ir", "1.0.0", "openvino", hashlib.sha256(graph.read_bytes()).hexdigest()
    )
    installer = ArtifactInstaller(tmp_path / "store")

    installed = installer.install(reference, graph, companions={"model.bin": weights})

    assert (installed.path.parent / "model.bin").read_bytes() == weights.read_bytes()
    assert installer.resolve_active("demo-ir").ready is True


def test_an_openvino_artifact_without_weights_is_refused(tmp_path):
    from omnitensor.plugins.artifacts import ArtifactReference

    graph, _weights = openvino_pair(tmp_path)
    reference = ArtifactReference(
        "demo-ir", "1.0.0", "openvino", hashlib.sha256(graph.read_bytes()).hexdigest()
    )

    with pytest.raises(ArtifactInstallationError, match="companion-missing"):
        ArtifactInstaller(tmp_path / "store").install(reference, graph)
