from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins import (
    ArtifactReference,
    ArtifactResolution,
    ArtifactResolver,
    artifact_filename,
    artifact_reference_error,
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
    artifact_id, version, digest,
):
    artifact = ArtifactReference(artifact_id, version, "onnx", digest)
    with tempfile.TemporaryDirectory() as root:
        resolution = ArtifactResolver([Path(root)]).resolve(artifact)
    assert resolution.path is None
    assert resolution.ready is False
