from __future__ import annotations

import hashlib
import json

import pytest

from omnitensor.preparation import (
    FORMAT_ACCELERATORS,
    PreparationError,
    file_digest,
    install_prepared,
    main,
    prepare_artifact,
)


def model_file(tmp_path, name="model.onnx", content=b"ONNX-ish bytes"):
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_a_model_file_is_described_as_the_contract_requires(tmp_path):
    source = model_file(tmp_path)

    prepared = prepare_artifact(source, artifact_id="minilm", version="1.0.0", model_format="onnx")

    assert prepared.reference.id == "minilm"
    assert prepared.reference.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert prepared.size_bytes == len(source.read_bytes())


def test_the_digest_is_computed_from_the_file_not_supplied(tmp_path):
    """A caller-provided digest only proves the caller can hash."""
    source = model_file(tmp_path, content=b"a" * 5000)

    prepared = prepare_artifact(source, artifact_id="m", version="1.0.0", model_format="onnx")

    assert prepared.reference.sha256 == file_digest(source)
    assert prepared.reference.sha256 == hashlib.sha256(b"a" * 5000).hexdigest()


def test_the_manifest_fragment_matches_the_installed_bytes(tmp_path):
    source = model_file(tmp_path)
    prepared = prepare_artifact(source, artifact_id="m", version="2.1.0", model_format="onnx")

    fragment = prepared.manifest_fragment()

    assert fragment == {
        "id": "m",
        "version": "2.1.0",
        "format": "onnx",
        "sha256": file_digest(source),
    }


def test_each_format_names_the_accelerator_it_can_run_on(tmp_path):
    """There is no CPU lane, so a format with no accelerator cannot be prepared."""
    for model_format, accelerator in FORMAT_ACCELERATORS.items():
        prepared = prepare_artifact(
            model_file(tmp_path, f"m.{model_format}"),
            artifact_id="m",
            version="1.0.0",
            model_format=model_format,
        )
        assert prepared.document()["accelerator"] == accelerator
    assert "cpu" not in set(FORMAT_ACCELERATORS.values())


def test_an_unsupported_format_is_refused(tmp_path):
    with pytest.raises(PreparationError, match="format-unsupported"):
        prepare_artifact(
            model_file(tmp_path), artifact_id="m", version="1.0.0", model_format="pickle"
        )


@pytest.mark.parametrize(
    "changes",
    [{"artifact_id": ""}, {"artifact_id": "x" * 200}, {"artifact_id": 7}, {"version": ""}],
)
def test_the_artifact_identity_is_validated(tmp_path, changes):
    options = {"artifact_id": "m", "version": "1.0.0", "model_format": "onnx", **changes}
    with pytest.raises(PreparationError, match="identity-invalid"):
        prepare_artifact(model_file(tmp_path), **options)


def test_a_missing_or_empty_file_is_refused(tmp_path):
    with pytest.raises(PreparationError, match="source-invalid"):
        prepare_artifact(
            tmp_path / "absent.onnx", artifact_id="m", version="1", model_format="onnx"
        )
    empty = model_file(tmp_path, "empty.onnx", b"")
    with pytest.raises(PreparationError, match="not a model"):
        prepare_artifact(empty, artifact_id="m", version="1", model_format="onnx")


def test_a_directory_is_not_a_model(tmp_path):
    directory = tmp_path / "model.onnx"
    directory.mkdir()
    with pytest.raises(PreparationError, match="source-invalid"):
        prepare_artifact(directory, artifact_id="m", version="1", model_format="onnx")


def test_a_prepared_artifact_installs_into_the_store_dispatch_reads(tmp_path):
    source = model_file(tmp_path, content=b"real model bytes")
    prepared = prepare_artifact(source, artifact_id="m", version="1.0.0", model_format="onnx")

    install_prepared(prepared, tmp_path / "artifacts")

    from omnitensor.plugins.artifact_installation import ArtifactInstaller

    resolution = ArtifactInstaller(tmp_path / "artifacts").resolve_active("m")
    assert resolution.ready, resolution.detail


def test_an_installed_artifact_is_rejected_when_the_digest_disagrees(tmp_path):
    """The store verifies what dispatch will verify, not what we claimed."""
    from omnitensor.plugins.artifact_installation import ArtifactInstaller
    from omnitensor.plugins.artifacts import ArtifactReference

    source = model_file(tmp_path, content=b"real model bytes")
    forged = ArtifactReference("m", "1.0.0", "onnx", "0" * 64)

    with pytest.raises(Exception) as failure:
        ArtifactInstaller(tmp_path / "artifacts").install(forged, source)
    assert "digest" in str(failure.value).lower() or "sha" in str(failure.value).lower()


def test_the_cli_reports_what_it_prepared(tmp_path, capsys):
    source = model_file(tmp_path, content=b"weights")

    code = main([str(source), "--id", "minilm", "--version", "1.0.0", "--format", "onnx"])

    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["artifact"]["sha256"] == file_digest(source)
    assert document["accelerator"] == "gpu"
    assert "installedAt" not in document


def test_the_cli_can_install_what_it_prepared(tmp_path, capsys):
    source = model_file(tmp_path, content=b"weights")

    code = main(
        [
            str(source),
            "--id",
            "minilm",
            "--version",
            "1.0.0",
            "--format",
            "onnx",
            "--install-root",
            str(tmp_path / "artifacts"),
        ]
    )

    assert code == 0
    assert "installedAt" in json.loads(capsys.readouterr().out)


def test_the_cli_reports_a_failure_without_a_traceback(tmp_path, capsys):
    code = main(
        [str(tmp_path / "absent.onnx"), "--id", "m", "--version", "1.0.0", "--format", "onnx"]
    )

    assert code == 1
    assert "preparation failed" in capsys.readouterr().err


def test_nothing_here_reaches_the_network():
    """Which weights to ship is a licensing decision, not a tool's to make."""
    import ast
    import inspect

    from omnitensor import preparation

    tree = ast.parse(inspect.getsource(preparation))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert not imported & {"urllib", "requests", "http", "socket", "ftplib"}


def test_an_installation_reports_where_the_artifact_landed(tmp_path):
    source = model_file(tmp_path, content=b"weights")
    prepared = prepare_artifact(source, artifact_id="m", version="1.0.0", model_format="onnx")

    landed = install_prepared(prepared, tmp_path / "artifacts")

    assert str(tmp_path) in str(landed)


def test_an_unreadable_file_is_reported_not_raised_as_an_oserror(tmp_path, monkeypatch):
    source = model_file(tmp_path, content=b"weights")

    def explode(self, *args, **kwargs):
        raise OSError("device error")

    monkeypatch.setattr("pathlib.Path.open", explode)

    with pytest.raises(PreparationError, match="cannot read"):
        prepare_artifact(source, artifact_id="m", version="1.0.0", model_format="onnx")
