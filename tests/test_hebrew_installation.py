from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import omnitensor.hebrew_installation as installation
from omnitensor.hebrew_installation import (
    HebrewInstallationError,
    install_hebrew_translation_model,
)
from omnitensor.plugins.artifacts import ArtifactReference

ROOT = Path(__file__).parents[1]


def _model(tmp_path: Path, content: bytes = b"dictalm") -> Path:
    path = (tmp_path / "dictalm.gguf").resolve()
    path.write_bytes(content)
    return path


def _pin(monkeypatch, content: bytes):
    reference = ArtifactReference(
        "dictalm-test", "1.0.0", "gguf", hashlib.sha256(content).hexdigest()
    )
    monkeypatch.setattr(installation, "DICTALM_REFERENCE", reference)
    monkeypatch.setattr(installation, "DICTALM_SIZE_BYTES", len(content))
    return reference


def test_installer_verifies_and_atomically_installs_the_optional_model(tmp_path, monkeypatch):
    content = b"dictalm"
    source = _model(tmp_path, content)
    reference = _pin(monkeypatch, content)

    receipt = install_hebrew_translation_model(
        tmp_path / "artifacts", source, accepted_license="Apache-2.0"
    )

    assert receipt["licenseAccepted"] == "Apache-2.0"
    assert receipt["artifact"] == {
        "id": reference.id,
        "version": reference.version,
        "format": reference.format,
        "sha256": reference.sha256,
        "path": str(tmp_path / "artifacts" / reference.id / reference.version / "model.gguf"),
    }
    assert Path(receipt["artifact"]["path"]).read_bytes() == content


@pytest.mark.parametrize("license_name", ["", "MIT", "apache-2.0"])
def test_installer_requires_exact_license_acceptance(tmp_path, license_name):
    with pytest.raises(HebrewInstallationError, match="explicitly name Apache-2.0"):
        install_hebrew_translation_model(
            tmp_path / "artifacts", tmp_path / "missing", accepted_license=license_name
        )


def test_installer_refuses_missing_relative_wrong_size_and_wrong_digest(tmp_path, monkeypatch):
    with pytest.raises(HebrewInstallationError, match="artifact is unavailable"):
        install_hebrew_translation_model(
            tmp_path / "artifacts",
            tmp_path / "missing",
            accepted_license="Apache-2.0",
        )
    source = _model(tmp_path, b"model")
    _pin(monkeypatch, b"other")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(HebrewInstallationError, match="absolute regular file"):
        install_hebrew_translation_model(
            tmp_path / "artifacts", Path(source.name), accepted_license="Apache-2.0"
        )
    monkeypatch.setattr(installation, "DICTALM_SIZE_BYTES", source.stat().st_size + 1)
    with pytest.raises(HebrewInstallationError, match="size does not match"):
        install_hebrew_translation_model(
            tmp_path / "artifacts", source, accepted_license="Apache-2.0"
        )
    monkeypatch.setattr(installation, "DICTALM_SIZE_BYTES", source.stat().st_size)
    with pytest.raises(HebrewInstallationError, match="digest does not match"):
        install_hebrew_translation_model(
            tmp_path / "artifacts", source, accepted_license="Apache-2.0"
        )


def test_cli_contract_and_receipt(tmp_path, monkeypatch, capsys):
    parser = installation._parser()
    actions = {
        action.dest: (tuple(action.option_strings), action.type, action.required)
        for action in parser._actions
        if action.dest != "help"
    }
    assert actions == {
        "artifact_root": (("--artifact-root",), Path, True),
        "model": (("--model",), Path, True),
        "accept_license": (("--accept-license",), None, True),
    }
    observed = {}

    def fake_install(root, model, **licenses):
        observed.update(root=root, model=model, licenses=licenses)
        return {"version": 1, "artifact": {}}

    monkeypatch.setattr(installation, "install_hebrew_translation_model", fake_install)
    installation.main(
        [
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--model",
            str(tmp_path / "model.gguf"),
            "--accept-license",
            "Apache-2.0",
        ]
    )
    assert observed == {
        "root": tmp_path / "artifacts",
        "model": tmp_path / "model.gguf",
        "licenses": {"accepted_license": "Apache-2.0"},
    }
    assert json.loads(capsys.readouterr().out) == {"version": 1, "artifact": {}}


def test_cli_exposes_stable_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(
        installation,
        "install_hebrew_translation_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(HebrewInstallationError("refused")),
    )
    with pytest.raises(SystemExit, match="refused"):
        installation.main(
            [
                "--artifact-root",
                str(tmp_path),
                "--model",
                str(tmp_path / "model"),
                "--accept-license",
                "Apache-2.0",
            ]
        )


def test_dictalm_catalog_pins_official_bytes_and_limits_its_route_to_hebrew_translation():
    catalog = json.loads((ROOT / "generation-models/dictalm2-hebrew.json").read_text())
    source = catalog["source"]
    revision = catalog["upstream"]["revision"]

    assert catalog["id"] == installation.DICTALM_REFERENCE.id
    assert catalog["license"]["spdx"] == "Apache-2.0"
    assert len(revision) == 40 and revision in source["uri"]
    assert source == {
        "uri": (
            f"https://huggingface.co/dicta-il/dictalm2.0-instruct-GGUF/resolve/{revision}/"
            "dictalm2.0-instruct-Q4_K_M.gguf"
        ),
        "filename": "dictalm2.0-instruct-Q4_K_M.gguf",
        "sha256": installation.DICTALM_REFERENCE.sha256,
        "sizeBytes": installation.DICTALM_SIZE_BYTES,
    }
    assert catalog["qualification"]["operation"] == (
        "selected-text translate with explicit Hebrew target only"
    )
    assert catalog["providers"] == [
        {
            "providerId": "dictalm2-hebrew-gpu",
            "accelerator": "gpu",
            "runtime": "llama.cpp-vulkan",
            "default": False,
            "status": "qualified-only-for-explicit-hebrew-translation",
        }
    ]
