from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import omnitensor.media_installation as installation
from omnitensor.media_installation import (
    MediaArtifactSources,
    MediaInstallationError,
    install_media_artifacts,
)
from omnitensor.plugins.artifacts import ArtifactReference


def _source(tmp_path: Path, name: str, content: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path.resolve()


def _small_contract(tmp_path: Path, monkeypatch):
    content = {"vision": b"vision", "projector": b"projector", "speech": b"speech"}
    paths = {name: _source(tmp_path, f"{name}.bin", value) for name, value in content.items()}
    projector_digest = hashlib.sha256(content["projector"]).hexdigest()
    vision = ArtifactReference(
        "vision-test",
        "1.0.0",
        "gguf",
        hashlib.sha256(content["vision"]).hexdigest(),
        (("mmproj.gguf", projector_digest),),
    )
    speech = ArtifactReference(
        "speech-test",
        "1.0.0",
        "ggml-whisper",
        hashlib.sha256(content["speech"]).hexdigest(),
    )
    monkeypatch.setattr(installation, "VISION_REFERENCE", vision)
    monkeypatch.setattr(installation, "SPEECH_REFERENCE", speech)
    monkeypatch.setattr(installation, "VISION_MODEL_SIZE_BYTES", len(content["vision"]))
    monkeypatch.setattr(installation, "VISION_PROJECTOR_SIZE_BYTES", len(content["projector"]))
    monkeypatch.setattr(installation, "SPEECH_MODEL_SIZE_BYTES", len(content["speech"]))
    sources = MediaArtifactSources(paths["vision"], paths["projector"], paths["speech"])
    return content, paths, sources, vision, speech


def test_installer_verifies_and_installs_pinned_vision_and_speech(tmp_path, monkeypatch):
    _content, _paths, sources, vision, speech = _small_contract(tmp_path, monkeypatch)

    receipt = install_media_artifacts(
        tmp_path / "artifacts",
        sources,
        accepted_qwen_license="Apache-2.0",
        accepted_whisper_license="MIT",
    )

    assert receipt["licensesAccepted"] == {"qwen": "Apache-2.0", "whisper": "MIT"}
    assert [item["id"] for item in receipt["artifacts"]] == [vision.id, speech.id]
    assert receipt["artifacts"][0]["companions"] == vision.declared_companions
    assert all(Path(item["path"]).is_file() for item in receipt["artifacts"])


@pytest.mark.parametrize(
    ("qwen", "whisper"),
    [("", "MIT"), ("Apache-2.0", ""), ("MIT", "Apache-2.0")],
)
def test_installer_requires_exact_license_acceptance(tmp_path, qwen, whisper):
    sources = MediaArtifactSources(*(tmp_path / name for name in ("a", "b", "c")))

    with pytest.raises(MediaInstallationError, match="explicitly name Apache-2.0 and MIT"):
        install_media_artifacts(
            tmp_path / "artifacts",
            sources,
            accepted_qwen_license=qwen,
            accepted_whisper_license=whisper,
        )


def test_all_sources_verify_before_store_creation(tmp_path, monkeypatch):
    _content, paths, sources, _vision, _speech = _small_contract(tmp_path, monkeypatch)
    paths["speech"].write_bytes(b"substituted")

    class UnexpectedInstaller:
        def __init__(self, _root):
            raise AssertionError("store touched before all source verification")

    monkeypatch.setattr(installation, "ArtifactInstaller", UnexpectedInstaller)
    with pytest.raises(MediaInstallationError, match="size does not match"):
        install_media_artifacts(
            tmp_path / "artifacts",
            sources,
            accepted_qwen_license="Apache-2.0",
            accepted_whisper_license="MIT",
        )


def test_source_verifier_refuses_missing_relative_size_and_digest(tmp_path, monkeypatch):
    with pytest.raises(MediaInstallationError, match="unavailable"):
        installation._verify_source(tmp_path / "missing", "0" * 64, 1)

    monkeypatch.chdir(tmp_path)
    relative = Path("relative.bin")
    relative.write_bytes(b"model")
    digest = hashlib.sha256(b"model").hexdigest()
    with pytest.raises(MediaInstallationError, match="absolute regular"):
        installation._verify_source(relative, digest, 5)
    with pytest.raises(MediaInstallationError, match="size does not match"):
        installation._verify_source(relative.resolve(), digest, 6)
    with pytest.raises(MediaInstallationError, match="digest does not match"):
        installation._verify_source(relative.resolve(), "0" * 64, 5)


def test_parser_exposes_exact_paths_and_licenses():
    actions = {
        action.dest: (tuple(action.option_strings), action.type, action.required)
        for action in installation._parser()._actions
        if action.dest != "help"
    }
    assert actions == {
        "artifact_root": (("--artifact-root",), Path, True),
        "vision_model": (("--vision-model",), Path, True),
        "vision_projector": (("--vision-projector",), Path, True),
        "speech_model": (("--speech-model",), Path, True),
        "accept_qwen_license": (("--accept-qwen-license",), None, True),
        "accept_whisper_license": (("--accept-whisper-license",), None, True),
    }


def test_cli_passes_sources_and_prints_receipt(tmp_path, monkeypatch, capsys):
    observed = {}

    def fake_install(root, sources, **licenses):
        observed.update(root=root, sources=sources, licenses=licenses)
        return {"version": 1, "artifacts": []}

    monkeypatch.setattr(installation, "install_media_artifacts", fake_install)
    values = [(tmp_path / name).resolve() for name in ("vision", "projector", "speech")]
    installation.main(
        [
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--vision-model",
            str(values[0]),
            "--vision-projector",
            str(values[1]),
            "--speech-model",
            str(values[2]),
            "--accept-qwen-license",
            "Apache-2.0",
            "--accept-whisper-license",
            "MIT",
        ]
    )
    assert observed["sources"] == MediaArtifactSources(*values)
    assert observed["licenses"] == {
        "accepted_qwen_license": "Apache-2.0",
        "accepted_whisper_license": "MIT",
    }
    assert json.loads(capsys.readouterr().out) == {"version": 1, "artifacts": []}


def test_cli_surfaces_stable_refusal(tmp_path, monkeypatch):
    def refused(*_args, **_kwargs):
        raise MediaInstallationError("refused")

    monkeypatch.setattr(installation, "install_media_artifacts", refused)
    argv = ["--artifact-root", str(tmp_path)]
    for option in ("--vision-model", "--vision-projector", "--speech-model"):
        argv.extend((option, str(tmp_path / option[2:])))
    argv.extend(("--accept-qwen-license", "Apache-2.0"))
    argv.extend(("--accept-whisper-license", "MIT"))
    with pytest.raises(SystemExit, match="refused"):
        installation.main(argv)


def test_pinned_references_match_catalogs():
    root = Path(__file__).parents[1]
    qwen = json.loads((root / "generation-models/qwen2-5-vl-7b.json").read_text())
    whisper = json.loads((root / "generation-models/whisper-small.json").read_text())
    sources = {item["role"]: item for item in qwen["sources"]}

    assert installation.VISION_REFERENCE.sha256 == sources["model"]["sha256"]
    assert sources["model"]["sizeBytes"] == installation.VISION_MODEL_SIZE_BYTES
    assert (
        installation.VISION_REFERENCE.declared_companions["mmproj.gguf"]
        == (sources["projector"]["sha256"])
    )
    assert sources["projector"]["sizeBytes"] == installation.VISION_PROJECTOR_SIZE_BYTES
    assert installation.SPEECH_REFERENCE.sha256 == whisper["source"]["sha256"]
    assert whisper["source"]["sizeBytes"] == installation.SPEECH_MODEL_SIZE_BYTES
