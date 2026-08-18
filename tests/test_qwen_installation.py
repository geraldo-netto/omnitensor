from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

import pytest

import omnitensor.qwen_installation as installation
from omnitensor.plugins import PluginMetadata, PluginSource, resolve_plugin_identities
from omnitensor.plugins.artifact_installation import PinnedArtifactInstallationError
from omnitensor.plugins.artifacts import ArtifactReference
from omnitensor.qwen_installation import (
    QWEN_REFERENCE,
    QwenArtifactSources,
    QwenInstallationError,
    install_qwen_artifacts,
)
from omnitensor.registry import validate_document

ROOT = Path(__file__).parents[1]
PROVIDERS = ROOT / "providers"
MANIFESTS = ROOT / "plugin-manifests"


def _source(tmp_path: Path, name: str, content: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path.resolve()


def _reference(artifact_id: str, model_format: str, content: bytes, companions=()):
    return ArtifactReference(
        artifact_id,
        "1.0.0",
        model_format,
        hashlib.sha256(content).hexdigest(),
        companions,
    )


def _small_sources(tmp_path: Path):
    content = {
        "qwen": b"shared-qwen",
        "param": b"bge-param",
        "bin": b"bge-bin",
        "tokenizer": b"bge-tokenizer",
    }
    paths = {name: _source(tmp_path, f"{name}.bin", value) for name, value in content.items()}
    qwen = _reference("qwen-shared", "gguf", content["qwen"])
    bge = _reference(
        "bge-ask",
        "ncnn",
        content["param"],
        (
            ("model.bin", hashlib.sha256(content["bin"]).hexdigest()),
            ("tokenizer.json", hashlib.sha256(content["tokenizer"]).hexdigest()),
        ),
    )
    sources = QwenArtifactSources(paths["qwen"], paths["param"], paths["bin"], paths["tokenizer"])
    return content, paths, sources, qwen, bge


def test_installer_verifies_and_atomically_installs_shared_qwen_and_bge(tmp_path, monkeypatch):
    content, _paths, sources, qwen, bge = _small_sources(tmp_path)
    monkeypatch.setattr(installation, "QWEN_REFERENCE", qwen)
    monkeypatch.setattr(installation, "BGE_REFERENCE", bge)
    monkeypatch.setattr(installation, "QWEN_SIZE_BYTES", len(content["qwen"]))

    document = install_qwen_artifacts(
        tmp_path / "artifacts",
        sources,
        accepted_qwen_license="Apache-2.0",
        accepted_bge_license="MIT",
    )

    assert document["licensesAccepted"] == {"bge": "MIT", "qwen": "Apache-2.0"}
    assert [item["id"] for item in document["artifacts"]] == [
        "qwen-shared",
        "bge-ask",
    ]
    assert document["artifacts"][1]["companions"] == bge.declared_companions
    for item in document["artifacts"]:
        assert Path(item["path"]).is_file()


@pytest.mark.parametrize(
    ("qwen_license", "bge_license"),
    [("", "MIT"), ("Apache-2.0", ""), ("MIT", "Apache-2.0")],
)
def test_installer_requires_exact_explicit_license_acceptance(tmp_path, qwen_license, bge_license):
    sources = QwenArtifactSources(*(tmp_path / name for name in ("a", "b", "c", "d")))

    with pytest.raises(QwenInstallationError, match="explicitly name Apache-2.0 and MIT"):
        install_qwen_artifacts(
            tmp_path / "artifacts",
            sources,
            accepted_qwen_license=qwen_license,
            accepted_bge_license=bge_license,
        )


def test_installer_verifies_every_source_before_creating_the_store(tmp_path, monkeypatch):
    content, paths, sources, qwen, bge = _small_sources(tmp_path)
    paths["tokenizer"].write_bytes(b"substituted")
    monkeypatch.setattr(installation, "QWEN_REFERENCE", qwen)
    monkeypatch.setattr(installation, "BGE_REFERENCE", bge)
    monkeypatch.setattr(installation, "QWEN_SIZE_BYTES", len(content["qwen"]))

    class UnexpectedInstaller:
        def __init__(self, _root):
            raise AssertionError("store must not be touched until all inputs verify")

    monkeypatch.setattr(installation, "ArtifactInstaller", UnexpectedInstaller)

    with pytest.raises(QwenInstallationError, match="digest does not match"):
        install_qwen_artifacts(
            tmp_path / "artifacts",
            sources,
            accepted_qwen_license="Apache-2.0",
            accepted_bge_license="MIT",
        )


def test_source_verification_refuses_missing_relative_wrong_size_and_wrong_digest(
    tmp_path, monkeypatch
):
    missing = tmp_path / "missing"
    with pytest.raises(QwenInstallationError) as missing_error:
        installation._verify_source(missing, "0" * 64, None)
    assert str(missing_error.value) == "provider artifact is unavailable"

    relative = Path("relative-model.gguf")
    monkeypatch.chdir(tmp_path)
    relative.write_bytes(b"model")
    with pytest.raises(QwenInstallationError) as relative_error:
        installation._verify_source(relative, hashlib.sha256(b"model").hexdigest(), None)
    assert str(relative_error.value) == "provider artifacts must be absolute regular files"

    absolute = relative.resolve()
    with pytest.raises(QwenInstallationError) as size_error:
        installation._verify_source(absolute, hashlib.sha256(b"model").hexdigest(), 6)
    assert str(size_error.value) == "Qwen GGUF size does not match its pinned release"
    with pytest.raises(QwenInstallationError) as digest_error:
        installation._verify_source(absolute, "0" * 64, 5)
    assert str(digest_error.value) == "provider artifact digest does not match its manifest"


def test_installed_document_is_the_exact_bounded_public_receipt(tmp_path):
    reference = ArtifactReference(
        "model-id",
        "2.3.4",
        "gguf",
        "a" * 64,
        (("tokenizer.json", "b" * 64),),
    )
    path = (tmp_path / "model.gguf").resolve()

    document = installation._installed_document(reference, path)
    assert document == {
        "id": "model-id",
        "version": "2.3.4",
        "format": "gguf",
        "sha256": "a" * 64,
        "path": str(path),
        "companions": {"tokenizer.json": "b" * 64},
    }
    assert list(document) == ["id", "version", "format", "sha256", "path", "companions"]


def test_qwen_error_name_is_a_compatible_shared_alias():
    assert QwenInstallationError is PinnedArtifactInstallationError
    assert str(QwenInstallationError("refused")) == "refused"


def test_qwen_verifier_preserves_digest_and_error_monkeypatch_hooks(tmp_path, monkeypatch):
    source = _source(tmp_path, "model.gguf", b"model")
    digest = hashlib.sha256(b"model").hexdigest()
    observed = []
    monkeypatch.setattr(
        installation,
        "file_digest",
        lambda path: observed.append(path) or digest,
    )
    installation._verify_source(source, digest, None)
    assert observed == [source]

    class HookError(PinnedArtifactInstallationError):
        pass

    monkeypatch.setattr(installation, "QwenInstallationError", HookError)
    monkeypatch.setattr(installation, "file_digest", lambda _path: "0" * 64)
    with pytest.raises(HookError, match="digest does not match"):
        installation._verify_source(source, digest, None)


def test_parser_exposes_the_exact_required_path_and_license_contract():
    parser = installation._parser()
    actions = {
        action.dest: (tuple(action.option_strings), action.type, action.required)
        for action in parser._actions
        if action.dest != "help"
    }

    assert parser.description == ("Verify and install pinned artifacts for Qwen workload providers")
    assert actions == {
        "artifact_root": (("--artifact-root",), Path, True),
        "qwen_model": (("--qwen-model",), Path, True),
        "bge_param": (("--bge-param",), Path, True),
        "bge_bin": (("--bge-bin",), Path, True),
        "bge_tokenizer": (("--bge-tokenizer",), Path, True),
        "accept_qwen_license": (("--accept-qwen-license",), None, True),
        "accept_bge_license": (("--accept-bge-license",), None, True),
    }


def test_cli_passes_all_sources_and_prints_stable_receipt(tmp_path, monkeypatch, capsys):
    observed = {}

    def fake_install(root, sources, **licenses):
        observed.update(root=root, sources=sources, licenses=licenses)
        return {"version": 1, "artifacts": []}

    monkeypatch.setattr(installation, "install_qwen_artifacts", fake_install)
    values = [str((tmp_path / name).resolve()) for name in ("qwen", "param", "bin", "tok")]
    installation.main(
        [
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--qwen-model",
            values[0],
            "--bge-param",
            values[1],
            "--bge-bin",
            values[2],
            "--bge-tokenizer",
            values[3],
            "--accept-qwen-license",
            "Apache-2.0",
            "--accept-bge-license",
            "MIT",
        ]
    )

    assert observed["sources"] == QwenArtifactSources(*(Path(value) for value in values))
    assert observed["licenses"] == {
        "accepted_qwen_license": "Apache-2.0",
        "accepted_bge_license": "MIT",
    }
    assert json.loads(capsys.readouterr().out) == {"version": 1, "artifacts": []}


def test_cli_exposes_stable_installation_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(
        installation,
        "install_qwen_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(QwenInstallationError("refused")),
    )
    repeated = ["--qwen-model", "--bge-param", "--bge-bin", "--bge-tokenizer"]
    argv = ["--artifact-root", str(tmp_path)]
    for option in repeated:
        argv.extend((option, str(tmp_path / option[2:])))
    argv.extend(("--accept-qwen-license", "Apache-2.0", "--accept-bge-license", "MIT"))

    with pytest.raises(SystemExit, match="refused"):
        installation.main(argv)


@pytest.mark.parametrize(
    ("directory", "plugin_id", "module", "artifact_ids"),
    [
        (
            "event-extraction",
            "event-extraction",
            "omnitensor_event_extraction",
            ["qwen3-5-9b-iq4-xs", "qwen3-8b-q4-k-m"],
        ),
        (
            "ask-selected-files",
            "ask-selected-files",
            "omnitensor_ask_selected_files",
            ["qwen3-5-9b-iq4-xs", "qwen3-8b-q4-k-m", "bge-small-en-v1-5-ask-gpu"],
        ),
        (
            "selected-text-tools",
            "selected-text-tools",
            "omnitensor_selected_text_tools",
            ["qwen3-5-9b-iq4-xs", "qwen3-8b-q4-k-m", "dictalm2-hebrew-q4-k-m"],
        ),
        (
            "file-organizer",
            "file-organizer",
            "omnitensor_file_organizer",
            ["qwen3-5-9b-iq4-xs", "qwen3-8b-q4-k-m"],
        ),
    ],
)
def test_provider_distribution_manifest_and_entry_point_identity_agree(
    directory, plugin_id, module, artifact_ids
):
    provider = PROVIDERS / directory
    project = tomllib.loads((provider / "pyproject.toml").read_text(encoding="utf-8"))
    manifest_path = MANIFESTS / f"{plugin_id}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry_points = project["project"]["entry-points"]["omnitensor.workloads"]
    force_include = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    assert entry_points == {plugin_id: f"{module}:create"}
    assert manifest["id"] == manifest["plugin"]["entryPoint"] == plugin_id
    assert [item["id"] for item in manifest["plugin"]["artifacts"]] == artifact_ids
    qwen_artifacts = [
        artifact
        for artifact in manifest["plugin"]["artifacts"]
        if artifact["id"] == QWEN_REFERENCE.id
    ]
    assert qwen_artifacts == [
        {
            "id": QWEN_REFERENCE.id,
            "version": QWEN_REFERENCE.version,
            "format": QWEN_REFERENCE.format,
            "sha256": QWEN_REFERENCE.sha256,
            "sourceUri": QWEN_REFERENCE.source_uri,
            "licenseSpdx": QWEN_REFERENCE.license_spdx,
        }
    ]
    assert "accelerator:gpu" in manifest["plugin"]["permissions"]
    assert validate_document("workload-manifest.schema.json", manifest) == []
    assert force_include == {
        f"../../plugin-manifests/{plugin_id}.json": f"{module}/omnitensor-plugin.json"
    }
    assert (provider / "src" / module / "__init__.py").is_file()

    catalog = resolve_plugin_identities(
        (
            PluginMetadata(
                PluginSource.EXTERNAL,
                plugin_id,
                entry_points[plugin_id],
                project["project"]["name"],
                project["project"]["version"],
                (manifest_path,),
                None,
                "",
            ),
        )
    )
    assert [item.plugin_id for item in catalog.plugins] == [plugin_id]
    assert catalog.rejections == ()


def test_qwen8b_catalog_pins_official_source_and_shared_qualification():
    catalog = json.loads((ROOT / "generation-models/qwen3-8b.json").read_text())

    assert catalog["id"] == "qwen3-8b-q4-k-m"
    assert catalog["license"]["spdx"] == "Apache-2.0"
    assert catalog["source"] == {
        "uri": (
            "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/"
            "7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q4_K_M.gguf"
        ),
        "filename": "Qwen3-8B-Q4_K_M.gguf",
        "sha256": installation.QWEN_REFERENCE.sha256,
        "sizeBytes": installation.QWEN_SIZE_BYTES,
    }
    assert catalog["qualification"] == {
        "receipt": ("providers/vulkan-runtime/src/omnitensor_vulkan_runtime/qualification.json"),
        "device": "AMD Radeon RX 6600 XT (RADV NAVI23)",
        "cpuFallback": "forbidden",
        "workloads": [
            "ask-selected-files",
            "event-extraction",
            "file-organizer",
            "selected-text-tools",
        ],
        "scope": (
            "frozen per-workload acceptance on the named GPU; arbitrary-domain quality not claimed"
        ),
    }
