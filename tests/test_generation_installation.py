from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path

import pytest

import omnitensor.generation_installation as installation
from omnitensor.generation_installation import (
    GENERATION_MODEL_REFERENCE,
    GenerationArtifactSources,
    GenerationInstallationError,
    install_generation_artifacts,
)
from omnitensor.plugins import PluginMetadata, PluginSource, resolve_plugin_identities
from omnitensor.plugins.artifact_installation import PinnedArtifactInstallationError
from omnitensor.plugins.artifacts import ArtifactReference
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
    sources = GenerationArtifactSources(
        paths["qwen"], paths["param"], paths["bin"], paths["tokenizer"]
    )
    return content, paths, sources, qwen, bge


def test_installer_verifies_and_atomically_installs_shared_qwen_and_bge(tmp_path, monkeypatch):
    content, _paths, sources, qwen, bge = _small_sources(tmp_path)
    monkeypatch.setattr(installation, "GENERATION_MODEL_REFERENCE", qwen)
    monkeypatch.setattr(installation, "BGE_REFERENCE", bge)
    monkeypatch.setattr(installation, "GENERATION_MODEL_SIZE_BYTES", len(content["qwen"]))

    document = install_generation_artifacts(
        tmp_path / "artifacts",
        sources,
        accepted_model_license="Apache-2.0",
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
    sources = GenerationArtifactSources(*(tmp_path / name for name in ("a", "b", "c", "d")))

    with pytest.raises(GenerationInstallationError, match="explicitly name Apache-2.0 and MIT"):
        install_generation_artifacts(
            tmp_path / "artifacts",
            sources,
            accepted_model_license=qwen_license,
            accepted_bge_license=bge_license,
        )


def test_installer_verifies_every_source_before_creating_the_store(tmp_path, monkeypatch):
    content, paths, sources, qwen, bge = _small_sources(tmp_path)
    paths["tokenizer"].write_bytes(b"substituted")
    monkeypatch.setattr(installation, "GENERATION_MODEL_REFERENCE", qwen)
    monkeypatch.setattr(installation, "BGE_REFERENCE", bge)
    monkeypatch.setattr(installation, "GENERATION_MODEL_SIZE_BYTES", len(content["qwen"]))

    class UnexpectedInstaller:
        def __init__(self, _root):
            raise AssertionError("store must not be touched until all inputs verify")

    monkeypatch.setattr(installation, "ArtifactInstaller", UnexpectedInstaller)

    with pytest.raises(GenerationInstallationError, match="digest does not match"):
        install_generation_artifacts(
            tmp_path / "artifacts",
            sources,
            accepted_model_license="Apache-2.0",
            accepted_bge_license="MIT",
        )


def test_source_verification_refuses_missing_relative_wrong_size_and_wrong_digest(
    tmp_path, monkeypatch
):
    missing = tmp_path / "missing"
    with pytest.raises(GenerationInstallationError) as missing_error:
        installation._verify_source(missing, "0" * 64, None)
    assert str(missing_error.value) == "provider artifact is unavailable"

    relative = Path("relative-model.gguf")
    monkeypatch.chdir(tmp_path)
    relative.write_bytes(b"model")
    with pytest.raises(GenerationInstallationError) as relative_error:
        installation._verify_source(relative, hashlib.sha256(b"model").hexdigest(), None)
    assert str(relative_error.value) == "provider artifacts must be absolute regular files"

    absolute = relative.resolve()
    with pytest.raises(GenerationInstallationError) as size_error:
        installation._verify_source(absolute, hashlib.sha256(b"model").hexdigest(), 6)
    assert str(size_error.value) == "Qwen GGUF size does not match its pinned release"
    with pytest.raises(GenerationInstallationError) as digest_error:
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
    assert GenerationInstallationError is PinnedArtifactInstallationError
    assert str(GenerationInstallationError("refused")) == "refused"


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

    monkeypatch.setattr(installation, "GenerationInstallationError", HookError)
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
        "accept_model_license": (("--accept-model-license",), None, True),
        "accept_bge_license": (("--accept-bge-license",), None, True),
    }


def test_cli_passes_all_sources_and_prints_stable_receipt(tmp_path, monkeypatch, capsys):
    observed = {}

    def fake_install(root, sources, **licenses):
        observed.update(root=root, sources=sources, licenses=licenses)
        return {"version": 1, "artifacts": []}

    monkeypatch.setattr(installation, "install_generation_artifacts", fake_install)
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
            "--accept-model-license",
            "Apache-2.0",
            "--accept-bge-license",
            "MIT",
        ]
    )

    assert observed["sources"] == GenerationArtifactSources(*(Path(value) for value in values))
    assert observed["licenses"] == {
        "accepted_model_license": "Apache-2.0",
        "accepted_bge_license": "MIT",
    }
    assert json.loads(capsys.readouterr().out) == {"version": 1, "artifacts": []}


def test_cli_exposes_stable_installation_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(
        installation,
        "install_generation_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(GenerationInstallationError("refused")),
    )
    repeated = ["--qwen-model", "--bge-param", "--bge-bin", "--bge-tokenizer"]
    argv = ["--artifact-root", str(tmp_path)]
    for option in repeated:
        argv.extend((option, str(tmp_path / option[2:])))
    argv.extend(("--accept-model-license", "Apache-2.0", "--accept-bge-license", "MIT"))

    with pytest.raises(SystemExit, match="refused"):
        installation.main(argv)


# What each workload's manifest mounts, keyed by workload. The keys are checked
# against the distributions actually in the tree below, in both directions: the
# list used to be four hand-typed tuples, which meant a fifth distribution — or
# a manifest with no distribution at all — was not a failing test but an absent
# one. `media-transcription` was absent exactly that way.
DECLARED_ARTIFACTS = {
    "ask-selected-files": [
        "qwen3-5-9b-iq4-xs",
        "qwen3-8b-q4-k-m",
        "bge-small-en-v1-5-ask-gpu",
        # The OCR extraction pair (OMNI-0615), fixed-role beside the embedder.
        "ppocrv6-medium-det",
        "ppocrv6-medium-rec",
    ],
    "document-translation": [
        "qwen3-5-9b-iq4-xs",
        "qwen3-8b-q4-k-m",
        "dictalm2-hebrew-q4-k-m",
    ],
    "event-extraction": ["qwen3-5-9b-iq4-xs", "qwen3-8b-q4-k-m"],
    "file-organizer": ["qwen3-5-9b-iq4-xs", "qwen3-8b-q4-k-m"],
    # Both vision models are selectable since OMNI-0587/0588 — the 9B is the
    # default (measured more robust, covers Hebrew), the 7B the alternative.
    "media-transcription": [
        "qwen2-5-vl-7b-instruct",
        "qwen3-5-9b-q4-k-m",
        "whisper-small-multilingual",
        # The OCR enrichment pair (OMNI-0607), fixed-role beside the
        # selectable vision models.
        "ppocrv6-medium-det",
        "ppocrv6-medium-rec",
    ],
    "selected-text-tools": ["qwen3-5-9b-iq4-xs", "qwen3-8b-q4-k-m", "dictalm2-hebrew-q4-k-m"],
}


def distributions() -> dict[str, tuple[Path, dict]]:
    """Every provider distribution that declares a workload entry point."""
    found: dict[str, tuple[Path, dict]] = {}
    for path in sorted(PROVIDERS.glob("*/pyproject.toml")):
        if "mutants" in path.parts:
            continue
        project = tomllib.loads(path.read_text(encoding="utf-8"))
        entry_points = project["project"].get("entry-points", {}).get("omnitensor.workloads", {})
        for workload_id in entry_points:
            assert workload_id not in found, f"{workload_id} is declared by two distributions"
            found[workload_id] = (path.parent, project)
    return found


def test_every_manifest_has_a_distribution_and_every_distribution_a_manifest():
    """Both directions of one correspondence.

    Only the first was ever checked, and only for a hand-typed four: a manifest
    the service publishes with nothing that installs it is a workload a person
    can see and never run, and a distribution with no manifest is discovered
    and then refuses to start.
    """
    shipped = {path.stem for path in MANIFESTS.glob("*.json")}

    assert set(distributions()) == shipped
    assert set(DECLARED_ARTIFACTS) == shipped


@pytest.mark.parametrize("plugin_id", sorted(distributions()))
def test_provider_distribution_manifest_and_entry_point_identity_agree(plugin_id):
    provider, project = distributions()[plugin_id]
    manifest_path = MANIFESTS / f"{plugin_id}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry_points = project["project"]["entry-points"]["omnitensor.workloads"]
    force_include = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    [module] = project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    module = module.removeprefix("src/")

    assert entry_points == {plugin_id: f"{module}:create"}
    assert manifest["id"] == manifest["plugin"]["entryPoint"] == plugin_id
    assert [item["id"] for item in manifest["plugin"]["artifacts"]] == DECLARED_ARTIFACTS[plugin_id]
    qwen_artifacts = [
        artifact
        for artifact in manifest["plugin"]["artifacts"]
        if artifact["id"] == GENERATION_MODEL_REFERENCE.id
    ]
    assert qwen_artifacts in (
        [],
        [
            {
                "id": GENERATION_MODEL_REFERENCE.id,
                "version": GENERATION_MODEL_REFERENCE.version,
                "format": GENERATION_MODEL_REFERENCE.format,
                "sha256": GENERATION_MODEL_REFERENCE.sha256,
                "sourceUri": GENERATION_MODEL_REFERENCE.source_uri,
                "licenseSpdx": GENERATION_MODEL_REFERENCE.license_spdx,
            }
        ],
    )
    assert "accelerator:gpu" in manifest["plugin"]["permissions"]
    assert validate_document("workload-manifest.schema.json", manifest) == []
    assert (
        force_include[f"../../plugin-manifests/{plugin_id}.json"]
        == f"{module}/omnitensor-plugin.json"
    )
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
        "sha256": installation.GENERATION_MODEL_REFERENCE.sha256,
        "sizeBytes": installation.GENERATION_MODEL_SIZE_BYTES,
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


def shim_distributions() -> dict[str, Path]:
    """Every distribution whose entry point is a re-export of a provider runtime.

    Derived, so a new one is covered by the rule the moment it exists, and no
    longer keyed to one runtime's package name: `ask-selected-files` stopped
    being a shim when its factory moved out of the generation wheel, and a
    rule that only recognised `omnitensor_vulkan_runtime` would have read that
    move as a regression. `media-transcription` is deliberately not one
    either: its shim re-exports from its own package, and its identity is
    checked by its own provider suite.
    """
    found = {}
    for provider, project in distributions().values():
        [module] = project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
        package = module.removeprefix("src/")
        source = (provider / "src" / package / "__init__.py").read_text(encoding="utf-8")
        if re.search(r"^from omnitensor_\w+ import \w+ as create$", source, re.MULTILINE):
            found[package] = provider
    return found


def composed_distributions() -> dict[str, Path]:
    """Every workload distribution that assembles rather than re-exports.

    Two are: `ask-selected-files`, which needs the llama.cpp runtime and the
    ncnn embedder and so builds its workload out of both, and
    `media-transcription`, whose provider is its own package. A distribution
    that is not a re-export states its own check, and this is where that is
    required rather than assumed.
    """
    shims = set(shim_distributions())
    found = {}
    for provider, project in distributions().values():
        [module] = project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
        package = module.removeprefix("src/")
        if package not in shims:
            found[package] = provider
    return found


def test_the_shim_identity_suites_state_their_one_check_once():
    """OMNI-0539: five byte-identical copies of the same file.

    Each said the same four structural things this module now derives from the
    tree, plus the one thing only an installed wheel can say. Reduced to that
    one, imported from `providers/shim_identity.py`, the copies differ in a
    single package name — so the next change to the shim contract is one edit
    rather than five that have to agree.
    """
    assert len(shim_distributions()) == 4

    bodies = set()
    for package, provider in shim_distributions().items():
        suite = provider / "tests" / "test_identity.py"
        text = suite.read_text(encoding="utf-8")
        assert f'PACKAGE = "{package}"' in text, f"{suite} does not name its own package"
        assert "from shim_identity import" in text, f"{suite} restates the shared check"
        bodies.add(text.replace(f'"{package}"', "<package>"))

    assert len(bodies) == 1, "the copies have drifted apart again"
    assert (PROVIDERS / "shim_identity.py").is_file()


def test_a_distribution_that_composes_rather_than_re_exports_states_its_own_check():
    """OMNI-0555: not every workload wheel is one import.

    `ask-selected-files` builds its workload from two distributions, so the
    shared shim check does not describe it and skipping it silently would
    leave the only assembled workload unproven where it is built.
    """
    composed = composed_distributions()

    assert set(composed) == {"omnitensor_ask_selected_files", "omnitensor_media_transcription"}
    for package, provider in composed.items():
        suites = sorted(path.name for path in (provider / "tests").glob("test_*.py"))
        assert suites, f"{package} states no check of its own"
