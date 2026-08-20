from __future__ import annotations

import ast
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
from importrules import forbidden_imports

import omnitensor.training.document_model as facade

# The tokenizer, the Vulkan runner and their types serve questions rather than
# produce models, so they live in the service package; only the producers that
# use them stayed behind in `training`.
from omnitensor import document_model_runners as runners
from omnitensor import document_model_types as types
from omnitensor.training import document_model_contracts as contracts
from omnitensor.training import document_model_cpu_reference as cpu_reference
from omnitensor.training import document_model_export as exporter
from omnitensor.training import document_model_gate as gate
from omnitensor.training import document_model_installation as installation

ROOT = Path(__file__).parents[1]
LEGACY_MODULE = "omnitensor.training.document_model"


def test_document_model_facade_preserves_type_identity_name_and_pickle_globals(tmp_path):
    owners = {
        "DocumentModelError": types.DocumentModelError,
        "TokenizedText": types.TokenizedText,
        "DocumentModelEvidence": types.DocumentModelEvidence,
        "InstalledDocumentModel": types.InstalledDocumentModel,
        "BgeTokenizer": runners.BgeTokenizer,
        "PortableBgeRunner": cpu_reference.PortableBgeRunner,
        "ProducerCpuBgeReferenceRunner": cpu_reference.ProducerCpuBgeReferenceRunner,
        "VulkanBgeRunner": runners.VulkanBgeRunner,
    }
    # Each says where it is defined. They used to claim
    # `omnitensor.training.document_model`, which stopped being true when the
    # trainers became their own distribution: a service-only machine has no
    # such module, so a pickle written by the serving path could not be read
    # back and a traceback named a file that was not installed.
    homes = {
        "DocumentModelError": "omnitensor.document_model_types",
        "TokenizedText": "omnitensor.document_model_types",
        "DocumentModelEvidence": "omnitensor.document_model_types",
        "InstalledDocumentModel": "omnitensor.document_model_types",
        "BgeTokenizer": "omnitensor.document_model_runners",
        "VulkanBgeRunner": "omnitensor.document_model_runners",
        "PortableBgeRunner": "omnitensor.training.document_model_cpu_reference",
        "ProducerCpuBgeReferenceRunner": "omnitensor.training.document_model_cpu_reference",
    }
    for name, owner in owners.items():
        assert getattr(facade, name) is owner
        assert owner.__module__ == homes[name]
        assert pickle.loads(pickle.dumps(owner)) is owner
    assert "omnitensor.training" not in pickle.dumps(types.TokenizedText((1,), (1,), (0,))).decode(
        "latin-1"
    )
    assert facade.PortableBgeRunner.__name__ == "PortableBgeRunner"
    assert facade.PortableBgeRunner is facade.ProducerCpuBgeReferenceRunner

    tokens = types.TokenizedText((1,), (1.0,), (0,))
    evidence = types.DocumentModelEvidence(0, "GPU", 2, 1.0, 1.0, 0.0, 1)
    installed = types.InstalledDocumentModel(
        tmp_path / "model", tmp_path / "binding", tmp_path / "report", evidence
    )
    for value in (tokens, evidence, installed):
        assert pickle.loads(pickle.dumps(value)) == value

    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        pickle.loads(pickle.dumps(types.DocumentModelError("code", "detail")))


def test_document_model_facade_maps_direct_leaf_owners_and_contains_no_classes():
    assert facade.source_path is contracts.source_path
    assert facade.append_ncnn_l2_normalization is exporter.append_ncnn_l2_normalization
    assert facade.maximum_embedding_error is gate.maximum_embedding_error
    assert facade.expected_retrieval_hits is gate.expected_retrieval_hits
    assert facade.report_document is gate.report_document
    assert facade.binding_document is installation.binding_document
    assert facade.native_tensor_contract is types.native_tensor_contract

    tree = ast.parse((ROOT / "src/omnitensor/training/document_model.py").read_text())
    assert not any(isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef)) for node in tree.body)


def test_document_model_private_facade_factories_resolve_live_legacy_seams(monkeypatch, tmp_path):
    fixed = object()
    monkeypatch.setattr(facade, "fixed_bge_model", lambda torch, encoder: (torch, encoder))
    assert facade._fixed_bge_model("torch", "encoder") == ("torch", "encoder")

    # One factory, resolved live: the branch that chose between this and a
    # rebound PortableBgeRunner could never take its second arm, since both
    # names are bound to the same object by the module that defines them.
    monkeypatch.setattr(facade, "_canonical_cpu_reference_factory", lambda model, tokenizer: fixed)
    assert facade._portable_reference_factory(tmp_path / "model", object()) is fixed
    assert facade.PortableBgeRunner is facade.ProducerCpuBgeReferenceRunner


def test_document_model_leaves_import_before_facade_and_never_import_it():
    served = ("document_model_types", "document_model_runners")
    modules = (
        "document_model_contracts",
        "document_model_cpu_reference",
        "document_model_export",
        "document_model_gate",
        "document_model_installation",
        "document_model_cli",
    )
    script = "\n".join(
        [
            "import importlib, sys",
            *(f"importlib.import_module('omnitensor.{name}')" for name in served),
            *(f"importlib.import_module('omnitensor.training.{name}')" for name in modules),
            "assert 'omnitensor.training.document_model' not in sys.modules",
        ]
    )
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        check=True,
    )

    assert (
        forbidden_imports(
            [ROOT / f"src/omnitensor/training/{name}.py" for name in modules],
            ["omnitensor.training.document_model"],
            source_root=ROOT / "src",
        )
        == []
    )


def test_cpu_reference_owner_is_absent_from_runtime_and_native_modules():
    allowed = {"document_model.py", "document_model_installation.py"}
    owners = {
        Path(violation.rsplit(":", 1)[0]).name
        for violation in forbidden_imports(
            (ROOT / "src/omnitensor/training").glob("*.py"),
            ["omnitensor.training.document_model_cpu_reference"],
            source_root=ROOT / "src",
        )
    }
    assert owners <= allowed

    # The embedder is its own distribution since OMNI-0517: BGE on ncnn shares
    # no code path with llama.cpp, so it left the generation wheel.
    provider = (
        ROOT / "providers/ncnn-embeddings/src/omnitensor_ncnn_embeddings/bge.py"
    ).read_text(encoding="utf-8")
    # The provider reaches the runner and its prefix in the service package, not
    # through the trainers: it is part of the serving path and must import on a
    # machine that never installed `omnitensor-training`.
    assert "from omnitensor.document_model_runners import BgeTokenizer, VulkanBgeRunner" in provider
    assert "from omnitensor.document_model_types import QUERY_PREFIX" in provider
    assert "omnitensor.training" not in provider.replace("`omnitensor.training`", "")
    assert "document_model_cpu_reference" not in provider


def test_document_model_cli_entrypoint_remains_on_stable_facade():
    project = (ROOT / "packaging" / "omnitensor-training" / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert (
        'omnitensor-install-document-model = "omnitensor.training.document_model:main"' in project
    )


def test_every_installation_dependency_says_what_it_must_accept():
    """OMNI-0471: seventeen fields annotated `object` documented nothing."""
    import typing

    from omnitensor.training.document_model_installation import (
        DocumentModelInstallationDependencies,
    )

    hints = typing.get_type_hints(DocumentModelInstallationDependencies)
    assert hints, "the dataclass must carry resolvable annotations"
    assert object not in hints.values()
    for name, hint in hints.items():
        assert "Callable" in str(hint), f"{name} is not described as a call"
