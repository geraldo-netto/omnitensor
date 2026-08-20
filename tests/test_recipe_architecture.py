from __future__ import annotations

import ast
import copy
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
from importrules import forbidden_imports, imported_modules

import omnitensor.registry as core_registry
import omnitensor.training.recipes as facade
from omnitensor.training import recipe_fetch as fetch
from omnitensor.training import recipe_model as model
from omnitensor.training import recipe_registry as registry
from omnitensor.training import recipe_transport as transport
from omnitensor.training import recipe_uri_policy as uri_policy
from omnitensor.training import recipe_validation as validation

ROOT = Path(__file__).parents[1]
LEGACY_MODULE = "omnitensor.training.recipes"


def test_facade_values_are_exact_leaf_objects_with_legacy_pickle_globals(tmp_path):
    names = (
        "ModelRecipeError",
        "ModelSource",
        "ModelLicense",
        "TargetClaim",
        "ModelRecipe",
        "FetchedModelSource",
        "SourceTransport",
    )
    for name in names:
        owner = getattr(model, name)
        assert getattr(facade, name) is owner
        assert owner.__module__ == LEGACY_MODULE
        assert pickle.loads(pickle.dumps(owner)) is owner

    assert facade.HttpsSourceTransport is transport.HttpsSourceTransport
    assert transport.HttpsSourceTransport.__module__ == LEGACY_MODULE
    adapter = transport.HttpsSourceTransport(2.5)
    restored = pickle.loads(pickle.dumps(adapter))
    assert type(restored) is transport.HttpsSourceTransport
    assert restored._timeout_seconds == 2.5

    recipe = facade.load_model_recipe(ROOT / "model-recipes/bge-small-en-v1-5.json")
    fetched = model.FetchedModelSource(recipe, tmp_path, tmp_path / "receipt.json")
    values = (
        recipe.model_source,
        recipe.license,
        next(iter(recipe.targets.values())),
        recipe,
        fetched,
    )
    for value in values:
        assert type(value).__module__ == LEGACY_MODULE
        assert pickle.loads(pickle.dumps(value)) == value

    error = model.ModelRecipeError("code", "detail")
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        pickle.loads(pickle.dumps(error))


def test_facade_keeps_public_and_private_compatibility_names():
    assert facade.load_model_recipe is registry.load_model_recipe
    assert facade._source_download_uri is uri_policy.source_download_uri
    assert facade._validate_download_response_uri is uri_policy.validate_download_response_uri
    assert facade._validate_https_uri is uri_policy.validate_https_uri
    assert facade._source_file_matches is fetch.source_file_matches
    assert facade._installed_source_matches is fetch.installed_source_matches
    assert facade._fetch_one is fetch.fetch_one
    assert facade._finite_json is validation.finite_json
    assert facade.MAX_BUNDLED_RECIPES == registry.MAX_BUNDLED_RECIPES
    assert facade.MAX_TOTAL_SOURCE_BYTES == validation.MAX_TOTAL_SOURCE_BYTES
    assert facade.DOWNLOAD_CHUNK_BYTES == transport.DOWNLOAD_CHUNK_BYTES


def test_direct_fetch_owners_wire_their_canonical_dependencies(monkeypatch, tmp_path):
    sentinel = object()
    calls = []

    def fetch_sources(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(fetch, "_fetch_model_sources", fetch_sources)
    assert (
        fetch.fetch_model_sources(
            "recipe.json",
            tmp_path,
            accepted_license="Apache-2.0",
            transport=None,
        )
        is sentinel
    )
    assert calls[0][0] == ("recipe.json", tmp_path)
    assert calls[0][1]["recipe_loader"] is registry.load_model_recipe
    assert calls[0][1]["installed_matcher"] is fetch.installed_source_matches
    assert calls[0][1]["rename"] is fetch.os.rename

    calls.clear()
    monkeypatch.setattr(fetch, "_open_fetched_model_source", fetch_sources)
    assert fetch.open_fetched_model_source("recipe.json", tmp_path) is sentinel
    assert calls[0][0] == ("recipe.json", tmp_path)
    assert calls[0][1] == {
        "recipe_loader": registry.load_model_recipe,
        "installed_matcher": fetch.installed_source_matches,
    }


def test_legacy_private_producer_wrappers_dispatch_through_registry():
    producer = {"sourceOutputShape": [1, 1], "outputShape": [1, 1]}

    facade._validate_producer_contract({}, "identity")
    facade._validate_specialized_producer({}, producer, [], "identity")


def test_workload_contract_accessor_uses_one_snapshot_and_returns_independent_definitions(
    monkeypatch,
):
    tensor = {"type": "object", "properties": {"inputs": {"type": "array"}}}
    output = {"type": "object", "properties": {"kind": {"type": "string"}}}
    schema = {
        "$defs": {
            "model": {
                "properties": {
                    "tensorContract": tensor,
                    "outputContract": output,
                }
            }
        }
    }
    calls = []

    def load(name):
        calls.append(name)
        return schema

    monkeypatch.setattr(core_registry, "load_schema", load)

    definitions = core_registry.workload_model_contract_schemas()

    assert calls == ["workload-manifest.schema.json"]
    assert definitions == {"tensorContract": tensor, "outputContract": output}
    assert definitions["tensorContract"] is not tensor
    assert definitions["outputContract"] is not output
    definitions["tensorContract"]["type"] = "changed"
    assert tensor["type"] == "object"


def test_producer_registry_is_closed_immutable_and_drives_explicit_identity():
    assert tuple(validation.PRODUCER_VALIDATORS) == (
        "identity",
        "sentence-embedding",
        "clip-image-embedding",
        "retinexformer-image-enhancement",
        "timeseries-point-forecast",
        "timeseries-quantile-forecast",
    )
    with pytest.raises(TypeError):
        validation.PRODUCER_VALIDATORS["new-kind"] = object()

    document = {
        "tensorContract": {"inputs": [{"shape": [1, 1]}]},
        "producer": {
            "kind": "identity",
            "inputNames": ["input"],
            "sourceOutputShape": [1, 1],
            "outputShape": [1, 1],
        },
    }
    validation.validate_producer(copy.deepcopy(document))
    document["producer"]["outputShape"] = [1, 2]
    with pytest.raises(model.ModelRecipeError, match="identity producer cannot change"):
        validation.validate_producer(document)


def test_recipe_leaves_are_independent_and_import_before_facade():
    leaf_names = (
        "recipe_model",
        "recipe_uri_policy",
        "recipe_transport",
        "recipe_validation",
        "recipe_registry",
        "recipe_fetch",
    )
    assert (
        forbidden_imports(
            [ROOT / f"src/omnitensor/training/{name}.py" for name in leaf_names],
            ["omnitensor.training.recipes"],
            source_root=ROOT / "src",
        )
        == []
    )

    statement = ";".join(f"import omnitensor.training.{name}" for name in (*leaf_names, "recipes"))
    completed = subprocess.run(
        [sys.executable, "-c", statement],
        check=False,
        capture_output=True,
        text=True,
    )
    assert (completed.returncode, completed.stderr) == (0, "")


def test_production_callers_use_leaf_owners_and_facade_remains_class_free():
    expected = {
        "recipe_cli.py": {"recipe_fetch", "recipe_model", "recipe_registry"},
        "production_pipeline.py": {"recipe_model"},
        "embedding_production.py": {"recipe_fetch", "recipe_model"},
        "clip_production.py": {"recipe_fetch", "recipe_model"},
        "retinexformer_production.py": {"recipe_fetch", "recipe_model"},
        "foundation_forecast_production.py": {"recipe_fetch", "recipe_model"},
        "document_model.py": {"recipe_fetch", "recipe_model", "recipe_registry"},
    }
    for filename, required in expected.items():
        path = ROOT / "src/omnitensor/training" / filename
        imports = {name.rsplit(".", 1)[-1] for name in imported_modules(path, ROOT / "src")}
        assert required <= imports
        assert (
            forbidden_imports([path], ["omnitensor.training.recipes"], source_root=ROOT / "src")
            == []
        )

    facade_tree = ast.parse(
        (ROOT / "src/omnitensor/training/recipes.py").read_text(encoding="utf-8")
    )
    assert not any(isinstance(node, ast.ClassDef) for node in facade_tree.body)
    validation_source = (ROOT / "src/omnitensor/training/recipe_validation.py").read_text(
        encoding="utf-8"
    )
    assert "load_schema(" not in validation_source
    assert "workload_model_contract_schemas()" in validation_source
