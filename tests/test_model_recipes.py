from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import urllib.error
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.training.recipe_cli import fetch_main, list_main
from omnitensor.training.recipes import (
    HttpsSourceTransport,
    ModelRecipeError,
    ModelSource,
    _source_file_matches,
    _validate_https_uri,
    bundled_model_recipe_root,
    fetch_model_sources,
    load_bundled_model_recipes,
    load_model_recipe,
    resolve_model_recipe_path,
)


def recipe_document(payload: bytes = b"portable-model") -> dict:
    revision = "0123456789abcdef0123456789abcdef01234567"
    return {
        "recipeVersion": 1,
        "id": "example-embedding",
        "version": "1.2.3",
        "family": "embedding",
        "profileIds": ["document-intelligence", "visual-library"],
        "sourceFormat": "onnx",
        "sources": [
            {
                "role": "model",
                "uri": f"https://models.example/releases/{revision}/model.onnx",
                "revision": revision,
                "filename": "model.onnx",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "sizeBytes": len(payload),
            }
        ],
        "license": {
            "spdx": "Apache-2.0",
            "termsUri": "https://models.example/LICENSE",
            "redistributionAllowed": True,
            "commercialUseAllowed": True,
            "attribution": "Example Authors, Apache-2.0",
        },
        "tensorContract": {
            "inputs": [{"shape": [1, 32], "dtype": "int64", "layout": "NC"}]
        },
        "outputContract": {"kind": "embedding"},
        "preprocessing": {
            "kind": "text",
            "version": 1,
            "description": "Token IDs in tokenizer order, padded to 32 tokens.",
        },
        "evaluation": [
            {
                "metric": "semantic-similarity",
                "comparator": "at-least",
                "target": 0.8,
                "unit": "spearman-rho",
                "description": "Frozen in-domain retrieval holdout correlation.",
            }
        ],
        "targets": {
            "tpu": {
                "status": "planned",
                "compiler": None,
                "fullyQuantized": False,
                "reason": "Requires a representative int8 calibration corpus.",
            },
            "npu": {
                "status": "convertible",
                "compiler": "ovc",
                "fullyQuantized": False,
                "reason": "ONNX-to-OpenVINO path exists; this graph still needs parity evidence.",
            },
            "gpu": {
                "status": "convertible",
                "compiler": "pnnx",
                "fullyQuantized": False,
                "reason": "ONNX-to-ncnn path exists; this graph still needs parity evidence.",
            },
        },
    }


def write_recipe(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def bundled_document(identifier: str) -> dict:
    recipes = {recipe.id: recipe.document for recipe in load_bundled_model_recipes()}
    return copy.deepcopy(recipes[identifier])


class FakeTransport:
    def __init__(self, payloads: dict[str, tuple[bytes, ...]]):
        self.payloads = payloads
        self.calls = []

    def chunks(self, uri, maximum_bytes):
        self.calls.append((uri, maximum_bytes))
        yield from self.payloads[uri]


def test_recipe_loads_exact_source_license_contract_and_target_claims(tmp_path):
    document = recipe_document()

    recipe = load_model_recipe(write_recipe(tmp_path, document))

    assert recipe.id == "example-embedding"
    assert recipe.profile_ids == ("document-intelligence", "visual-library")
    assert recipe.model_source.filename == "model.onnx"
    assert recipe.license.spdx == "Apache-2.0"
    assert recipe.tensor_contract == document["tensorContract"]
    assert recipe.output_contract == {"kind": "embedding"}
    assert recipe.targets["tpu"].status == "planned"
    assert recipe.targets["npu"].compiler == "ovc"
    assert len(recipe.document_sha256) == 64


def test_bundled_embedding_catalog_pins_sources_semantics_and_target_truth():
    recipes = load_bundled_model_recipes()

    assert tuple(recipe.id for recipe in recipes) == (
        "all-minilm-l6-v2",
        "bge-small-en-v1-5",
        "clip-vit-b-32-image",
    )
    assert {recipe.id: recipe.model_source.sha256 for recipe in recipes} == {
        "all-minilm-l6-v2": "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
        "bge-small-en-v1-5": "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
        "clip-vit-b-32-image": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
    }
    for recipe in recipes:
        assert recipe.producer is not None
        assert recipe.output_contract == {"kind": "embedding"}
        assert recipe.producer["outputShape"][0] == 1
        assert set(recipe.targets) == {"tpu", "npu", "gpu"}
        assert {claim.status for claim in recipe.targets.values()} == {"planned"}
        assert not any(claim.fully_quantized for claim in recipe.targets.values())
        assert all(source.revision in source.uri.split("/") for source in recipe.sources)
        assert len({source.role for source in recipe.sources}) == len(recipe.sources)
        assert all(len(source.sha256) == 64 for source in recipe.sources)
        assert all(source.size_bytes > 0 for source in recipe.sources)

    sentence_recipes = recipes[:2]
    postprocessing = {
        "all-minilm-l6-v2": "attention-mask-mean-pool-l2",
        "bge-small-en-v1-5": "cls-token-l2",
    }
    for recipe in sentence_recipes:
        assert recipe.source_format == "onnx"
        assert recipe.profile_ids == ("document-intelligence",)
        assert recipe.producer == {
            "kind": "sentence-embedding",
            "version": 1,
            "inputNames": ["input_ids", "attention_mask", "token_type_ids"],
            "sourceOutputName": "last_hidden_state",
            "sourceOutputShape": [1, 128, 384],
            "outputShape": [1, 384],
            "postprocessing": postprocessing[recipe.id],
            "description": recipe.producer["description"],
        }

    clip = recipes[2]
    assert clip.source_format == "torchscript"
    assert clip.profile_ids == ("visual-library",)
    image = clip.tensor_contract["inputs"][0]
    assert image["shape"] == [1, 3, 224, 224]
    assert image["preprocess"] == {
        "channelOrder": "RGB",
        "mean": [122.7709383, 116.7460125, 104.09373615],
        "scale": [
            0.01459842661924292,
            0.015007768493717056,
            0.014220065717024088,
        ],
        "resize": {"filter": "bicubic", "fit": "cover"},
    }


def test_bundled_recipe_reference_resolves_id_path_and_refuses_unknown(tmp_path):
    bundled = resolve_model_recipe_path("bge-small-en-v1-5")
    explicit = tmp_path / "custom.json"
    explicit.touch()

    assert bundled == bundled_model_recipe_root() / "bge-small-en-v1-5.json"
    assert resolve_model_recipe_path(explicit) == explicit
    assert resolve_model_recipe_path("nested/recipe.json") == Path("nested/recipe.json")
    with pytest.raises(ModelRecipeError) as missing:
        resolve_model_recipe_path("not-a-recipe")
    assert missing.value.code == "recipe-not-found"
    assert missing.value.detail == "no bundled model recipe is named not-a-recipe"


def test_bundled_recipe_root_reports_missing_catalog(tmp_path, monkeypatch):
    pretend_module = tmp_path / "site/omnitensor/training/recipes.py"
    pretend_module.parent.mkdir(parents=True)
    monkeypatch.setattr("omnitensor.training.recipes.__file__", str(pretend_module))

    with pytest.raises(ModelRecipeError) as missing:
        bundled_model_recipe_root()

    assert missing.value.code == "recipe-catalog-missing"
    assert missing.value.detail == "bundled model recipe catalog is absent"


def test_bundled_recipe_root_prefers_installed_then_uses_source_layout(tmp_path, monkeypatch):
    pretend_module = tmp_path / "site/omnitensor/training/recipes.py"
    pretend_module.parent.mkdir(parents=True)
    package_root = pretend_module.resolve().parents[1]
    installed = package_root / "model-recipes"
    source = package_root.parents[1] / "model-recipes"
    installed.mkdir()
    source.mkdir()
    monkeypatch.setattr("omnitensor.training.recipes.__file__", str(pretend_module))

    assert bundled_model_recipe_root() == installed

    installed.rmdir()
    assert bundled_model_recipe_root() == source


def test_bundled_catalog_rejects_empty_oversize_symlink_and_misnamed_entry(
    tmp_path, monkeypatch
):
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    monkeypatch.setattr("omnitensor.training.recipes.bundled_model_recipe_root", lambda: catalog)

    with pytest.raises(ModelRecipeError) as empty:
        load_bundled_model_recipes()
    assert empty.value.code == "recipe-catalog-invalid"
    assert empty.value.detail == "bundled model recipe catalog is empty"

    valid = Path(__file__).parents[1] / "model-recipes"
    source = valid / "bge-small-en-v1-5.json"
    link = catalog / "linked.json"
    link.symlink_to(source)
    with pytest.raises(ModelRecipeError) as unsafe:
        load_bundled_model_recipes()
    assert unsafe.value.code == "recipe-catalog-invalid"
    assert unsafe.value.detail == "unsafe recipe entry: linked.json"

    link.unlink()
    wrong = catalog / "wrong.json"
    wrong.write_bytes(source.read_bytes())
    with pytest.raises(ModelRecipeError) as mismatched:
        load_bundled_model_recipes()
    assert mismatched.value.code == "recipe-catalog-invalid"
    assert mismatched.value.detail == "recipe filename does not match id: wrong.json"

    wrong.rename(catalog / "bge-small-en-v1-5.json")
    monkeypatch.setattr("omnitensor.training.recipes.MAX_BUNDLED_RECIPES", 1)
    assert [recipe.id for recipe in load_bundled_model_recipes()] == ["bge-small-en-v1-5"]

    monkeypatch.setattr("omnitensor.training.recipes.MAX_BUNDLED_RECIPES", 0)
    with pytest.raises(ModelRecipeError) as oversized:
        load_bundled_model_recipes()
    assert oversized.value.code == "recipe-catalog-invalid"
    assert oversized.value.detail == "bundled model recipe catalog is too large"


@pytest.mark.parametrize(
    ("change", "detail"),
    [
        (
            lambda item: item["producer"].update(inputNames=["input_ids"]),
            "producer inputNames must match tensorContract input order",
        ),
        (
            lambda item: item.update(outputContract={"kind": "raw"}),
            "embedding producers require an embedding output contract",
        ),
        (
            lambda item: item["producer"].update(outputShape=[2, 384]),
            "producer outputShape must have batch size 1",
        ),
        (
            lambda item: item.update(sourceFormat="torchscript"),
            "sentence embedding source must be ONNX with canonical input order",
        ),
        (
            lambda item: item["producer"].update(
                inputNames=["attention_mask", "input_ids", "token_type_ids"]
            ),
            "sentence embedding source must be ONNX with canonical input order",
        ),
        (
            lambda item: item["producer"].update(sourceOutputName="pooler_output"),
            "sentence embedding source output must be last_hidden_state",
        ),
        (
            lambda item: item["tensorContract"]["inputs"][2].update(shape=[1, 64]),
            "sentence embedding inputs must share one fixed token shape",
        ),
        (
            lambda item: item["tensorContract"]["inputs"][1].update(dtype="int32"),
            "sentence embedding inputs must be int64",
        ),
        (
            lambda item: item["producer"].update(sourceOutputShape=[1, 64, 384]),
            "sentence embedding pooling shapes or postprocessing disagree",
        ),
        (
            lambda item: item["producer"].update(outputShape=[1, 768]),
            "sentence embedding pooling shapes or postprocessing disagree",
        ),
        (
            lambda item: item["producer"].update(postprocessing="identity"),
            "sentence embedding pooling shapes or postprocessing disagree",
        ),
    ],
)
def test_sentence_embedding_producer_semantics_fail_closed(tmp_path, change, detail):
    document = bundled_document("bge-small-en-v1-5")
    change(document)

    with pytest.raises(ModelRecipeError) as invalid:
        load_model_recipe(write_recipe(tmp_path, document))

    assert invalid.value.code == "recipe-invalid"
    assert invalid.value.detail == detail


@pytest.mark.parametrize(
    ("change", "detail"),
    [
        (
            lambda item: item.update(sourceFormat="onnx"),
            "CLIP image embedding source must be TorchScript with image input",
        ),
        (
            lambda item: item["producer"].update(inputNames=["pixels"]),
            "CLIP image embedding source must be TorchScript with image input",
        ),
        (
            lambda item: item["tensorContract"]["inputs"][0].update(
                shape=[1, 3, 256, 256]
            ),
            "CLIP image embedding input must be fixed NCHW 224x224",
        ),
        (
            lambda item: item["producer"].update(sourceOutputName="forward"),
            "CLIP image output must preserve and L2-normalize encode_image",
        ),
        (
            lambda item: item["producer"].update(postprocessing="identity"),
            "CLIP image output must preserve and L2-normalize encode_image",
        ),
        (
            lambda item: item["producer"].update(outputShape=[1, 256]),
            "CLIP image output must preserve and L2-normalize encode_image",
        ),
    ],
)
def test_clip_image_producer_semantics_fail_closed(tmp_path, change, detail):
    document = bundled_document("clip-vit-b-32-image")
    change(document)

    with pytest.raises(ModelRecipeError) as invalid:
        load_model_recipe(write_recipe(tmp_path, document))

    assert invalid.value.code == "recipe-invalid"
    assert invalid.value.detail == detail


def test_identity_producer_cannot_relabel_an_output_shape(tmp_path):
    document = recipe_document()
    document["producer"] = {
        "kind": "identity",
        "version": 1,
        "inputNames": ["input"],
        "sourceOutputName": "output",
        "sourceOutputShape": [1, 384],
        "outputShape": [1, 385],
        "postprocessing": "identity",
        "description": "Preserve the portable output without a semantic transform.",
    }

    with pytest.raises(ModelRecipeError) as invalid:
        load_model_recipe(write_recipe(tmp_path, document))

    assert invalid.value.code == "recipe-invalid"
    assert invalid.value.detail == "identity producer cannot change the source output shape"

    document["producer"]["outputShape"] = [1, 384]
    assert load_model_recipe(write_recipe(tmp_path, document)).producer == document["producer"]


@pytest.mark.parametrize(
    ("change", "detail"),
    [
        (lambda item: item.update(extra=True), "Additional properties"),
        (lambda item: item["license"].update(redistributionAllowed=False), "True was expected"),
        (lambda item: item["license"].update(commercialUseAllowed=False), "True was expected"),
        (lambda item: item.update(profileIds=[]), "should be non-empty"),
        (lambda item: item.update(sourceFormat="pickle"), "is not one of"),
        (lambda item: item["targets"].pop("tpu"), "'tpu' is a required property"),
        (lambda item: item["evaluation"][0].update(target=float("inf")), "must be finite"),
    ],
)
def test_recipe_rejects_unshareable_unbounded_or_ambiguous_documents(tmp_path, change, detail):
    document = recipe_document()
    change(document)

    with pytest.raises(ModelRecipeError) as captured:
        load_model_recipe(write_recipe(tmp_path, document))

    assert captured.value.code == "recipe-invalid"
    assert detail.lower() in captured.value.detail.lower()


@pytest.mark.parametrize(
    ("mutate", "detail"),
    [
        (
            lambda item: item["sources"].append(copy.deepcopy(item["sources"][0])),
            "exactly one model source is required",
        ),
        (
            lambda item: item["sources"].append(
                {**item["sources"][0], "role": "labels"}
            ),
            "source filenames must be unique",
        ),
        (
            lambda item: item["sources"][0].update(uri="https://models.example/model.onnx"),
            "source URI does not pin revision 0123456789abcdef0123456789abcdef01234567",
        ),
        (
            lambda item: item["sources"][0].update(
                uri="https://user:secret@models.example/0123456789abcdef0123456789abcdef01234567/model.onnx"
            ),
            "source URI must be an HTTPS URL without credentials, query, or fragment",
        ),
        (
            lambda item: item["targets"]["tpu"].update(
                status="validated", compiler="edgetpu_compiler", evidenceSha256="0" * 64
            ),
            "validated TPU compatibility requires full quantization",
        ),
        (
            lambda item: item["tensorContract"]["inputs"][0].update(dtype="bytes"),
            "model contract is invalid",
        ),
    ],
)
def test_recipe_semantics_fail_closed(tmp_path, mutate, detail):
    document = recipe_document()
    mutate(document)

    with pytest.raises(ModelRecipeError) as captured:
        load_model_recipe(write_recipe(tmp_path, document))

    assert captured.value.code == "recipe-invalid"
    if detail == "model contract is invalid":
        assert captured.value.detail == (
            "model contract is invalid at tensorContract: inputs/0/dtype: 'bytes' is not one of "
            "['float32', 'float64', 'int32', 'int64', 'uint8']"
        )
    else:
        assert captured.value.detail == detail


def test_recipe_validates_output_contract_with_exact_path(tmp_path):
    document = recipe_document()
    document["outputContract"] = {"kind": "classification", "topK": 0}

    with pytest.raises(ModelRecipeError) as captured:
        load_model_recipe(write_recipe(tmp_path, document))

    assert captured.value.code == "recipe-invalid"
    assert captured.value.detail == (
        "model contract is invalid at outputContract: topK: 0 is less than the minimum of 1"
    )


def test_recipe_requires_unique_roles_and_pinned_preprocessing_sources(tmp_path):
    duplicate = recipe_document()
    duplicate["sources"].append(
        {
            **duplicate["sources"][0],
            "role": "config",
            "filename": "config.json",
        }
    )
    duplicate["sources"].append(
        {
            **duplicate["sources"][0],
            "role": "config",
            "filename": "other-config.json",
        }
    )
    with pytest.raises(ModelRecipeError) as repeated:
        load_model_recipe(write_recipe(tmp_path, duplicate))
    assert repeated.value.code == "recipe-invalid"
    assert repeated.value.detail == "source roles must be unique"

    missing = recipe_document()
    missing["preprocessing"]["artifacts"] = ["tokenizer"]
    with pytest.raises(ModelRecipeError) as absent:
        load_model_recipe(write_recipe(tmp_path, missing))
    assert absent.value.code == "recipe-invalid"
    assert absent.value.detail == "preprocessing artifact has no pinned source: tokenizer"


def test_recipe_combined_source_boundary_is_exact(tmp_path):
    document = recipe_document()
    document["sources"][0]["sizeBytes"] = 2 * 1024 * 1024 * 1024
    document["sources"].append(
        {
            **document["sources"][0],
            "role": "labels",
            "filename": "labels.txt",
        }
    )
    assert load_model_recipe(write_recipe(tmp_path, document)).id == "example-embedding"

    document["sources"].append(
        {
            **document["sources"][0],
            "role": "config",
            "filename": "config.json",
        }
    )
    with pytest.raises(ModelRecipeError) as too_large:
        load_model_recipe(write_recipe(tmp_path, document))
    assert too_large.value.code == "recipe-invalid"
    assert too_large.value.detail == "combined source size exceeds 4 GiB"


def test_fetch_requires_exact_license_then_publishes_verified_atomic_receipt(tmp_path):
    payload = b"portable-model"
    document = recipe_document(payload)
    path = write_recipe(tmp_path, document)
    uri = document["sources"][0]["uri"]
    transport = FakeTransport({uri: (b"portable-", b"model")})

    with pytest.raises(ModelRecipeError) as unaccepted:
        fetch_model_sources(
            path, tmp_path / "sources", accepted_license="MIT", transport=transport
        )
    assert unaccepted.value.code == "license-not-accepted"
    assert transport.calls == []

    fetched = fetch_model_sources(
        path,
        tmp_path / "sources",
        accepted_license="Apache-2.0",
        transport=transport,
    )

    assert fetched.root == tmp_path / "sources/example-embedding/1.2.3"
    assert fetched.document() == {
        "recipeId": "example-embedding",
        "recipeVersion": "1.2.3",
        "recipeSha256": fetched.recipe.document_sha256,
        "root": str(fetched.root),
        "receipt": str(fetched.receipt_path),
        "files": ["model.onnx"],
    }
    assert (fetched.root / "model.onnx").read_bytes() == payload
    receipt = json.loads(fetched.receipt_path.read_text())
    assert receipt == {
        "version": 1,
        "recipe": {
            "id": "example-embedding",
            "version": "1.2.3",
            "sha256": fetched.recipe.document_sha256,
        },
        "license": {
            "spdx": "Apache-2.0",
            "termsUri": "https://models.example/LICENSE",
            "attribution": "Example Authors, Apache-2.0",
        },
        "sources": [
            {
                "role": "model",
                "revision": document["sources"][0]["revision"],
                "filename": "model.onnx",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "sizeBytes": len(payload),
            }
        ],
    }
    assert not list((tmp_path / "sources/example-embedding").glob(".model-source-*"))


def test_matching_fetch_is_idempotent_and_tampering_is_a_conflict(tmp_path):
    payload = b"portable-model"
    document = recipe_document(payload)
    path = write_recipe(tmp_path, document)
    uri = document["sources"][0]["uri"]
    transport = FakeTransport({uri: (payload,)})
    first = fetch_model_sources(
        path, tmp_path / "sources", accepted_license="Apache-2.0", transport=transport
    )

    second = fetch_model_sources(
        path,
        tmp_path / "sources",
        accepted_license="Apache-2.0",
        transport=FakeTransport({}),
    )
    assert second.root == first.root
    assert second.recipe == first.recipe
    assert second.receipt_path == first.root / "source-receipt.json"

    (first.root / "model.onnx").write_bytes(b"tampered")
    with pytest.raises(ModelRecipeError) as conflict:
        fetch_model_sources(
            path,
            tmp_path / "sources",
            accepted_license="Apache-2.0",
            transport=FakeTransport({}),
        )
    assert conflict.value.code == "source-conflict"


@pytest.mark.parametrize("matching", [True, False])
def test_concurrent_source_publish_rechecks_the_winner(tmp_path, monkeypatch, matching):
    payload = b"portable-model"
    document = recipe_document(payload)
    path = write_recipe(tmp_path, document)
    uri = document["sources"][0]["uri"]
    checked = []

    def already_published(destination, recipe):
        checked.append((destination, recipe.id))
        return matching

    monkeypatch.setattr(
        "omnitensor.training.recipes.os.rename",
        lambda *_args: (_ for _ in ()).throw(FileExistsError()),
    )
    monkeypatch.setattr(
        "omnitensor.training.recipes._installed_source_matches", already_published
    )

    if matching:
        fetched = fetch_model_sources(
            path,
            tmp_path / "sources",
            accepted_license="Apache-2.0",
            transport=FakeTransport({uri: (payload,)}),
        )
        assert fetched.recipe.id == "example-embedding"
        assert fetched.root == tmp_path / "sources/example-embedding/1.2.3"
        assert fetched.receipt_path == fetched.root / "source-receipt.json"
    else:
        with pytest.raises(ModelRecipeError) as conflict:
            fetch_model_sources(
                path,
                tmp_path / "sources",
                accepted_license="Apache-2.0",
                transport=FakeTransport({uri: (payload,)}),
            )
        assert conflict.value.code == "source-conflict"
        assert conflict.value.detail == (
            "source version appeared concurrently: "
            f"{tmp_path / 'sources/example-embedding/1.2.3'}"
        )
    assert checked == [
        (tmp_path / "sources/example-embedding/1.2.3", "example-embedding")
    ]
    assert not list((tmp_path / "sources/example-embedding").glob(".model-source-*"))


@pytest.mark.parametrize("tamper", ["receipt", "size", "digest"])
def test_existing_incomplete_or_changed_source_never_shortcuts_verification(tmp_path, tamper):
    payload = b"portable-model"
    document = recipe_document(payload)
    path = write_recipe(tmp_path, document)
    uri = document["sources"][0]["uri"]
    fetched = fetch_model_sources(
        path,
        tmp_path / "sources",
        accepted_license="Apache-2.0",
        transport=FakeTransport({uri: (payload,)}),
    )
    if tamper == "receipt":
        fetched.receipt_path.unlink()
    elif tamper == "size":
        (fetched.root / "model.onnx").write_bytes(b"short")
    else:
        (fetched.root / "model.onnx").write_bytes(b"portable-modeX")

    with pytest.raises(ModelRecipeError) as conflict:
        fetch_model_sources(
            path,
            tmp_path / "sources",
            accepted_license="Apache-2.0",
            transport=FakeTransport({}),
        )
    assert conflict.value.code == "source-conflict"


@pytest.mark.parametrize("target", ["receipt", "model"])
def test_existing_source_symlinks_never_satisfy_the_receipt(tmp_path, target):
    payload = b"portable-model"
    document = recipe_document(payload)
    path = write_recipe(tmp_path, document)
    uri = document["sources"][0]["uri"]
    fetched = fetch_model_sources(
        path,
        tmp_path / "sources",
        accepted_license="Apache-2.0",
        transport=FakeTransport({uri: (payload,)}),
    )
    selected = fetched.receipt_path if target == "receipt" else fetched.root / "model.onnx"
    backup = tmp_path / selected.name
    selected.replace(backup)
    selected.symlink_to(backup)

    with pytest.raises(ModelRecipeError) as conflict:
        fetch_model_sources(
            path,
            tmp_path / "sources",
            accepted_license="Apache-2.0",
            transport=FakeTransport({}),
        )
    assert conflict.value.code == "source-conflict"


def test_source_file_match_checks_type_size_digest_and_read_errors(tmp_path, monkeypatch):
    payload = b"portable-model"
    path = tmp_path / "model.onnx"
    path.write_bytes(payload)
    source = ModelSource(
        "model",
        "https://models.example/1234567/model.onnx",
        "1234567",
        "model.onnx",
        hashlib.sha256(payload).hexdigest(),
        len(payload),
    )

    assert _source_file_matches(path, source) is True
    assert _source_file_matches(path, replace(source, size_bytes=len(payload) + 1)) is False
    assert _source_file_matches(path, replace(source, sha256="0" * 64)) is False
    assert _source_file_matches(tmp_path / "missing.onnx", source) is False

    link = tmp_path / "linked.onnx"
    link.symlink_to(path)
    assert _source_file_matches(link, source) is False

    def unreadable(_self, *_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "open", unreadable)
    assert _source_file_matches(path, source) is False


@pytest.mark.parametrize(
    ("chunks", "code"),
    [
        ((b"short",), "source-size-mismatch"),
        ((b"portable-model!",), "source-too-large"),
        ((b"portable-modeX",), "source-digest-mismatch"),
        (("not-bytes",), "source-invalid"),
        ((b"",), "source-invalid"),
    ],
)
def test_failed_fetch_removes_partial_version(tmp_path, chunks, code):
    document = recipe_document()
    path = write_recipe(tmp_path, document)
    uri = document["sources"][0]["uri"]

    with pytest.raises(ModelRecipeError) as captured:
        fetch_model_sources(
            path,
            tmp_path / "sources",
            accepted_license="Apache-2.0",
            transport=FakeTransport({uri: chunks}),
        )

    assert captured.value.code == code
    expected_detail = {
        "source-size-mismatch": "model.onnx has 5 bytes; expected 14",
        "source-too-large": "model.onnx exceeds sizeBytes",
        "source-digest-mismatch": "model.onnx digest does not match",
        "source-invalid": "transport returned an invalid chunk",
    }
    assert captured.value.detail == expected_detail[code]
    assert not (tmp_path / "sources/example-embedding/1.2.3").exists()
    assert not list((tmp_path / "sources/example-embedding").glob(".model-source-*"))


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, *, uri: str, length: str | None = None):
        super().__init__(payload)
        self._uri = uri
        self.headers = {} if length is None else {"Content-Length": length}

    def geturl(self):
        return self._uri

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_https_transport_bounds_headers_body_and_redirects(monkeypatch):
    uri = "https://models.example/releases/1234567/model.onnx"
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: FakeResponse(b"abc", uri=uri, length="3"),
    )
    assert b"".join(HttpsSourceTransport().chunks(uri, 3)) == b"abc"

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: FakeResponse(b"", uri=uri, length="4"),
    )
    with pytest.raises(ModelRecipeError, match="exceeds"):
        tuple(HttpsSourceTransport().chunks(uri, 3))

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: FakeResponse(b"abc", uri="http://redirect.invalid/model"),
    )
    with pytest.raises(ModelRecipeError, match="HTTPS URL"):
        tuple(HttpsSourceTransport().chunks(uri, 3))


def test_https_transport_refuses_invalid_timeout_header_network_and_unbounded_body(monkeypatch):
    uri = "https://models.example/releases/1234567/model.onnx"
    for timeout in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError) as invalid:
            HttpsSourceTransport(timeout)
        assert str(invalid.value) == "download timeout must be a positive finite number"
    assert isinstance(HttpsSourceTransport(0.5), HttpsSourceTransport)

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )
    with pytest.raises(ModelRecipeError) as unavailable:
        tuple(HttpsSourceTransport().chunks(uri, 3))
    assert unavailable.value.code == "source-unavailable"
    assert unavailable.value.detail == "cannot fetch source: <urlopen error offline>"

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: FakeResponse(b"", uri=uri, length="many"),
    )
    with pytest.raises(ModelRecipeError) as malformed:
        tuple(HttpsSourceTransport().chunks(uri, 3))
    assert malformed.value.code == "source-invalid"
    assert malformed.value.detail == "source Content-Length is invalid"

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: FakeResponse(b"", uri=uri, length="0"),
    )
    assert tuple(HttpsSourceTransport().chunks(uri, 3)) == ()

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: FakeResponse(b"abcd", uri=uri),
    )
    with pytest.raises(ModelRecipeError) as oversized:
        tuple(HttpsSourceTransport().chunks(uri, 3))
    assert oversized.value.code == "source-too-large"
    assert oversized.value.detail == "source exceeds its declared bound"


def test_https_transport_passes_exact_uri_timeout_and_accumulates_chunks(monkeypatch):
    uri = "https://models.example/releases/1234567/model.onnx"
    called = []

    class ChunkedResponse(FakeResponse):
        def __init__(self):
            super().__init__(b"", uri=uri)
            self.parts = iter((b"ab", b"cd"))
            self.read_sizes = []

        def read(self, size=-1):
            self.read_sizes.append(size)
            return next(self.parts, b"")

    response = ChunkedResponse()

    def open_source(*args, **kwargs):
        called.append((args, kwargs))
        return response

    monkeypatch.setattr("urllib.request.urlopen", open_source)
    with pytest.raises(ModelRecipeError) as oversized:
        tuple(HttpsSourceTransport(2.5).chunks(uri, 3))
    assert called == [((uri,), {"timeout": 2.5})]
    assert response.read_sizes == [4, 2]
    assert oversized.value.code == "source-too-large"
    assert oversized.value.detail == "source exceeds its declared bound"


@pytest.mark.parametrize(
    "uri",
    [
        "http://models.example/1234567/model.onnx",
        "https:///1234567/model.onnx",
        "https://user@models.example/1234567/model.onnx",
        "https://user:secret@models.example/1234567/model.onnx",
        "https://models.example/1234567/model.onnx?token=secret",
        "https://models.example/1234567/model.onnx#fragment",
    ],
)
def test_recipe_uri_conditions_are_independently_rejected(uri):
    with pytest.raises(ModelRecipeError) as invalid:
        _validate_https_uri(uri, "source URI")

    assert invalid.value.code == "recipe-invalid"
    assert invalid.value.detail == (
        "source URI must be an HTTPS URL without credentials, query, or fragment"
    )


def test_fetch_cli_reports_success_and_failure_without_traceback(tmp_path, monkeypatch, capsys):
    fetched = type(
        "Fetched",
        (),
        {"document": lambda self: {"recipeId": "example", "files": ["model.onnx"]}},
    )()
    calls = []

    def fetch(*args, **kwargs):
        calls.append((args, kwargs))
        return fetched

    monkeypatch.setattr("omnitensor.training.recipe_cli.fetch_model_sources", fetch)
    assert fetch_main(
        [
            str(tmp_path / "recipe.json"),
            "--accept-license",
            "MIT",
            "--source-root",
            str(tmp_path / "sources"),
        ]
    ) == 0
    assert calls == [
        (
            (tmp_path / "recipe.json", tmp_path / "sources"),
            {"accepted_license": "MIT"},
        )
    ]
    assert capsys.readouterr().out == (
        '{\n  "recipeId": "example",\n  "files": [\n    "model.onnx"\n  ]\n}\n'
    )

    monkeypatch.setattr(
        "omnitensor.training.recipe_cli.fetch_model_sources",
        lambda *args, **kwargs: (_ for _ in ()).throw(ModelRecipeError("bad", "unsafe")),
    )
    assert fetch_main(["missing.json", "--accept-license", "MIT"]) == 1
    assert capsys.readouterr().err == "model source fetch failed: bad: unsafe\n"


def test_fetch_cli_help_is_an_exact_operator_contract(capsys):
    with pytest.raises(SystemExit) as stopped:
        fetch_main(["--help"])

    assert stopped.value.code == 0
    assert capsys.readouterr().out == (
        "usage: omnitensor-fetch-model-source [-h] --accept-license ACCEPT_LICENSE\n"
        "                                     [--source-root SOURCE_ROOT]\n"
        "                                     recipe\n"
        "\n"
        "Fetch and verify a pinned portable model recipe outside the service\n"
        "\n"
        "positional arguments:\n"
        "  recipe                bundled recipe id or path to a model-recipe JSON\n"
        "                        document\n"
        "\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
        "  --accept-license ACCEPT_LICENSE\n"
        "                        exact SPDX identifier shown in the reviewed recipe\n"
        "  --source-root SOURCE_ROOT\n"
    )


def test_list_cli_reports_exact_catalog_and_failure(monkeypatch, capsys):
    assert list_main([]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed == [
        {
            "id": "all-minilm-l6-v2",
            "version": "1.0.0",
            "family": "embedding",
            "profileIds": ["document-intelligence"],
            "sourceFormat": "onnx",
            "license": "Apache-2.0",
            "targets": {"tpu": "planned", "npu": "planned", "gpu": "planned"},
        },
        {
            "id": "bge-small-en-v1-5",
            "version": "1.0.0",
            "family": "embedding",
            "profileIds": ["document-intelligence"],
            "sourceFormat": "onnx",
            "license": "MIT",
            "targets": {"tpu": "planned", "npu": "planned", "gpu": "planned"},
        },
        {
            "id": "clip-vit-b-32-image",
            "version": "1.0.0",
            "family": "embedding",
            "profileIds": ["visual-library"],
            "sourceFormat": "torchscript",
            "license": "MIT",
            "targets": {"tpu": "planned", "npu": "planned", "gpu": "planned"},
        },
    ]

    monkeypatch.setattr(
        "omnitensor.training.recipe_cli.load_bundled_model_recipes",
        lambda: (_ for _ in ()).throw(ModelRecipeError("catalog", "broken")),
    )
    assert list_main([]) == 1
    assert capsys.readouterr().err == "model recipe catalog failed: catalog: broken\n"


def test_list_cli_help_is_an_exact_operator_contract(capsys):
    with pytest.raises(SystemExit) as stopped:
        list_main(["--help"])

    assert stopped.value.code == 0
    assert capsys.readouterr().out == (
        "usage: omnitensor-list-model-recipes [-h]\n"
        "\n"
        "List reviewed portable source recipes and honest target status\n"
        "\n"
        "options:\n"
        "  -h, --help  show this help message and exit\n"
    )


@given(
    identifier=st.from_regex(r"[a-z0-9]+(?:-[a-z0-9]+){0,3}", fullmatch=True).filter(
        lambda value: len(value) <= 120
    ),
    version=st.tuples(
        st.integers(min_value=0, max_value=999),
        st.integers(min_value=0, max_value=999),
        st.integers(min_value=0, max_value=999),
    ),
)
def test_valid_bounded_recipe_identities_round_trip(identifier, version):
    document = recipe_document()
    document["id"] = identifier
    document["version"] = ".".join(str(part) for part in version)

    with tempfile.TemporaryDirectory() as directory:
        recipe = load_model_recipe(write_recipe(Path(directory), document))

    assert recipe.id == identifier
    assert recipe.version == document["version"]


def test_service_does_not_import_model_fetching():
    source = Path(__file__).parents[1] / "src/omnitensor/service.py"
    assert "training.recipes" not in source.read_text(encoding="utf-8")
