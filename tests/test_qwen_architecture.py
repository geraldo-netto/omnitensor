from __future__ import annotations

import ast
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import omnitensor.plugins.qwen as facade
from omnitensor.plugins import acceptance_kit, generation
from omnitensor.plugins import qwen_catalog as catalog
from omnitensor.plugins import qwen_contracts as contracts
from omnitensor.plugins import qwen_providers as providers
from omnitensor.plugins import qwen_qualification as qualification

ROOT = Path(__file__).parents[1]
LEGACY_MODULE = "omnitensor.plugins.qwen"


def _contract_values():
    source = contracts.QwenSource(
        "model",
        "https://example.invalid/model",
        "a" * 40,
        "model.gguf",
        "b" * 64,
        1,
    )
    policy = contracts.EventQualificationPolicy()
    evaluation = contracts.QwenEvaluation("corpus.json", policy)
    catalog_value = contracts.QwenCatalog(
        "qwen",
        "1.0.0",
        "a" * 40,
        "Apache-2.0",
        (source,),
        (("provider", "gpu", "llama.cpp-vulkan", True, "qualified"),),
        evaluation,
    )
    case = contracts.FrozenEventCase(
        "case",
        "text",
        "content",
        False,
        (("title", "2026-01-01T00:00:00+00:00", "UTC", ""),),
    )
    corpus = contracts.FrozenEventCorpus("corpus", "c" * 64, (case,))
    observation = contracts.EventProviderObservation("case", {}, 1, 1)
    load = acceptance_kit.NativeLoadReport("llama.cpp-vulkan", "Vulkan", 1, 1, False)
    evidence = contracts.EventProviderEvidence(
        "provider",
        corpus.sha256,
        "runtime",
        "device",
        load,
        (observation,),
        1,
    )
    report = contracts.EventQualificationReport(
        "provider", "gpu", corpus.sha256, "runtime", "device", 1.0, 1.0, 1, 1, 1, True
    )
    return (
        source,
        catalog_value,
        case,
        corpus,
        observation,
        evidence,
        policy,
        evaluation,
        report,
        load,
    )


def test_facade_classes_are_exact_new_owner_objects_with_legacy_globals():
    names = (
        "QwenProviderError",
        "QwenSource",
        "QwenCatalog",
        "NativeQwenRuntime",
        "FrozenEventCase",
        "FrozenEventCorpus",
        "EventProviderObservation",
        "EventProviderEvidence",
        "EventQualificationPolicy",
        "QwenEvaluation",
        "EventQualificationReport",
    )

    for name in names:
        owner = getattr(contracts, name)
        assert getattr(facade, name) is owner
        assert owner.__module__ == LEGACY_MODULE
        assert pickle.loads(pickle.dumps(owner)) is owner

    assert facade._QwenWorker is providers._QwenWorker
    assert facade.LlamaCppVulkanQwenWorker is providers.LlamaCppVulkanQwenWorker
    assert facade.OpenVinoNpuQwenWorker is providers.OpenVinoNpuQwenWorker
    for worker in (
        providers._QwenWorker,
        providers.LlamaCppVulkanQwenWorker,
        providers.OpenVinoNpuQwenWorker,
    ):
        assert worker.__module__ == LEGACY_MODULE
        assert pickle.loads(pickle.dumps(worker)) is worker


def test_every_serializable_qwen_value_keeps_its_legacy_pickle_global():
    for value in _contract_values():
        assert type(value).__module__ == LEGACY_MODULE
        assert pickle.loads(pickle.dumps(value)) == value

    error = contracts.QwenProviderError("code", "detail")
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        pickle.loads(pickle.dumps(error))


def test_facade_keeps_exact_exports_and_live_compatibility_aliases():
    assert facade.__all__ == [
        "EventProviderEvidence",
        "EventProviderObservation",
        "EventQualificationPolicy",
        "EventQualificationReport",
        "FrozenEventCorpus",
        "LlamaCppVulkanQwenWorker",
        "NativeLoadReport",
        "NativeQwenRuntime",
        "OpenVinoNpuQwenWorker",
        "QwenCatalog",
        "QwenEvaluation",
        "QwenProviderError",
        "load_event_corpus",
        "load_qwen_catalog",
        "qualify_event_provider",
    ]
    assert facade.NativeLoadReport is acceptance_kit.NativeLoadReport
    assert facade.ProviderGenerationError is generation.ProviderGenerationError
    assert facade._validate_gpu_load is acceptance_kit.validate_gpu_load
    assert facade._validate_npu_load is acceptance_kit.validate_npu_load
    assert facade.MAX_CATALOG_BYTES == catalog.MAX_CATALOG_BYTES
    assert facade.MAX_CORPUS_BYTES == qualification.MAX_CORPUS_BYTES
    assert facade.qualify_event_provider is qualification.qualify_event_provider


def test_facade_validator_and_policy_monkeypatches_dispatch_at_call_time(monkeypatch):
    gpu_report = acceptance_kit.NativeLoadReport(
        "llama.cpp-vulkan", "Vulkan", 1, 1, False
    )
    npu_report = acceptance_kit.NativeLoadReport("openvino-genai", "NPU", 1, 1, False)
    calls = []

    monkeypatch.setattr(facade, "_validate_gpu_load", lambda report: calls.append(("gpu", report)))
    monkeypatch.setattr(facade, "_validate_npu_load", lambda report: calls.append(("npu", report)))

    providers.LlamaCppVulkanQwenWorker._validate_load(object(), gpu_report)
    qualification._validate_lane_load("gpu", gpu_report)
    providers.OpenVinoNpuQwenWorker._validate_load(object(), npu_report)
    qualification._validate_lane_load("npu", npu_report)

    policy_calls = []

    def validate_policy(policy):
        policy_calls.append(policy)
        contracts.validate_event_policy(policy)

    monkeypatch.setattr(facade, "_validate_policy", validate_policy)
    catalog.load_qwen_catalog()
    monkeypatch.setattr(qualification, "_validate_evidence_identity", lambda *_args: None)
    monkeypatch.setattr(
        qualification,
        "_score_observations",
        lambda *_args: (1, 0, 0, [1], 1),
    )
    monkeypatch.setattr(qualification, "_validate_lane_load", lambda *_args: None)
    policy = contracts.EventQualificationPolicy()
    qualification.qualify_event_provider(
        SimpleNamespace(provider_id="provider", accelerator="gpu"),
        SimpleNamespace(sha256="a" * 64),
        SimpleNamespace(
            runtime_version="runtime",
            device_name="device",
            cancellation_latency_ms=1,
            load=gpu_report,
        ),
        policy,
    )

    assert calls == [
        ("gpu", gpu_report),
        ("gpu", gpu_report),
        ("npu", npu_report),
        ("npu", npu_report),
    ]
    assert policy_calls == [contracts.EventQualificationPolicy(), policy]


def test_legacy_callback_falls_back_when_direct_leaf_import_has_no_facade(monkeypatch):
    def fallback(value):
        return value

    monkeypatch.delitem(sys.modules, LEGACY_MODULE)

    assert contracts.legacy_qwen_callback("callback", fallback) is fallback


def test_direct_catalog_and_qualification_owners_match_facade_results():
    assert catalog.load_qwen_catalog() == facade.load_qwen_catalog()
    assert qualification.load_event_corpus() == facade.load_event_corpus()


def test_qwen_leaves_are_independent_of_the_legacy_facade():
    leaf_names = (
        "qwen_contracts.py",
        "qwen_catalog.py",
        "qwen_qualification.py",
        "qwen_providers.py",
    )
    for name in leaf_names:
        path = ROOT / "src/omnitensor/plugins" / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert "qwen" not in imports
        assert "omnitensor.plugins.qwen" not in imports

    facade_tree = ast.parse(
        (ROOT / "src/omnitensor/plugins/qwen.py").read_text(encoding="utf-8")
    )
    assert not any(isinstance(node, ast.ClassDef) for node in facade_tree.body)
    assert {
        node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)
    } == {"load_qwen_catalog", "load_event_corpus", "_catalog_path", "_corpus_path"}


def test_qwen_leaves_import_before_the_facade_in_a_fresh_interpreter():
    statement = ";".join(
        f"import omnitensor.plugins.{name}"
        for name in (
            "qwen_contracts",
            "qwen_catalog",
            "qwen_qualification",
            "qwen_providers",
            "qwen",
        )
    )

    completed = subprocess.run(
        [sys.executable, "-c", statement],
        check=False,
        capture_output=True,
        text=True,
    )

    assert (completed.returncode, completed.stderr) == (0, "")


def test_production_callers_use_new_owners_without_bypassing_the_facade_contract():
    expected = {
        "src/omnitensor/event_cli.py": {
            "plugins.qwen_catalog",
            "plugins.qwen_contracts",
        },
        "providers/qwen-vulkan-runtime/src/omnitensor_qwen_runtime/factories.py": {
            "omnitensor.plugins.qwen_providers"
        },
        "providers/qwen-vulkan-runtime/src/omnitensor_qwen_runtime/hebrew.py": {
            "omnitensor.plugins.generation"
        },
        "providers/qwen-vulkan-runtime/src/omnitensor_qwen_runtime/runtime.py": {
            "omnitensor.plugins.acceptance_kit",
            "omnitensor.plugins.generation",
        },
    }
    for relative, required in expected.items():
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert required <= imports
        assert "omnitensor.plugins.qwen" not in imports
