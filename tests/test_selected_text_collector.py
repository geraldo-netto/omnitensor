from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins.selected_text_acceptance import (
    HEBREW_MODEL_SHA256,
    PRIMARY_MODEL_SHA256,
    SelectedTextAcceptanceError,
    parse_selected_text_worker_load_receipt,
)

ROOT = Path(__file__).parents[1]
DEFAULT_PRIMARY_MODEL_ID = "qwen3-5-9b-iq4-xs"


def _manifest() -> dict:
    return json.loads((ROOT / "plugin-manifests/selected-text-tools.json").read_text())


DEFAULT_PRIMARY_MODEL_SHA256 = next(
    artifact["sha256"]
    for artifact in _manifest()["plugin"]["artifacts"]
    if artifact["id"] == DEFAULT_PRIMARY_MODEL_ID
)


def _collector_module():
    path = ROOT / "scripts/collect-selected-text-acceptance.py"
    spec = importlib.util.spec_from_file_location("selected_text_collector", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(digest: str, layers: int) -> dict[str, object]:
    return {
        "modelSha256": digest,
        "runtimeVersion": "llama-cpp-python-0.3.34",
        "deviceName": "AMD Radeon RX 6600 XT (RADV NAVI23)",
        "load": {
            "backend": "llama.cpp-vulkan",
            "device": "Vulkan",
            "totalModelLayers": layers,
            "acceleratorLayers": layers,
            "cpuFallback": False,
        },
    }


def _receipt(primary_digest: str = PRIMARY_MODEL_SHA256, primary_layers: int = 37):
    return parse_selected_text_worker_load_receipt(
        {
            "receiptVersion": 1,
            "pluginId": "selected-text-tools",
            "models": {
                "primary": _model(primary_digest, primary_layers),
                "hebrewTranslation": _model(HEBREW_MODEL_SHA256, 33),
            },
        }
    )


def _inventory(primary_id: str = "qwen3-8b-q4-k-m") -> str:
    return json.dumps(
        {
            "plugins": [
                {
                    "id": "selected-text-tools",
                    "version": "1.2.0",
                    "workerState": "ready",
                    "artifacts": [
                        {"id": primary_id, "ready": True},
                        {"id": "dictalm2-hebrew-q4-k-m", "ready": True},
                    ],
                }
            ]
        }
    )


class _Control:
    """A fake ``call_control`` that answers the inventory handshake."""

    def __init__(self, primary_id: str = "qwen3-8b-q4-k-m") -> None:
        self.calls = []
        self.primary_id = primary_id

    async def __call__(self, method, params):
        self.calls.append((method, params))
        assert method == "describe-plugins"
        return json.loads(_inventory(self.primary_id))


def _install_control(monkeypatch, collector, primary_id: str = "qwen3-8b-q4-k-m"):
    control = _Control(primary_id)
    monkeypatch.setattr(collector, "call_control", control)
    return control


def test_collector_accepts_the_configured_default_after_digest_agreement(monkeypatch):
    collector = _collector_module()
    receipt = _receipt(DEFAULT_PRIMARY_MODEL_SHA256, 33)
    control = _install_control(monkeypatch, collector, DEFAULT_PRIMARY_MODEL_ID)
    case = SimpleNamespace(case_id="case-1", expected_provider_id="qwen3-workloads-gpu")
    corpus = SimpleNamespace(sha256="c" * 64, cases=(case,))

    monkeypatch.setattr(collector, "load_selected_text_worker_load_receipt", lambda _path: receipt)
    monkeypatch.setattr(collector, "load_selected_text_corpus", lambda _path: corpus)
    monkeypatch.setattr(
        collector,
        "file_digest",
        lambda path: (
            DEFAULT_PRIMARY_MODEL_SHA256 if path.name == "qwen.gguf" else HEBREW_MODEL_SHA256
        ),
    )

    async def run_case(_interface, _case, _index, _timeout):
        return {
            "caseId": "case-1",
            "result": {"providerId": "qwen3-workloads-gpu"},
            "latencyMs": 4,
        }

    async def cancellation(_interface, _timeout):
        return 7

    monkeypatch.setattr(collector, "_run_case", run_case)
    monkeypatch.setattr(collector, "_cancellation_latency", cancellation)
    arguments = SimpleNamespace(
        corpus=Path("corpus.json"),
        qwen_model=Path("qwen.gguf"),
        hebrew_model=Path("hebrew.gguf"),
        worker_load_receipt=Path("receipt.json"),
        timeout_seconds=1.0,
    )

    document = asyncio.run(collector._collect(arguments))

    assert document["models"] == receipt.models_document()
    assert document["safety"] == {
        "cancellationLatencyMs": 7,
        "privateFragmentsDiscarded": True,
        "defaultRoutePreserved": True,
    }
    assert [method for method, _params in control.calls] == ["describe-plugins"]


def test_collector_accepts_every_artifact_the_manifest_declares():
    """A worker that installed all three manifest artifacts is ready, not broken."""
    collector = _collector_module()
    manifest = _manifest()
    declared = [artifact["id"] for artifact in manifest["plugin"]["artifacts"]]
    assert "qwen3-5-9b-iq4-xs" in declared

    collector._require_ready(
        {
            "plugins": [
                {
                    "id": "selected-text-tools",
                    "version": "1.2.0",
                    "workerState": "ready",
                    "artifacts": [{"id": identifier, "ready": True} for identifier in declared],
                }
            ]
        },
        _receipt(DEFAULT_PRIMARY_MODEL_SHA256, 33),
        manifest,
    )


@given(st.permutations((DEFAULT_PRIMARY_MODEL_ID, "dictalm2-hebrew-q4-k-m", "unused")))
def test_receipt_selected_readiness_is_independent_of_inventory_order(artifact_ids):
    collector = _collector_module()
    inventory = {
        "plugins": [
            {
                "id": "selected-text-tools",
                "version": "1.2.0",
                "workerState": "ready",
                "artifacts": [
                    {"id": artifact_id, "ready": artifact_id != "unused"}
                    for artifact_id in artifact_ids
                ],
            }
        ]
    }

    collector._require_ready(
        inventory,
        _receipt(DEFAULT_PRIMARY_MODEL_SHA256, 33),
        _manifest(),
    )


@pytest.mark.parametrize(
    ("artifacts", "detail"),
    [
        (
            [{"id": DEFAULT_PRIMARY_MODEL_ID, "ready": True}],
            "not installed: dictalm2-hebrew-q4-k-m",
        ),
        (
            [
                {"id": "qwen3-8b-q4-k-m", "ready": True},
                {"id": DEFAULT_PRIMARY_MODEL_ID, "ready": False},
                {"id": "dictalm2-hebrew-q4-k-m", "ready": False},
            ],
            f"not ready: dictalm2-hebrew-q4-k-m, {DEFAULT_PRIMARY_MODEL_ID}",
        ),
    ],
)
def test_collector_names_the_routed_artifact_that_is_missing_or_unready(artifacts, detail):
    collector = _collector_module()

    with pytest.raises(RuntimeError, match=detail):
        collector._require_ready(
            {
                "plugins": [
                    {
                        "id": "selected-text-tools",
                        "version": "1.2.0",
                        "workerState": "ready",
                        "artifacts": artifacts,
                    }
                ]
            },
            _receipt(DEFAULT_PRIMARY_MODEL_SHA256, 33),
            _manifest(),
        )


def test_collector_refuses_a_receipt_model_the_manifest_does_not_declare():
    collector = _collector_module()

    with pytest.raises(RuntimeError, match="declared selectable primary"):
        collector._require_ready(
            json.loads(_inventory(DEFAULT_PRIMARY_MODEL_ID)),
            _receipt("0" * 64, 33),
            _manifest(),
        )


def test_collector_invalid_receipt_aborts_before_corpus_jobs_and_output(tmp_path, monkeypatch):
    collector = _collector_module()
    control = _install_control(monkeypatch, collector)
    writes = []

    def refuse_receipt(_path):
        raise SelectedTextAcceptanceError(
            "receipt-invalid", "worker load receipt fields are invalid"
        )

    monkeypatch.setattr(collector, "load_selected_text_worker_load_receipt", refuse_receipt)
    monkeypatch.setattr(
        collector,
        "load_selected_text_corpus",
        lambda _path: pytest.fail("corpus was loaded"),
    )
    monkeypatch.setattr(collector, "file_digest", lambda _path: pytest.fail("model was digested"))
    monkeypatch.setattr(
        collector,
        "write_json_atomic",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )

    with pytest.raises(SelectedTextAcceptanceError, match="receipt fields"):
        collector.main(
            [
                "--output",
                str(tmp_path / "evidence.json"),
                "--qwen-model",
                str(tmp_path / "qwen.gguf"),
                "--hebrew-model",
                str(tmp_path / "hebrew.gguf"),
                "--worker-load-receipt",
                str(tmp_path / "receipt.json"),
            ]
        )

    assert writes == []
    assert not (tmp_path / "evidence.json").exists()
    assert [method for method, _params in control.calls] == ["describe-plugins"]


@pytest.mark.parametrize(
    ("wrong_name", "detail"),
    [
        ("qwen.gguf", "Qwen model path"),
        ("hebrew.gguf", "Hebrew model path"),
    ],
)
def test_collector_refuses_local_model_bytes_that_disagree_with_receipt(
    wrong_name, detail, monkeypatch
):
    collector = _collector_module()
    receipt = _receipt(DEFAULT_PRIMARY_MODEL_SHA256, 33)
    control = _install_control(monkeypatch, collector, DEFAULT_PRIMARY_MODEL_ID)
    monkeypatch.setattr(collector, "load_selected_text_worker_load_receipt", lambda _path: receipt)
    monkeypatch.setattr(
        collector,
        "load_selected_text_corpus",
        lambda _path: pytest.fail("corpus was loaded"),
    )

    def digest(path):
        if path.name == wrong_name:
            return "0" * 64
        if path.name == "qwen.gguf":
            return DEFAULT_PRIMARY_MODEL_SHA256
        return HEBREW_MODEL_SHA256

    monkeypatch.setattr(collector, "file_digest", digest)
    arguments = SimpleNamespace(
        corpus=Path("corpus.json"),
        qwen_model=Path("qwen.gguf"),
        hebrew_model=Path("hebrew.gguf"),
        worker_load_receipt=Path("receipt.json"),
        timeout_seconds=1.0,
    )

    with pytest.raises(RuntimeError, match=detail):
        asyncio.run(collector._collect(arguments))

    assert [method for method, _params in control.calls] == ["describe-plugins"]


def test_collector_main_publishes_only_the_completed_document(tmp_path, monkeypatch, capsys):
    collector = _collector_module()
    output = tmp_path / "evidence.json"
    document = {"evidenceVersion": 1, "models": _receipt().models_document()}
    writes = []

    async def collect(_arguments):
        return document

    monkeypatch.setattr(collector, "_collect", collect)
    monkeypatch.setattr(
        collector,
        "write_json_atomic",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )

    collector.main(
        [
            "--output",
            str(output),
            "--qwen-model",
            "qwen.gguf",
            "--hebrew-model",
            "hebrew.gguf",
            "--worker-load-receipt",
            "receipt.json",
        ]
    )

    assert writes == [((output, document), {"prefix": ".selected-text-evidence-"})]
    assert json.loads(capsys.readouterr().out) == document


def test_collector_requires_an_explicit_worker_load_receipt():
    collector = _collector_module()

    with pytest.raises(SystemExit) as caught:
        collector._arguments(
            [
                "--output",
                "evidence.json",
                "--qwen-model",
                "qwen.gguf",
                "--hebrew-model",
                "hebrew.gguf",
            ]
        )

    assert caught.value.code == 2
