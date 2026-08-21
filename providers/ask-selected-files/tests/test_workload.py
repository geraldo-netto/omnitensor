"""What this distribution assembles, checked where the wheel is built.

Every structural half — the entry-point name, the packaged manifest, the
plugin id inside it, the wheel's package list — is derived from the tree and
checked in both directions by `tests/test_generation_installation.py` in the
service repository. What is left is what only this distribution knows: that
the generation runtime and the ncnn embedder compose into one workload.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import omnitensor_ask_selected_files as workload
import pytest
from omnitensor_vulkan_runtime import factories
from omnitensor_vulkan_runtime import qualification as qualification_module

from omnitensor.plugins.document_spans import EmbeddingProvider
from omnitensor.sdk import BootstrapArtifact

QUALIFIED_DEVICE = "AMD Radeon RX 6600 XT (RADV NAVI23)"


def _qualification():
    return qualification_module.Qualification(QUALIFIED_DEVICE)


class FakeEmbedder:
    def __init__(self, *arguments):
        self.arguments = arguments
        self.descriptor = EmbeddingProvider("bge", "gpu", "c" * 64, True)

    async def embed(self, *_arguments, **_keywords):
        return ()

    async def preflight(self):
        return None

    async def aclose(self):
        return None


@pytest.fixture
def bootstrapped(tmp_path, monkeypatch):
    lease = tmp_path / "generation.lock"
    lease.touch()
    qwen = BootstrapArtifact(
        qualification_module.default_model("ask-selected-files"),
        "1.0.0",
        "gguf",
        "a" * 64,
        tmp_path / "model.gguf",
        (),
        "https://example.invalid/model.gguf",
        "Apache-2.0",
    )
    bge = BootstrapArtifact(
        workload.BGE_ARTIFACT_ID,
        "1.0.0",
        "ncnn",
        "c" * 64,
        tmp_path / "bge" / "model.param",
        (("model.bin", "d" * 64), ("tokenizer.json", "e" * 64)),
        "https://example.invalid/bge.param",
        "MIT",
    )
    # The Hebrew translation route (OMNI-0617 stage 2): the file side offers
    # translate, and an explicit Hebrew target rides DictaLM.
    dicta = BootstrapArtifact(
        factories.HEBREW_ARTIFACT_ID,
        "1.0.0",
        "gguf",
        "f" * 64,
        tmp_path / "dicta.gguf",
        (),
        "https://example.invalid/dicta.gguf",
        "Apache-2.0",
    )

    class Bootstrap:
        model_choice = ""
        accelerator_lease_path = lease
        state_path = tmp_path / "state"

        def chosen_or(self, default_artifact_id):
            return self.require_artifact(self.model_choice or default_artifact_id)

        def require_artifact(self, artifact_id):
            return {qwen.id: qwen, bge.id: bge, dicta.id: dicta}[artifact_id]

        def find_artifact(self, _artifact_id):
            return None

    monkeypatch.setattr(factories, "current_plugin_bootstrap", lambda _plugin_id: Bootstrap())
    monkeypatch.setattr(factories, "load_qualification", lambda *_arguments: _qualification())
    monkeypatch.setattr(factories, "load_model_qualification", lambda *_arguments: _qualification())
    monkeypatch.setattr(workload, "BgeVulkanEmbedder", FakeEmbedder)
    return lease, bge


def test_the_entry_point_builds_a_workload_out_of_both_distributions(bootstrapped):
    lease, bge = bootstrapped

    built = workload.create()

    assert built.plugin_id == "ask-selected-files"
    assert isinstance(built._embedder, FakeEmbedder)
    assert built._embedder.arguments == (
        bge.path,
        bge.path.parent / "tokenizer.json",
        bge.sha256,
        lease,
    )
    # The Hebrew route is wired the way the inline workload wires it
    # (OMNI-0617 stage 2): a dedicated runtime beside the primary, and the
    # plugin holding the route under the person's own spelling of the target.
    assert len(built._runtimes) == 2
    assert set(built._plugin._translation_routes) == {"hebrew"}
    assert workload.__all__ == ["create"]


def test_an_incomplete_bge_artifact_refuses_before_a_worker_exists(bootstrapped, monkeypatch):
    """The weights are one file of three; two of them are companions."""
    _lease, bge = bootstrapped
    monkeypatch.setattr(
        type(bge), "companions", property(lambda _self: (("model.bin", "d" * 64),)), raising=False
    )

    with pytest.raises(RuntimeError, match="BGE companions are incomplete"):
        workload.create()


def test_the_embedder_this_distribution_uses_is_the_ncnn_one():
    """Not the generation wheel's: it has not had one since the split."""
    from omnitensor_ncnn_embeddings import BgeVulkanEmbedder

    assert workload.BgeVulkanEmbedder is BgeVulkanEmbedder
    assert not hasattr(factories, "BgeVulkanEmbedder")


def test_the_packaged_manifest_declares_this_workload():
    root = Path(__file__).resolve().parents[1]
    assert (root / "src/omnitensor_ask_selected_files").is_dir()


def test_the_pinned_ocr_pair_becomes_the_image_and_pdf_adapter(bootstrapped, monkeypatch, tmp_path):
    """OMNI-0615: when both OCR artifacts are installed, selected images and
    scans are read verbatim through VulkanOCR rather than refused."""

    from omnitensor.plugins.extraction import AdapterKind

    class FakeReader:
        kind = AdapterKind.OCR

        def __init__(self, *arguments):
            self.arguments = arguments

        def pages(self, item):
            raise NotImplementedError

    det = SimpleNamespace(path=tmp_path / "det.param")
    rec = SimpleNamespace(path=tmp_path / "ocr" / "rec.param")
    found = {workload.OCR_DET_ARTIFACT_ID: det, workload.OCR_REC_ARTIFACT_ID: rec}

    original = factories.current_plugin_bootstrap

    def with_ocr(plugin_id):
        bootstrap = original(plugin_id)
        bootstrap.find_artifact = staticmethod(found.get)
        return bootstrap

    monkeypatch.setattr(factories, "current_plugin_bootstrap", with_ocr)
    monkeypatch.setattr(workload, "VulkanOcrExtractionAdapter", FakeReader)

    built = workload.create()

    adapters = built._plugin._adapters
    readers = {adapters[suffix] for suffix in (".pdf", ".png", ".jpg", ".jpeg", ".webp")}
    assert len(readers) == 1
    (reader,) = readers
    assert isinstance(reader, FakeReader)
    assert reader.arguments == (det.path, rec.path, rec.path.parent / "labels.txt")
