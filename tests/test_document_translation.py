"""OMNI-0520/0521: a whole selected document, translated span by span."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from omnitensor.plugins.document_translation import (
    MAX_LANGUAGE_CHARACTERS,
    PLUGIN_ID,
    DocumentTranslationError,
    DocumentTranslationPlugin,
    document_translation_task,
    translated_documents_result,
)
from omnitensor.plugins.event_workload import MemoryFragmentStore
from omnitensor.plugins.generation import (
    ArtifactProvenance,
    GenerationProviderDescriptor,
    GenerationRouter,
)
from omnitensor.plugins.protocol import (
    PluginContext,
    PluginRequest,
    PluginResultStatus,
)
from omnitensor.registry import validate_document
from omnitensor.sdk import CancellationController

ROOT = Path(__file__).parents[1]


class Progress:
    def __init__(self):
        self.items = []

    async def report(self, item):
        self.items.append(item)


class TranslatingWorker:
    descriptor = GenerationProviderDescriptor(
        "dictalm-gpu",
        "gpu",
        "llama.cpp-vulkan",
        True,
        ArtifactProvenance(
            "dictalm2",
            "1",
            "a" * 64,
            "https://example.invalid/dictalm.gguf",
            "Apache-2.0",
        ),
    )

    def __init__(self, store, *, mutate=None):
        self.store = store
        self.mutate = mutate
        self.spans = []

    async def generate(self, task, request, cancellation, progress):
        cancellation.raise_if_cancelled()
        await progress.report(type("ProviderProgress", (), {"fraction": 0.5})())
        control = self.store.resolve(request.request_id, request.content_references[0])
        assert control.reference.endswith(":target-language")
        reference = request.content_references[1]
        fragment = self.store.resolve(request.request_id, reference)
        self.spans.append(fragment.text)
        document = {
            "version": 1,
            "requestId": request.request_id,
            "translation": f"[{control.text}] {fragment.text}",
            "evidence": {
                "sourceRef": reference,
                "textSha256": fragment.text_sha256,
            },
        }
        if self.mutate is not None:
            self.mutate(document)
        return json.dumps(document)

    async def terminate(self, _request_id):
        return None


def plugin(store, *, mutate=None):
    return DocumentTranslationPlugin(
        GenerationRouter((TranslatingWorker(store, mutate=mutate),)), store
    )


def request(source: Path, **payload) -> PluginRequest:
    return PluginRequest(
        "job-1",
        PLUGIN_ID,
        "manual",
        {"sources": [str(source)], "targetLanguage": "Hebrew", **payload},
        1,
        None,
    )


async def started(store, *, mutate=None):
    subject = plugin(store, mutate=mutate)
    await subject.start(
        PluginContext(PLUGIN_ID, 1, {}, frozenset({"files:read-selected", "accelerator:gpu"}))
    )
    return subject


def document(tmp_path: Path, paragraphs: int = 3) -> Path:
    source = tmp_path / "report.txt"
    source.write_text(
        "\n\n".join(f"Paragraph {index + 1} says something." for index in range(paragraphs)),
        encoding="utf-8",
    )
    return source


@pytest.mark.asyncio
async def test_a_whole_document_is_translated_and_reassembled(tmp_path):
    store = MemoryFragmentStore()
    subject = await started(store)
    source = document(tmp_path)

    result = await subject.execute(request(source), CancellationController(), Progress())

    assert result.status is PluginResultStatus.SUCCEEDED
    output = result.output
    assert validate_document("document-translation-result.schema.json", output) == []
    [translated] = output["documents"]
    assert output["targetLanguage"] == "Hebrew"
    assert output["accelerator"] == "gpu"
    assert translated["documentId"] == "selected-document-1"
    assert translated["fileName"] == "report.txt"
    assert translated["spanCount"] >= 1
    # Every paragraph is present, in order, and each was translated.
    for index in range(3):
        assert f"Paragraph {index + 1} says something." in translated["text"]
    assert translated["text"].count("[Hebrew]") == translated["spanCount"]
    assert (
        translated["translationSha256"]
        == hashlib.sha256(translated["text"].encode("utf-8")).hexdigest()
    )


@pytest.mark.asyncio
async def test_progress_is_reported_and_the_private_material_is_discarded(tmp_path):
    store = MemoryFragmentStore()
    subject = await started(store)
    progress = Progress()

    await subject.execute(request(document(tmp_path)), CancellationController(), progress)

    stages = [item.stage for item in progress.items]
    assert stages[0] == "extract"
    assert stages[-1] == "terminal"
    assert progress.items[-1].fraction == 1.0
    assert all(0.0 <= item.fraction <= 1.0 for item in progress.items)


@pytest.mark.asyncio
async def test_a_span_answered_from_other_material_is_refused(tmp_path):
    store = MemoryFragmentStore()

    def cite_elsewhere(answer):
        answer["evidence"]["textSha256"] = "f" * 64

    subject = await started(store, mutate=cite_elsewhere)

    result = await subject.execute(
        request(document(tmp_path)), CancellationController(), Progress()
    )

    assert result.status is PluginResultStatus.FAILED
    assert result.detail == "translation-invalid"


@pytest.mark.asyncio
async def test_an_empty_translation_stops_the_job_rather_than_losing_a_span(tmp_path):
    store = MemoryFragmentStore()

    def empty(answer):
        answer["translation"] = ""

    subject = await started(store, mutate=empty)

    result = await subject.execute(
        request(document(tmp_path)), CancellationController(), Progress()
    )

    assert result.status is PluginResultStatus.FAILED
    # Refused at the provider boundary, by the same schema the plugin checks:
    # an empty translation is a span silently lost from the document.
    assert result.detail == "provider-output-invalid"


@pytest.mark.asyncio
async def test_the_request_must_name_a_language_and_be_manual(tmp_path):
    store = MemoryFragmentStore()
    subject = await started(store)
    source = document(tmp_path)

    blank = await subject.execute(
        request(source, targetLanguage="  "), CancellationController(), Progress()
    )
    assert blank.status is PluginResultStatus.FAILED
    assert blank.detail == "language-invalid"

    overlong = await subject.execute(
        request(source, targetLanguage="x" * (MAX_LANGUAGE_CHARACTERS + 1)),
        CancellationController(),
        Progress(),
    )
    assert overlong.detail == "language-invalid"

    automatic = PluginRequest(
        "job-2",
        PLUGIN_ID,
        "automatic",
        {"sources": [str(source)], "targetLanguage": "Hebrew"},
        1,
        None,
    )
    assert (
        await subject.execute(automatic, CancellationController(), Progress())
    ).detail == "request-invalid"


@pytest.mark.asyncio
async def test_a_cancelled_translation_reaches_a_terminal_result(tmp_path):
    store = MemoryFragmentStore()
    subject = await started(store)
    cancellation = CancellationController()
    cancellation.cancel()

    result = await subject.execute(request(document(tmp_path)), cancellation, Progress())

    assert result.status is PluginResultStatus.CANCELLED
    assert result.detail == "document translation cancelled"


def test_the_result_contract_refuses_a_document_it_cannot_describe():
    with pytest.raises(DocumentTranslationError) as error:
        translated_documents_result("job-1", "Hebrew", (), provider_id="p", accelerator="gpu")
    assert error.value.code == "result-invalid"


def test_the_task_declares_no_output_ceiling():
    """A translation is as long as the document; only the window bounds it."""

    task = document_translation_task()
    assert task.task_id == PLUGIN_ID
    assert task.limits.output_tokens is None
    assert task.limits.context_tokens == 32_768


def test_the_manifest_declares_the_published_contracts():
    manifest = json.loads(
        (ROOT / "plugin-manifests" / "document-translation.json").read_text(encoding="utf-8")
    )
    assert validate_document("workload-manifest.schema.json", manifest) == []
    assert manifest["id"] == PLUGIN_ID
    assert manifest["plugin"]["entryPoint"] == PLUGIN_ID
    assert manifest["plugin"]["triggers"] == ["manual"]
    assert manifest["plugin"]["schemas"]["output"] == {
        "$ref": "document-translation-result.schema.json"
    }
    assert set(manifest["plugin"]["schemas"]["input"]["required"]) == {
        "sources",
        "targetLanguage",
    }
    assert "files:read-selected" in manifest["plugin"]["permissions"]


@pytest.mark.asyncio
async def test_a_language_with_its_own_qualified_model_is_answered_by_that_model(tmp_path):
    """OMNI-0522: the Hebrew route the distribution wires, proved here.

    The provider mounts a second, separately qualified model for one target.
    A workload that took a single router would have sent Hebrew to the general
    model anyway, and nothing would have said so.
    """

    store = MemoryFragmentStore()
    general = TranslatingWorker(store)
    hebrew = TranslatingWorker(store)
    subject = DocumentTranslationPlugin(
        GenerationRouter((general,)),
        store,
        translation_routes={"Hebrew": GenerationRouter((hebrew,))},
    )
    await subject.start(
        PluginContext(PLUGIN_ID, 1, {}, frozenset({"files:read-selected", "accelerator:gpu"}))
    )
    source = document(tmp_path)

    # The target is matched the way the request is normalised, not by spelling.
    assert (
        await subject.execute(
            request(source, targetLanguage="hebrew"), CancellationController(), Progress()
        )
    ).status is PluginResultStatus.SUCCEEDED
    assert general.spans == []
    assert hebrew.spans != []

    hebrew.spans.clear()
    assert (
        await subject.execute(
            PluginRequest(
                "job-2",
                PLUGIN_ID,
                "manual",
                {"sources": [str(source)], "targetLanguage": "French"},
                1,
                None,
            ),
            CancellationController(),
            Progress(),
        )
    ).status is PluginResultStatus.SUCCEEDED
    assert hebrew.spans == []
    assert general.spans != []


def test_a_route_the_workload_cannot_address_is_refused_at_construction():
    store = MemoryFragmentStore()
    router = GenerationRouter((TranslatingWorker(store),))

    for routes in (
        {"": router},
        {"Hebr3w": router},
        {"x" * (MAX_LANGUAGE_CHARACTERS + 1): router},
    ):
        with pytest.raises(DocumentTranslationError) as invalid:
            DocumentTranslationPlugin(router, store, translation_routes=routes)
        assert invalid.value.code == "provider-invalid"

    with pytest.raises(DocumentTranslationError) as duplicated:
        DocumentTranslationPlugin(
            router, store, translation_routes={"Hebrew": router, "hebrew": router}
        )
    assert duplicated.value.code == "provider-invalid"

    with pytest.raises(DocumentTranslationError) as unrouted:
        DocumentTranslationPlugin(router, store, translation_routes={"Hebrew": object()})
    assert unrouted.value.code == "provider-invalid"

    with pytest.raises(DocumentTranslationError) as unmapped:
        DocumentTranslationPlugin(router, store, translation_routes=["Hebrew"])
    assert unmapped.value.code == "provider-invalid"


@pytest.mark.asyncio
async def test_the_plugin_refuses_the_targets_its_own_manifest_refuses(tmp_path):
    """The input contract and the plugin have to state one rule, not two.

    The manifest declares `^[A-Za-z][A-Za-z -]*$`; the plugin checked only the
    length, so a target the published contract refuses reached the prompt.
    """

    store = MemoryFragmentStore()
    subject = await started(store)
    source = document(tmp_path)
    pattern = json.loads(
        (ROOT / "plugin-manifests" / "document-translation.json").read_text(encoding="utf-8")
    )["plugin"]["schemas"]["input"]["properties"]["targetLanguage"]["pattern"]
    assert pattern == "^[A-Za-z][A-Za-z -]*$"

    for target in ("He9rew", "עברית", "-Hebrew"):
        refused = await subject.execute(
            request(source, targetLanguage=target), CancellationController(), Progress()
        )
        assert refused.detail == "language-invalid", target
