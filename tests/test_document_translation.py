"""OMNI-0520/0521: a whole selected document, translated span by span.

Since OMNI-0617 stage 3a the shipped workload is `DocumentTranslationAlias`,
the same public contract answered by the file-side translate pipeline; the
alias suite lives at the end of this file. The original plugin's suite stays
until stage 3c retires the class it pins.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins.document_translation import (
    MAX_LANGUAGE_CHARACTERS,
    PLUGIN_ID,
    DocumentTranslationAlias,
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
from omnitensor.plugins.target_language import TARGET_LANGUAGE_PATTERN
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
        # A digit inside a name is a language tag — `es-419` is one — so what
        # is refused is a name that does not begin as a name.
        {"3Hebrew": router},
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

    The plugin used to check only the length, so a target the published
    contract refuses reached the prompt. Both now name the same pattern, and
    that pattern is a person's own words for a language rather than an ASCII
    subset of them (G5).
    """

    store = MemoryFragmentStore()
    subject = await started(store)
    source = document(tmp_path)
    pattern = json.loads(
        (ROOT / "plugin-manifests" / "document-translation.json").read_text(encoding="utf-8")
    )["plugin"]["schemas"]["input"]["properties"]["targetLanguage"]["pattern"]
    assert pattern == TARGET_LANGUAGE_PATTERN

    for target in ("9Hebrew", "'Hebrew'", "-Hebrew"):
        refused = await subject.execute(
            request(source, targetLanguage=target), CancellationController(), Progress()
        )
        assert refused.detail == "language-invalid", target


# --- OMNI-0617 stage 3a: the compatibility alias over the file-side translate ---


class FileSideWorker:
    """Answers the file-operations contract: reads the `{operation, language}`
    control, translates the one content fragment, and cites it exactly."""

    descriptor = TranslatingWorker.descriptor

    def __init__(self, store, *, mutate=None):
        self.store = store
        self.mutate = mutate
        self.controls = []
        self.spans = []

    async def generate(self, task, request, cancellation, progress):
        cancellation.raise_if_cancelled()
        await progress.report(type("ProviderProgress", (), {"fraction": 0.5})())
        control = json.loads(
            self.store.resolve(request.request_id, request.content_references[0]).text
        )
        self.controls.append(control)
        fragment = self.store.resolve(request.request_id, request.content_references[1])
        self.spans.append(fragment.text)
        document = {
            "version": 1,
            "requestId": request.request_id,
            "operation": "translate",
            "answer": f"[{control['language']}] {fragment.text}",
            "citations": [
                {
                    "sourceRef": fragment.reference,
                    "sourceSha256": fragment.source_sha256,
                    "page": fragment.page,
                    "span": {"start": 0, "end": len(fragment.text)},
                    "textSha256": fragment.text_sha256,
                }
            ],
        }
        if self.mutate is not None:
            self.mutate(document)
        return json.dumps(document)

    async def terminate(self, _request_id):
        return None


async def started_alias(store, *, mutate=None, translation_routes=None):
    worker = FileSideWorker(store, mutate=mutate)
    subject = DocumentTranslationAlias(
        GenerationRouter((worker,)), store, translation_routes=translation_routes
    )
    await subject.start(
        PluginContext(PLUGIN_ID, 1, {}, frozenset({"files:read-selected", "accelerator:gpu"}))
    )
    return subject, worker


@pytest.mark.asyncio
async def test_the_alias_answers_the_unchanged_result_contract(tmp_path):
    """The stage-2 pipeline underneath, the stage-0 contract on the wire."""

    store = MemoryFragmentStore()
    subject, worker = await started_alias(store)
    source = document(tmp_path)

    result = await subject.execute(request(source), CancellationController(), Progress())

    assert result.status is PluginResultStatus.SUCCEEDED
    output = result.output
    assert validate_document("document-translation-result.schema.json", output) == []
    assert set(output) == {
        "version",
        "requestId",
        "targetLanguage",
        "providerId",
        "accelerator",
        "documents",
    }
    assert output["targetLanguage"] == "Hebrew"
    assert output["accelerator"] == "gpu"
    [translated] = output["documents"]
    assert translated["documentId"] == "selected-document-1"
    assert translated["fileName"] == "report.txt"
    assert translated["spanCount"] >= 1
    for index in range(3):
        assert f"Paragraph {index + 1} says something." in translated["text"]
    assert translated["text"].count("[Hebrew]") == translated["spanCount"]
    assert (
        translated["translationSha256"]
        == hashlib.sha256(translated["text"].encode("utf-8")).hexdigest()
    )
    # The alias pinned the operation and mapped targetLanguage onto language.
    assert worker.controls == [{"operation": "translate", "language": "Hebrew"}] * len(
        worker.controls
    )
    assert worker.controls != []


@pytest.mark.asyncio
async def test_the_alias_sends_hebrew_to_the_dedicated_route(tmp_path):
    """The Dicta rule survives the aliasing: an explicit Hebrew target rides
    the dedicated route, every other target the general one."""

    store = MemoryFragmentStore()
    general = FileSideWorker(store)
    hebrew = FileSideWorker(store)
    subject = DocumentTranslationAlias(
        GenerationRouter((general,)),
        store,
        translation_routes={"Hebrew": GenerationRouter((hebrew,))},
    )
    await subject.start(
        PluginContext(PLUGIN_ID, 1, {}, frozenset({"files:read-selected", "accelerator:gpu"}))
    )
    source = document(tmp_path)

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


@pytest.mark.asyncio
async def test_the_alias_keeps_the_request_contract(tmp_path):
    store = MemoryFragmentStore()
    subject, _worker = await started_alias(store)
    source = document(tmp_path)

    blank = await subject.execute(
        request(source, targetLanguage="  "), CancellationController(), Progress()
    )
    assert blank.status is PluginResultStatus.FAILED
    assert blank.detail == "language-invalid"

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
async def test_a_cancelled_alias_job_reaches_a_terminal_result(tmp_path):
    store = MemoryFragmentStore()
    subject, _worker = await started_alias(store)
    cancellation = CancellationController()
    cancellation.cancel()

    result = await subject.execute(request(document(tmp_path)), cancellation, Progress())

    assert result.status is PluginResultStatus.CANCELLED
    assert result.detail == "document translation cancelled"


@pytest.mark.asyncio
async def test_a_file_side_refusal_is_the_alias_result_not_a_crash(tmp_path):
    """A `DocumentQuestionError` from the pipeline underneath lands inside the
    alias's own terminal result, the way the old plugin's refusals did."""

    store = MemoryFragmentStore()

    def cite_elsewhere(answer):
        answer["citations"][0]["textSha256"] = "f" * 64

    subject, _worker = await started_alias(store, mutate=cite_elsewhere)

    result = await subject.execute(
        request(document(tmp_path)), CancellationController(), Progress()
    )

    assert result.status is PluginResultStatus.FAILED
    assert result.detail == "answer-invalid"


# The manifest's own shape: begins as a name, continues in word characters,
# apostrophes, hyphens, and inner spaces (`target_language.py`).
_LANGUAGES = st.builds(
    lambda first, rest: (first + rest).strip() or first,
    st.characters(whitelist_categories=("Lu", "Ll", "Lo")),
    st.text(
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Lo", "Nd"), whitelist_characters=" '-"
        ),
        max_size=30,
    ),
)


@given(language=_LANGUAGES)
def test_every_valid_target_reaches_the_core_as_language(language):
    """OMNI-0617 stage 3a boundary: whatever `targetLanguage` a person may
    name, the control fragment carries it as the core's `language` and the
    result carries it back as `targetLanguage`, unchanged."""

    async def run():
        store = MemoryFragmentStore()
        subject, worker = await started_alias(store)
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "report.txt"
            source.write_text("One paragraph is enough.", encoding="utf-8")
            result = await subject.execute(
                request(source, targetLanguage=language), CancellationController(), Progress()
            )
        return result, worker

    result, worker = asyncio.run(run())
    assert result.status is PluginResultStatus.SUCCEEDED, result.detail
    assert result.output["targetLanguage"] == language.strip()
    assert worker.controls[0] == {"operation": "translate", "language": language.strip()}
