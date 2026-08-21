"""The one workload that embeds as well as generates.

Every other workload wheel is a re-export of a factory in the shared Vulkan
runtime. This one is not, because it needs two distributions: the llama.cpp
generation runtime and the ncnn embedder. Assembling it here is what keeps
ncnn out of the wheel that only generates — the factory used to live beside
the runtime, so the `[documents]` extra put an embedding engine into every
installation that wanted answers.
"""

from omnitensor_ncnn_embeddings import BgeVulkanEmbedder
from omnitensor_vulkan_runtime import QualifiedWorkload, generation_context

from omnitensor.plugins.document_qa import DocumentQuestionPlugin
from omnitensor.plugins.ocr_extraction import VulkanOcrExtractionAdapter

BGE_ARTIFACT_ID = "bge-small-en-v1-5-ask-gpu"
OCR_DET_ARTIFACT_ID = "ppocrv6-medium-det"
OCR_REC_ARTIFACT_ID = "ppocrv6-medium-rec"


def create() -> QualifiedWorkload:
    bootstrap, store, runtime, model, qualification, router = generation_context(
        "ask-selected-files"
    )
    bge = bootstrap.require_artifact(BGE_ARTIFACT_ID)
    companions = dict(bge.companions)
    if set(companions) != {"model.bin", "tokenizer.json"}:
        raise RuntimeError("BGE companions are incomplete")
    embedder = BgeVulkanEmbedder(
        bge.path,
        bge.path.parent / "tokenizer.json",
        bge.sha256,
        bootstrap.accelerator_lease_path,
    )
    # Selected images and scanned pages become citable fragments through the
    # VulkanOCR engine (OMNI-0615) — verbatim source text, never a vision
    # model's claim about it. Found rather than required: without the pinned
    # pair the previous adapters stand, and a selected image is refused the
    # way it always was rather than the worker failing to start.
    ocr_det = bootstrap.find_artifact(OCR_DET_ARTIFACT_ID)
    ocr_rec = bootstrap.find_artifact(OCR_REC_ARTIFACT_ID)
    adapters = None
    if ocr_det is not None and ocr_rec is not None:
        reader = VulkanOcrExtractionAdapter(
            ocr_det.path, ocr_rec.path, ocr_rec.path.parent / "labels.txt"
        )
        adapters = dict.fromkeys((".pdf", ".png", ".jpg", ".jpeg", ".webp"), reader)
    return QualifiedWorkload(
        DocumentQuestionPlugin(embedder, router, store, adapters=adapters),
        runtime,
        model.path,
        qualification,
        embedder,
    )


__all__ = ["create"]
