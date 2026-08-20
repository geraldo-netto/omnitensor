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

BGE_ARTIFACT_ID = "bge-small-en-v1-5-ask-gpu"


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
    return QualifiedWorkload(
        DocumentQuestionPlugin(embedder, router, store),
        runtime,
        model.path,
        qualification,
        embedder,
    )


__all__ = ["create"]
