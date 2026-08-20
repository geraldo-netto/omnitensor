"""BGE embeddings on ncnn, as a distribution of its own.

This shares no code path with llama.cpp: a different engine, a different
model format, and its own `[documents]` dependencies — `ncnn`, `numpy`,
`tokenizers` — which every wheel that wanted generation used to carry because
the embedder lived inside the Vulkan generation runtime.
"""

from .bge import QUALIFIED_DEVICE, BgeVulkanEmbedder, vulkan_device_index

__all__ = ["QUALIFIED_DEVICE", "BgeVulkanEmbedder", "vulkan_device_index"]
