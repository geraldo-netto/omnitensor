"""Lazy BGE-small ncnn worker sharing the cross-process GPU lease."""

from __future__ import annotations

import asyncio
import fcntl
import math
from collections.abc import Sequence
from pathlib import Path

from omnitensor.plugins.document_qa import EmbeddingProvider
from omnitensor.plugins.protocol import CancellationToken
from omnitensor.training.document_model import QUERY_PREFIX, BgeTokenizer, VulkanBgeRunner

QUALIFIED_DEVICE = "AMD Radeon RX 6600 XT (RADV NAVI23)"


class BgeVulkanEmbedder:
    def __init__(
        self,
        model: Path,
        tokenizer: Path,
        model_sha256: str,
        lease_path: Path,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        if not lease_path.is_file():
            raise ValueError("accelerator lease is unavailable")
        self._lease_path = lease_path
        self._descriptor = EmbeddingProvider("bge-small-documents-gpu", "gpu", model_sha256, True)

    @property
    def descriptor(self) -> EmbeddingProvider:
        return self._descriptor

    async def preflight(self) -> None:
        await asyncio.to_thread(self._embed_sync, ("BGE startup probe",), False, None)

    async def embed(
        self,
        texts: Sequence[str],
        *,
        query: bool,
        cancellation: CancellationToken,
    ) -> Sequence[Sequence[float]]:
        cancellation.raise_if_cancelled()
        vectors = await asyncio.to_thread(self._embed_sync, tuple(texts), query, cancellation)
        cancellation.raise_if_cancelled()
        return vectors

    def _embed_sync(
        self,
        texts: tuple[str, ...],
        query: bool,
        cancellation: CancellationToken | None,
    ) -> tuple[tuple[float, ...], ...]:
        with self._lease_path.open("a+b") as lease:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
            tokenizer = BgeTokenizer(self._tokenizer)
            runner = VulkanBgeRunner(
                self._model,
                tokenizer,
                device_index=_qualified_device_index(),
            )
            if runner.device_name != QUALIFIED_DEVICE:
                raise ValueError("BGE is not qualified on this Vulkan device")
            vectors = []
            for text in texts:
                if cancellation is not None:
                    cancellation.raise_if_cancelled()
                prepared = f"{QUERY_PREFIX}{text}" if query else text
                vector = tuple(float(value) for value in runner.embed(prepared))
                if len(vector) != 384 or any(not math.isfinite(value) for value in vector):
                    raise ValueError("BGE returned an invalid embedding")
                vectors.append(vector)
            return tuple(vectors)


def _qualified_device_index() -> int:
    try:
        import ncnn
    except ImportError as error:  # pragma: no cover - dependency boundary
        raise ValueError("ncnn is unavailable") from error
    matches = tuple(
        index
        for index in range(ncnn.get_gpu_count())
        if ncnn.get_gpu_info(index).device_name() == QUALIFIED_DEVICE
    )
    if len(matches) != 1:
        raise ValueError("qualified BGE Vulkan device is unavailable")
    return matches[0]
