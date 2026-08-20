"""Lazy BGE-small ncnn worker sharing the cross-process GPU lease."""

from __future__ import annotations

import asyncio
import fcntl
import math
import threading
from collections.abc import Sequence
from pathlib import Path

# The tokenizer and the Vulkan runner are inference, not production: they run
# on every question a person asks. They used to live in `omnitensor.training`
# because that is where the model was built, which made this provider — part
# of the serving path — unable to start on a machine that installed the
# service without the trainers.
from omnitensor.document_model_runners import BgeTokenizer, VulkanBgeRunner
from omnitensor.document_model_types import QUERY_PREFIX
from omnitensor.executors.vulkan import VulkanSelectionError, hardware_vulkan_devices
from omnitensor.plugins.document_spans import EmbeddingProvider
from omnitensor.plugins.protocol import CancellationToken

# The device the embedder was measured on. A preference, not a requirement:
# it used to be matched exactly and anything else refused, so a person whose
# 6600 XT was busy and who moved the work to the integrated 610M got
# "qualified BGE Vulkan device is unavailable" and no answer at all. What the
# receipt records is where the numbers came from; what runs is the person's to
# choose.
QUALIFIED_DEVICE = "AMD Radeon RX 6600 XT (RADV NAVI23)"


class BgeVulkanEmbedder:
    def __init__(
        self,
        model: Path,
        tokenizer: Path,
        model_sha256: str,
        lease_path: Path,
        device: str = "",
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        if not lease_path.is_file():
            raise ValueError("accelerator lease is unavailable")
        self._lease_path = lease_path
        self._descriptor = EmbeddingProvider("bge-small-documents-gpu", "gpu", model_sha256, True)
        self._runner = None
        self._runner_lock = threading.Lock()

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

    async def aclose(self) -> None:
        """Release the loaded model; safe to call more than once."""
        await asyncio.to_thread(self.close)

    def close(self) -> None:
        """Unload under the lease, so no embed is running on what is released."""
        with self._lease_path.open("a+b") as lease:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
            with self._runner_lock:
                runner, self._runner = self._runner, None
        if runner is not None:
            release = getattr(runner, "close", None)
            if callable(release):
                release()

    def _runner_under_lease(self):
        """The tokenizer, device and loaded model, built at most once.

        This was rebuilt on every embed call: a fresh tokenizer, a fresh ncnn
        ``Net`` — reading the weights and compiling every layer's shaders —
        and a fresh Vulkan enumeration, which initialises the loader and
        queries every driver. That is the 60–190 ms
        ``omnitensor.executors.vulkan`` documents and caches, paid per
        question rather than per worker, and nothing released the runner it
        replaced either.

        The caller holds the cross-process lease, so the model is loaded while
        this worker owns the GPU; the lock only keeps two of this process's
        embed threads from loading it twice.
        """
        with self._runner_lock:
            if self._runner is None:
                tokenizer = BgeTokenizer(self._tokenizer)
                self._runner = VulkanBgeRunner(
                    self._model,
                    tokenizer,
                    device_index=vulkan_device_index(self._device or QUALIFIED_DEVICE),
                )
            return self._runner

    def _embed_sync(
        self,
        texts: tuple[str, ...],
        query: bool,
        cancellation: CancellationToken | None,
    ) -> tuple[tuple[float, ...], ...]:
        with self._lease_path.open("a+b") as lease:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
            runner = self._runner_under_lease()
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


def vulkan_device_index(preferred: str) -> int:
    """The GPU to embed on: the one asked for, or the only hardware one there is.

    This matched one hardcoded device name and refused everything else, so the
    embedder ran on exactly one card in the world and `ask-selected-files` was
    unavailable on any other machine — including the same machine's second GPU.
    A named preference that is present is honoured; otherwise a sole GPU is the
    obvious answer. Only an ambiguous choice is refused, and it says what the
    alternatives were.

    Which devices exist is asked of `select_vulkan_device`'s enumeration, not
    of a second one here: that one drops software (CPU) Vulkan devices, and
    the count it also reports separates "only llvmpipe is present" — the
    no-CPU rule refusing, which used to embed on the host CPU whenever a
    software device was the only one — from no Vulkan device at all.
    """
    try:
        import ncnn  # noqa: PLC0415 - deferred: an optional or heavy dependency
    except ImportError as error:  # pragma: no cover - dependency boundary
        raise ValueError("ncnn is unavailable") from error
    try:
        count, candidates = hardware_vulkan_devices(ncnn)
    except VulkanSelectionError as error:
        raise ValueError(error.reason) from error
    devices = [device for _preference, _index, device in candidates]
    if not devices:
        if count > 0:
            raise ValueError(
                "only a software (CPU) Vulkan device is present; CPU execution is not used"
            )
        raise ValueError("no Vulkan device is available")
    matches = [device for device in devices if device.name == preferred]
    if len(matches) == 1:
        return matches[0].index
    if len(devices) == 1:
        return devices[0].index
    names = ", ".join(sorted({device.name for device in devices}))
    raise ValueError(f"name the Vulkan device to embed on, one of: {names}")
