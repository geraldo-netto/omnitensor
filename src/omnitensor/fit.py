"""Whether a model fits a card, and what it would leave behind if it did.

Every model question on this desk has ended at the same wall: an 8B at Q4_K_M
with a 32,768-token context needs about 7.6 GiB, the discrete card has 8.0, and
so "why not something bigger" has been answered by arithmetic in a commit
message rather than by anything that runs. This is that arithmetic, in code,
checkable against a real card.

Three terms, and only three:

* **Weights** — the whole file. Nothing here runs a model partly on the CPU,
  so a model that does not fit does not run slower, it does not run.
* **The key/value cache** — the term people forget, and the one that decides
  the answer. It grows with the context the *task* asks for, so it is a
  property of the pairing rather than of the model: the same model needs 2.4
  GiB of cache for 32k tokens and 0.6 for 8k.
* **Everything else the runtime allocates** — graph and compute buffers. This
  is the one term estimated rather than derived, so it is stated separately and
  never folded into the others: a measured probe replaces it with what the
  driver actually reported, and an estimate that hides its guess is worse than
  no estimate.

Nothing here proposes shrinking a context or coarsening a cache to make
something fit. Both are ways of making an answer worse, and this exists to
report a tradeoff rather than to take one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .gguf import ModelShape

# Bytes per cached element, by llama.cpp cache type. The quantized blocks hold
# 32 elements plus their scales, so these are not whole numbers: q8_0 is 34
# bytes per 32 elements, q4_0 is 18.
CACHE_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q5_1": 24 / 32,
    "q5_0": 22 / 32,
    "q4_1": 20 / 32,
    "q4_0": 18 / 32,
}

# What the runtime allocates beyond weights and cache. Observed rather than
# derived, which is why a measured probe supersedes it; stated as its own term
# so nobody mistakes it for a calculation.
DEFAULT_OVERHEAD_BYTES = 512 * 1024 * 1024

# A card that is exactly full is a card that will fail on the first allocation
# nobody counted. This much has to remain unclaimed for a fit to be called one.
DEFAULT_MARGIN_BYTES = 192 * 1024 * 1024

_DRM_ROOT = Path("/sys/class/drm")


class UnknownCacheError(ValueError):
    """A cache precision llama.cpp does not have."""


@dataclass(frozen=True, slots=True)
class Estimate:
    """What one (model, context, cache) would take on a card."""

    model: str
    context_tokens: int
    cache: str
    weight_bytes: int
    cache_bytes: int
    overhead_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.cache_bytes + self.overhead_bytes

    def fits_in(self, free_bytes: int, margin_bytes: int = DEFAULT_MARGIN_BYTES) -> bool:
        return self.total_bytes + margin_bytes <= free_bytes

    def headroom_in(self, free_bytes: int) -> int:
        """What would be left over. Negative is how much it is short by."""
        return free_bytes - self.total_bytes


def estimate(
    shape: ModelShape,
    context_tokens: int,
    *,
    cache: str = "q8_0",
    overhead_bytes: int = DEFAULT_OVERHEAD_BYTES,
) -> Estimate:
    """What this model at this context and cache precision would occupy."""
    if context_tokens < 1:
        raise ValueError("a context of no tokens is not a context")
    per_element = CACHE_BYTES.get(cache)
    if per_element is None:
        raise UnknownCacheError(f"unknown cache precision: {cache}")
    return Estimate(
        model=shape.name,
        context_tokens=context_tokens,
        cache=cache,
        weight_bytes=shape.file_bytes,
        cache_bytes=int(shape.kv_bytes_per_token * per_element * context_tokens),
        overhead_bytes=max(0, overhead_bytes),
    )


@dataclass(frozen=True, slots=True)
class DeviceMemory:
    """What one card has, and what is already spoken for.

    Two pools, because an integrated GPU has two very different answers. A
    discrete card's VRAM is its own memory. An integrated one carves a small
    VRAM aperture out of system RAM — 4 GiB here — while its real capacity is
    GTT, the system memory it may map, which on this desk is 45 GiB. Reading
    only VRAM would report the 610M as too small for a 4B model that it can in
    fact hold comfortably.

    GTT alone cannot tell the two apart: a *discrete* card also maps system
    memory over PCIe and reports 45 GiB of it here, and running weights from
    there means streaming them across the bus for every token, which is not a
    fit by any useful definition. The signal that does separate them is the
    memory vendor the driver names — this card says "samsung", meaning real
    GDDR6 soldered to it, and the integrated one names nobody because its
    memory is the machine's.
    """

    device_id: str
    total_bytes: int
    used_bytes: int
    mapped_total_bytes: int = 0
    mapped_used_bytes: int = 0
    memory_vendor: str = ""

    @property
    def free_bytes(self) -> int:
        return max(0, self.total_bytes - self.used_bytes)

    @property
    def mapped_free_bytes(self) -> int:
        return max(0, self.mapped_total_bytes - self.mapped_used_bytes)

    @property
    def integrated(self) -> bool:
        vendor = self.memory_vendor.strip().lower()
        return vendor in ("", "unknown") and self.mapped_total_bytes > self.total_bytes

    @property
    def usable_free_bytes(self) -> int:
        """What a model may actually occupy on this card.

        For an integrated GPU that is system memory, at system-memory
        bandwidth — which is the whole question OMNI-0364 asks and this number
        deliberately does not answer: it says what fits, never how fast.
        """
        return max(self.free_bytes, self.mapped_free_bytes) if self.integrated else self.free_bytes


def device_memory(root: Path = _DRM_ROOT) -> tuple[DeviceMemory, ...]:
    """Every render node's memory, as its own driver reports it.

    Read here rather than through a GPU API because it must work with no
    process holding the card, and because the number that matters is what is
    free *now*, alongside whatever else the desktop is running.
    """
    found = []
    for node in sorted(root.glob("renderD*")):
        total = _read_int(node / "device/mem_info_vram_total")
        if total is None:
            continue
        found.append(
            DeviceMemory(
                device_id=f"gpu-{node.name}",
                total_bytes=total,
                used_bytes=_read_int(node / "device/mem_info_vram_used") or 0,
                mapped_total_bytes=_read_int(node / "device/mem_info_gtt_total") or 0,
                mapped_used_bytes=_read_int(node / "device/mem_info_gtt_used") or 0,
                memory_vendor=_read_text(node / "device/mem_info_vram_vendor"),
            )
        )
    return tuple(found)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()[:32]
    except OSError:
        return ""


def _read_int(path: Path) -> int | None:
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return value if value >= 0 else None


@dataclass(frozen=True, slots=True)
class Verdict:
    """One pairing of a model with a card, and whether it can run there."""

    estimate: Estimate
    device: DeviceMemory
    fits: bool
    headroom_bytes: int

    @property
    def short_by_bytes(self) -> int:
        return max(0, -self.headroom_bytes)


def verdict(
    estimated: Estimate,
    device: DeviceMemory,
    *,
    margin_bytes: int = DEFAULT_MARGIN_BYTES,
) -> Verdict:
    free = device.usable_free_bytes
    return Verdict(
        estimate=estimated,
        device=device,
        fits=estimated.fits_in(free, margin_bytes),
        headroom_bytes=estimated.headroom_in(free),
    )


def gibibytes(value: float) -> str:
    """Bytes as a person compares them: one number, two decimals, a unit."""
    return f"{value / (1024**3):.2f} GiB"


__all__ = [
    "CACHE_BYTES",
    "DEFAULT_MARGIN_BYTES",
    "DEFAULT_OVERHEAD_BYTES",
    "DeviceMemory",
    "Estimate",
    "UnknownCacheError",
    "Verdict",
    "device_memory",
    "estimate",
    "gibibytes",
    "verdict",
]
