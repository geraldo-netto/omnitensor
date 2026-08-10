"""Executor contract shared by every backend.

An executor either reports itself unavailable with a concrete reason
(missing runtime library, missing execution provider, missing hardware) or
executes a compatible model.  Runtime modules are injected so the complete
execution path is unit-testable without accelerator hardware; production
wiring injects the real libraries.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Machine-readable counterparts to the sentences below, so a consumer can
# choose a remedy without matching on English prose. The sentence stays: it
# says which device or package, which the code deliberately does not.
DEVICE_ABSENT = "device-absent"
RUNTIME_MISSING = "runtime-missing"
RUNTIME_UNUSABLE = "runtime-unusable"
FORMAT_UNSUPPORTED = "format-unsupported"


@dataclass(frozen=True)
class Availability:
    available: bool
    reason: str = ""
    code: str = ""


@dataclass(frozen=True)
class InferenceResult:
    outputs: list
    duration_ms: float


class Executor(Protocol):
    backend: str
    model_formats: frozenset[str]

    def availability(self) -> Availability:
        """Report whether this executor can run right now, with a reason."""

    def run(self, model_path: str, inputs: list) -> InferenceResult:
        """Execute one inference; raises RuntimeError when unavailable."""


@runtime_checkable
class ModelAwareAvailability(Protocol):
    """Optional executor capability for format-specific runtime health."""

    def availability_for(self, model: dict | None) -> Availability:
        """Availability of the lane that can execute ``model``."""


# Ordered by what a user can actually do about it: install a package, repair
# an installed one, supply a different artifact, obtain hardware. A profile
# blocked on several backends is reported by the most actionable of them,
# because that is the only one worth offering as a remedy.
_ACTIONABILITY = (RUNTIME_MISSING, RUNTIME_UNUSABLE, FORMAT_UNSUPPORTED, DEVICE_ABSENT)


def most_actionable(codes: Iterable[str]) -> str:
    """The code a user has the best chance of resolving, or ``""``."""
    present = {code for code in codes if code}
    for code in _ACTIONABILITY:
        if code in present:
            return code
    return min(present, default="")


def require_available(executor: Executor) -> None:
    availability = executor.availability()
    if not availability.available:
        raise RuntimeError(f"{executor.backend} executor unavailable: {availability.reason}")


def supports_model(executor: Executor, model: dict | None) -> bool:
    """A workload without a model spec is compatible with any backend."""
    return model is None or model.get("format") in executor.model_formats


def availability_for_model(executor: Executor, model: dict | None) -> Availability:
    """Use format-aware health when the executor exposes it."""
    if isinstance(executor, ModelAwareAvailability):
        return executor.availability_for(model)
    return executor.availability()


DEFAULT_MAX_CACHED_MODELS = 4


class ModelCache:
    """Bounded, revision-aware cache of expensive per-model runtime state.

    Compiled models and interpreters are costly to build, so caching them is
    right; caching them forever is not.  Unbounded, the cache grows with every
    distinct model a long-running service is ever asked for.  Keyed on the path
    alone, it keeps serving the old model after an artifact is replaced in
    place — silently returning results from a model the operator has retired.

    Entries are therefore bounded by count with least-recently-used eviction,
    and revalidated against the file's modification time and size on every
    lookup.  Callers hold their own lock; this class does no locking.
    """

    def __init__(self, max_entries: int = DEFAULT_MAX_CACHED_MODELS) -> None:
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = max_entries
        self._entries: OrderedDict[str, tuple[tuple[int, int], object]] = OrderedDict()

    def get_or_build(self, model_path: str, build: Callable[[], object]) -> object:
        revision = _model_revision(model_path)
        cached = self._entries.get(model_path)
        if revision is not None and cached is not None and cached[0] == revision:
            self._entries.move_to_end(model_path)
            return cached[1]
        value = build()
        if revision is None:
            # A model whose file cannot be stat'd cannot be revalidated later,
            # so it is never cached: serving it again could mean serving a
            # model that has since been replaced or withdrawn.
            self._entries.pop(model_path, None)
            return value
        self._entries[model_path] = (revision, value)
        self._entries.move_to_end(model_path)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
        return value

    def __len__(self) -> int:
        return len(self._entries)

    def paths(self) -> tuple[str, ...]:
        """Cached model paths, least recently used first."""
        return tuple(self._entries)


def _model_revision(model_path: str) -> tuple[int, int] | None:
    """Identify the model file on disk, or ``None`` when it cannot be read."""
    try:
        status = os.stat(model_path)
    except OSError:
        return None
    return status.st_mtime_ns, status.st_size
