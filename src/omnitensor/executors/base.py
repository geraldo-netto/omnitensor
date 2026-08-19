"""Executor contract shared by every backend.

An executor either reports itself unavailable with a concrete reason
(missing runtime library, missing execution provider, missing hardware) or
executes a compatible model.  Runtime modules are injected so the complete
execution path is unit-testable without accelerator hardware; production
wiring injects the real libraries.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

LOGGER = logging.getLogger(__name__)

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


@runtime_checkable
class FormatAwareExecutor(Protocol):
    """Optional executor capability for routing by a declared model format."""

    def run_for_format(
        self,
        model_format: str,
        model_path: str,
        inputs: list,
    ) -> InferenceResult:
        """Execute ``model_path`` using the lane for ``model_format``."""


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


def run_executor(
    executor: Executor,
    model_path: str,
    inputs: list,
    *,
    model_format: str | None = None,
) -> InferenceResult:
    """Run through an optional declared-format capability when available."""
    if model_format is not None and isinstance(executor, FormatAwareExecutor):
        return executor.run_for_format(model_format, model_path, inputs)
    return executor.run(model_path, inputs)


DEFAULT_MAX_CACHED_MODELS = 4


class ModelCache:
    """Bounded, revision-aware cache of expensive per-model runtime state.

    Compiled models and interpreters are costly to build, so caching them is
    right; caching them forever is not.  Unbounded, the cache grows with every
    distinct model a long-running service is ever asked for.  Keyed on the path
    alone, it keeps serving the old model after an artifact is replaced in
    place — silently returning results from a model the operator has retired.

    Entries are therefore bounded by count with least-recently-used eviction,
    and revalidated against the model and any companion files' modification
    times and sizes on every lookup. Callers hold their own lock; this class
    does no locking.

    Evicting an entry hands it to ``release`` rather than merely dropping the
    reference: what these entries hold — an ONNX Runtime session, an OpenVINO
    compiled model, an ncnn ``Net`` holding Vulkan device memory — is released
    by the garbage collector at a time and in an order nobody chose otherwise.
    ``release`` is not called from here for an entry a caller may still be
    using; deciding that is :class:`ModelStore`'s job.
    """

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_CACHED_MODELS,
        *,
        release: Callable[[object], None] | None = None,
    ) -> None:
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = max_entries
        self._release = release
        self._entries: OrderedDict[str, tuple[tuple[tuple[int, int], ...], object]] = OrderedDict()

    def get_or_build(
        self,
        model_path: str,
        build: Callable[[], object],
        *,
        companion_paths: Iterable[str] = (),
    ) -> object:
        revision = _model_revision(model_path, companion_paths)
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
        replaced = self._entries.get(model_path)
        self._entries[model_path] = (revision, value)
        self._entries.move_to_end(model_path)
        if replaced is not None:
            self._releasing(replaced[1])
        while len(self._entries) > self._max_entries:
            self._releasing(self._entries.popitem(last=False)[1][1])
        return value

    def clear(self) -> None:
        """Release every entry and empty the cache."""
        entries, self._entries = self._entries, OrderedDict()
        for _revision, value in entries.values():
            self._releasing(value)

    def _releasing(self, value: object) -> None:
        if self._release is not None:
            self._release(value)

    def __len__(self) -> int:
        return len(self._entries)

    def paths(self) -> tuple[str, ...]:
        """Cached model paths, least recently used first."""
        return tuple(self._entries)


# The teardown method each runtime happens to spell differently: ONNX Runtime
# and tflite hand the object back to the allocator on ``close``, OpenVINO on
# ``release_memory``, ncnn on ``clear``.  Executors inject their own releaser
# where they need an order; this is the default for a plain cached object.
_RELEASE_METHODS = ("close", "release_memory", "clear")


def release_runtime_value(value: object) -> None:
    """Hand one cached runtime object back to its library, if it can be."""
    for name in _RELEASE_METHODS:
        method = getattr(value, name, None)
        if callable(method):
            try:
                method()
            except Exception:  # noqa: BLE001 - native teardown fails arbitrarily
                LOGGER.debug("releasing %s via %s() failed", type(value).__name__, name)
            return


def close_executor(executor: object) -> None:
    """Close an executor that is being discarded, if it knows how."""
    close = getattr(executor, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:  # noqa: BLE001 - backend teardown fails arbitrarily
        LOGGER.exception("Could not close the %s executor", getattr(executor, "backend", "?"))


class ModelStore:
    """A :class:`ModelCache`, the lock guarding it, and its teardown order.

    Releasing a cached runtime object is not simply "do it on eviction".
    Another thread can be mid-inference on the exact object being evicted, and
    tearing an ncnn ``Net``'s allocators down under a running extractor is a
    fault this package has already had once — it surfaces as freed device
    memory handed to a later job, not as a crash.  So a release requested
    while a job is in flight is parked, and performed by the last job to
    leave; a release requested while the store is idle happens at once.

    ``close`` therefore never blocks on a running job, which matters because
    the executors are discarded on the event loop thread whenever discovery
    rebuilds them.
    """

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_CACHED_MODELS,
        *,
        release: Callable[[object], None] = release_runtime_value,
    ) -> None:
        self._release = release
        self._cache = ModelCache(max_entries, release=self._releasing)
        self._cache_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._in_flight = 0
        self._pending: list[object] = []

    def get_or_build(
        self,
        model_path: str,
        build: Callable[[], object],
        *,
        companion_paths: Iterable[str] = (),
    ) -> object:
        with self._cache_lock:
            return self._cache.get_or_build(model_path, build, companion_paths=companion_paths)

    @contextmanager
    def running(self) -> Iterator[None]:
        """Mark one job in flight, so nothing is released underneath it."""
        with self._state_lock:
            self._in_flight += 1
        try:
            yield
        finally:
            with self._state_lock:
                self._in_flight -= 1
                parked = self._pending if self._in_flight == 0 else []
                if parked:
                    self._pending = []
            for value in parked:
                self._release(value)

    def close(self) -> None:
        """Release everything held; safe to call more than once."""
        with self._cache_lock:
            self._cache.clear()

    def _releasing(self, value: object) -> None:
        with self._state_lock:
            if self._in_flight:
                self._pending.append(value)
                return
        self._release(value)

    def __len__(self) -> int:
        return len(self._cache)

    def paths(self) -> tuple[str, ...]:
        return self._cache.paths()


def _model_revision(
    model_path: str,
    companion_paths: Iterable[str] = (),
) -> tuple[tuple[int, int], ...] | None:
    """Identify every file needed by a model, or ``None`` if one is unreadable."""
    revision = []
    for path in (model_path, *companion_paths):
        try:
            status = os.stat(path)
        except OSError:
            return None
        revision.append((status.st_mtime_ns, status.st_size))
    return tuple(revision)
