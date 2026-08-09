"""Executor contract shared by every backend.

An executor either reports itself unavailable with a concrete reason
(missing runtime library, missing execution provider, missing hardware) or
executes a compatible model.  Runtime modules are injected so the complete
execution path is unit-testable without accelerator hardware; production
wiring injects the real libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Availability:
    available: bool
    reason: str = ""


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
