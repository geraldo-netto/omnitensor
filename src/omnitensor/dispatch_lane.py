"""Immutable accelerator decision carried from resolution into dispatch."""

from __future__ import annotations

from dataclasses import dataclass

from .discovery import BACKENDS
from .plugins.artifacts import ArtifactReference, artifact_reference_error


@dataclass(frozen=True, slots=True)
class PreparedDispatchLane:
    """Exact model and physical lane selected for one pipeline job."""

    backend: str
    device_id: str
    model_reference: ArtifactReference

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"prepared lane backend must be one of {', '.join(BACKENDS)}")
        if (
            not isinstance(self.device_id, str)
            or not self.device_id
            or len(self.device_id) > 120
        ):
            raise ValueError("prepared lane device_id must contain 1-120 characters")
        if not isinstance(self.model_reference, ArtifactReference):
            raise TypeError("prepared lane model_reference must be an ArtifactReference")
        error = artifact_reference_error(self.model_reference)
        if error:
            raise ValueError(f"prepared lane model reference is invalid: {error}")

    @property
    def scheduler_lane(self) -> str:
        """Queue identity derived from, never independent of, the physical lane."""
        return self.device_id if self.backend == "gpu" else self.backend


__all__ = ["PreparedDispatchLane"]
