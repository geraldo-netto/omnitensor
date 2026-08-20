"""A scheduler's figures, frozen on the event loop for an off-loop reader."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SchedulerObservation:
    """A scheduler's published figures, frozen on the event loop.

    Snapshot building runs on a worker thread (`run_off_loop`), and reading the
    live scheduler from there walked `profiles` and `_running` while the loop
    deleted drained profiles and finished jobs, raising `RuntimeError` past a
    handler that catches only `OSError` and `ValueError` — one unlucky
    interleaving tore the daemon down mid-job. `tick()` also folds busy time
    into the load EMA and resets it, which is a mutation a function named
    "build a snapshot" should not perform, and which raced the running job
    still adding to it. Both now happen on the loop; the thread only reads.
    """

    _stats: dict
    _profile_stats: dict
    _degraded: dict

    @classmethod
    def of(cls, scheduler) -> SchedulerObservation:
        scheduler.tick()
        return cls(scheduler.stats(), scheduler.profile_stats(), scheduler.degraded_backends())

    def stats(self) -> dict:
        return self._stats

    def profile_stats(self) -> dict:
        return self._profile_stats

    def degraded_backends(self) -> dict:
        """Backends that failed outside job execution, frozen with the rest."""
        return self._degraded
