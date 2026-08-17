"""Which executor a profile runs on, once a person has chosen a device.

A backend name — "gpu" — is not a lane. Two cards are two lanes, and a profile
pinned to the second one must queue behind that card's work rather than behind
whatever "gpu" happens to mean. The executor collection knows how to answer
that; every method here asks it if it can and falls back to the backend name if
it cannot, because an executor set assembled by an older build (or by a test)
is a plain mapping with no device knowledge and must keep working.
"""

from __future__ import annotations

from collections.abc import Callable


class DeviceLanes:
    """Resolve a profile's device choice into executors, lanes, and identities."""

    __slots__ = ("_choice_of", "_executors_of")

    def __init__(
        self,
        executors_of: Callable[[], object],
        choice_of: Callable[[str | None], str | None],
    ) -> None:
        self._executors_of = executors_of
        self._choice_of = choice_of

    def executors_for(self, profile_id: str) -> dict:
        """The executors this profile may use, narrowed to its chosen device."""
        executors = self._executors_of()
        selector = getattr(executors, "for_device", None)
        return selector(self._choice_of(profile_id)) if callable(selector) else dict(executors)

    def lane(self, profile_id: str, backend: str) -> str:
        """The scheduler queue this profile's work belongs in."""
        executors = self._executors_of()
        selector = getattr(executors, "lane_key", None)
        return selector(backend, self._choice_of(profile_id)) if callable(selector) else backend

    def identity(self, profile_id: str, backend: str) -> str | None:
        """The stable device id behind that lane, or None when there is none."""
        executors = self._executors_of()
        selector = getattr(executors, "device_id", None)
        return selector(backend, self._choice_of(profile_id)) if callable(selector) else backend

    def executor_for(self, backend: str, device_id: str):
        """The executor owning exactly that device, or None when it is gone."""
        executors = self._executors_of()
        selector = getattr(executors, "executor_for_device", None)
        return selector(backend, device_id) if callable(selector) else executors.get(backend)


__all__ = ["DeviceLanes"]
