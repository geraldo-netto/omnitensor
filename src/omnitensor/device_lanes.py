"""Which executor a profile runs on, once a person has chosen a device.

A backend name — "gpu" — is not a lane. Two cards are two lanes, and a profile
pinned to the second one must queue behind that card's work rather than behind
whatever "gpu" happens to mean. A :class:`~omnitensor.ports.DeviceAwareExecutors`
collection knows how to answer that; an executor set assembled by an older build
(or by a test) is a plain mapping with no device knowledge, and must keep
working.

What a plain mapping may fall back to differs per question. A queue name is
free to be the backend, because "gpu" is a real queue. A *device identity* is
not: answering "gpu" to "which card is this?" hands a backend name to a caller
that will compare it against physical ids, and an executor looked up by backend
answers for a device nobody asked about. Those two say ``None`` instead.
"""

from __future__ import annotations

from collections.abc import Callable

from .ports import DeviceAwareExecutors


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

    def _device_aware(self) -> DeviceAwareExecutors | None:
        executors = self._executors_of()
        return executors if isinstance(executors, DeviceAwareExecutors) else None

    def executors_for(self, profile_id: str) -> dict:
        """The executors this profile may use, narrowed to its chosen device."""
        executors = self._device_aware()
        if executors is None:
            return dict(self._executors_of())
        return executors.for_device(self._choice_of(profile_id))

    def lane(self, profile_id: str, backend: str) -> str:
        """The scheduler queue this profile's work belongs in."""
        executors = self._device_aware()
        if executors is None:
            return backend
        return executors.lane_key(backend, self._choice_of(profile_id))

    def identity(self, profile_id: str, backend: str) -> str | None:
        """The stable device id behind that lane, or None when there is none.

        A collection that cannot tell two cards apart has no device id to give;
        the backend name is not one, so the honest answer is ``None``.
        """
        executors = self._device_aware()
        if executors is None:
            return None
        return executors.device_id(backend, self._choice_of(profile_id))

    def executor_for(self, backend: str, device_id: str):
        """The executor owning exactly that device, or None when it is gone.

        Never the backend's executor as a stand-in: the caller asked about one
        physical device, and an executor that may belong to another card is a
        wrong answer, not a lenient one.
        """
        executors = self._device_aware()
        if executors is None:
            return None
        return executors.executor_for_device(backend, device_id)


__all__ = ["DeviceLanes"]
