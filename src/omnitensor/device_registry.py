"""What accelerators are present, as an object rather than a service field.

The detected list is read by four different collaborators — the executors,
the scheduler lane, the plugin runtime's device mounts, and the control
service's answer about which GPUs may be chosen — and it changes while the
runtime runs, when the kernel reports a card arriving or leaving. Held as a
field on the service, "what cards are here" could only be asked of the
service, which is what kept the plugin runtime and the control service
un-buildable anywhere else.
"""

from __future__ import annotations

from collections.abc import Sequence


class DeviceRegistry:
    """The detected devices, replaced whole when discovery says so."""

    __slots__ = ("_devices",)

    def __init__(self, devices: Sequence[object] = ()) -> None:
        self._devices = list(devices)

    @property
    def devices(self) -> list[object]:
        """The live list.

        Returned rather than copied: `build_executors` compares it against the
        previous one by identity of contents, and the callers that hold it
        expect the same object the service publishes from.
        """
        return self._devices

    def replace(self, devices: Sequence[object]) -> bool:
        """Adopt a newly detected list; True when it differs from the last."""
        changed = list(devices) != self._devices
        self._devices = list(devices)
        return changed

    def gpu_ids(self) -> tuple[str, ...]:
        return tuple(device.id for device in self._devices if device.backend == "gpu")

    def __bool__(self) -> bool:
        return bool(self._devices)


__all__ = ["DeviceRegistry"]
