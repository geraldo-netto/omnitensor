"""Which Vulkan device is which, before anything is loaded onto one.

`GGML_VK_VISIBLE_DEVICES` takes indices into the *physical* device list and
llama.cpp then renumbers what survives from zero, so "device 1" means nothing
on its own — on this desk index 1 is the discrete RX 6600 XT and index 0 is the
integrated 610M, the opposite of the order sysfs reports the render nodes in.
A run of the benchmark on 2026-08-17 asked for index 1 believing it meant the
integrated card, measured the discrete one for twenty minutes, and labelled the
results "integrated". Nothing in the code could have caught that, because
nothing compared what was asked for with what answered.

So this module names devices instead of numbering them. It also explains the
other half of that morning: with no variable set, llama.cpp lists *one* device
here — the discrete card — and the integrated one is not reachable by
`main_gpu` at all. The variable is the only lever, and it is honoured.

Enumeration costs a subprocess per index because the Vulkan backend initialises
once per process and reads the variable at that moment; a second look inside
the same process would see the first look's answer.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass

# `ggml_vulkan: 0 = AMD Radeon 610M (RADV RAPHAEL_MENDOCINO) (radv) | uma: 1 | ...`
_LISTED = re.compile(r"^ggml_vulkan:\s+\d+\s+=\s+(.+?)\s+\|\s+uma:\s+([01])", re.MULTILINE)
_INVALID = "Invalid device index"
# ` (radv)`, ` (llvmpipe)` — the driver, appended only by the enumeration line.
_DRIVER_TAG = re.compile(r"\s*\([a-z][a-z0-9_-]*\)\s*$")

# Nobody has this many cards; the ceiling is here so a driver that answers
# strangely cannot spin forever.
MAX_PHYSICAL_DEVICES = 16

_PROBE = """
import os, sys
from llama_cpp import llama_cpp
lines = []

@llama_cpp.llama_log_callback
def note(level, text, data):
    lines.append(text.decode("utf-8", errors="replace"))

llama_cpp.llama_log_set(note, None)
llama_cpp.llama_backend_init()
sys.stdout.write("".join(lines))
"""


class DeviceError(RuntimeError):
    """A device was asked for that this machine cannot offer."""


@dataclass(frozen=True, slots=True)
class VulkanDevice:
    """One physical device, at the index that selects it."""

    index: int
    name: str
    integrated: bool

    def __str__(self) -> str:
        return f"{self.index}: {self.name}{' (integrated)' if self.integrated else ''}"


def _named(listed: str) -> str:
    """The device name without the driver tag the enumeration line appends.

    Enumeration says `AMD Radeon RX 6600 XT (RADV NAVI23) (radv)`; the load
    says `AMD Radeon RX 6600 XT (RADV NAVI23)`. Same card, and the two must
    compare equal or the confirmation below would refuse every load. The card's
    own bracket stays — it is upper case and part of the name.
    """
    return _DRIVER_TAG.sub("", listed).strip()


def _probe(index: int, python: str) -> tuple[str, bool] | None:
    completed = subprocess.run(  # noqa: S603 - fixed argv, our own interpreter
        [python, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={
            "GGML_VK_VISIBLE_DEVICES": str(index),
            "PATH": "/usr/bin:/bin",
            "HOME": "/nonexistent",
        },
    )
    output = completed.stdout + completed.stderr
    if _INVALID in output:
        return None
    listed = _LISTED.search(output)
    if listed is None:
        return None
    return _named(listed.group(1)), listed.group(2) == "1"


def devices(python: str | None = None) -> tuple[VulkanDevice, ...]:
    """Every physical Vulkan device, in the order the variable indexes them."""
    interpreter = python or sys.executable
    found: list[VulkanDevice] = []
    for index in range(MAX_PHYSICAL_DEVICES):
        answer = _probe(index, interpreter)
        if answer is None:
            break
        name, integrated = answer
        found.append(VulkanDevice(index, name, integrated))
    return tuple(found)


def select(wanted: str, available: tuple[VulkanDevice, ...]) -> VulkanDevice:
    """The one device the name asks for — never a guess between two.

    A bare number still works, because a person reading llama.cpp's own output
    sees numbers; it is checked against the list rather than trusted.
    """
    if not available:
        raise DeviceError("no Vulkan device answered; is a driver installed?")
    if wanted.isdecimal():
        index = int(wanted)
        for device in available:
            if device.index == index:
                return device
        raise DeviceError(
            f"no Vulkan device at index {index}; this machine has "
            + ", ".join(str(device) for device in available)
        )
    folded = wanted.casefold()
    matches = [device for device in available if folded in device.name.casefold()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise DeviceError(
            f"no Vulkan device matches {wanted!r}; this machine has "
            + ", ".join(str(device) for device in available)
        )
    raise DeviceError(
        f"{wanted!r} matches more than one device: " + ", ".join(str(device) for device in matches)
    )


def confirm(device: VulkanDevice, loaded_name: str) -> None:
    """Refuse a load that landed on a card nobody asked for."""
    if loaded_name.strip().casefold() != device.name.strip().casefold():
        raise DeviceError(
            f"asked for {device.name!r} but the model loaded on {loaded_name!r}; "
            "the measurement would describe the wrong card"
        )


__all__ = [
    "MAX_PHYSICAL_DEVICES",
    "DeviceError",
    "VulkanDevice",
    "confirm",
    "devices",
    "select",
]
