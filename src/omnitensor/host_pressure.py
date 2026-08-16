"""How loaded the host is, and whether it can take more work.

This is what replaces the per-worker ceilings. Those guessed, one worker at a
time, what the machine could hold — and guessed wrong in both directions: a
512 MiB memory budget killed a Qwen worker that legitimately held 831 MB, while
nothing at all stopped four such workers starting together on a busy laptop.

The question a person actually cares about is not "is this worker big?" but "is
my machine still usable?", and that is one measurement of the whole host:

* **Memory including swap.** Thrashing here is memory-driven — this service has
  peaked at 12 GB — and the number that predicts a stall is how much is
  *available*, not how much is free: reclaimable cache is not pressure.
* **Swap in use.** A machine already swapping is one where the next model load
  is paid for in disk, so swap counts against the same budget rather than being
  treated as more room.
* **CPU as a secondary signal.** Load average per core says the machine is
  busy, which is often a job working exactly as intended. It can stop the pool
  widening; it does not hold work on its own.

Nothing here refuses a job. It answers a question, and the admission queue
decides — the whole point being that work waits for room instead of being
turned away or thrashing the machine to run now.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# The share of memory (including swap) and of CPU at which the host is
# considered full. Both configurable; both default to the same 80% because
# that is the number an operator can reason about without reading this file.
DEFAULT_MEMORY_PERCENT = 80.0
DEFAULT_CPU_PERCENT = 80.0
MEMORY_PERCENT_VARIABLE = "OMNITENSOR_MEMORY_PRESSURE_PERCENT"
CPU_PERCENT_VARIABLE = "OMNITENSOR_CPU_PRESSURE_PERCENT"
# Below 10% the gate would hold work permanently on an ordinary desktop; above
# 99% it is not a gate. Neither bound is a policy, only a guard against a
# configuration that can never be satisfied.
MIN_PERCENT = 10.0
MAX_PERCENT = 99.0


@dataclass(frozen=True, slots=True)
class HostPressure:
    """One reading of how full the host is."""

    memory_percent: float
    cpu_percent: float
    memory_limit: float
    cpu_limit: float

    @property
    def memory_saturated(self) -> bool:
        return self.memory_percent >= self.memory_limit

    @property
    def cpu_saturated(self) -> bool:
        return self.cpu_percent >= self.cpu_limit

    @property
    def saturated(self) -> bool:
        """Whether the host is too full to start more work.

        Memory decides. CPU at the limit is a machine doing what it was asked
        to do, and holding work for it would stall the runtime exactly when it
        is being useful.
        """
        return self.memory_saturated

    def detail(self) -> str:
        return (
            f"memory {self.memory_percent:.0f}% of {self.memory_limit:.0f}%, "
            f"cpu {self.cpu_percent:.0f}% of {self.cpu_limit:.0f}%"
        )


def bounded_percent(value: object, fallback: float) -> float:
    """A configured threshold, or the default when it cannot be one.

    An unreadable setting is the default rather than a startup failure: a typo
    in a tuning knob must not take the runtime down, and the value it would
    take down is the one protecting the machine.
    """
    try:
        percent = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    if percent != percent or percent < MIN_PERCENT or percent > MAX_PERCENT:
        return fallback
    return percent


def configured_limits(environ=None) -> tuple[float, float]:
    """The memory and CPU thresholds this process runs with."""
    source = os.environ if environ is None else environ
    return (
        bounded_percent(source.get(MEMORY_PERCENT_VARIABLE), DEFAULT_MEMORY_PERCENT),
        bounded_percent(source.get(CPU_PERCENT_VARIABLE), DEFAULT_CPU_PERCENT),
    )


def _meminfo(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        name, _, rest = line.partition(":")
        fields = rest.split()
        if fields and fields[0].isdigit():
            values[name] = int(fields[0]) * 1024
    return values


def memory_percent(path: Path = Path("/proc/meminfo")) -> float:
    """Memory in use as a share of memory plus swap, or 0 when unreadable.

    ``MemAvailable`` rather than ``MemFree``: the kernel's own estimate of what
    a new allocation can have, which counts reclaimable cache as available. Free
    memory alone reads as 90% used on any machine that has been up a while, and
    a gate built on it would hold work forever.
    """
    try:
        values = _meminfo(path)
    except (OSError, ValueError):
        return 0.0
    total = values.get("MemTotal", 0) + values.get("SwapTotal", 0)
    if total <= 0:
        return 0.0
    available = values.get("MemAvailable", values.get("MemFree", 0)) + values.get("SwapFree", 0)
    return max(0.0, min(100.0, 100.0 * (total - available) / total))


def cpu_percent(path: Path = Path("/proc/loadavg"), cores: int | None = None) -> float:
    """One-minute load average as a share of the cores available.

    Load average rather than an instantaneous sample: a runqueue that has been
    long for a minute is a busy machine, while one instant of 100% is a
    process starting.
    """
    count = cores if cores is not None else (os.cpu_count() or 1)
    try:
        first = Path(path).read_text(encoding="ascii").split()[0]
        load = float(first)
    except (OSError, ValueError, IndexError):
        return 0.0
    if count <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * load / count))


def read_pressure(
    *,
    environ=None,
    meminfo: Path = Path("/proc/meminfo"),
    loadavg: Path = Path("/proc/loadavg"),
    cores: int | None = None,
) -> HostPressure:
    """One reading, against the thresholds this process was configured with.

    An unreadable ``/proc`` reads as no pressure. The alternative — treating a
    failed read as a full machine — would stop every job on a host whose
    ``/proc`` is not what this expects, which is a worse failure than not
    throttling on one that is.
    """
    memory_limit, cpu_limit = configured_limits(environ)
    return HostPressure(
        memory_percent=memory_percent(meminfo),
        cpu_percent=cpu_percent(loadavg, cores),
        memory_limit=memory_limit,
        cpu_limit=cpu_limit,
    )
