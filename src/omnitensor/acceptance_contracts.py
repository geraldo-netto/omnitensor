"""Stable value and probe contracts for installation acceptance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Check:
    """One named verification outcome with the evidence behind it."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class InstallationReport:
    checks: tuple[Check, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(check for check in self.checks if not check.ok)

    def render(self) -> str:
        return "\n".join(
            f"{'PASS' if check.ok else 'FAIL'}  {check.name}: {check.detail}"
            for check in self.checks
        )


@runtime_checkable
class ServiceProbe(Protocol):
    """Whether the service unit is installed and currently running."""

    def unit_state(self) -> tuple[bool, str]: ...


@runtime_checkable
class ControlProbe(Protocol):
    """Whether the D-Bus surface answers with a contract-valid reply."""

    def apply_command(self, text: str) -> str: ...
