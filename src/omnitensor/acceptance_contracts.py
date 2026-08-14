"""Stable value and probe contracts for installation acceptance."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

_LEGACY_MODULE = "omnitensor.acceptance"


def legacy_acceptance_value(name: str, fallback):
    """Resolve a facade override without importing the facade from an owner."""
    facade = sys.modules.get(_LEGACY_MODULE)
    return getattr(facade, name, fallback) if facade is not None else fallback


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
class BusProbe(Protocol):
    """Whether the D-Bus surface answers with a contract-valid reply."""

    def apply_command(self, text: str) -> str: ...


for _legacy_type in (Check, InstallationReport, ServiceProbe, BusProbe):
    _legacy_type.__module__ = _LEGACY_MODULE
