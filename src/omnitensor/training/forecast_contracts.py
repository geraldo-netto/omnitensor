"""Stable forecast runner contracts shared by client and orchestration."""

from __future__ import annotations

from typing import Protocol

from ..stable_error import StableError

_LEGACY_MODULE = "omnitensor.training.runner"


class ForecastRunError(StableError, ValueError):
    """Stable refusal from the trusted forecast path."""


class ForecastClient(Protocol):
    async def describe_contract(self) -> str: ...

    async def submit_job(self, request: str) -> str: ...

    async def get_job_result(self, request: str) -> str: ...

    def close(self) -> None: ...


for _legacy_type in (ForecastRunError, ForecastClient):
    _legacy_type.__module__ = _LEGACY_MODULE
