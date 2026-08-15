"""Live control-socket client for trusted local forecasts."""

from __future__ import annotations

import json

from .forecast_contracts import ForecastRunError

_LEGACY_MODULE = "omnitensor.training.runner"


class SocketForecastClient:
    """Thin async adapter over the local control socket.

    Each call is one request over one connection — the runner's traffic is a
    handshake, a submission, and a poll loop measured in seconds, so holding a
    connection open would preserve nothing but failure modes.  Every transport
    failure surfaces as ``runtime-unavailable`` because that is the only
    diagnosis the forecast runner can act on.
    """

    def __init__(self, caller=None) -> None:
        self._caller = caller or self._live_call

    @classmethod
    async def connect(cls) -> SocketForecastClient:
        # Nothing to connect eagerly; the method exists so the CLI seam stays
        # "make a client, use it, close it" whatever the transport is.
        return cls()

    def close(self) -> None:
        return None

    @staticmethod
    async def _live_call(method: str, params: dict) -> dict:  # pragma: no cover - live wiring
        from ..socket_transport import call_control  # noqa: PLC0415

        return await call_control(method, params)

    async def _call(self, method: str, params: dict) -> str:
        try:
            result = await self._caller(method, params)
        except Exception as error:
            raise ForecastRunError("runtime-unavailable", str(error)) from error
        return json.dumps(result, separators=(",", ":"))

    async def describe_contract(self) -> str:
        return await self._call("describe-contract", {})

    async def submit_job(self, request: str) -> str:
        return await self._call("submit-job", json.loads(request))

    async def get_job_result(self, request: str) -> str:
        return await self._call("get-job-result", json.loads(request))


SocketForecastClient.__module__ = _LEGACY_MODULE
