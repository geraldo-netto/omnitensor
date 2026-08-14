"""Live session-bus client for trusted local forecasts."""

from __future__ import annotations

from .forecast_contracts import ForecastRunError

BUS_NAME = "org.cinnamon.OmniTensor1"
OBJECT_PATH = "/org/cinnamon/OmniTensor1"
_LEGACY_MODULE = "omnitensor.training.runner"


class DbusForecastClient:
    """Thin async adapter over the local session-bus interface."""

    def __init__(self, bus, interface) -> None:
        self._bus = bus
        self._interface = interface

    @classmethod
    async def connect(cls) -> DbusForecastClient:  # pragma: no cover - needs live bus
        from dbus_fast import BusType  # noqa: PLC0415
        from dbus_fast.aio import MessageBus  # noqa: PLC0415

        try:
            bus = await MessageBus(bus_type=BusType.SESSION).connect()
            introspection = await bus.introspect(BUS_NAME, OBJECT_PATH)
            proxy = bus.get_proxy_object(BUS_NAME, OBJECT_PATH, introspection)
            return cls(bus, proxy.get_interface(BUS_NAME))
        except Exception as error:
            if "bus" in locals():
                bus.disconnect()
            raise ForecastRunError("runtime-unavailable", str(error)) from error

    def close(self) -> None:
        self._bus.disconnect()

    async def describe_contract(self) -> str:
        try:
            return await self._interface.call_describe_contract()
        except Exception as error:
            raise ForecastRunError("runtime-unavailable", str(error)) from error

    async def submit_job(self, request: str) -> str:
        try:
            return await self._interface.call_submit_job(request)
        except Exception as error:
            raise ForecastRunError("runtime-unavailable", str(error)) from error

    async def get_job_result(self, request: str) -> str:
        try:
            return await self._interface.call_get_job_result(request)
        except Exception as error:
            raise ForecastRunError("runtime-unavailable", str(error)) from error


DbusForecastClient.__module__ = _LEGACY_MODULE
