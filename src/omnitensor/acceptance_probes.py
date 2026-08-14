"""Live service, D-Bus, and accelerator probes for installation acceptance."""

from __future__ import annotations

from collections.abc import Sequence

from .acceptance_contracts import legacy_acceptance_value


def _executor_for(module: str):
    """The executor that would actually run this runtime."""
    from .executors.gpu import GpuExecutor  # noqa: PLC0415 - probe-only imports
    from .executors.npu import NpuExecutor  # noqa: PLC0415
    from .executors.tpu import TpuExecutor  # noqa: PLC0415
    from .executors.vulkan import VulkanGpuExecutor  # noqa: PLC0415

    builders = {
        "ncnn": VulkanGpuExecutor,
        "onnxruntime": GpuExecutor,
        "openvino": NpuExecutor,
        "tflite_runtime": TpuExecutor,
    }
    builder = builders.get(module)
    return None if builder is None else builder(True)


def _runtime_verdict(module: str) -> str | None:
    """Why this runtime cannot be used, or ``None`` when it can."""
    from .executors.base import DEVICE_ABSENT  # noqa: PLC0415 - probe-only import

    executor_for = legacy_acceptance_value("_executor_for", _executor_for)
    executor = executor_for(module)
    if executor is None:
        return None
    try:
        availability = executor.availability()
    except Exception as error:  # noqa: BLE001 - a probe must not fail the report
        return type(error).__name__
    if availability.available or availability.code == DEVICE_ABSENT:
        return None
    return availability.reason


class SystemdUserServiceProbe:
    """:class:`ServiceProbe` over ``systemctl --user``."""

    def __init__(self, unit: str = "omnitensor.service", *, runner=None):
        self._unit = unit
        self._runner = runner or legacy_acceptance_value("_run_command", _run_command)

    def unit_state(self) -> tuple[bool, str]:
        code, output = self._runner(["systemctl", "--user", "is-active", self._unit])
        state = output.strip() or "unknown"
        return code == 0 and state == "active", f"{self._unit} is {state}"


class DbusApplyCommandProbe:
    """:class:`BusProbe` over the session bus."""

    def __init__(
        self,
        *,
        bus_name: str = "org.cinnamon.OmniTensor1",
        timeout_s: float = 30.0,
        retry_interval_s: float = 0.25,
        caller=None,
    ):
        if timeout_s <= 0 or retry_interval_s < 0:
            raise ValueError("D-Bus probe timing must be positive")
        self._bus_name = bus_name
        self._timeout_s = timeout_s
        self._retry_interval_s = retry_interval_s
        self._caller = caller

    def apply_command(self, text: str) -> str:
        import asyncio  # noqa: PLC0415

        object_path = "/" + self._bus_name.replace(".", "/")

        async def call_once() -> str:  # pragma: no cover - live adapter wiring
            if self._caller is not None:
                return await self._caller(text)
            from dbus_fast import BusType  # noqa: PLC0415
            from dbus_fast.aio import MessageBus  # noqa: PLC0415

            bus = await MessageBus(bus_type=BusType.SESSION).connect()
            try:
                introspection = await bus.introspect(self._bus_name, object_path)
                proxy = bus.get_proxy_object(self._bus_name, object_path, introspection)
                interface = proxy.get_interface(self._bus_name)
                return await interface.call_apply_command(text)
            finally:
                bus.disconnect()

        async def call() -> str:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._timeout_s
            while True:
                remaining = deadline - loop.time()
                try:
                    return await asyncio.wait_for(call_once(), timeout=max(remaining, 0.001))
                except Exception:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise
                    await asyncio.sleep(min(self._retry_interval_s, remaining))

        return asyncio.run(call())


def _run_command(argv: Sequence[str]) -> tuple[int, str]:  # pragma: no cover - thin shim
    import subprocess  # noqa: PLC0415

    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    return completed.returncode, completed.stdout


for _legacy_function in (_executor_for, _runtime_verdict, _run_command):
    _legacy_function.__module__ = "omnitensor.acceptance"


for _legacy_type in (SystemdUserServiceProbe, DbusApplyCommandProbe):
    _legacy_type.__module__ = "omnitensor.acceptance"
