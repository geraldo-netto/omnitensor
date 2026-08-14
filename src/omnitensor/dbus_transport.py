"""D-Bus adapter for the transport-neutral OmniTensor runtime handler."""

from __future__ import annotations

from dbus_fast import BusType, RequestNameReply
from dbus_fast.aio import MessageBus
from dbus_fast.service import ServiceInterface, method

from .callers import CallerIdentityResolver, caller_capture_handler, unix_user_lookup
from .contract import RUNTIME_METHODS
from .ports import RuntimeHandler

BUS_NAME = "org.cinnamon.OmniTensor1"
OBJECT_PATH = "/org/cinnamon/OmniTensor1"

# Named once because the handshake announces exactly what the adapter exports.
BUS_METHODS = RUNTIME_METHODS


class OmniTensorInterface(ServiceInterface):
    """Transport-only D-Bus shim; all logic stays in the injected handler."""

    def __init__(self, runtime: RuntimeHandler) -> None:
        super().__init__(BUS_NAME)
        self._runtime = runtime

    @method()
    async def ApplyCommand(self, command: s) -> s:  # noqa: F821, N802 - wire API
        return await self._runtime.apply_command_text(command)

    @method()
    async def SubmitJob(self, request: s) -> s:  # noqa: F821, N802 - wire API
        return await self._runtime.submit_job_text(request)

    @method()
    async def CancelJob(self, request: s) -> s:  # noqa: F821, N802 - wire API
        return await self._runtime.cancel_job_text(request)

    @method()
    async def GetJobResult(self, request: s) -> s:  # noqa: F821, N802 - wire API
        return await self._runtime.job_result_text(request)

    @method()
    def DescribePlugins(self) -> s:  # noqa: F821, N802 - wire API
        return self._runtime.describe_plugins_text()

    @method()
    def DescribeContract(self) -> s:  # noqa: F821, N802 - wire API
        return self._runtime.describe_contract_text()


class DbusControlTransport:
    """:class:`~omnitensor.ports.ControlTransport` over the session bus."""

    def __init__(
        self,
        bus_type: BusType = BusType.SESSION,
        *,
        bus_factory=None,
        bus_name: str = BUS_NAME,
        callers: CallerIdentityResolver | None = None,
    ) -> None:
        self._bus_type = bus_type
        self._bus_factory = bus_factory
        self._bus_name = bus_name
        self._callers = callers
        self._bus = None

    async def start(self, handler: RuntimeHandler) -> None:
        bus = await self._connect()
        try:
            # Bind callers before export so no dispatch can inherit a sender.
            bus.add_message_handler(caller_capture_handler())
            if self._callers is not None:
                self._callers.attach(unix_user_lookup(bus))
            bus.export(OBJECT_PATH, OmniTensorInterface(handler))
            reply = await bus.request_name(self._bus_name)
            self._require_primary_owner(reply)
        except BaseException:
            bus.disconnect()
            raise
        self._bus = bus

    async def _connect(self):
        if self._bus_factory is not None:
            return await self._bus_factory()
        return await MessageBus(bus_type=self._bus_type).connect()

    def _require_primary_owner(self, reply: RequestNameReply) -> None:
        if reply == RequestNameReply.PRIMARY_OWNER:
            return
        raise RuntimeError(
            f"{self._bus_name} is already owned"
            f" (request_name reply: {reply.name});"
            " another omnitensor instance is running",
        )

    async def stop(self) -> None:
        if self._bus is not None:
            self._bus.disconnect()
            self._bus = None
