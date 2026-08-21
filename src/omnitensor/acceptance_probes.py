"""Live service, control-socket, and accelerator probes for installation acceptance."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


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


USABLE = "usable"
DEVICE_MISSING = "device-missing"
UNUSABLE = "unusable"


def runtime_state(module: str) -> tuple[str, str]:
    """What this runtime is: usable, waiting for its device, or unusable.

    Three states, not two. Absent hardware and a broken runtime used to give
    the same answer — ``None``, "nothing wrong" — so a machine with no Coral
    plugged in reported "accelerator runtimes usable: tpu:tflite_runtime" and
    then refused the first job with ``device-absent``. An install waiting for
    hardware is not an install that is wrong, but it is not a usable lane
    either, and the report has to be able to say which.
    """
    from .executors.base import DEVICE_ABSENT  # noqa: PLC0415 - probe-only import

    executor_for = _executor_for
    executor = executor_for(module)
    if executor is None:
        return USABLE, ""
    try:
        availability = executor.availability()
    except Exception as error:  # noqa: BLE001 - a probe must not fail the report
        return UNUSABLE, type(error).__name__
    if availability.available:
        return USABLE, ""
    if availability.code == DEVICE_ABSENT:
        return DEVICE_MISSING, availability.reason
    return UNUSABLE, availability.reason


def _runtime_verdict(module: str) -> str | None:
    """Why this runtime cannot be used, or ``None`` when it can.

    Kept for callers that only branch on usable/unusable; ``runtime_state``
    is the one that can tell absent hardware apart.
    """
    state, reason = runtime_state(module)
    return reason if state == UNUSABLE else None


class SystemdUserServiceProbe:
    """:class:`ServiceProbe` over ``systemctl --user``."""

    def __init__(self, unit: str = "omnitensor.service", *, runner=None):
        self._unit = unit
        self._runner = runner or _run_command

    def unit_state(self) -> tuple[bool, str]:
        code, output = self._runner(["systemctl", "--user", "is-active", self._unit])
        state = output.strip() or "unknown"
        return code == 0 and state == "active", f"{self._unit} is {state}"


def _diagnosed(last: BaseException, first: BaseException) -> BaseException:
    """The exception to raise once the retry budget is spent.

    Attaching the first failure as the cause keeps the diagnosis a caller can
    act on ("the name has no owner") attached to the outcome the probe observed
    ("it never answered"), instead of replacing one with the other.
    """
    if last is first or (type(last) is type(first) and str(last) == str(first)):
        return last
    # Set rather than `raise ... from`, because the caller re-raises the value
    # this returns and an explicit `from` there would overwrite it.
    last.__cause__ = first
    last.__suppress_context__ = True
    return last


class SocketApplyCommandProbe:
    """:class:`ControlProbe` over the control socket."""

    def __init__(
        self,
        *,
        socket_path=None,
        timeout_s: float = 30.0,
        retry_interval_s: float = 0.25,
        caller=None,
    ):
        if timeout_s <= 0 or retry_interval_s < 0:
            raise ValueError("control probe timing must be positive")
        self._socket_path = socket_path
        self._timeout_s = timeout_s
        self._retry_interval_s = retry_interval_s
        self._caller = caller

    def apply_command(self, text: str) -> str:
        import asyncio  # noqa: PLC0415
        import json  # noqa: PLC0415

        async def call_once() -> str:  # pragma: no cover - live adapter wiring
            if self._caller is not None:
                return await self._caller(text)
            from .socket_transport import call_control  # noqa: PLC0415

            # The probe's text is deliberately not a valid command; wrapping a
            # bare string keeps the envelope valid so the rejection under test
            # is the method's, not the transport's.
            result = await call_control(
                "apply-command",
                json.loads(text) if text.startswith("{") else {"probe": text},
                socket_path=self._socket_path,
                timeout_s=self._timeout_s,
            )
            return json.dumps(result, separators=(",", ":"))

        async def call() -> str:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._timeout_s
            # The readiness signal, not blind connect churn (XTPU-0191): run
            # right after `systemctl --user restart`, the probe used to spend
            # its whole budget against a service still creating its socket.
            # The socket file appearing IS the service saying it is ready to
            # be dialled, so existence is awaited first — within the same
            # deadline, and only when this probe dials a real socket. A path
            # that never appears falls through to the call phase, whose
            # refusal names the actual cause.
            if self._caller is None:
                await self._socket_present(loop, deadline)
            # A probe that retries for its whole budget and then reports the
            # deadline says "TimeoutError" for a service that never created its
            # socket — accurate about the probe and useless about the cause.
            # The first real failure is the one worth carrying out.
            first_error: BaseException | None = None
            while True:
                remaining = deadline - loop.time()
                try:
                    return await asyncio.wait_for(call_once(), timeout=max(remaining, 0.001))
                except Exception as error:  # noqa: BLE001 - containment: this must not escape into the caller
                    first_error = first_error or error
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise _diagnosed(error, first_error)  # noqa: B904 - cause is set above
                    await asyncio.sleep(min(self._retry_interval_s, remaining))

        return asyncio.run(call())

    async def _socket_present(self, loop, deadline: float) -> None:
        import asyncio  # noqa: PLC0415

        from .socket_transport import default_socket_path  # noqa: PLC0415

        path = Path(self._socket_path) if self._socket_path else default_socket_path()
        while not path.exists():
            if loop.time() >= deadline:
                return
            await asyncio.sleep(min(self._retry_interval_s or 0.05, 0.25))


# `systemctl --user is-active` answers immediately or not at all: a stalled
# manager leaves it waiting, and an unbounded wait here hangs the whole
# install report with no diagnostic. The sibling socket probe already bounds
# itself, so this matches rather than being the one unbounded call.
DEFAULT_COMMAND_TIMEOUT_SECONDS = 10.0


def _run_command(
    argv: Sequence[str], *, timeout_s: float = DEFAULT_COMMAND_TIMEOUT_SECONDS
) -> tuple[int, str]:
    import subprocess  # noqa: PLC0415

    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, check=False, timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        # A probe reports; it does not raise into the report it is filling.
        return 1, f"{argv[0]} did not answer within {timeout_s:g}s"
    return completed.returncode, completed.stdout
