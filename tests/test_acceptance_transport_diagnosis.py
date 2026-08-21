"""A probe that retries must not report the deadline as the diagnosis.

`omnitensor-verify-install` reported `apply-command raised TimeoutError` for a
service that was still starting and had not claimed the bus name. That is true
about the probe and useless about the cause: it reads like a wedged service.
"""

import asyncio

import pytest

from omnitensor.acceptance_checks import check_bus
from omnitensor.acceptance_probes import SocketApplyCommandProbe


def probe(caller, *, timeout_s: float = 0.3):
    return SocketApplyCommandProbe(timeout_s=timeout_s, retry_interval_s=0.01, caller=caller)


def failing(error: Exception):
    async def call(_text: str) -> str:
        raise error

    return call


def slow(delay: float = 5.0):
    async def call(_text: str) -> str:
        await asyncio.sleep(delay)
        return "{}"

    return call


def first_then(error: Exception, follow):
    state = {"first": True}

    async def call(text: str) -> str:
        if state["first"]:
            state["first"] = False
            raise error
        return await follow(text)

    return call


def test_a_transport_that_never_answers_names_the_first_real_failure():
    detail = check_bus(probe(first_then(ConnectionError("name has no owner"), slow()))).detail
    assert "name has no owner" in detail, detail
    assert "TimeoutError after ConnectionError" in detail, detail


def test_a_repeated_failure_is_reported_once_rather_than_chained_to_itself():
    detail = check_bus(probe(failing(ConnectionError("name has no owner")))).detail
    assert detail == "apply-command raised ConnectionError: name has no owner"


def test_a_genuinely_unresponsive_service_still_reports_the_deadline():
    # No earlier failure exists to name, and `wait_for` chains its own
    # cancellation, which describes nothing.
    detail = check_bus(probe(slow(), timeout_s=0.2)).detail
    assert detail == "apply-command raised TimeoutError"


def test_an_answering_transport_is_reported_with_its_revision():
    async def call(_text: str) -> str:
        return (
            '{"version":2,"commandId":"invalid","status":"rejected","revision":7,'
            '"appliedAt":1,"message":"Command does not match the version 2 contract",'
            '"portfolio":{"paused":false,"profiles":{},"deviceChoices":{},"modelChoices":{}}}'
        )

    check = check_bus(probe(call))
    assert check.ok is True
    assert "revision 7" in check.detail


def test_a_reply_that_is_not_the_contract_is_refused():
    async def not_json(_text: str) -> str:
        return "<html>"

    assert check_bus(probe(not_json)).detail == "apply-command did not return JSON"

    async def wrong_shape(_text: str) -> str:
        return '{"version":2}'

    assert "violates contract" in check_bus(probe(wrong_shape)).detail


def test_probe_timing_must_be_positive():
    with pytest.raises(ValueError, match="timing"):
        SocketApplyCommandProbe(timeout_s=0)
    with pytest.raises(ValueError, match="timing"):
        SocketApplyCommandProbe(retry_interval_s=-1)


def test_a_command_that_never_answers_is_reported_rather_than_waited_on():
    """A stalled user bus used to hang the whole install report."""
    from omnitensor.acceptance_probes import _run_command

    code, output = _run_command(["sleep", "5"], timeout_s=0.2)
    assert code == 1
    assert "did not answer within" in output


def test_the_service_probe_reports_a_unit_that_never_answers():
    from omnitensor.acceptance_checks import check_service
    from omnitensor.acceptance_probes import SystemdUserServiceProbe

    probe = SystemdUserServiceProbe(
        "omnitensor.service",
        runner=lambda _argv: (1, "systemctl did not answer within 10s"),
    )
    check = check_service(probe)
    assert check.ok is False
    assert "did not answer" in check.detail


def test_the_probe_waits_for_the_socket_before_dialling(tmp_path):
    """The socket file appearing is the readiness signal (XTPU-0191).

    Run right after a restart, the probe used to spend its whole budget
    dialling a socket the service had not created yet; now it watches for
    the file and dials the moment it exists.
    """
    import asyncio
    import threading
    import time

    from omnitensor.acceptance_probes import SocketApplyCommandProbe

    socket_path = tmp_path / "control.sock"
    dialled = []

    probe = SocketApplyCommandProbe(socket_path=socket_path, timeout_s=5.0, retry_interval_s=0.05)

    async def watch_only():
        loop = asyncio.get_running_loop()
        await probe._socket_present(loop, loop.time() + 5.0)
        dialled.append(time.monotonic())

    def create_later():
        time.sleep(0.2)
        socket_path.write_bytes(b"")

    creator = threading.Thread(target=create_later)
    started = time.monotonic()
    creator.start()
    asyncio.run(watch_only())
    creator.join()
    waited = dialled[0] - started
    # It waited for the file, then moved on promptly — well inside the
    # budget it used to burn whole.
    assert 0.15 <= waited < 1.0


def test_a_socket_that_never_appears_falls_through_to_the_call_phase(tmp_path):
    import asyncio

    from omnitensor.acceptance_probes import SocketApplyCommandProbe

    probe = SocketApplyCommandProbe(
        socket_path=tmp_path / "never.sock", timeout_s=0.3, retry_interval_s=0.05
    )

    async def watch_only():
        loop = asyncio.get_running_loop()
        await probe._socket_present(loop, loop.time() + 0.3)

    # Returns rather than raising: the call phase owns the refusal.
    asyncio.run(watch_only())
