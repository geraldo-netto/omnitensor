from __future__ import annotations

import socket

import pytest

from omnitensor.readiness import READY, STOPPING, notify, notify_ready, notify_stopping


def test_nothing_is_sent_when_nobody_asked_to_be_told():
    """A service started by hand has no NOTIFY_SOCKET, which is not an error."""
    assert notify_ready(environ={}) is False
    assert notify(READY, environ={"NOTIFY_SOCKET": ""}) is False


def test_readiness_reaches_a_listening_path_socket(tmp_path):
    address = tmp_path / "notify.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as listener:
        listener.bind(str(address))
        assert notify_ready(environ={"NOTIFY_SOCKET": str(address)}) is True
        assert listener.recv(64) == READY.encode("utf-8")

        assert notify_stopping(environ={"NOTIFY_SOCKET": str(address)}) is True
        assert listener.recv(64) == STOPPING.encode("utf-8")


def test_readiness_reaches_an_abstract_socket_named_with_an_at_sign():
    name = "\0omnitensor-readiness-test"
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as listener:
        listener.bind(name)
        assert notify(READY, environ={"NOTIFY_SOCKET": "@" + name[1:]}) is True
        assert listener.recv(64) == READY.encode("utf-8")


def test_an_undeliverable_notification_never_stops_a_ready_service(tmp_path):
    assert notify(READY, environ={"NOTIFY_SOCKET": str(tmp_path / "absent.sock")}) is False


@pytest.mark.parametrize("state", ["", None, 1])
def test_an_empty_state_is_a_programming_error_rather_than_a_silent_no_op(state):
    with pytest.raises(ValueError, match="non-empty string"):
        notify(state, environ={"NOTIFY_SOCKET": "/tmp/whatever"})
