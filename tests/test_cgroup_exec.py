from __future__ import annotations

import os

import pytest

from omnitensor.plugins import cgroup_exec


def test_cgroup_exec_joins_before_replacing_itself(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cgroup_exec,
        "join_worker_cgroup",
        lambda path: calls.append(("join", path)),
    )
    monkeypatch.setattr(
        os,
        "execvp",
        lambda executable, argv: calls.append(("exec", executable, argv)),
    )

    assert cgroup_exec.main((str(tmp_path), "worker", "--serve")) == 127
    assert calls == [
        ("join", tmp_path),
        ("exec", "worker", ("worker", "--serve")),
    ]


@pytest.mark.parametrize("arguments", [(), ("/cgroup",)])
def test_cgroup_exec_requires_a_cgroup_and_command(arguments):
    with pytest.raises(SystemExit, match="usage: cgroup_exec"):
        cgroup_exec.main(arguments)
