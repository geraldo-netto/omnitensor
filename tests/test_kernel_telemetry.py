from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
from pathlib import Path

import pytest

from omnitensor.plugins.kernel_telemetry import (
    FORBIDDEN_FIELDS,
    AbsentAggregateSource,
    KernelTelemetryError,
    KernelTelemetryState,
    UnixSocketAggregateSource,
    forbidden_field,
    parse_aggregate,
)

HELPER = "helpers/bpf/omnitensor-bpf-helper.py"
BPF_SOURCE = "helpers/bpf/runq_latency.bpf.c"


def document(**changes):
    base = {
        "version": 1,
        "collectedAtMs": 1_700_000_000_000,
        "histograms": [{"name": "runq_latency_us", "unit": "us", "buckets": [3, 9, 1]}],
        "counters": [{"name": "block_rq_completed", "value": 42}],
    }
    base.update(changes)
    return base


def serve_once(path, payload):
    """A stand-in helper: one connection, one aggregate."""
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)

    def run():
        connection, _ = server.accept()
        with connection:
            connection.sendall(payload)
        server.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_an_aggregate_of_histograms_and_counters_is_accepted():
    aggregate = parse_aggregate(document())

    assert aggregate.usable
    assert aggregate.state is KernelTelemetryState.READY
    assert aggregate.histograms[0].buckets == (3, 9, 1)
    assert aggregate.histograms[0].total == 13
    assert aggregate.counters[0].value == 42


@pytest.mark.parametrize("field", sorted(FORBIDDEN_FIELDS))
def test_an_aggregate_carrying_identity_is_refused_not_stripped(field):
    """eBPF can read all of this; the helper is supposed to aggregate first."""
    with pytest.raises(KernelTelemetryError, match="aggregate-identifying"):
        parse_aggregate(document(**{field: "anything"}))


def test_identity_nested_inside_a_series_is_still_refused():
    """A pid one level down is still a pid."""
    with pytest.raises(KernelTelemetryError, match="must not carry pid"):
        parse_aggregate(
            document(histograms=[{"name": "x", "unit": "us", "buckets": [1], "pid": 1234}])
        )
    with pytest.raises(KernelTelemetryError, match="must not carry comm"):
        parse_aggregate(document(counters=[{"name": "x", "value": 1, "comm": "firefox"}]))


def test_an_ordinary_aggregate_is_not_mistaken_for_an_identifying_one():
    assert forbidden_field({"name": "runq", "buckets": [1, 2]}) == ""
    assert forbidden_field({"nested": {"deep": [{"name": "ok"}]}}) == ""


@pytest.mark.parametrize(
    "changes",
    [
        {"version": 2},
        {"collectedAtMs": -1},
        {"collectedAtMs": True},
        {"histograms": {}},
        {"histograms": [{"name": "", "unit": "us", "buckets": []}]},
        {"histograms": [{"name": "x", "unit": "us", "buckets": [-1]}]},
        {"histograms": [{"name": "x", "unit": "us", "buckets": [True]}]},
        {"histograms": [{"name": "x", "unit": "us", "buckets": "many"}]},
        {"histograms": ["not-an-object"]},
        {"histograms": [{"name": "x", "unit": "us", "buckets": [0] * 100}]},
        {"counters": {}},
        {"counters": ["not-an-object"]},
        {"counters": [{"name": "x", "value": -1}]},
        {"counters": [{"name": "x", "value": "many"}]},
    ],
)
def test_a_malformed_aggregate_is_refused(changes):
    with pytest.raises(KernelTelemetryError, match="aggregate-invalid"):
        parse_aggregate(document(**changes))


def test_a_non_object_aggregate_is_refused():
    with pytest.raises(KernelTelemetryError, match="aggregate-invalid"):
        parse_aggregate(["not", "an", "object"])


def test_an_absent_helper_says_so_rather_than_reporting_zero(tmp_path):
    """Zeroes would be indistinguishable from a quiet kernel."""
    source = UnixSocketAggregateSource(tmp_path / "missing.sock")

    aggregate = source.read()

    assert aggregate.state is KernelTelemetryState.HELPER_ABSENT
    assert not aggregate.usable
    assert aggregate.histograms == ()
    assert str(tmp_path) in aggregate.detail


def test_an_aggregate_is_read_from_the_helper_socket(tmp_path):
    path = tmp_path / "bpf.sock"
    serve_once(path, json.dumps(document()).encode())

    aggregate = UnixSocketAggregateSource(path).read()

    assert aggregate.usable
    assert aggregate.histograms[0].name == "runq_latency_us"


def test_a_helper_that_sends_rubbish_is_reported_not_trusted(tmp_path):
    path = tmp_path / "bpf.sock"
    serve_once(path, b"{ not json")

    aggregate = UnixSocketAggregateSource(path).read()

    assert aggregate.state is KernelTelemetryState.HELPER_INVALID
    assert not aggregate.usable


def test_a_helper_that_sends_identity_is_reported_as_invalid(tmp_path):
    path = tmp_path / "bpf.sock"
    serve_once(path, json.dumps(document(pid=1234)).encode())

    aggregate = UnixSocketAggregateSource(path).read()

    assert aggregate.state is KernelTelemetryState.HELPER_INVALID
    assert "must not carry pid" in aggregate.detail


def test_an_oversized_aggregate_is_refused(tmp_path):
    path = tmp_path / "bpf.sock"
    serve_once(path, b"x" * 5000)

    aggregate = UnixSocketAggregateSource(path, max_aggregate_bytes=64).read()

    assert aggregate.state is KernelTelemetryState.HELPER_UNREACHABLE
    assert "exceeds 64 bytes" in aggregate.detail


def test_a_socket_that_refuses_the_connection_is_reported(tmp_path):
    path = tmp_path / "bpf.sock"
    path.write_text("not a socket")

    aggregate = UnixSocketAggregateSource(path).read()

    assert aggregate.state is KernelTelemetryState.HELPER_UNREACHABLE


@pytest.mark.parametrize(
    "bounds", [{"max_aggregate_bytes": 0}, {"timeout_seconds": 0}, {"timeout_seconds": True}]
)
def test_the_reader_bounds_are_validated(bounds):
    with pytest.raises(KernelTelemetryError, match="bounds-invalid"):
        UnixSocketAggregateSource("/run/x.sock", **bounds)


def test_an_unconfigured_service_reports_absence_explicitly():
    aggregate = AbsentAggregateSource().read()
    assert aggregate.state is KernelTelemetryState.HELPER_ABSENT
    assert not aggregate.usable
    assert aggregate.document()["state"] == "helper-absent"


def test_the_reader_reports_the_socket_it_watches():
    assert UnixSocketAggregateSource("/run/omnitensor/x.sock").socket_path.name == "x.sock"


def test_the_bpf_program_reads_no_process_identity():
    """A pid that reached userspace would be a leak; these helpers never run."""
    source = Path(BPF_SOURCE).read_text()

    for helper in (
        "bpf_get_current_comm",
        "bpf_probe_read_user_str",
        "bpf_get_current_uid_gid",
        "bpf_get_current_cgroup_id",
        "bpf_perf_event_output",
        "bpf_ringbuf",
    ):
        assert helper not in source, f"{helper} would export per-process data"
    # The one pid used is a transient hash key for the wakeup delta, deleted
    # the moment it is consumed and never placed in an exported map.
    assert "bpf_map_delete_elem(&wakeup_at" in source


def load_helper():
    import importlib.util

    spec = importlib.util.spec_from_file_location("omnitensor_bpf_helper", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_helper_exports_only_aggregate_series(monkeypatch):
    """Checked by running it, not by reading its source for words."""
    helper = load_helper()
    monkeypatch.setattr(helper, "read_histogram", lambda pin_dir, name: [1, 2, 3])

    exported = helper.aggregate(Path("/sys/fs/bpf/omnitensor"))

    assert set(exported) == {"version", "collectedAtMs", "histograms", "counters"}
    assert forbidden_field(exported) == ""
    assert parse_aggregate(exported).usable
    assert [item["name"] for item in exported["histograms"]] == [
        "runq_latency_us",
        "block_latency_us",
    ]


def test_the_helper_aggregate_survives_the_readers_own_validation(monkeypatch):
    """The two sides agree on the contract, checked against each other."""
    helper = load_helper()
    monkeypatch.setattr(helper, "read_histogram", lambda pin_dir, name: [0] * 27)

    aggregate = parse_aggregate(helper.aggregate(Path("/sys/fs/bpf/omnitensor")))

    assert aggregate.usable
    assert len(aggregate.histograms) == 2


@pytest.mark.skipif(
    not all(
        subprocess.run(["which", tool], capture_output=True).returncode == 0
        for tool in ("clang", "bpftool")
    )
    or not os.path.exists("/sys/kernel/btf/vmlinux"),
    reason="the BPF toolchain or kernel BTF is unavailable on this host",
)
def test_the_bpf_object_builds_against_this_kernel(tmp_path):
    """CO-RE against generated BTF, so a stale vmlinux.h cannot go unnoticed."""
    result = subprocess.run(
        ["helpers/bpf/build.sh", str(tmp_path)], capture_output=True, text=True, timeout=600
    )

    assert result.returncode == 0, result.stderr[-800:]
    assert (tmp_path / "runq_latency.bpf.o").is_file()


def test_an_aggregate_describes_itself_for_the_snapshot():
    aggregate = parse_aggregate(document())

    described = aggregate.document()

    assert described["histograms"] == [
        {"name": "runq_latency_us", "unit": "us", "buckets": [3, 9, 1]}
    ]
    assert described["counters"] == [{"name": "block_rq_completed", "value": 42}]
    assert described["state"] == "ready"


def test_identity_nested_in_a_plain_mapping_is_refused():
    assert forbidden_field({"outer": {"inner": {"uid": 1000}}}) == "uid"
