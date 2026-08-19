from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins.kernel_telemetry import (
    FORBIDDEN_FIELDS,
    AbsentAggregateSource,
    KernelAggregate,
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


def test_aggregate_bounds_accept_their_exact_safe_limits():
    histograms = [{"name": f"h-{index}", "unit": "us", "buckets": []} for index in range(64)]
    histograms[0]["buckets"] = [0] * 63 + [2**53 - 1]
    counters = [{"name": f"c-{index}", "value": 0} for index in range(64)]
    counters[-1]["value"] = 2**53 - 1

    aggregate = parse_aggregate(document(histograms=histograms, counters=counters))

    assert len(aggregate.histograms) == 64
    assert len(aggregate.histograms[0].buckets) == 64
    assert aggregate.histograms[0].buckets[-1] == 2**53 - 1
    assert len(aggregate.counters) == 64
    assert aggregate.counters[0].value == 0
    assert aggregate.counters[-1].value == 2**53 - 1


def test_an_omitted_histogram_unit_has_the_declared_nanosecond_default():
    aggregate = parse_aggregate(document(histograms=[{"name": "latency", "buckets": [1]}]))
    assert aggregate.histograms[0].unit == "ns"


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

    spec = importlib.util.spec_from_file_location("helpers.bpf.omnitensor-bpf-helper", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_privileged_helper_pins_every_executable_path():
    helper = load_helper()
    source = Path(HELPER).read_text(encoding="utf-8")
    unit = Path("helpers/bpf/omnitensor-bpf.service").read_text(encoding="utf-8")

    assert source.startswith("#!/usr/bin/python3\n")
    assert helper.BPFTOOL == "/usr/sbin/bpftool"
    assert 'BPFTOOL = "/usr/sbin/bpftool"' in source
    assert '"bpftool"' not in source
    assert "Environment=PATH=\n" in unit


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


def test_the_helper_autoattaches_and_verifies_every_pinned_object(monkeypatch, tmp_path):
    helper = load_helper()
    pin_dir = tmp_path / "pins"
    object_path = tmp_path / "runq_latency.bpf.o"
    calls = []

    def run(arguments, **options):
        calls.append((arguments, options))
        if arguments[1:3] == ["prog", "loadall"]:
            pin_dir.mkdir()
        name = Path(arguments[-2] if arguments[-1] == "-j" else arguments[-1]).name
        layout = helper.REQUIRED_MAPS.get(name)
        stdout = ""
        if layout is not None:
            stdout = json.dumps(
                {
                    "name": name[: helper.BPF_OBJECT_NAME_MAX],
                    "type": layout[0],
                    "max_entries": layout[1],
                }
            )
        return subprocess.CompletedProcess(arguments, 0, stdout=stdout)

    monkeypatch.setattr(helper.subprocess, "run", run)

    helper.load_probes(object_path, pin_dir)

    assert calls[0] == (
        [
            helper.BPFTOOL,
            "prog",
            "loadall",
            str(object_path),
            str(pin_dir),
            "pinmaps",
            str(pin_dir),
            "autoattach",
        ],
        {"check": True, "capture_output": True, "timeout": 30.0},
    )
    assert [arguments[1:4] for arguments, _options in calls[1:]] == [
        ["map", "show", "pinned"],
        ["map", "show", "pinned"],
        ["map", "show", "pinned"],
        ["link", "show", "pinned"],
        ["link", "show", "pinned"],
        ["link", "show", "pinned"],
    ]
    assert [
        Path(arguments[-2] if arguments[-1] == "-j" else arguments[-1]).name
        for arguments, _options in calls[1:]
    ] == [
        *helper.REQUIRED_MAPS,
        *helper.REQUIRED_LINKS,
    ]
    assert calls[0][1] == {"check": True, "capture_output": True, "timeout": 30.0}
    assert all(
        options == {"check": True, "capture_output": True, "text": True, "timeout": 10.0}
        for _, options in calls[1:4]
    )
    assert all(
        options == {"check": True, "capture_output": True, "timeout": 10.0}
        for _, options in calls[4:]
    )


@pytest.mark.parametrize(
    "changed",
    [
        {"name": "foreign"},
        {"type": "hash"},
        {"max_entries": 2**32},
        {"document": []},
    ],
)
def test_the_helper_refuses_foreign_or_malformed_map_layouts(monkeypatch, tmp_path, changed):
    helper = load_helper()
    pin_dir = tmp_path / "pins"
    pin_dir.mkdir()

    def run(arguments, **_options):
        name = Path(arguments[-2]).name
        expected_type, expected_entries = helper.REQUIRED_MAPS[name]
        document = {
            "name": name[: helper.BPF_OBJECT_NAME_MAX],
            "type": expected_type,
            "max_entries": expected_entries,
        }
        if "document" in changed:
            document = changed["document"]
        else:
            document.update(changed)
        return subprocess.CompletedProcess(arguments, 0, stdout=json.dumps(document))

    monkeypatch.setattr(helper.subprocess, "run", run)

    with pytest.raises(ValueError, match="BPF map layout is invalid"):
        helper.verify_pins(pin_dir)


def test_the_helper_refuses_partial_existing_pins_instead_of_reloading(monkeypatch, tmp_path):
    helper = load_helper()
    pin_dir = tmp_path / "pins"
    pin_dir.mkdir()
    calls = []

    def run(arguments, **options):
        calls.append((arguments, options))
        raise subprocess.CalledProcessError(1, arguments, stderr=b"not a pinned map")

    monkeypatch.setattr(helper.subprocess, "run", run)

    with pytest.raises(subprocess.CalledProcessError):
        helper.load_probes(tmp_path / "runq_latency.bpf.o", pin_dir)

    assert len(calls) == 1
    assert calls[0][0][1:4] == ["map", "show", "pinned"]


def test_the_helper_refuses_an_existing_non_directory_pin_path(tmp_path):
    helper = load_helper()
    pin_path = tmp_path / "pins"
    pin_path.write_text("not bpffs", encoding="utf-8")

    with pytest.raises(OSError, match="BPF pin directory is absent"):
        helper.load_probes(tmp_path / "runq_latency.bpf.o", pin_path)


def test_the_helper_decodes_little_endian_map_words_and_preserves_empty_slots(monkeypatch):
    helper = load_helper()
    output = json.dumps(
        [
            {"key": ["0x00", "0x00"], "value": ["0x02", "0x00"]},
            {"key": [2, 0], "value": [5, 0]},
        ]
    )
    calls = []

    def run(arguments, **options):
        calls.append((arguments, options))
        return subprocess.CompletedProcess(arguments, 0, stdout=output)

    monkeypatch.setattr(helper.subprocess, "run", run)

    assert helper._packed(["0x34", "0x12"]) == 0x1234
    assert helper._packed([0x34, 0x12]) == 0x1234
    buckets = helper.read_histogram(Path("/pins"), "runq_latency_us")
    assert buckets[:3] == [2, 0, 5]
    assert buckets[3:] == [0] * (helper.MAX_SLOTS - 3)
    assert calls == [
        (
            [helper.BPFTOOL, "map", "dump", "pinned", "/pins/runq_latency_us", "-j"],
            {"check": True, "capture_output": True, "text": True, "timeout": 10.0},
        )
    ]


def test_the_helper_bounds_a_foreign_histogram_key_without_allocating(monkeypatch):
    helper = load_helper()
    output = json.dumps([{"key": [255] * 8, "value": [1]}])
    monkeypatch.setattr(
        helper.subprocess,
        "run",
        lambda *_arguments, **_options: subprocess.CompletedProcess(
            helper.BPFTOOL, 0, stdout=output
        ),
    )

    assert helper.read_histogram(Path("/pins"), "runq_latency_us") == [0] * helper.MAX_SLOTS


class FakeConnection:
    def __init__(self, send_error=None):
        self.payloads = []
        self.send_error = send_error

    def __enter__(self):
        return self

    def __exit__(self, *_arguments):
        return None

    def sendall(self, payload):
        if self.send_error is not None:
            raise self.send_error
        self.payloads.append(payload)


class FakeServer:
    def __init__(self, connection):
        self.connection = connection
        self.bound = None
        self.backlog = None
        self.accepted = False

    def __enter__(self):
        return self

    def __exit__(self, *_arguments):
        return None

    def bind(self, path):
        self.bound = path

    def listen(self, backlog):
        self.backlog = backlog

    def accept(self):
        if self.accepted:
            raise KeyboardInterrupt
        self.accepted = True
        return self.connection, None


@pytest.mark.parametrize("failure", [None, subprocess.CalledProcessError(1, ["bpftool"])])
def test_the_helper_socket_serves_aggregates_or_bounded_errors(monkeypatch, tmp_path, failure):
    helper = load_helper()
    socket_path = tmp_path / "bpf.sock"
    socket_path.write_text("stale", encoding="utf-8")
    connection = FakeConnection()
    server = FakeServer(connection)
    modes = []
    monkeypatch.setattr(helper.socket, "socket", lambda *arguments: server)
    monkeypatch.setattr(helper.os, "chmod", lambda path, mode: modes.append((path, mode)))

    def aggregate(_pin_dir):
        if failure is not None:
            raise failure
        return {"version": 1, "histograms": [], "counters": []}

    monkeypatch.setattr(helper, "aggregate", aggregate)

    with pytest.raises(KeyboardInterrupt):
        helper.serve(socket_path, Path("/pins"))

    assert server.bound == str(socket_path)
    assert server.backlog == 8
    assert modes == [(socket_path, 0o666)]
    document = json.loads(connection.payloads[0])
    if failure is None:
        assert document == {"version": 1, "histograms": [], "counters": []}
    else:
        assert document["version"] == 1
        assert document["error"].startswith("Command '['bpftool']' returned non-zero")
        assert len(document["error"]) <= 200


@pytest.mark.parametrize("send_error", [BrokenPipeError("closed"), ConnectionResetError("reset")])
def test_the_helper_survives_a_client_disconnect_while_writing(monkeypatch, tmp_path, send_error):
    helper = load_helper()
    connection = FakeConnection(send_error)
    server = FakeServer(connection)
    monkeypatch.setattr(helper.socket, "socket", lambda *_arguments: server)
    monkeypatch.setattr(helper.os, "chmod", lambda *_arguments: None)
    monkeypatch.setattr(
        helper,
        "aggregate",
        lambda _pin_dir: {"version": 1, "histograms": [], "counters": []},
    )

    with pytest.raises(KeyboardInterrupt):
        helper.serve(tmp_path / "bpf.sock", Path("/pins"))

    assert server.accepted is True
    assert connection.payloads == []


def test_the_helper_cli_fails_closed_prints_once_and_serves(monkeypatch, tmp_path, capsys):
    helper = load_helper()
    object_path = tmp_path / "probe.o"
    pin_dir = tmp_path / "pins"
    socket_path = tmp_path / "bpf.sock"
    monkeypatch.setattr(helper, "DEFAULT_OBJECT", str(object_path))
    monkeypatch.setattr(helper, "DEFAULT_PIN_DIR", str(pin_dir))
    monkeypatch.setattr(helper, "DEFAULT_SOCKET", str(socket_path))

    for failure in (OSError("denied"), ValueError("invalid map")):
        monkeypatch.setattr(
            helper,
            "load_probes",
            lambda *_arguments, failure=failure: (_ for _ in ()).throw(failure),
        )
        assert helper.main([]) == 1
        assert capsys.readouterr().err == f"could not load BPF probes: {failure}\n"

    monkeypatch.setattr(helper, "load_probes", lambda *_arguments: None)
    monkeypatch.setattr(
        helper,
        "aggregate",
        lambda observed: {"version": 1, "pin": str(observed)},
    )
    assert helper.main(["--print-once"]) == 0
    assert json.loads(capsys.readouterr().out) == {"version": 1, "pin": str(pin_dir)}

    served = []
    monkeypatch.setattr(helper, "serve", lambda *arguments: served.append(arguments))
    assert helper.main([]) == 0
    assert served == [(socket_path, pin_dir)]

    with pytest.raises(SystemExit) as excinfo:
        helper.main(["--object", str(object_path)])
    assert excinfo.value.code == 2


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


def test_scheduler_features_summarize_both_latency_histograms_without_identity():
    aggregate = parse_aggregate(
        document(
            histograms=[
                {"name": "runq_latency_us", "unit": "us", "buckets": [1, 1, 8]},
                {"name": "block_latency_us", "unit": "us", "buckets": [5, 5]},
            ]
        )
    )

    assert aggregate.scheduler_features() == {
        "kernelRunQueueSamples": 10.0,
        "kernelRunQueueP50UpperUs": 8.0,
        "kernelRunQueueP95UpperUs": 8.0,
        "kernelBlockIoSamples": 10.0,
        "kernelBlockIoP50UpperUs": 2.0,
        "kernelBlockIoP95UpperUs": 4.0,
    }


def test_a_nanosecond_histogram_is_converted_before_it_becomes_a_feature():
    """The parser defaults `unit` to ns; the feature names promise us."""
    aggregate = parse_aggregate(
        document(histograms=[{"name": "runq_latency_us", "buckets": [1, 1, 8]}])
    )

    assert aggregate.histograms[0].unit == "ns"
    assert aggregate.scheduler_features() == {
        "kernelRunQueueSamples": 10.0,
        "kernelRunQueueP50UpperUs": 0.008,
        "kernelRunQueueP95UpperUs": 0.008,
    }


def test_an_unconvertible_unit_yields_no_feature_rather_than_a_wrong_one():
    aggregate = parse_aggregate(
        document(
            histograms=[
                {"name": "runq_latency_us", "unit": "ticks", "buckets": [1, 1, 8]},
                {"name": "block_latency_us", "unit": "ms", "buckets": [5, 5]},
            ]
        )
    )

    assert aggregate.scheduler_features() == {
        "kernelBlockIoSamples": 10.0,
        "kernelBlockIoP50UpperUs": 2000.0,
        "kernelBlockIoP95UpperUs": 4000.0,
    }


@given(st.lists(st.integers(min_value=0, max_value=10_000), max_size=64))
def test_scheduler_percentiles_are_bounded_for_every_valid_histogram(buckets):
    aggregate = parse_aggregate(
        document(histograms=[{"name": "runq_latency_us", "unit": "us", "buckets": buckets}])
    )

    features = aggregate.scheduler_features()

    assert len(features) == 3
    assert all(0 <= value <= 2**53 - 1 for value in features.values())


def test_unusable_telemetry_never_invents_scheduler_features():
    assert AbsentAggregateSource().read().scheduler_features() == {}


def test_snapshot_detail_is_bounded_even_when_an_os_error_is_not():
    aggregate = KernelAggregate(KernelTelemetryState.HELPER_UNREACHABLE, "x" * 500)
    assert len(aggregate.document()["detail"]) == 240


def test_identity_nested_in_a_plain_mapping_is_refused():
    assert forbidden_field({"outer": {"inner": {"uid": 1000}}}) == "uid"


def test_every_bpftool_call_in_the_helper_is_bounded():
    """A hung bpftool would wedge a privileged, single-connection server.

    The helper reads its maps per request, so a call that never returns leaves
    later collectors blocked in the listen backlog with no error, and
    ``Restart=on-failure`` cannot recover a process that is hung rather than
    dead.
    """
    import ast
    from pathlib import Path

    source = Path(HELPER).read_text()
    tree = ast.parse(source)

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
    ]
    assert calls, "the helper is expected to shell out to bpftool"
    for call in calls:
        keywords = {keyword.arg for keyword in call.keywords}
        assert "timeout" in keywords, f"unbounded subprocess.run at line {call.lineno}"

    # And an expiry has to be answered rather than escaping the accept loop.
    assert "TimeoutExpired" in source


def test_a_worker_failure_reason_reaches_the_inventory_only_when_there_is_one():
    """Dropping it made every failed launch read as an unqualified provider.

    The applet's record is closed, so an entry that always carried the field —
    even as null — is a field an older applet rejects, taking the whole catalog
    down with it. It is present only when the runtime has something to say.
    """
    from omnitensor.inspection import plugin_inventory_entry
    from omnitensor.registry import validate_document
    from tests.conftest import sample_plugin_manifest  # noqa: PLC0415

    class Resolution:
        ready = True
        reason = ""

    plugin = type(
        "Plugin",
        (),
        {
            "plugin_id": "events",
            "version": "1.0.0",
            "source": "external",
            "distribution_name": "omnitensor-events",
            "manifest": sample_plugin_manifest("events"),
        },
    )()

    def entry(**kwargs):
        return plugin_inventory_entry(
            plugin,
            resolve_artifact=lambda _artifact_id: Resolution(),
            granted_permissions=(),
            **kwargs,
        )

    silent = entry(worker_state="ready")
    assert "workerDetail" not in silent

    failed = entry(worker_state="failed", worker_detail="ModuleNotFoundError: qwen")
    assert failed["workerDetail"] == "ModuleNotFoundError: qwen"

    # Bounded, because it is rendered in a panel menu.
    long = entry(worker_state="failed", worker_detail="x" * 500)
    assert len(long["workerDetail"]) == 300

    for document in (silent, failed, long):
        assert (
            validate_document(
                "plugin-inventory.schema.json",
                {
                    "version": 1,
                    "generatedAt": 1,
                    "plugins": [document],
                },
            )
            == []
        )
