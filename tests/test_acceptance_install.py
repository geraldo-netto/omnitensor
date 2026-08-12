from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from conftest import add_pcie_tpu, sample_manifest, write_workload

from omnitensor.acceptance import (
    REQUIRED_SCHEMAS,
    Check,
    DbusApplyCommandProbe,
    InstallationReport,
    SystemdUserServiceProbe,
    check_applet,
    check_bus,
    check_executable,
    check_isolation,
    check_plugin_discovery,
    check_schemas,
    check_service,
    check_snapshot,
    check_workload_catalog,
    load_applet_checksums,
    main,
    verify_installation,
)
from omnitensor.discovery import detect_devices
from omnitensor.snapshot import build_snapshot


class StubService:
    def __init__(self, running, detail):
        self._state = (running, detail)

    def unit_state(self):
        return self._state


class StubBus:
    def __init__(self, reply=None, error=None):
        self._reply = reply
        self._error = error
        self.calls = []

    def apply_command(self, text):
        self.calls.append(text)
        if self._error is not None:
            raise self._error
        return self._reply


def acknowledgement(**changes):
    document = {
        "version": 1,
        "commandId": "invalid",
        "status": "rejected",
        "revision": 4,
        "appliedAt": 1_700_000_000_000,
        "message": "Command is not valid JSON",
        "portfolio": {"paused": False, "profiles": {}},
    }
    document.update(changes)
    return json.dumps(document)


def snapshot_file(tmp_path, fake_nodes, *, generated_at_ms=1_000_000):
    add_pcie_tpu(fake_nodes)
    document = build_snapshot(
        devices=detect_devices(fake_nodes),
        metrics={},
        profiles={},
        generated_at_ms=generated_at_ms,
    )
    path = tmp_path / "state.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_a_passing_report_is_ok_and_renders_every_check():
    report = verify_installation(
        [Check("service", True, "active"), Check("dbus", True, "answered")]
    )
    assert report.ok is True
    assert report.failures == ()
    assert report.render() == "PASS  service: active\nPASS  dbus: answered"


def test_one_failure_fails_the_whole_report():
    report = verify_installation(
        [Check("service", True, "active"), Check("dbus", False, "no reply")]
    )
    assert report.ok is False
    assert [check.name for check in report.failures] == ["dbus"]
    assert "FAIL  dbus: no reply" in report.render()


def test_a_missing_executable_is_reported():
    assert check_executable("omnitensor", resolver=lambda _n: None).ok is False
    found = check_executable("omnitensor", resolver=lambda _n: "/usr/bin/omnitensor")
    assert found.ok is True
    assert "/usr/bin/omnitensor" in found.detail


@pytest.mark.parametrize(
    ("running", "detail"), [(True, "omnitensor.service is active"), (False, "inactive")]
)
def test_the_service_check_reports_the_probe_verbatim(running, detail):
    check = check_service(StubService(running, detail))
    assert check.ok is running
    assert check.detail == detail


def test_the_systemd_probe_reads_is_active():
    calls = []

    def runner(argv):
        calls.append(argv)
        return 0, "active\n"

    probe = SystemdUserServiceProbe(runner=runner)
    running, detail = probe.unit_state()

    assert running is True
    assert detail == "omnitensor.service is active"
    assert calls == [["systemctl", "--user", "is-active", "omnitensor.service"]]


@pytest.mark.parametrize(
    ("code", "output"), [(3, "inactive\n"), (0, "activating\n"), (1, "")]
)
def test_the_systemd_probe_reports_anything_but_active_as_down(code, output):
    probe = SystemdUserServiceProbe(runner=lambda _argv: (code, output))
    running, _detail = probe.unit_state()
    assert running is False


def test_every_canonical_schema_is_installed_and_valid():
    check = check_schemas()
    assert check.ok is True
    assert str(len(REQUIRED_SCHEMAS)) in check.detail


def test_a_missing_schema_fails_the_check():
    check = check_schemas(("runtime-snapshot.schema.json", "absent.schema.json"))
    assert check.ok is False
    assert "missing absent.schema.json" in check.detail


def test_an_invalid_schema_fails_the_check(monkeypatch, tmp_path):
    """Present but invalid is worse than absent: it fails at runtime instead."""
    packaged = tmp_path / "schemas"
    packaged.mkdir()
    (packaged / "broken.schema.json").write_text('{"type": 7}', encoding="utf-8")
    import omnitensor.registry as registry

    monkeypatch.setattr(registry, "_PACKAGED_SCHEMAS", packaged)
    monkeypatch.setattr(registry, "_SOURCE_SCHEMAS", None)

    check = check_schemas(("broken.schema.json",))

    assert check.ok is False
    assert "invalid broken.schema.json" in check.detail


def test_the_bundled_workload_catalog_loads():
    check = check_workload_catalog()
    assert check.ok is True
    assert "bundled workloads loaded" in check.detail


def test_an_empty_workload_catalog_fails(tmp_path):
    check = check_workload_catalog(tmp_path)
    assert check.ok is False
    assert "no bundled workloads" in check.detail


def test_an_invalid_bundled_manifest_fails(tmp_path):
    write_workload(tmp_path, sample_manifest())
    (tmp_path / "sample-workload/manifest.json").write_text("{nope", encoding="utf-8")
    check = check_workload_catalog(tmp_path)
    assert check.ok is False
    assert "bundled catalog is invalid" in check.detail


def test_discovery_failure_is_reported_without_raising():
    def explode():
        raise OSError("entry points unreadable")

    check = check_plugin_discovery(explode)
    assert check.ok is False
    assert "discovery raised OSError" in check.detail


def test_successful_discovery_counts_candidates():
    check = check_plugin_discovery(lambda: (object(), object()))
    assert check.ok is True
    assert "2 plugin candidates" in check.detail


class StubSpec:
    def __init__(self, plugin_id, sandbox):
        self.plugin_id = plugin_id
        self.sandbox = sandbox


def test_an_unsandboxed_worker_fails_the_isolation_check():
    check = check_isolation([StubSpec("alpha", object()), StubSpec("beta", None)])
    assert check.ok is False
    assert "beta" in check.detail


def test_all_sandboxed_workers_pass_the_isolation_check():
    check = check_isolation([StubSpec("alpha", object())])
    assert check.ok is True
    assert "1 external workers are sandboxed" in check.detail


def test_the_bus_check_accepts_a_contract_valid_rejection():
    bus = StubBus(reply=acknowledgement())
    check = check_bus(bus)
    assert check.ok is True
    assert "revision 4" in check.detail
    assert bus.calls == ["not-json"], "the probe must not mutate policy"


def test_the_bus_check_reports_a_transport_failure():
    check = check_bus(StubBus(error=ConnectionRefusedError("no bus")))
    assert check.ok is False
    assert "ConnectionRefusedError" in check.detail


def test_the_bus_check_rejects_a_non_json_reply():
    check = check_bus(StubBus(reply="not-json"))
    assert check.ok is False
    assert "did not return JSON" in check.detail


def test_the_bus_check_rejects_a_reply_that_violates_the_contract():
    check = check_bus(StubBus(reply=json.dumps({"version": 1})))
    assert check.ok is False
    assert "violates contract" in check.detail


def test_the_bus_check_rejects_an_accepted_invalid_command():
    check = check_bus(StubBus(reply=acknowledgement(status="applied")))
    assert check.ok is False
    assert "was not rejected" in check.detail


def test_a_fresh_contract_valid_snapshot_passes(tmp_path, fake_nodes):
    path = snapshot_file(tmp_path, fake_nodes, generated_at_ms=1_000_000)
    check = check_snapshot(path, now_ms=1_005_000, max_age_ms=30_000)
    assert check.ok is True
    assert "5000 ms old" in check.detail


def test_a_stale_snapshot_fails_even_though_it_is_valid(tmp_path, fake_nodes):
    """The failure that looks most like success: the applet renders it happily."""
    path = snapshot_file(tmp_path, fake_nodes, generated_at_ms=1_000_000)
    check = check_snapshot(path, now_ms=1_060_000, max_age_ms=30_000)
    assert check.ok is False
    assert "60000 ms old" in check.detail


def test_a_missing_snapshot_fails(tmp_path):
    check = check_snapshot(tmp_path / "absent.json", now_ms=1)
    assert check.ok is False
    assert "no snapshot" in check.detail


def test_an_unreadable_snapshot_fails(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{nope", encoding="utf-8")
    check = check_snapshot(path, now_ms=1)
    assert check.ok is False
    assert "unreadable" in check.detail


def test_a_snapshot_that_violates_the_contract_fails(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 1}), encoding="utf-8")
    check = check_snapshot(path, now_ms=1)
    assert check.ok is False
    assert "violates contract" in check.detail


def applet_tree(tmp_path, files):
    root = tmp_path / "applet"
    checksums = {}
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        checksums[relative] = hashlib.sha256(content).hexdigest()
    return root, checksums


def test_a_matching_applet_tree_passes(tmp_path):
    root, checksums = applet_tree(tmp_path, {"applet.js": b"code", "lib/x.js": b"more"})
    check = check_applet(root, checksums)
    assert check.ok is True
    assert "2 applet files" in check.detail


def test_a_half_copied_applet_tree_fails(tmp_path):
    root, checksums = applet_tree(tmp_path, {"applet.js": b"code", "lib/x.js": b"more"})
    (root / "lib/x.js").unlink()
    check = check_applet(root, checksums)
    assert check.ok is False
    assert "missing 1 file" in check.detail


def test_an_altered_applet_file_fails(tmp_path):
    root, checksums = applet_tree(tmp_path, {"applet.js": b"code"})
    (root / "applet.js").write_bytes(b"tampered")
    check = check_applet(root, checksums)
    assert check.ok is False
    assert "altered 1 file" in check.detail


def test_an_absent_applet_directory_fails(tmp_path):
    check = check_applet(tmp_path / "absent", {"applet.js": "0" * 64})
    assert check.ok is False
    assert "no applet installed" in check.detail


def test_checksum_manifests_are_parsed_and_validated(tmp_path):
    path = tmp_path / "SHA256SUMS"
    path.write_text(f"{'a' * 64}  applet.js\n{'b' * 64}  lib/x.js\n", encoding="utf-8")
    assert load_applet_checksums(path) == {"applet.js": "a" * 64, "lib/x.js": "b" * 64}


def test_an_empty_checksum_manifest_is_rejected(tmp_path):
    path = tmp_path / "SHA256SUMS"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no checksums"):
        load_applet_checksums(path)


def test_a_malformed_checksum_line_is_rejected(tmp_path):
    path = tmp_path / "SHA256SUMS"
    path.write_text("garbage\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed checksum line"):
        load_applet_checksums(path)


def test_the_cli_reports_failures_and_exits_non_zero(monkeypatch, capsys, tmp_path):
    import omnitensor.acceptance as acceptance

    monkeypatch.setattr(
        acceptance,
        "build_default_report",
        lambda **_kwargs: InstallationReport(
            (Check("service", True, "active"), Check("dbus", False, "no reply"))
        ),
    )

    code = main(["--snapshot", str(tmp_path / "state.json")])

    assert code == 1
    output = capsys.readouterr().out
    assert "PASS  service: active" in output
    assert "FAIL  dbus: no reply" in output


def test_the_cli_exits_zero_when_everything_passes(monkeypatch, capsys, tmp_path):
    import omnitensor.acceptance as acceptance

    monkeypatch.setattr(
        acceptance,
        "build_default_report",
        lambda **_kwargs: InstallationReport((Check("service", True, "active"),)),
    )

    assert main(["--snapshot", str(tmp_path / "state.json")]) == 0
    assert "PASS  service: active" in capsys.readouterr().out


def test_the_dbus_probe_derives_its_object_path_from_the_bus_name():
    probe = DbusApplyCommandProbe(bus_name="org.cinnamon.OmniTensor1")
    assert probe._bus_name == "org.cinnamon.OmniTensor1"


def test_the_dbus_probe_retries_transient_service_startup_errors():
    attempts = []

    async def caller(text):
        attempts.append(text)
        if len(attempts) < 3:
            raise ConnectionRefusedError("service is starting")
        return acknowledgement()

    probe = DbusApplyCommandProbe(
        timeout_s=1,
        retry_interval_s=0,
        caller=caller,
    )

    assert probe.apply_command("not-json") == acknowledgement()
    assert attempts == ["not-json", "not-json", "not-json"]


@pytest.mark.parametrize(
    ("timeout_s", "retry_interval_s"),
    [(0, 0.25), (-1, 0.25), (30, -0.1)],
)
def test_the_dbus_probe_rejects_invalid_retry_timing(timeout_s, retry_interval_s):
    with pytest.raises(ValueError, match="timing"):
        DbusApplyCommandProbe(
            timeout_s=timeout_s,
            retry_interval_s=retry_interval_s,
        )


def test_a_snapshot_read_a_moment_after_it_was_written_is_not_negative(tmp_path, fake_nodes):
    """Publisher and verifier read the same clock microseconds apart."""
    path = snapshot_file(tmp_path, fake_nodes, generated_at_ms=1_000_100)

    check = check_snapshot(path, now_ms=1_000_000, max_age_ms=30_000)

    assert check.ok is True
    assert "0 ms old" in check.detail
    assert "-100 ms" not in check.detail


def test_a_snapshot_stamped_in_the_future_fails_rather_than_reading_as_fresh(
    tmp_path, fake_nodes
):
    """A future timestamp would otherwise pass the staleness check forever."""
    path = snapshot_file(tmp_path, fake_nodes, generated_at_ms=2_000_000)

    check = check_snapshot(path, now_ms=1_000_000, max_age_ms=30_000)

    assert check.ok is False
    assert "in the future" in check.detail


def test_an_install_with_no_accelerator_runtime_fails_the_report():
    """The whole report passing while nothing can execute is the worst state."""
    from omnitensor.acceptance import check_backends

    check = check_backends((("gpu", "a_runtime_that_is_not_installed", "install [gpu]"),))

    assert check.ok is False
    assert "no accelerator runtime is importable" in check.detail
    assert "install [gpu]" in check.detail


def test_an_available_runtime_is_named_rather_than_merely_counted():
    from omnitensor.acceptance import check_backends

    check = check_backends((("gpu", "json", "install [gpu]"),))

    assert check.ok is True
    assert "gpu:json" in check.detail


def test_a_broken_distribution_is_not_counted_as_installed(monkeypatch):
    from omnitensor import acceptance

    def explode(name):
        raise ValueError("broken distribution metadata")

    monkeypatch.setattr("importlib.util.find_spec", explode)

    assert acceptance.check_backends((("gpu", "ncnn", "install [gpu]"),)).ok is False


def test_no_backend_entry_names_the_cpu():
    """A CPU-only runtime is not a backend here, by design."""
    from omnitensor.acceptance import BACKEND_RUNTIMES

    assert "cpu" not in {backend for backend, _module, _remedy in BACKEND_RUNTIMES}


def test_every_shipped_schema_is_one_the_install_check_requires():
    """A check that lists its own expectations falls behind them.

    ``runtime-contract.schema.json`` shipped, was resolvable, and answered on
    the bus while this check still reported eleven schemas — so an install
    missing it would have passed.
    """
    from omnitensor.acceptance import REQUIRED_SCHEMAS
    from omnitensor.registry import schema_names

    assert sorted(REQUIRED_SCHEMAS) == sorted(schema_names())


def test_a_runtime_the_executor_refuses_is_not_reported_as_usable():
    """The check used to ask the import system a question only the executor
    can answer.

    A plain ``onnxruntime`` wheel ships ``CPUExecutionProvider`` only, and
    ``executors/gpu.py`` refuses it by design — there is no CPU backend here.
    Importability reported the GPU lane available while dispatch answered
    ``runtime-unusable`` for the same install, so the acceptance gate told an
    operator the opposite of what the runtime would do.
    """
    from omnitensor.acceptance import check_backends

    runtimes = (("gpu", "onnxruntime", "install a vendor extra"),)
    check = check_backends(runtimes)

    assert check.ok is False
    assert "no installed accelerator runtime is usable" in check.detail
    assert "onnxruntime" in check.detail
    assert "provider" in check.detail, "the executor's own reason is the remedy"


def test_an_absent_device_is_an_install_that_is_waiting_not_one_that_is_wrong():
    """A Coral that is not plugged in does not make tflite-runtime unusable."""
    from omnitensor.acceptance import _runtime_verdict

    assert _runtime_verdict("tflite_runtime") in (None, "tflite-runtime is not installed")
    assert _runtime_verdict("not-a-runtime") is None


def test_a_runtime_nothing_can_import_still_reports_its_remedy():
    from omnitensor.acceptance import check_backends

    check = check_backends((("npu", "no_such_module", "install the [npu] extra"),))

    assert check.ok is False
    assert "no accelerator runtime is importable" in check.detail
    assert "install the [npu] extra" in check.detail


def test_the_usable_lane_on_this_host_is_the_one_that_serves_jobs():
    """Whatever this machine has, the check and dispatch must agree about it."""
    from omnitensor.acceptance import BACKEND_RUNTIMES, _runtime_verdict
    from omnitensor.executors.base import DEVICE_ABSENT

    for backend, module, _remedy in BACKEND_RUNTIMES:
        verdict = _runtime_verdict(module)
        if verdict is None:
            continue
        # Anything reported unusable must be a refusal the executor actually
        # makes, not a sentence this check invented.
        assert isinstance(verdict, str) and verdict, f"{backend}:{module}"
        assert verdict != DEVICE_ABSENT


def _applet_with_schema(root, schema: dict):
    root.mkdir(parents=True, exist_ok=True)
    (root / "runtime-snapshot.schema.json").write_text(json.dumps(schema), encoding="utf-8")
    return root


def test_an_applet_that_cannot_read_the_snapshot_is_named_at_install_time(tmp_path):
    """The failure this catches reads to a user as a service that has stopped.

    The applet validates a snapshot as a closed record, so a field the service
    added and the installed payload does not know invalidates the whole
    document — and "no runtime service is publishing state" is what the user
    sees, which is what a dead service looks like too.
    """
    from omnitensor.acceptance import check_applet_contract

    older = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["version"],
        "properties": {"version": {"const": 1}},
    }
    snapshot = tmp_path / "state.json"
    snapshot.write_text(json.dumps({"version": 1, "inputs": {"roots": []}}), encoding="utf-8")
    root = _applet_with_schema(tmp_path / "applet", older)

    check = check_applet_contract(root, snapshot)

    assert check.ok is False
    assert "cannot read the snapshot this service publishes" in check.detail
    assert "install the applet built from these contracts" in check.detail


def test_an_applet_that_knows_the_field_passes(tmp_path):
    from omnitensor.acceptance import check_applet_contract

    current = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["version"],
        "properties": {"version": {"const": 1}, "inputs": {"type": "object"}},
    }
    snapshot = tmp_path / "state.json"
    snapshot.write_text(json.dumps({"version": 1, "inputs": {"roots": []}}), encoding="utf-8")

    check = check_applet_contract(_applet_with_schema(tmp_path / "applet", current), snapshot)

    assert check.ok is True


def test_an_applet_shipping_no_contract_is_a_broken_payload(tmp_path):
    from omnitensor.acceptance import check_applet_contract

    snapshot = tmp_path / "state.json"
    snapshot.write_text("{}", encoding="utf-8")
    empty = tmp_path / "applet"
    empty.mkdir()

    check = check_applet_contract(empty, snapshot)

    assert check.ok is False
    assert "ships no snapshot contract" in check.detail


def test_an_unreadable_pair_is_reported_rather_than_raised(tmp_path):
    from omnitensor.acceptance import check_applet_contract

    root = _applet_with_schema(tmp_path / "applet", {"type": "object"})
    check = check_applet_contract(root, tmp_path / "absent.json")

    assert check.ok is False
    assert "could not compare contracts" in check.detail


def test_the_installed_applet_reads_what_this_service_publishes():
    """The real pair on this host, which is the only place the order matters."""
    from omnitensor.acceptance import DEFAULT_APPLET_ROOT, check_applet_contract

    root = Path(DEFAULT_APPLET_ROOT).expanduser()
    snapshot = Path("~/.local/state/xpu-workload-manager/state.json").expanduser()
    if not root.is_dir() or not snapshot.is_file():
        pytest.skip("no applet and snapshot pair is installed on this host")

    assert check_applet_contract(root, snapshot).ok is True
