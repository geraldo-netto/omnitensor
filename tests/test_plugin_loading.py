from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from conftest import sample_plugin_manifest

from omnitensor.discovery import Device
from omnitensor.plugins import (
    DEFAULT_MAX_FRAME_BYTES,
    MAX_WORKER_IMPORT_PATHS,
    HandshakeOffer,
    InstalledPluginRuntime,
    IPCProtocolError,
    PluginSource,
    ResolvedPlugin,
    WorkerMessageType,
    WorkerState,
    decode_frame,
    encode_frame,
    entry_points_from_distributions,
    external_worker_specs,
    handshake_frame,
)
from omnitensor.plugins import worker as worker_module
from omnitensor.plugins.protocol import (
    PluginHealth,
    PluginHealthStatus,
    WorkloadPlugin,
)
from omnitensor.plugins.worker import (
    ExternalPluginLoadError,
    _read_exact,
    _read_frame,
    load_external_plugin,
    serve_worker,
)
from omnitensor.service import OmniTensorService


class TrackingPlugin:
    plugin_id = "external-example"

    def __init__(self):
        self.events = []

    async def start(self, context):
        self.events.append(("start", context))

    async def health(self):
        return PluginHealth(PluginHealthStatus.READY, "ready", 0)

    async def stop(self):
        self.events.append(("stop", None))

    async def execute(self, request, cancellation, progress):
        raise AssertionError("worker bootstrap must not execute jobs")


class FakeDistribution:
    def __init__(self, name="external-dist"):
        self.name = name


class FakeEntryPoint:
    def __init__(
        self,
        *,
        name="external-example",
        value="external_package:Plugin",
        distribution="external-dist",
        factory=TrackingPlugin,
    ):
        self.name = name
        self.value = value
        self.dist = FakeDistribution(distribution) if distribution is not None else None
        self._factory = factory
        self.load_calls = 0

    def load(self):
        self.load_calls += 1
        return self._factory


def _resolved(source=PluginSource.EXTERNAL, tmp_path=Path("/tmp")):
    manifest = sample_plugin_manifest("external-example")
    manifest["plugin"]["protocol"] = {
        "minimum": 2,
        "maximum": 4,
        "capabilities": ["cancel", "health", "progress"],
    }
    return ResolvedPlugin(
        "external-example",
        "1.0.0",
        source,
        "external-dist",
        "1.0.0",
        "external-example",
        "external_package:Plugin" if source is PluginSource.EXTERNAL else None,
        tmp_path / "omnitensor-plugin.json",
        manifest,
    )


def _frames(data: bytes):
    frames = []
    offset = 0
    while offset < len(data):
        size = int.from_bytes(data[offset : offset + 4], "big")
        end = offset + 4 + size
        frames.append(decode_frame(data[offset:end]))
        offset = end
    return frames


def test_external_worker_specs_are_deterministic_and_do_not_import_plugins(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    external = _resolved(tmp_path=tmp_path)
    bundled = _resolved(PluginSource.BUNDLED, tmp_path)

    specs = external_worker_specs(
        (bundled, external),
        python_executable="/usr/bin/python3",
        worker_import_paths=(site,),
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec.plugin_id == "external-example"
    assert spec.minimum_protocol == 2
    assert spec.maximum_protocol == 4
    assert spec.capabilities == frozenset({"cancel", "health"})
    assert spec.sandbox is not None
    assert spec.sandbox.python_path == str(
        Path(__file__).resolve().parents[1] / "src"
    )
    assert spec.argv == (
        "/usr/bin/python3",
        "-m",
        "omnitensor.plugins.worker",
        "--plugin-id",
        "external-example",
        "--entry-point",
        "external-example",
        "--target",
        "external_package:Plugin",
        "--distribution",
        "external-dist",
        "--import-path",
        str(site),
    )


def test_external_worker_specs_pass_only_granted_declared_permissions(tmp_path):
    site = tmp_path / "site"
    source = tmp_path / "source"
    site.mkdir()
    source.mkdir()
    plugin = _resolved(tmp_path=tmp_path)
    permission = f"read:{source}"
    plugin.manifest["plugin"]["permissions"] = [permission]

    [spec] = external_worker_specs(
        (plugin,),
        worker_import_paths=(site,),
        granted_permissions={plugin.plugin_id: {permission}},
    )

    assert spec.argv[-2:] == ("--permission", permission)
    assert spec.sandbox is not None
    assert spec.sandbox.read_paths == (str(source),)


def test_external_worker_specs_fail_closed_on_undeclared_grant(tmp_path):
    plugin = _resolved(tmp_path=tmp_path)
    with pytest.raises(ValueError, match="undeclared"):
        external_worker_specs(
            (plugin,),
            granted_permissions={plugin.plugin_id: {f"read:{tmp_path}"}},
        )


def test_worker_spec_inputs_are_bounded(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    plugin = _resolved(tmp_path=tmp_path)

    with pytest.raises(ValueError, match="python_executable"):
        external_worker_specs((plugin,), python_executable="")
    with pytest.raises(ValueError, match="python_executable"):
        external_worker_specs((plugin,), python_executable="bad\0path")
    with pytest.raises(ValueError, match="absolute directories"):
        external_worker_specs((plugin,), worker_import_paths=(Path("relative"),))
    with pytest.raises(ValueError, match="absolute directories"):
        external_worker_specs((plugin,), worker_import_paths=(tmp_path / "missing",))
    with pytest.raises(ValueError, match="unique"):
        external_worker_specs((plugin,), worker_import_paths=(site, site))
    with pytest.raises(ValueError, match=f"at most {MAX_WORKER_IMPORT_PATHS}"):
        external_worker_specs(
            (plugin,),
            worker_import_paths=(site,) * (MAX_WORKER_IMPORT_PATHS + 1),
        )


def test_worker_loads_only_the_exact_validated_entry_point():
    entry_point = FakeEntryPoint()
    plugin = load_external_plugin(
        "external-example",
        "external-example",
        "external_package:Plugin",
        "external-dist",
        entry_points_provider=lambda **selection: [entry_point]
        if selection == {"group": "omnitensor.workloads"}
        else [],
    )

    assert isinstance(plugin, TrackingPlugin)
    assert entry_point.load_calls == 1


@pytest.mark.parametrize(
    "entry_point",
    [
        FakeEntryPoint(name="substitute"),
        FakeEntryPoint(value="other:Plugin"),
        FakeEntryPoint(distribution="other-dist"),
        FakeEntryPoint(distribution=None),
    ],
)
def test_worker_rejects_changed_installed_identity(entry_point):
    with pytest.raises(ExternalPluginLoadError) as error:
        load_external_plugin(
            "external-example",
            "external-example",
            "external_package:Plugin",
            "external-dist",
            entry_points_provider=lambda **_selection: [entry_point],
        )
    assert str(error.value) == "installed entry-point identity changed"
    assert entry_point.load_calls == 0


def test_worker_contains_enumeration_load_and_contract_failures():
    def fail_enumeration(**_selection):
        raise OSError("private")

    def fail_factory():
        raise RuntimeError("private")

    with pytest.raises(ExternalPluginLoadError, match="enumeration failed: OSError"):
        load_external_plugin("p", "p", "p:P", "p", entry_points_provider=fail_enumeration)
    with pytest.raises(ExternalPluginLoadError, match="load failed: RuntimeError"):
        load_external_plugin(
            "external-example",
            "external-example",
            "external_package:Plugin",
            "external-dist",
            entry_points_provider=lambda **_selection: [
                FakeEntryPoint(factory=fail_factory)
            ],
        )
    with pytest.raises(ExternalPluginLoadError) as error:
        load_external_plugin(
            "external-example",
            "external-example",
            "external_package:Plugin",
            "external-dist",
            entry_points_provider=lambda **_selection: [FakeEntryPoint(factory=dict)],
        )
    assert str(error.value) == "entry point does not implement the declared plugin"
    duplicate = FakeEntryPoint()
    with pytest.raises(ExternalPluginLoadError, match="identity changed"):
        load_external_plugin(
            "external-example",
            "external-example",
            "external_package:Plugin",
            "external-dist",
            entry_points_provider=lambda **_selection: [duplicate, duplicate],
        )


def test_worker_acknowledges_the_handshake_before_starting_the_plugin():
    service_offer = HandshakeOffer(
        "external-example", 1, 3, frozenset({"cancel", "progress"})
    )
    health = encode_frame(
        handshake_frame(service_offer).__class__(
            1, WorkerMessageType.HEALTH, "request-1", {}
        )
    )
    cancel = encode_frame(
        handshake_frame(service_offer).__class__(
            1, WorkerMessageType.CANCEL, None, {"reason": "shutdown"}
        )
    )
    reader = io.BytesIO(encode_frame(handshake_frame(service_offer)) + health + cancel)
    writer = io.BytesIO()
    plugin = TrackingPlugin()

    agreement = serve_worker(
        plugin,
        reader,
        writer,
        minimum_protocol=2,
        maximum_protocol=4,
    )

    assert agreement.protocol_version == 3
    assert agreement.capabilities == frozenset({"cancel"})
    assert [event[0] for event in plugin.events] == ["start", "stop"]
    assert plugin.events[0][1].protocol_version == 3
    assert plugin.events[0][1].plugin_id == "external-example"
    assert plugin.events[0][1].configuration == {}
    assert plugin.events[0][1].permissions == frozenset()
    output = _frames(writer.getvalue())
    assert [frame.type for frame in output] == [
        WorkerMessageType.HELLO,
        WorkerMessageType.READY,
        WorkerMessageType.ERROR,
    ]
    assert output[1].payload == {"pluginId": "external-example"}
    assert output[2].request_id == "request-1"
    assert output[2].payload == {
        "code": "unsupported-message",
        "detail": "message is not implemented",
    }


def test_worker_receives_only_the_service_supplied_active_permissions():
    service_offer = HandshakeOffer("external-example", 1, 1, frozenset())
    reader = io.BytesIO(encode_frame(handshake_frame(service_offer)))
    plugin = TrackingPlugin()

    serve_worker(
        plugin,
        reader,
        io.BytesIO(),
        permissions=frozenset({"read:/allowed"}),
    )

    assert plugin.events[0][1].permissions == frozenset({"read:/allowed"})


def test_worker_default_protocol_and_eof_shutdown():
    service_offer = HandshakeOffer(
        "external-example", 1, 1, frozenset({"cancel", "health"})
    )
    plugin = TrackingPlugin()
    writer = io.BytesIO()

    agreement = serve_worker(
        plugin,
        io.BytesIO(encode_frame(handshake_frame(service_offer))),
        writer,
    )

    assert agreement.protocol_version == 1
    assert [event[0] for event in plugin.events] == ["start", "stop"]
    assert [frame.type for frame in _frames(writer.getvalue())] == [
        WorkerMessageType.HELLO,
        WorkerMessageType.READY,
    ]


class _ChunkedReader:
    def __init__(self, chunks):
        self._chunks = iter(chunks)
        self.requests = []

    def read(self, size):
        self.requests.append(size)
        return next(self._chunks, b"")


def test_worker_blocking_frame_reader_bounds_and_reports_truncation():
    reader = _ChunkedReader((b"a", b"bc", b"d"))
    assert _read_exact(reader, 4) == b"abcd"
    assert reader.requests == [4, 3, 1]
    with pytest.raises(EOFError):
        _read_exact(io.BytesIO(), 1)

    truncated = _ChunkedReader((b"a", b""))
    with pytest.raises(IPCProtocolError) as error:
        _read_exact(truncated, 2)
    assert error.value.code == "truncated-frame"
    assert error.value.detail == "expected 2 bytes; received 1"

    oversized = (DEFAULT_MAX_FRAME_BYTES + 1).to_bytes(4, "big")
    with pytest.raises(IPCProtocolError) as error:
        _read_frame(io.BytesIO(oversized))
    assert error.value.code == "frame-too-large"
    assert error.value.detail == (
        f"declared {DEFAULT_MAX_FRAME_BYTES + 1} bytes; "
        f"limit is {DEFAULT_MAX_FRAME_BYTES}"
    )


def test_worker_main_parses_identity_and_import_roots(monkeypatch, tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    plugin = TrackingPlugin()
    loaded = []
    served = []

    def load(*identity):
        loaded.append(identity)
        return plugin

    def serve(candidate, reader, writer, **options):
        served.append((candidate, reader, writer, options))

    stdin = type("Input", (), {"buffer": io.BytesIO()})()
    channel = io.BytesIO()
    claims = []
    monkeypatch.setattr(worker_module, "load_external_plugin", load)
    monkeypatch.setattr(worker_module, "serve_worker", serve)
    monkeypatch.setattr(
        worker_module,
        "claim_frame_channel",
        lambda: claims.append(len(loaded)) or channel,
    )
    monkeypatch.setattr(worker_module.sys, "stdin", stdin)
    monkeypatch.setattr(worker_module.sys, "path", list(sys.path))

    # --no-seccomp: installing the real filter here would confine the test
    # process itself, and a seccomp filter cannot be removed once applied.
    worker_module.main(
        [
            "--no-seccomp",
            "--plugin-id",
            "external-example",
            "--entry-point",
            "external-example",
            "--target",
            "external_package:Plugin",
            "--distribution",
            "external-dist",
            "--import-path",
            str(site),
        ]
    )

    assert loaded == [
        (
            "external-example",
            "external-example",
            "external_package:Plugin",
            "external-dist",
        )
    ]
    assert served == [(plugin, stdin.buffer, channel, {"permissions": frozenset()})]
    assert claims == [0], "the frame channel must be claimed before the plugin is loaded"
    assert worker_module.sys.path[0] == str(site)


def _write_external_wheel(path: Path) -> None:
    manifest = sample_plugin_manifest("third-party-plugin")
    manifest["version"] = "1.0.0"
    package = """\
from omnitensor.sdk import ManagedPlugin, cancelled_result

class ThirdPartyPlugin(ManagedPlugin):
    plugin_id = "third-party-plugin"

    async def execute(self, request, cancellation, progress):
        return cancelled_result(request, "fixture")
"""
    dist_info = "third_party_omnitensor-1.0.0.dist-info"
    files = {
        "third_party_plugin/__init__.py": package,
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\nName: third-party-omnitensor\nVersion: 1.0.0\n"
        ),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: omnitensor-test\n"
            "Root-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        f"{dist_info}/entry_points.txt": (
            "[omnitensor.workloads]\n"
            "third-party-plugin = third_party_plugin:ThirdPartyPlugin\n"
        ),
        "omnitensor-plugin.json": json.dumps(manifest),
    }
    record = "".join(f"{name},,\n" for name in (*files, f"{dist_info}/RECORD"))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for name, contents in files.items():
            wheel.writestr(name, contents)
        wheel.writestr(f"{dist_info}/RECORD", record)


class _Discovery:
    def detect(self):
        return [Device("tpu-pcie-0", "tpu", "Test TPU", "pcie")]

    def utilization(self, _device):
        return None


class _Publisher:
    def publish(self, _snapshot):
        return None

    def retract(self):
        return None


class _Transport:
    async def start(self, _handler):
        return None

    async def stop(self):
        return None


def test_installed_wheel_is_discovered_and_loaded_after_service_restart(tmp_path):
    wheel = tmp_path / "third_party_omnitensor-1.0.0-py3-none-any.whl"
    site = tmp_path / "site"
    workloads = tmp_path / "workloads"
    site.mkdir()
    workloads.mkdir()
    _write_external_wheel(wheel)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-compile",
            "--no-deps",
            "--no-index",
            "--target",
            str(site),
            str(wheel),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    provider = entry_points_from_distributions((site,))

    async def run_once(index):
        runtime = InstalledPluginRuntime(
            workloads,
            entry_points_provider=provider,
            worker_import_paths=(site,),
            python_executable=sys.executable,
        )
        assert runtime.snapshot.catalog.plugins == ()
        assert runtime.snapshot.workers == ()
        service = OmniTensorService(
            tmp_path / f"snapshot-{index}.json",
            tmp_path / f"policy-{index}.json",
            workloads,
            discovery=_Discovery(),
            publisher=_Publisher(),
            transport=_Transport(),
            plugin_runtime=runtime,
            publish_interval_s=0.01,
            discovery_interval_s=60,
        )
        runner = asyncio.create_task(service.run())
        try:
            for _ in range(200):
                if runtime.snapshot.workers:
                    break
                await asyncio.sleep(0.005)
            started = runtime.snapshot
            assert started.workers
        finally:
            service._stopping.set()
            await asyncio.wait_for(runner, timeout=2)
        return started, runtime.snapshot

    async def scenario():
        return await run_once(1), await run_once(2)

    first, restarted = asyncio.run(scenario())
    for started, stopped in (first, restarted):
        external = [
            plugin
            for plugin in started.catalog.plugins
            if plugin.source is PluginSource.EXTERNAL
        ]
        assert [plugin.plugin_id for plugin in external] == ["third-party-plugin"]
        assert started.catalog.rejections == ()
        assert len(started.workers) == 1
        assert started.workers[0].state is WorkerState.READY
        assert stopped.workers[0].state is WorkerState.STOPPED


def test_worker_rejects_a_plugin_that_omits_its_identity():
    class Anonymous:
        async def start(self, context): ...

        async def health(self): ...

        async def stop(self): ...

        async def execute(self, request, cancellation, progress): ...

    assert isinstance(Anonymous(), WorkloadPlugin)
    with pytest.raises(ExternalPluginLoadError) as error:
        load_external_plugin(
            "external-example",
            "external-example",
            "external_package:Plugin",
            "external-dist",
            entry_points_provider=lambda **_selection: [FakeEntryPoint(factory=Anonymous)],
        )
    assert str(error.value) == "entry point does not implement the declared plugin"


_CHATTY_PLUGIN = '''\
import sys

print("noise from import", flush=True)
sys.stdout.write("more noise\\n")

from omnitensor.plugins.protocol import PluginHealth, PluginHealthStatus


class Plugin:
    plugin_id = "chatty-plugin"

    async def start(self, context):
        print("noise from start", flush=True)

    async def health(self):
        return PluginHealth(PluginHealthStatus.READY, "ready", 0)

    async def stop(self):
        print("noise from stop", flush=True)

    async def execute(self, request, cancellation, progress):
        raise AssertionError("worker bootstrap must not execute jobs")
'''

_CHATTY_BOOTSTRAP = '''\
from omnitensor.plugins import worker


def _load(plugin_id, entry_point, target, distribution):
    import chatty_plugin

    return chatty_plugin.Plugin()


worker.load_external_plugin = _load
worker.main(
    [
        "--plugin-id",
        "chatty-plugin",
        "--entry-point",
        "chatty-plugin",
        "--target",
        "chatty_plugin:Plugin",
        "--distribution",
        "chatty-dist",
    ]
)
'''


def test_plugin_stdout_never_reaches_the_frame_channel(tmp_path):
    """A single stray print used to desynchronise the service's frame reader."""
    (tmp_path / "chatty_plugin.py").write_text(_CHATTY_PLUGIN, encoding="utf-8")
    (tmp_path / "bootstrap.py").write_text(_CHATTY_BOOTSTRAP, encoding="utf-8")
    offer = HandshakeOffer("chatty-plugin", 1, 1, frozenset({"cancel"}))

    worker = subprocess.Popen(
        [sys.executable, str(tmp_path / "bootstrap.py")],
        cwd=tmp_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
    )
    try:
        worker.stdin.write(encode_frame(handshake_frame(offer)))
        worker.stdin.flush()
        agreement = _read_frame(worker.stdout)
        ready = _read_frame(worker.stdout)
        worker.stdin.close()
        stdout_tail = worker.stdout.read()
        stderr = worker.stderr.read().decode()
    finally:
        worker.kill()
        worker.wait(timeout=30)

    assert agreement.payload["pluginId"] == "chatty-plugin"
    assert ready.type is WorkerMessageType.READY
    assert stdout_tail == b"", "plugin output leaked into the frame channel"
    assert "noise from import" in stderr
    assert "noise from start" in stderr


def test_worker_answers_the_handshake_before_its_plugin_starts():
    """Startup time used to be charged to the service's handshake deadline."""
    service_offer = HandshakeOffer("external-example", 1, 1, frozenset({"cancel"}))
    writer = io.BytesIO()

    class ObservantPlugin(TrackingPlugin):
        frames_at_start = None

        async def start(self, context):
            type(self).frames_at_start = _frames(writer.getvalue())
            await super().start(context)

    serve_worker(
        ObservantPlugin(),
        io.BytesIO(encode_frame(handshake_frame(service_offer))),
        writer,
    )

    assert [frame.type for frame in ObservantPlugin.frames_at_start] == [
        WorkerMessageType.HELLO
    ]
    assert [frame.type for frame in _frames(writer.getvalue())] == [
        WorkerMessageType.HELLO,
        WorkerMessageType.READY,
    ]


def test_the_worker_confines_itself_before_it_imports_the_plugin(monkeypatch, tmp_path):
    """Importing a plugin already runs its code; a later filter is too late."""
    order = []

    def install():
        order.append("install")
        return 15

    def load(*identity):
        order.append("load")
        return TrackingPlugin()

    monkeypatch.setattr(worker_module, "install_filter", install)
    monkeypatch.setattr(worker_module, "load_external_plugin", load)
    monkeypatch.setattr(worker_module, "serve_worker", lambda *a, **k: order.append("serve"))
    monkeypatch.setattr(
        worker_module, "claim_frame_channel", lambda: io.BytesIO()
    )
    monkeypatch.setattr(
        worker_module.sys, "stdin", type("Input", (), {"buffer": io.BytesIO()})()
    )

    worker_module.main(
        [
            "--plugin-id",
            "external-example",
            "--entry-point",
            "external-example",
            "--target",
            "external_package:Plugin",
            "--distribution",
            "external-dist",
        ]
    )

    assert order == ["install", "load", "serve"]


def test_a_worker_told_not_to_confine_itself_does_not(monkeypatch):
    """Only for kernels that cannot install one; it is never the default."""
    installed = []
    monkeypatch.setattr(worker_module, "install_filter", lambda: installed.append(1))
    monkeypatch.setattr(worker_module, "load_external_plugin", lambda *a: TrackingPlugin())
    monkeypatch.setattr(worker_module, "serve_worker", lambda *a, **k: None)
    monkeypatch.setattr(worker_module, "claim_frame_channel", lambda: io.BytesIO())
    monkeypatch.setattr(
        worker_module.sys, "stdin", type("Input", (), {"buffer": io.BytesIO()})()
    )

    worker_module.main(
        [
            "--no-seccomp",
            "--plugin-id",
            "external-example",
            "--entry-point",
            "external-example",
            "--target",
            "external_package:Plugin",
            "--distribution",
            "external-dist",
        ]
    )

    assert installed == []
