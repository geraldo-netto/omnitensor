from __future__ import annotations

import asyncio
import io
import json
import os
import stat

import pytest

from omnitensor.plugins.ipc import (
    HandshakeOffer,
    encode_frame,
    handshake_frame,
)
from omnitensor.plugins.protocol import PluginContext, PluginResult, WorkloadPlugin
from omnitensor.plugins.worker import serve_worker_requests
from omnitensor.plugins.worker_configuration import (
    CONFIGURATION_FILENAME,
    configuration_path,
    publish_worker_configuration,
    read_worker_configuration,
)


class RecordingPlugin(WorkloadPlugin):
    plugin_id = "external-example"

    def __init__(self) -> None:
        self.context: PluginContext | None = None

    async def start(self, context: PluginContext) -> None:
        self.context = context

    async def execute(self, request, cancellation, progress) -> PluginResult:  # pragma: no cover
        raise AssertionError("this suite never executes work")

    async def stop(self) -> None:
        return None


def serve(plugin, **kwargs):
    offer = HandshakeOffer("external-example", 1, 1, frozenset({"execute"}))
    reader = io.BytesIO(encode_frame(handshake_frame(offer)))
    return asyncio.run(serve_worker_requests(plugin, reader, io.BytesIO(), **kwargs))


def test_the_worker_starts_with_the_configuration_it_was_given():
    plugin = RecordingPlugin()
    serve(plugin, configuration={"guidance": "cite sources"})
    assert plugin.context.configuration == {"guidance": "cite sources"}


def test_a_worker_given_nothing_starts_with_an_empty_configuration():
    plugin = RecordingPlugin()
    serve(plugin)
    assert plugin.context.configuration == {}


def test_the_document_is_private_to_the_plugin_that_reads_it(tmp_path):
    """A tunable is free text somebody typed; the file mode is what protects it."""
    publish_worker_configuration(tmp_path, {"guidance": "personal"})
    path = configuration_path(tmp_path)
    assert path.name == CONFIGURATION_FILENAME
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text(encoding="utf-8")) == {"guidance": "personal"}


def test_what_was_written_is_what_is_read_back(tmp_path):
    publish_worker_configuration(tmp_path, {"guidance": "cite", "length": 4})
    assert read_worker_configuration(tmp_path) == {"guidance": "cite", "length": 4}


def test_clearing_a_configuration_removes_the_document(tmp_path):
    """A cleared setting must not survive as the file the next worker reads."""
    publish_worker_configuration(tmp_path, {"guidance": "cite"})
    publish_worker_configuration(tmp_path, {})
    assert not configuration_path(tmp_path).exists()
    assert read_worker_configuration(tmp_path) == {}


def test_removing_a_configuration_that_was_never_written_is_not_an_error(tmp_path):
    publish_worker_configuration(tmp_path, None)
    assert not configuration_path(tmp_path).exists()


def test_a_worker_with_no_state_directory_has_no_configuration_channel():
    assert read_worker_configuration(None) == {}
    publish_worker_configuration(None, {"guidance": "ignored"})


@pytest.mark.parametrize(
    "content",
    [b"not json", b"[]", b'"text"', b"3", b"null", b""],
)
def test_a_document_that_is_not_an_object_reads_as_no_configuration(tmp_path, content):
    """A workload lost to an unreadable tunable would be the worse answer."""
    configuration_path(tmp_path).write_bytes(content)
    assert read_worker_configuration(tmp_path) == {}


def test_a_document_larger_than_the_store_would_hold_reads_as_no_configuration(tmp_path):
    from omnitensor.plugins.settings import DEFAULT_MAX_SETTINGS_BYTES

    oversized = {"guidance": "x" * (DEFAULT_MAX_SETTINGS_BYTES + 1)}
    configuration_path(tmp_path).write_text(json.dumps(oversized), encoding="utf-8")
    assert read_worker_configuration(tmp_path) == {}


def test_the_configuration_never_reaches_the_command_line(tmp_path):
    """`/proc/<pid>/cmdline` is world-readable, so an argument would publish it."""
    from omnitensor.plugins.worker_specs import external_worker_spec

    manifest = {
        "plugin": {
            "protocol": {"minimum": 1, "maximum": 1, "capabilities": []},
            "permissions": [],
            "artifacts": [],
            "schemas": {"configuration": {}, "input": {}, "output": {}},
        }
    }

    from omnitensor.plugins.discovery import PluginSource
    from omnitensor.plugins.identity import ResolvedPlugin

    plugin = ResolvedPlugin(
        "external-example",
        "1.0.0",
        PluginSource.EXTERNAL,
        "omnitensor-external-example",
        "1.0.0",
        "external-example",
        "module:create",
        tmp_path / "manifest.json",
        manifest,
    )
    state_root = tmp_path / "state"
    state_root.mkdir(mode=0o700)
    spec = external_worker_spec(
        plugin,
        executable="/usr/bin/python3",
        import_paths=(),
        granted=frozenset(),
        selected_files_root=None,
        worker_state_root=state_root,
        resolve_artifact=None,
        accelerator_devices={},
        configuration={"guidance": "personal"},
    )
    assert not any("personal" in argument for argument in spec.argv)
    state_path = state_root / "external-example"
    assert read_worker_configuration(state_path) == {"guidance": "personal"}
    assert stat.S_IMODE(os.stat(state_path).st_mode) == 0o700


def test_the_runtime_reads_the_store_it_was_given_rather_than_asking_its_host(tmp_path):
    """OMNI-0552: the host callback could only ask this object for the contract.

    A workload's spec comes from a manifest this runtime owns, so a host asked
    for the values had to call back into it for the contract first — a cycle
    that had to be broken before either object could move to the composition
    root.
    """
    from omnitensor.plugins import PluginSettingsStore, manifest_configuration_spec
    from omnitensor.plugins.loading import InstalledPluginRuntime

    manifest = {
        "plugin": {
            "schemas": {
                "configuration": {
                    "type": "object",
                    "properties": {"guidance": {"type": "string", "default": ""}},
                },
                "input": {},
                "output": {},
            }
        }
    }
    store = PluginSettingsStore(tmp_path / "settings")
    store.update(
        manifest_configuration_spec("external-example", manifest),
        expected_revision=0,
        configuration={"guidance": "cite sources"},
    )
    runtime = InstalledPluginRuntime(tmp_path / "bundled", settings_store=store)
    runtime.configuration_spec = lambda plugin_id: manifest_configuration_spec(plugin_id, manifest)

    assert runtime.stored_configuration("external-example") == {"guidance": "cite sources"}


def test_a_runtime_with_no_store_starts_its_workers_untuned(tmp_path):
    from omnitensor.plugins.loading import InstalledPluginRuntime

    runtime = InstalledPluginRuntime(tmp_path / "bundled")

    assert runtime.stored_configuration("external-example") is None


def test_a_store_that_refuses_starts_the_worker_on_the_manifests_defaults(tmp_path, caplog):
    """A workload lost to an unreadable tunable would be the worse answer."""
    import logging

    from omnitensor.plugins import manifest_configuration_spec
    from omnitensor.plugins.loading import InstalledPluginRuntime

    manifest = {"plugin": {"schemas": {"configuration": {}, "input": {}, "output": {}}}}

    class Refusing:
        def load(self, _spec):
            raise OSError("unreadable")

    runtime = InstalledPluginRuntime(tmp_path / "bundled", settings_store=Refusing())
    runtime.configuration_spec = lambda plugin_id: manifest_configuration_spec(plugin_id, manifest)

    with caplog.at_level(logging.WARNING, logger="omnitensor.plugins.loading"):
        assert runtime.stored_configuration("external-example") is None
    assert "starts on the" in caplog.text
