from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import omnitensor.sdk.smoke as smoke_module
from omnitensor.sdk import (
    MAX_SMOKE_DOCUMENT_BYTES,
    ManagedPlugin,
    PluginProgress,
    PluginSmokeError,
    cancelled_result,
    parse_smoke_document,
    run_installed_plugin_smoke,
    succeeded_result,
)


class SmokePlugin(ManagedPlugin):
    plugin_id = "smoke-plugin"

    async def execute(self, request, cancellation, progress):
        assert self.context.plugin_id == "smoke-plugin"
        assert self.context.protocol_version == 1
        assert self.context.permissions == frozenset()
        assert request.job_id == "installed-smoke-job"
        assert request.trigger == "manual"
        assert request.submitted_at_ms == 0
        assert request.deadline_at_ms == 30_000
        if cancellation.cancelled:
            return cancelled_result(request, "cancelled", completed_at_ms=0)
        await progress.report(PluginProgress(request.job_id, "execute", 0.5, "smoke", 0))
        return succeeded_result(
            request,
            {"value": request.payload["value"] * self.configuration.optional("scale", int, 1)},
            detail="smoke complete",
            completed_at_ms=0,
        )


class CancelledPlugin(SmokePlugin):
    async def execute(self, request, cancellation, progress):
        return cancelled_result(request, "fixture cancellation", completed_at_ms=0)


class FakeEntryPoint:
    def __init__(self, name="smoke-plugin", factory=SmokePlugin):
        self.name = name
        self._factory = factory
        self.load_calls = 0

    def load(self):
        self.load_calls += 1
        return self._factory


def test_installed_plugin_smoke_executes_a_complete_job():
    entry_point = FakeEntryPoint()
    selections = []

    async def scenario():
        return await run_installed_plugin_smoke(
            "smoke-plugin",
            {"value": 3},
            {"scale": 2},
            entry_points_provider=lambda **selection: selections.append(selection) or [entry_point],
        )

    result = asyncio.run(scenario())

    assert selections == [{"group": "omnitensor.workloads"}]
    assert entry_point.load_calls == 1
    assert result == {
        "pluginId": "smoke-plugin",
        "status": "succeeded",
        "output": {"value": 6},
        "detail": "smoke complete",
        "progress": [{"stage": "execute", "fraction": 0.5}],
    }


@pytest.mark.parametrize("plugin_id", [None, "", 1])
def test_smoke_validates_plugin_identity(plugin_id):
    with pytest.raises(ValueError) as error:
        asyncio.run(run_installed_plugin_smoke(plugin_id, {}))
    assert str(error.value) == "plugin_id must be a non-empty string"


def test_smoke_validates_mapping_boundaries():
    with pytest.raises(TypeError) as error:
        asyncio.run(run_installed_plugin_smoke("plugin", []))
    assert str(error.value) == "payload must be a mapping"
    with pytest.raises(TypeError) as error:
        asyncio.run(run_installed_plugin_smoke("plugin", {}, []))
    assert str(error.value) == "configuration must be a mapping"


def test_smoke_contains_entry_point_enumeration_and_identity_failures():
    def fail(**_selection):
        raise OSError("private")

    with pytest.raises(PluginSmokeError) as error:
        asyncio.run(run_installed_plugin_smoke("smoke-plugin", {}, entry_points_provider=fail))
    assert str(error.value) == "entry-point enumeration failed: OSError"

    for entries in ([], [FakeEntryPoint(), FakeEntryPoint()]):
        with pytest.raises(PluginSmokeError) as error:
            asyncio.run(
                run_installed_plugin_smoke(
                    "smoke-plugin",
                    {},
                    entry_points_provider=lambda entries=entries, **_selection: entries,
                )
            )
        assert str(error.value) == "plugin entry point is missing or ambiguous"


def test_smoke_contains_load_contract_and_terminal_failures():
    def fail_factory():
        raise RuntimeError("private")

    cases = [
        (FakeEntryPoint(factory=fail_factory), "plugin load failed: RuntimeError"),
        (
            FakeEntryPoint(factory=dict),
            "entry point does not implement the declared plugin",
        ),
        (
            FakeEntryPoint(factory=CancelledPlugin),
            "plugin smoke job ended with status cancelled",
        ),
    ]
    for entry_point, detail in cases:
        with pytest.raises(PluginSmokeError) as error:
            asyncio.run(
                run_installed_plugin_smoke(
                    "smoke-plugin",
                    {"value": 1},
                    entry_points_provider=lambda entry_point=entry_point, **_selection: [
                        entry_point
                    ],
                )
            )
        assert str(error.value) == detail

    with pytest.raises(PluginSmokeError, match="does not implement"):
        asyncio.run(
            run_installed_plugin_smoke(
                "other-plugin",
                {},
                entry_points_provider=lambda **_selection: [
                    FakeEntryPoint(name="other-plugin", factory=SmokePlugin)
                ],
            )
        )


def test_smoke_json_documents_are_finite_bounded_objects():
    assert parse_smoke_document('{"value":1}', "payload") == {"value": 1}
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_smoke_document("[]", "payload")
    with pytest.raises(ValueError, match="must be finite JSON"):
        parse_smoke_document('{"value":NaN}', "payload")
    with pytest.raises(ValueError, match="must be finite JSON"):
        parse_smoke_document("{", "payload")
    with pytest.raises(ValueError, match=f"exceeds {MAX_SMOKE_DOCUMENT_BYTES}"):
        parse_smoke_document(
            json.dumps({"value": "x" * MAX_SMOKE_DOCUMENT_BYTES}),
            "payload",
        )
    exact = '{"value":"' + "x" * (MAX_SMOKE_DOCUMENT_BYTES - 12) + '"}'
    assert len(exact.encode()) == MAX_SMOKE_DOCUMENT_BYTES
    assert parse_smoke_document(exact, "payload")["value"].startswith("x")


def test_smoke_cli_prints_stable_json(monkeypatch, capsys):
    async def run(plugin_id, payload, configuration):
        assert (plugin_id, payload, configuration) == (
            "smoke-plugin",
            {"value": 2},
            {"scale": 3},
        )
        return {"status": "succeeded", "output": {"value": 6}}

    monkeypatch.setattr(smoke_module, "run_installed_plugin_smoke", run)
    smoke_module.main(
        [
            "--plugin-id",
            "smoke-plugin",
            "--payload",
            '{"value":2}',
            "--configuration",
            '{"scale":3}',
        ]
    )
    assert capsys.readouterr().out == '{"output":{"value":6},"status":"succeeded"}\n'


def test_external_plugin_workflow_builds_real_wheels_and_runs_outside_checkout():
    workflow = Path(".github/workflows/external-plugin.yml").read_text()
    script = Path("scripts/run-external-plugin-smoke.sh").read_text()

    # The steps moved into the script so a developer can run the gate that
    # fails on them; the workflow's job is to call it.
    assert "./scripts/run-external-plugin-smoke.sh" in workflow
    assert "-m build --wheel --no-isolation" in script
    assert "examples/omnitensor-plugin-template" in script
    assert "-m venv" in script
    assert "omnitensor-plugin-smoke" in script
    assert "--plugin-id template-workload" in script
    # Still outside the checkout: an accidental relative import of the source
    # tree has to fail rather than pass quietly.
    assert 'cd "$work"' in script
