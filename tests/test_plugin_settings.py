from __future__ import annotations

import json
import multiprocessing
import tempfile
import time
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins import (
    MAX_MIGRATIONS,
    ConfigurationMigration,
    PluginConfigurationSpec,
    PluginSettings,
    PluginSettingsError,
    PluginSettingsStore,
)

SCHEMA = {
    "type": "object",
    "properties": {
        "mode": {"enum": ["safe", "fast"]},
        "threshold": {"type": "integer", "minimum": 0, "maximum": 10},
        "label": {"type": "string", "maxLength": 20},
    },
    "required": ["mode", "threshold"],
    "additionalProperties": False,
}


def spec(
    version="1.0.0",
    *,
    schema=SCHEMA,
    defaults=None,
    migrations=(),
    plugin_id="echo-plugin",
):
    return PluginConfigurationSpec(
        plugin_id,
        version,
        schema,
        defaults or {"mode": "safe", "threshold": 2},
        tuple(migrations),
    )


def settings_path(tmp_path):
    return tmp_path / "echo-plugin.json"


def write_settings(
    tmp_path,
    *,
    version="1.0.0",
    revision=1,
    configuration=None,
    **overrides,
):
    document = {
        "documentVersion": 1,
        "pluginId": "echo-plugin",
        "pluginVersion": version,
        "revision": revision,
        "configuration": configuration or {"mode": "safe", "threshold": 2},
    }
    document.update(overrides)
    settings_path(tmp_path).write_text(json.dumps(document))
    return document


def test_missing_settings_return_deterministic_defaults_without_creating_state(tmp_path):
    defaults = {"mode": "safe", "threshold": 2}
    plugin_spec = spec(defaults=defaults)
    store = PluginSettingsStore(tmp_path)

    first = store.load(plugin_spec)
    first.configuration["threshold"] = 9
    second = store.load(plugin_spec)

    assert first == PluginSettings("echo-plugin", "1.0.0", 0, {"mode": "safe", "threshold": 9})
    assert second == PluginSettings("echo-plugin", "1.0.0", 0, {"mode": "safe", "threshold": 2})
    assert not settings_path(tmp_path).exists()


def test_update_validates_and_atomically_persists_a_new_revision(tmp_path):
    store = PluginSettingsStore(tmp_path)
    updated = store.update(
        spec(),
        expected_revision=0,
        configuration={"mode": "fast", "threshold": 7},
    )

    assert updated == PluginSettings(
        "echo-plugin",
        "1.0.0",
        1,
        {"mode": "fast", "threshold": 7},
    )
    assert store.load(spec()) == updated
    assert json.loads(settings_path(tmp_path).read_text()) == {
        "documentVersion": 1,
        "pluginId": "echo-plugin",
        "pluginVersion": "1.0.0",
        "revision": 1,
        "configuration": {"mode": "fast", "threshold": 7},
    }


def test_revision_mismatch_preserves_the_committed_document(tmp_path):
    store = PluginSettingsStore(tmp_path)
    store.update(spec(), expected_revision=0, configuration={"mode": "safe", "threshold": 3})
    before = settings_path(tmp_path).read_bytes()

    with pytest.raises(PluginSettingsError) as excinfo:
        store.update(
            spec(),
            expected_revision=0,
            configuration={"mode": "fast", "threshold": 4},
        )

    assert str(excinfo.value) == "revision-mismatch: expected 0; current revision is 1"
    assert settings_path(tmp_path).read_bytes() == before


@pytest.mark.parametrize(
    "configuration",
    [
        [],
        {},
        {"mode": "turbo", "threshold": 2},
        {"mode": "safe", "threshold": -1},
        {"mode": "safe", "threshold": 2, "unknown": True},
    ],
)
def test_invalid_configuration_is_rejected_before_persistence(tmp_path, configuration):
    store = PluginSettingsStore(tmp_path)
    with pytest.raises(PluginSettingsError) as excinfo:
        store.update(spec(), expected_revision=0, configuration=configuration)
    assert excinfo.value.code == "invalid-configuration"
    assert not settings_path(tmp_path).exists()


def test_configuration_must_remain_json_serializable_after_schema_validation(tmp_path):
    open_spec = spec(schema={"type": "object"}, defaults={})
    with pytest.raises(PluginSettingsError) as excinfo:
        PluginSettingsStore(tmp_path).update(
            open_spec,
            expected_revision=0,
            configuration={"value": object()},
        )
    assert excinfo.value.code == "invalid-configuration"
    assert not settings_path(tmp_path).exists()


def test_schema_and_defaults_are_validated_before_state_access(tmp_path):
    store = PluginSettingsStore(tmp_path)
    with pytest.raises(PluginSettingsError) as non_object:
        store.load(spec(schema=[]))
    assert non_object.value.code == "invalid-schema"

    with pytest.raises(PluginSettingsError) as bad_schema:
        store.load(spec(schema={"type": "not-a-json-schema-type"}))
    assert bad_schema.value.code == "invalid-schema"

    with pytest.raises(PluginSettingsError) as bad_defaults:
        store.load(spec(defaults={"mode": "unsafe", "threshold": 2}))
    assert bad_defaults.value.code == "invalid-configuration"


def test_upgrade_applies_an_explicit_multi_step_route_and_commits_once(tmp_path):
    store = PluginSettingsStore(tmp_path)
    original = store.update(
        spec("1.0.0"),
        expected_revision=0,
        configuration={"mode": "safe", "threshold": 3},
    )
    seen = []

    def one_to_two(configuration):
        seen.append(("1.0.0", dict(configuration)))
        configuration["threshold"] += 1
        return configuration

    def two_to_three(configuration):
        seen.append(("2.0.0", dict(configuration)))
        return {**configuration, "label": "migrated"}

    upgraded_spec = spec(
        "3.0.0",
        migrations=(
            ConfigurationMigration("1.0.0", "2.0.0", one_to_two),
            ConfigurationMigration("2.0.0", "3.0.0", two_to_three),
        ),
    )

    upgraded = store.load(upgraded_spec)

    assert original.configuration == {"mode": "safe", "threshold": 3}
    assert upgraded == PluginSettings(
        "echo-plugin",
        "3.0.0",
        2,
        {"mode": "safe", "threshold": 4, "label": "migrated"},
    )
    assert seen == [
        ("1.0.0", {"mode": "safe", "threshold": 3}),
        ("2.0.0", {"mode": "safe", "threshold": 4}),
    ]
    assert store.load(upgraded_spec) == upgraded
    assert json.loads(settings_path(tmp_path).read_text())["pluginVersion"] == "3.0.0"


@pytest.mark.parametrize(
    ("target_version", "migrations", "code"),
    [
        ("2.0.0", (), "migration-missing"),
        (
            "3.0.0",
            (
                ConfigurationMigration("1.0.0", "2.0.0", lambda value: value),
                ConfigurationMigration("2.0.0", "1.0.0", lambda value: value),
            ),
            "migration-cycle",
        ),
        (
            "2.0.0",
            (
                ConfigurationMigration(
                    "1.0.0",
                    "2.0.0",
                    lambda _value: (_ for _ in ()).throw(RuntimeError("private")),
                ),
            ),
            "migration-failed",
        ),
        (
            "2.0.0",
            (ConfigurationMigration("1.0.0", "2.0.0", lambda _value: []),),
            "migration-failed",
        ),
        (
            "2.0.0",
            (
                ConfigurationMigration(
                    "1.0.0",
                    "2.0.0",
                    lambda value: {**value, "threshold": -1},
                ),
            ),
            "invalid-configuration",
        ),
    ],
)
def test_failed_migration_never_replaces_last_valid_state(
    tmp_path,
    target_version,
    migrations,
    code,
):
    write_settings(tmp_path)
    before = settings_path(tmp_path).read_bytes()

    with pytest.raises(PluginSettingsError) as excinfo:
        PluginSettingsStore(tmp_path).load(spec(target_version, migrations=migrations))

    assert excinfo.value.code == code
    assert "private" not in str(excinfo.value)
    assert settings_path(tmp_path).read_bytes() == before


@pytest.mark.parametrize(
    ("migrations", "detail"),
    [
        (
            (ConfigurationMigration("1.0.0", "1.0.0", lambda value: value),),
            "migration cannot target itself",
        ),
        (
            (
                ConfigurationMigration("1.0.0", "2.0.0", lambda value: value),
                ConfigurationMigration("1.0.0", "3.0.0", lambda value: value),
            ),
            "ambiguous migration from 1.0.0",
        ),
        (
            (ConfigurationMigration("1.0.0", "2.0.0", None),),
            "migration transform must be callable",
        ),
    ],
)
def test_migration_graph_must_be_unambiguous_and_callable(tmp_path, migrations, detail):
    with pytest.raises(PluginSettingsError) as excinfo:
        PluginSettingsStore(tmp_path).load(spec(migrations=migrations))
    assert excinfo.value.code == "invalid-migrations"
    assert excinfo.value.detail == detail


def test_migration_count_and_versions_are_bounded(tmp_path):
    too_many = tuple(
        ConfigurationMigration(f"0.0.{index}", f"0.0.{index + 1}", lambda value: value)
        for index in range(MAX_MIGRATIONS + 1)
    )
    with pytest.raises(PluginSettingsError, match="at most 32 migrations are allowed"):
        PluginSettingsStore(tmp_path).load(spec(migrations=too_many))

    with pytest.raises(PluginSettingsError) as invalid_source:
        PluginSettingsStore(tmp_path).load(
            spec(migrations=(ConfigurationMigration("v1", "2.0.0", lambda value: value),))
        )
    assert invalid_source.value.code == "invalid-version"

    with pytest.raises(PluginSettingsError) as invalid_target:
        PluginSettingsStore(tmp_path).load(
            spec(migrations=(ConfigurationMigration("1.0.0", "v2", lambda value: value),))
        )
    assert invalid_target.value.code == "invalid-version"


@pytest.mark.parametrize(
    ("replacement", "code"),
    [
        ({"extra": True}, "invalid-settings"),
        ({"documentVersion": 2}, "settings-version-incompatible"),
        ({"documentVersion": True}, "settings-version-incompatible"),
        ({"pluginId": "other-plugin"}, "plugin-identity-mismatch"),
        ({"pluginVersion": "v1"}, "invalid-version"),
        ({"revision": -1}, "invalid-revision"),
        ({"revision": True}, "invalid-revision"),
        ({"configuration": []}, "invalid-settings"),
        ({"configuration": {"mode": "unsafe", "threshold": 2}}, "invalid-configuration"),
    ],
)
def test_stored_document_contract_is_strict(tmp_path, replacement, code):
    document = write_settings(tmp_path)
    if "extra" in replacement:
        document.update(replacement)
    else:
        document.update(replacement)
    settings_path(tmp_path).write_text(json.dumps(document))

    with pytest.raises(PluginSettingsError) as excinfo:
        PluginSettingsStore(tmp_path).load(spec())
    assert excinfo.value.code == code


@pytest.mark.parametrize("content", ["not-json", "[]", "\ud800"])
def test_unreadable_or_non_object_state_fails_closed(tmp_path, content):
    settings_path(tmp_path).write_text(content, errors="surrogatepass")
    expected = "invalid-settings" if content == "[]" else "settings-unreadable"
    with pytest.raises(PluginSettingsError) as excinfo:
        PluginSettingsStore(tmp_path).load(spec())
    assert excinfo.value.code == expected


def test_read_and_write_size_limit_is_exact(tmp_path):
    generous = PluginSettingsStore(tmp_path)
    generous.update(spec(), expected_revision=0, configuration={"mode": "safe", "threshold": 2})
    size = settings_path(tmp_path).stat().st_size

    assert PluginSettingsStore(tmp_path, max_settings_bytes=size).load(spec()).revision == 1
    with pytest.raises(PluginSettingsError) as read_limit:
        PluginSettingsStore(tmp_path, max_settings_bytes=size - 1).load(spec())
    assert read_limit.value.code == "settings-too-large"

    other = tmp_path / "other"
    with pytest.raises(PluginSettingsError) as write_limit:
        PluginSettingsStore(other, max_settings_bytes=size - 1).update(
            spec(),
            expected_revision=0,
            configuration={"mode": "safe", "threshold": 2},
        )
    assert write_limit.value.code == "settings-too-large"
    assert not settings_path(other).exists()


def test_atomic_writer_failure_preserves_previous_revision(tmp_path, monkeypatch):
    store = PluginSettingsStore(tmp_path)
    store.update(spec(), expected_revision=0, configuration={"mode": "safe", "threshold": 2})
    before = settings_path(tmp_path).read_bytes()

    def fail_write(_path, _document, prefix):
        assert prefix == ".echo-plugin-settings-"
        raise OSError("disk full")

    monkeypatch.setattr("omnitensor.plugins.settings.write_json_atomic", fail_write)
    with pytest.raises(OSError, match="disk full"):
        store.update(spec(), expected_revision=1, configuration={"mode": "fast", "threshold": 2})
    assert settings_path(tmp_path).read_bytes() == before


@pytest.mark.parametrize("plugin_id", ["", "../escape", "Upper", None])
def test_plugin_identity_cannot_escape_the_settings_root(tmp_path, plugin_id):
    with pytest.raises(PluginSettingsError) as excinfo:
        PluginSettingsStore(tmp_path).load(spec(plugin_id=plugin_id))
    assert excinfo.value.code == "invalid-plugin-id"


@pytest.mark.parametrize("version", ["", "v1", "01.0.0", None])
def test_current_plugin_version_is_strict_semver(tmp_path, version):
    with pytest.raises(PluginSettingsError) as excinfo:
        PluginSettingsStore(tmp_path).load(spec(version))
    assert excinfo.value.code == "invalid-version"


@pytest.mark.parametrize("revision", [-1, True, 1.5, "1", None])
def test_expected_revision_is_a_non_negative_integer(tmp_path, revision):
    with pytest.raises(PluginSettingsError) as excinfo:
        PluginSettingsStore(tmp_path).update(
            spec(),
            expected_revision=revision,
            configuration={"mode": "safe", "threshold": 2},
        )
    assert excinfo.value.code == "invalid-revision"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "1"])
def test_settings_byte_limit_is_a_positive_integer(tmp_path, limit):
    with pytest.raises(ValueError, match="max_settings_bytes must be a positive integer"):
        PluginSettingsStore(tmp_path, max_settings_bytes=limit)


@given(threshold=st.integers(min_value=0, max_value=10))
def test_valid_schema_domain_round_trips_every_revision(threshold):
    with tempfile.TemporaryDirectory() as directory:
        store = PluginSettingsStore(Path(directory))
        expected = store.update(
            spec(),
            expected_revision=0,
            configuration={"mode": "safe", "threshold": threshold},
        )
        assert store.load(spec()) == expected


def _hold_lock_then_bump(root: str, started, hold: float) -> None:
    """Commit revision 1 while holding the settings lock for ``hold`` seconds."""
    from omnitensor.atomicio import write_json_atomic
    from omnitensor.plugins.settings import SETTINGS_LOCK_FILE
    from omnitensor.storelock import store_lock

    with store_lock(Path(root), SETTINGS_LOCK_FILE):
        started.set()
        time.sleep(hold)
        write_json_atomic(
            Path(root) / "echo-plugin.json",
            {
                "documentVersion": 1,
                "pluginId": "echo-plugin",
                "pluginVersion": "1.0.0",
                "revision": 1,
                "configuration": {"mode": "fast", "threshold": 9},
            },
            prefix=".echo-plugin-settings-",
        )


def test_update_rechecks_the_revision_under_the_store_lock(tmp_path):
    """An unlocked check-then-write would read revision 0 and clobber revision 1."""
    store = PluginSettingsStore(tmp_path)
    store.update(spec(), expected_revision=0, configuration={"mode": "safe", "threshold": 1})
    (tmp_path / "echo-plugin.json").unlink()

    context = multiprocessing.get_context("spawn")
    started = context.Event()
    writer = context.Process(target=_hold_lock_then_bump, args=(str(tmp_path), started, 0.5))
    writer.start()
    try:
        assert started.wait(timeout=30)
        with pytest.raises(PluginSettingsError) as excinfo:
            store.update(
                spec(),
                expected_revision=0,
                configuration={"mode": "safe", "threshold": 3},
            )
    finally:
        writer.join(timeout=30)
    assert excinfo.value.code == "revision-mismatch"
    assert store.load(spec()).configuration == {"mode": "fast", "threshold": 9}


def test_concurrent_updaters_cannot_both_commit_the_same_revision(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready = context.Barrier(2)
    results = context.Queue()
    workers = [
        context.Process(target=_race_update, args=(str(tmp_path), ready, results, threshold))
        for threshold in (4, 7)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=60)
    outcomes = sorted(results.get() for _ in workers)
    assert outcomes == ["ok", "revision-mismatch"]
    assert PluginSettingsStore(tmp_path).load(spec()).revision == 1


def _race_update(root: str, ready, results, threshold: int) -> None:
    store = PluginSettingsStore(Path(root))
    ready.wait(timeout=30)
    try:
        store.update(
            spec(),
            expected_revision=0,
            configuration={"mode": "safe", "threshold": threshold},
        )
    except PluginSettingsError as error:
        results.put(error.code)
    else:
        results.put("ok")
