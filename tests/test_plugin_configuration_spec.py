from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins import (
    DEFAULT_CONFIGURATION_VERSION,
    PluginSettingsError,
    PluginSettingsStore,
    manifest_configuration_spec,
)
from omnitensor.registry import load_schema

ROOT = Path(__file__).parents[1]
MANIFESTS = ROOT / "plugin-manifests"


def manifest(configuration=None, *, version=None, plugin_id="echo-plugin"):
    schemas = {
        "configuration": {"type": "object"} if configuration is None else configuration,
        "input": {"type": "object"},
        "output": {"type": "object"},
    }
    if version is not None:
        schemas["configurationVersion"] = version
    return {"id": plugin_id, "plugin": {"schemas": schemas}}


def test_an_undeclared_version_is_the_first_one():
    spec = manifest_configuration_spec("echo-plugin", manifest())
    assert spec.plugin_version == DEFAULT_CONFIGURATION_VERSION
    assert spec.plugin_id == "echo-plugin"
    assert spec.migrations == ()


def test_the_declared_version_is_the_contract_version_not_the_release():
    spec = manifest_configuration_spec("echo-plugin", manifest(version="2.1.0"))
    assert spec.plugin_version == "2.1.0"


def test_defaults_come_from_the_schema_the_manifest_declares():
    spec = manifest_configuration_spec(
        "echo-plugin",
        manifest(
            {
                "type": "object",
                "properties": {
                    "guidance": {"type": "string", "default": "be brief"},
                    "length": {"type": "integer", "default": 3},
                    "unset": {"type": "string"},
                },
            }
        ),
    )
    assert spec.defaults == {"guidance": "be brief", "length": 3}


def test_a_nested_default_is_carried_whole_and_never_synthesised():
    """The manifest states the object or it states nothing.

    Building one out of the defaults inside it would put a value in the store
    that the plugin never declared.
    """
    spec = manifest_configuration_spec(
        "echo-plugin",
        manifest(
            {
                "type": "object",
                "properties": {
                    "synthesised": {
                        "type": "object",
                        "properties": {"inner": {"type": "string", "default": "x"}},
                    },
                    "stated": {"type": "object", "default": {"inner": "x"}},
                },
            }
        ),
    )
    assert spec.defaults == {"stated": {"inner": "x"}}


def test_a_required_property_with_no_default_is_named_by_the_manifest_not_the_store():
    with pytest.raises(PluginSettingsError) as error:
        manifest_configuration_spec(
            "echo-plugin",
            manifest(
                {
                    "type": "object",
                    "properties": {"mode": {"enum": ["safe", "fast"]}},
                    "required": ["mode"],
                }
            ),
        )
    assert error.value.code == "configuration-defaults-missing"
    assert "mode" in error.value.detail


@pytest.mark.parametrize("version", ["", "1.0", "v1.0.0", "1.0.0-rc1", "01.0.0", 1])
def test_a_version_that_is_not_semver_is_refused(version):
    with pytest.raises(PluginSettingsError) as error:
        manifest_configuration_spec("echo-plugin", manifest(version=version))
    assert error.value.code == "invalid-version"


def test_an_explicit_null_version_is_refused_rather_than_read_as_absent():
    """Absent means the first version; stated-as-nothing is a manifest defect."""
    document = manifest()
    document["plugin"]["schemas"]["configurationVersion"] = None
    with pytest.raises(PluginSettingsError) as error:
        manifest_configuration_spec("echo-plugin", document)
    assert error.value.code == "invalid-version"


@pytest.mark.parametrize(
    ("document", "code"),
    [
        ({"id": "echo-plugin"}, "configuration-undeclared"),
        ({"plugin": {}}, "configuration-undeclared"),
        ({"plugin": {"schemas": {}}}, "invalid-schema"),
        ({"plugin": {"schemas": {"configuration": []}}}, "invalid-schema"),
    ],
)
def test_a_manifest_that_declares_no_holdable_contract_is_refused(document, code):
    with pytest.raises(PluginSettingsError) as error:
        manifest_configuration_spec("echo-plugin", document)
    assert error.value.code == code


def test_the_derived_spec_is_one_a_store_can_actually_hold(tmp_path):
    """The point of the whole item: an external plugin's settings persist."""
    spec = manifest_configuration_spec(
        "echo-plugin",
        manifest(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"guidance": {"type": "string", "default": ""}},
            },
            version="1.2.0",
        ),
    )
    store = PluginSettingsStore(tmp_path)
    assert store.load(spec).configuration == {"guidance": ""}

    updated = store.update(spec, expected_revision=0, configuration={"guidance": "cite sources"})
    assert updated.revision == 1
    assert store.load(spec).configuration == {"guidance": "cite sources"}


def test_a_contract_change_refuses_rather_than_guessing_a_migration(tmp_path):
    """A manifest has nowhere to declare a callable, so a derived spec has none.

    The settings a person chose are refused by name rather than transformed by
    a guess, which is why the version is the contract's and not the release's.
    """
    schema = {"type": "object", "properties": {"guidance": {"type": "string", "default": ""}}}
    store = PluginSettingsStore(tmp_path)
    store.update(
        manifest_configuration_spec("echo-plugin", manifest(schema, version="1.0.0")),
        expected_revision=0,
        configuration={"guidance": "cite sources"},
    )
    with pytest.raises(PluginSettingsError) as error:
        store.load(manifest_configuration_spec("echo-plugin", manifest(schema, version="2.0.0")))
    assert error.value.code == "migration-missing"


def test_a_release_that_leaves_the_contract_alone_keeps_the_settings(tmp_path):
    schema = {"type": "object", "properties": {"guidance": {"type": "string", "default": ""}}}
    store = PluginSettingsStore(tmp_path)
    spec = manifest_configuration_spec("echo-plugin", manifest(schema))
    store.update(spec, expected_revision=0, configuration={"guidance": "cite sources"})

    # The same contract, shipped by a later release of the plugin.
    reloaded = manifest_configuration_spec("echo-plugin", manifest(schema))
    assert store.load(reloaded).configuration == {"guidance": "cite sources"}


@given(
    st.text(
        alphabet=st.characters(blacklist_categories=("Cs",)),
        min_size=0,
        max_size=12,
    )
)
def test_no_string_is_accepted_as_a_version_unless_it_is_semver(version):
    import re

    semver = re.fullmatch(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", version)
    if semver:
        assert manifest_configuration_spec("p", manifest(version=version)).plugin_version == version
        return
    with pytest.raises(PluginSettingsError):
        manifest_configuration_spec("p", manifest(version=version))


def test_the_manifest_schema_carries_the_new_field_and_still_rejects_unknown_ones():
    schema = load_schema("workload-manifest.schema.json")
    schemas = schema["properties"]["plugin"]["properties"]["schemas"]
    assert "configurationVersion" in schemas["properties"]
    assert "configurationVersion" not in schemas["required"]
    assert schemas["additionalProperties"] is False


def test_every_shipped_manifest_declares_a_contract_this_can_derive():
    for path in sorted(MANIFESTS.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator(load_schema("workload-manifest.schema.json")).validate(
            document
        )
        spec = manifest_configuration_spec(document["id"], document)
        assert spec.plugin_version == DEFAULT_CONFIGURATION_VERSION, path
