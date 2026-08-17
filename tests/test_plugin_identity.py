from __future__ import annotations

import json
from pathlib import Path

from conftest import sample_manifest, sample_plugin_manifest

from omnitensor.plugins import (
    MAX_DISTRIBUTION_NAME_LENGTH,
    MAX_DISTRIBUTION_VERSION_LENGTH,
    MAX_ENTRY_POINT_TARGET_LENGTH,
    PluginMetadata,
    PluginRejectionCode,
    PluginSource,
    resolve_plugin_identities,
)
from omnitensor.registry import MAX_MANIFEST_BYTES


def external_candidate(
    tmp_path: Path,
    plugin_id: str = "external-plugin",
    *,
    distribution_name: str | None = "external-package",
    distribution_version: str | None = "1.2.0",
    target: str | None = "external_plugin.runtime:Plugin",
    manifest: object | None = None,
    metadata_error: str = "",
    paths: tuple[Path, ...] | None = None,
) -> PluginMetadata:
    path = tmp_path / f"{plugin_id}-omnitensor-plugin.json"
    document = sample_plugin_manifest(plugin_id) if manifest is None else manifest
    path.write_text(json.dumps(document), encoding="utf-8")
    return PluginMetadata(
        source=PluginSource.EXTERNAL,
        entry_point_name=plugin_id,
        entry_point_value=target,
        distribution_name=distribution_name,
        distribution_version=distribution_version,
        manifest_paths=(path,) if paths is None else paths,
        bundled_manifest=None,
        metadata_error=metadata_error,
    )


def bundled_candidate(tmp_path: Path, plugin_id: str = "bundled-plugin") -> PluginMetadata:
    manifest = sample_manifest(plugin_id)
    return PluginMetadata(
        source=PluginSource.BUNDLED,
        entry_point_name=plugin_id,
        entry_point_value=None,
        distribution_name="omnitensor",
        distribution_version=None,
        manifest_paths=(tmp_path / plugin_id / "manifest.json",),
        bundled_manifest=manifest,
        metadata_error="",
    )


def forecast_manifest(plugin_id: str) -> dict:
    manifest = sample_plugin_manifest(plugin_id)
    manifest["requirements"]["accelerator"] = "gpu"
    manifest["requirements"]["acceleratorPreference"] = ["gpu"]
    manifest["requirements"]["model"] = {
        "id": "local-forecast-gpu",
        "version": "1.0.0",
        "format": "ncnn",
        "sha256": "b" * 64,
        "fullyQuantized": False,
        "minimumCompilerVersion": "0.0.0",
        "minimumRuntimeVersion": "0.0.0",
        "tensorContract": {"inputs": [{"shape": [1, 2], "dtype": "float32", "layout": "NC"}]},
        "featureContract": {
            "version": 1,
            "recipe": "forecast-v1",
            "featureNames": ["load", "queue"],
            "targetFeature": "load",
            "window": 1,
            "horizon": 1,
            "observationOrder": "oldest-first",
            "flattenOrder": "observations-then-features",
        },
        "outputContract": {"kind": "raw"},
    }
    return manifest


def test_resolves_bundled_v1_and_external_v2_identities(tmp_path):
    catalog = resolve_plugin_identities(
        (
            external_candidate(tmp_path),
            bundled_candidate(tmp_path),
        )
    )

    assert [plugin.plugin_id for plugin in catalog.plugins] == [
        "bundled-plugin",
        "external-plugin",
    ]
    bundled, external = catalog.plugins
    assert bundled.source is PluginSource.BUNDLED
    assert bundled.distribution_name == "omnitensor"
    assert bundled.distribution_version is None
    assert bundled.entry_point_value is None
    assert bundled.entry_point_name == "bundled-plugin"
    assert bundled.manifest_path == tmp_path / "bundled-plugin" / "manifest.json"
    assert bundled.version == "1.0.0"
    assert external.source is PluginSource.EXTERNAL
    assert external.distribution_name == "external-package"
    assert external.distribution_version == "1.2.0"
    assert external.entry_point_value == "external_plugin.runtime:Plugin"
    assert external.entry_point_name == "external-plugin"
    assert external.manifest_path == tmp_path / "external-plugin-omnitensor-plugin.json"
    assert external.manifest["manifestVersion"] == 2
    assert catalog.rejections == ()


def test_broken_candidates_are_rejected_without_hiding_valid_plugin(tmp_path):
    missing = external_candidate(tmp_path, "missing", paths=())
    multiple_path = tmp_path / "second.json"
    multiple_path.write_text("{}", encoding="utf-8")
    multiple = external_candidate(
        tmp_path,
        "multiple",
        paths=(tmp_path / "one.json", multiple_path),
    )
    metadata = external_candidate(
        tmp_path,
        "metadata",
        metadata_error="distribution metadata unavailable: RuntimeError",
    )

    catalog = resolve_plugin_identities((missing, bundled_candidate(tmp_path), multiple, metadata))

    assert [plugin.plugin_id for plugin in catalog.plugins] == ["bundled-plugin"]
    assert [rejection.entry_point_name for rejection in catalog.rejections] == [
        "metadata",
        "missing",
        "multiple",
    ]
    assert [rejection.code for rejection in catalog.rejections] == [
        PluginRejectionCode.METADATA_UNAVAILABLE,
        PluginRejectionCode.MANIFEST_COUNT,
        PluginRejectionCode.MANIFEST_COUNT,
    ]
    assert catalog.rejections[0].detail == ("distribution metadata unavailable: RuntimeError")
    assert catalog.rejections[1].detail == "candidate must provide exactly one manifest"
    assert catalog.rejections[1].distribution_name == "external-package"


def test_external_manifest_read_failures_are_bounded_and_redacted(tmp_path):
    missing_path = tmp_path / "missing.json"
    missing = external_candidate(tmp_path, "missing-file", paths=(missing_path,))
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("not json: private", encoding="utf-8")
    invalid = external_candidate(tmp_path, "invalid-json", paths=(invalid_path,))
    list_document = external_candidate(tmp_path, "list-root", manifest=[])
    oversized_path = tmp_path / "oversized.json"
    oversized_path.write_bytes(b"x" * (MAX_MANIFEST_BYTES + 1))
    oversized = external_candidate(tmp_path, "oversized", paths=(oversized_path,))

    catalog = resolve_plugin_identities((oversized, invalid, list_document, missing))

    assert [(item.entry_point_name, item.code) for item in catalog.rejections] == [
        ("invalid-json", PluginRejectionCode.MANIFEST_IO),
        ("list-root", PluginRejectionCode.MANIFEST_INVALID),
        ("missing-file", PluginRejectionCode.MANIFEST_IO),
        ("oversized", PluginRejectionCode.MANIFEST_INVALID),
    ]
    assert catalog.rejections[0].detail == "manifest cannot be read: JSONDecodeError"
    assert catalog.rejections[1].detail == "manifest root must be an object"
    assert catalog.rejections[2].detail == "manifest cannot be read: FileNotFoundError"
    assert catalog.rejections[3].detail == "manifest exceeds 65536 bytes"


def test_rejects_invalid_schema_and_external_v1_manifest(tmp_path):
    invalid_schema = sample_plugin_manifest("invalid-schema")
    invalid_schema["version"] = "latest"

    catalog = resolve_plugin_identities(
        (
            external_candidate(tmp_path, "invalid-schema", manifest=invalid_schema),
            external_candidate(tmp_path, "legacy", manifest=sample_manifest("legacy")),
        )
    )

    assert [(item.entry_point_name, item.code, item.detail) for item in catalog.rejections] == [
        (
            "invalid-schema",
            PluginRejectionCode.MANIFEST_INVALID,
            "manifest violates the workload schema",
        ),
        (
            "legacy",
            PluginRejectionCode.INCOMPATIBLE,
            "external plugins require manifest version 2",
        ),
    ]


def test_external_identity_rejects_schema_valid_but_ambiguous_feature_semantics(tmp_path):
    manifest = forecast_manifest("ambiguous-forecast")
    manifest["requirements"]["model"]["featureContract"]["targetFeature"] = "queue"

    catalog = resolve_plugin_identities(
        (external_candidate(tmp_path, "ambiguous-forecast", manifest=manifest),)
    )

    [rejection] = catalog.rejections
    assert rejection.code is PluginRejectionCode.MANIFEST_INVALID
    assert rejection.detail == "manifest violates the workload schema"


def test_rejects_manifest_and_entry_point_identity_mismatches(tmp_path):
    wrong_id = sample_plugin_manifest("other-id")
    wrong_declared = sample_plugin_manifest("wrong-declared")
    wrong_declared["plugin"]["entryPoint"] = "other-entry"

    catalog = resolve_plugin_identities(
        (
            external_candidate(tmp_path, "installed-name", manifest=wrong_id),
            external_candidate(tmp_path, "wrong-declared", manifest=wrong_declared),
        )
    )

    assert [(item.entry_point_name, item.code, item.detail) for item in catalog.rejections] == [
        (
            "installed-name",
            PluginRejectionCode.IDENTITY_MISMATCH,
            "manifest ID and entry-point name differ",
        ),
        (
            "wrong-declared",
            PluginRejectionCode.IDENTITY_MISMATCH,
            "manifest and installed entry-point identities differ",
        ),
    ]


def test_rejects_invalid_distribution_and_target_metadata(tmp_path):
    candidates = (
        external_candidate(tmp_path, "missing-name", distribution_name=None),
        external_candidate(tmp_path, "bad-name", distribution_name="bad/name"),
        external_candidate(
            tmp_path,
            "long-name",
            distribution_name="a" * (MAX_DISTRIBUTION_NAME_LENGTH + 1),
        ),
        external_candidate(tmp_path, "missing-version", distribution_version=None),
        external_candidate(
            tmp_path,
            "long-version",
            distribution_version="1" * (MAX_DISTRIBUTION_VERSION_LENGTH + 1),
        ),
        external_candidate(tmp_path, "missing-target", target=None),
        external_candidate(tmp_path, "bad-target", target="bad target"),
        external_candidate(
            tmp_path,
            "long-target",
            target="a" * (MAX_ENTRY_POINT_TARGET_LENGTH + 1),
        ),
    )

    catalog = resolve_plugin_identities(candidates)

    assert len(catalog.rejections) == len(candidates)
    by_name = {item.entry_point_name: item for item in catalog.rejections}
    for name in ("missing-name", "bad-name", "long-name", "missing-version", "long-version"):
        assert by_name[name].code is PluginRejectionCode.METADATA_UNAVAILABLE
        assert by_name[name].detail == "distribution identity is missing or invalid"
    for name in ("missing-target", "bad-target", "long-target"):
        assert by_name[name].code is PluginRejectionCode.IDENTITY_MISMATCH
        assert by_name[name].detail == "entry-point target is missing or invalid"


def test_identity_metadata_and_manifest_size_accept_exact_bounds(tmp_path):
    candidate = external_candidate(
        tmp_path,
        "boundary-plugin",
        distribution_name="a" * MAX_DISTRIBUTION_NAME_LENGTH,
        distribution_version="1" * MAX_DISTRIBUTION_VERSION_LENGTH,
        target="a" * MAX_ENTRY_POINT_TARGET_LENGTH,
    )
    manifest_path = candidate.manifest_paths[0]
    document = manifest_path.read_bytes()
    manifest_path.write_bytes(document + b" " * (MAX_MANIFEST_BYTES - len(document)))

    catalog = resolve_plugin_identities((candidate,))

    assert [plugin.plugin_id for plugin in catalog.plugins] == ["boundary-plugin"]
    assert catalog.rejections == ()


def test_bundled_identity_must_belong_to_omnitensor(tmp_path):
    wrong_distribution = bundled_candidate(tmp_path, "wrong-distribution")
    wrong_distribution = PluginMetadata(
        source=wrong_distribution.source,
        entry_point_name=wrong_distribution.entry_point_name,
        entry_point_value=wrong_distribution.entry_point_value,
        distribution_name="impersonator",
        distribution_version=wrong_distribution.distribution_version,
        manifest_paths=wrong_distribution.manifest_paths,
        bundled_manifest=wrong_distribution.bundled_manifest,
        metadata_error=wrong_distribution.metadata_error,
    )
    missing_manifest = bundled_candidate(tmp_path, "missing-manifest")
    missing_manifest = PluginMetadata(
        source=missing_manifest.source,
        entry_point_name=missing_manifest.entry_point_name,
        entry_point_value=missing_manifest.entry_point_value,
        distribution_name=missing_manifest.distribution_name,
        distribution_version=missing_manifest.distribution_version,
        manifest_paths=missing_manifest.manifest_paths,
        bundled_manifest=None,
        metadata_error=missing_manifest.metadata_error,
    )

    catalog = resolve_plugin_identities((wrong_distribution, missing_manifest))

    assert all(
        rejection.code is PluginRejectionCode.IDENTITY_MISMATCH for rejection in catalog.rejections
    )
    assert all(
        rejection.detail == "bundled candidate does not belong to OmniTensor"
        for rejection in catalog.rejections
    )


def test_bundled_plugin_wins_external_duplicate(tmp_path):
    catalog = resolve_plugin_identities(
        (
            external_candidate(tmp_path, "same-plugin"),
            bundled_candidate(tmp_path, "same-plugin"),
        )
    )

    assert len(catalog.plugins) == 1
    assert catalog.plugins[0].source is PluginSource.BUNDLED
    assert [(item.code, item.detail) for item in catalog.rejections] == [
        (
            PluginRejectionCode.DUPLICATE_ID,
            "plugin ID conflicts with a bundled plugin",
        )
    ]
    assert catalog.rejections[0].entry_point_name == "same-plugin"
    assert catalog.rejections[0].distribution_name == "external-package"


def test_all_ambiguous_external_duplicates_are_rejected(tmp_path):
    first = external_candidate(tmp_path, "duplicate", distribution_name="first-package")
    second = external_candidate(tmp_path, "duplicate", distribution_name="second-package")

    catalog = resolve_plugin_identities((second, first))

    assert catalog.plugins == ()
    assert [item.distribution_name for item in catalog.rejections] == [
        "first-package",
        "second-package",
    ]
    assert all(item.code is PluginRejectionCode.DUPLICATE_ID for item in catalog.rejections)
    assert all(
        item.detail == "plugin ID is declared by multiple candidates" for item in catalog.rejections
    )
