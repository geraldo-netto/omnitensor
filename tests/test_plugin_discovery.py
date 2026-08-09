from __future__ import annotations

from pathlib import Path

import pytest
from conftest import sample_manifest, sample_plugin_manifest, write_workload

from omnitensor.plugins import (
    MAX_DISTRIBUTION_FILES,
    MAX_PLUGIN_CANDIDATES,
    PLUGIN_ENTRY_POINT_GROUP,
    PluginDiscoveryError,
    PluginSource,
    discover_plugin_metadata,
)


class FakeDistribution:
    def __init__(self, root, name="external-dist", version="2.0.0", files=()):
        self.root = root
        self.name = name
        self.version = version
        self.files = files

    def locate_file(self, path):
        return self.root / path


class FakeEntryPoint:
    load_calls = 0

    def __init__(self, name, value, dist=None):
        self.name = name
        self.value = value
        self.dist = dist

    def load(self):
        type(self).load_calls += 1
        raise AssertionError("metadata discovery imported plugin code")


def test_discovery_combines_sorted_bundled_and_external_metadata_without_loading(tmp_path):
    bundled = tmp_path / "bundled"
    write_workload(bundled, sample_manifest("zeta-bundled"))
    plugin_manifest = sample_plugin_manifest("alpha-bundled")
    plugin_manifest["plugin"]["entryPoint"] = "custom-entry"
    write_workload(bundled, plugin_manifest)

    dist_root = tmp_path / "site"
    manifests = [
        Path("pkg/omnitensor-plugin.json"),
        Path("other/ignored.json"),
    ]
    distribution = FakeDistribution(dist_root, files=manifests)
    external = [
        FakeEntryPoint("zeta-external", "zeta.module:Plugin", distribution),
        FakeEntryPoint("alpha-external", "alpha.module:Plugin", distribution),
    ]
    calls = []

    def entry_points_provider(**selection):
        calls.append(selection)
        return reversed(external)

    FakeEntryPoint.load_calls = 0
    discovered = discover_plugin_metadata(
        bundled_root=bundled,
        entry_points_provider=entry_points_provider,
    )

    assert calls == [{"group": PLUGIN_ENTRY_POINT_GROUP}]
    assert FakeEntryPoint.load_calls == 0
    assert [(item.source, item.entry_point_name) for item in discovered] == [
        (PluginSource.BUNDLED, "custom-entry"),
        (PluginSource.BUNDLED, "zeta-bundled"),
        (PluginSource.EXTERNAL, "alpha-external"),
        (PluginSource.EXTERNAL, "zeta-external"),
    ]
    assert discovered[0].bundled_manifest == plugin_manifest
    assert discovered[0].manifest_paths == (bundled / "alpha-bundled" / "manifest.json",)
    assert discovered[0].distribution_name == "omnitensor"
    assert discovered[0].distribution_version is None
    assert discovered[0].metadata_error == ""
    assert discovered[1].entry_point_value is None
    assert discovered[2].distribution_name == "external-dist"
    assert discovered[2].distribution_version == "2.0.0"
    assert discovered[2].entry_point_value == "alpha.module:Plugin"
    assert discovered[2].manifest_paths == (
        (dist_root / "pkg/omnitensor-plugin.json").resolve(),
    )
    assert discovered[2].bundled_manifest is None
    assert discovered[2].metadata_error == ""


def test_external_sort_key_is_name_then_distribution_then_target(tmp_path):
    first_dist = FakeDistribution(tmp_path, name="a-dist", files=None)
    second_dist = FakeDistribution(tmp_path, name="b-dist", files=None)
    candidates = [
        FakeEntryPoint("same", "z.module:Plugin", second_dist),
        FakeEntryPoint("same", "b.module:Plugin", first_dist),
        FakeEntryPoint("same", "a.module:Plugin", first_dist),
    ]

    discovered = discover_plugin_metadata(
        bundled_root=tmp_path / "missing",
        entry_points_provider=lambda **_selection: candidates,
    )

    assert [
        (item.distribution_name, item.entry_point_value) for item in discovered
    ] == [
        ("a-dist", "a.module:Plugin"),
        ("a-dist", "b.module:Plugin"),
        ("b-dist", "z.module:Plugin"),
    ]
    assert all(item.manifest_paths == () for item in discovered)


def test_external_sort_key_handles_missing_distribution_and_target(tmp_path):
    distribution = FakeDistribution(tmp_path, name="a-dist", files=None)
    candidates = [
        FakeEntryPoint("same", "z.module:Plugin", None),
        FakeEntryPoint("same", "b.module:Plugin", distribution),
        FakeEntryPoint("same", None, None),
    ]

    discovered = discover_plugin_metadata(
        bundled_root=tmp_path / "missing",
        entry_points_provider=lambda **_selection: candidates,
    )

    assert [
        (item.distribution_name, item.entry_point_value) for item in discovered
    ] == [
        (None, None),
        (None, "z.module:Plugin"),
        ("a-dist", "b.module:Plugin"),
    ]


def test_duplicate_candidates_remain_visible_for_identity_validation(tmp_path):
    distribution = FakeDistribution(tmp_path)
    candidates = [
        FakeEntryPoint("duplicate", "one:Plugin", distribution),
        FakeEntryPoint("duplicate", "two:Plugin", distribution),
    ]

    discovered = discover_plugin_metadata(
        bundled_root=tmp_path / "bundled",
        entry_points_provider=lambda **_selection: candidates,
    )

    assert len(discovered) == 2
    assert [candidate.entry_point_value for candidate in discovered] == [
        "one:Plugin",
        "two:Plugin",
    ]


def test_missing_or_broken_distribution_metadata_isolated_per_candidate(tmp_path):
    class BrokenEntryPoint:
        name = "broken"
        value = "broken:Plugin"

        @property
        def dist(self):
            raise RuntimeError("private metadata")

    candidates = [
        FakeEntryPoint("missing-dist", "missing:Plugin", None),
        BrokenEntryPoint(),
    ]

    discovered = discover_plugin_metadata(
        bundled_root=tmp_path / "bundled",
        entry_points_provider=lambda **_selection: candidates,
    )

    assert [(item.entry_point_name, item.metadata_error) for item in discovered] == [
        ("broken", "distribution metadata unavailable: RuntimeError"),
        ("missing-dist", "entry point has no distribution"),
    ]
    assert all(item.distribution_name is None for item in discovered)
    assert all(item.distribution_version is None for item in discovered)


def test_multiple_distribution_manifests_are_sorted_for_later_rejection(tmp_path):
    distribution = FakeDistribution(
        tmp_path,
        files=[
            Path("z/omnitensor-plugin.json"),
            Path("a/omnitensor-plugin.json"),
        ],
    )
    discovered = discover_plugin_metadata(
        bundled_root=tmp_path / "bundled",
        entry_points_provider=lambda **_selection: [
            FakeEntryPoint("external", "external:Plugin", distribution)
        ],
    )
    assert discovered[0].manifest_paths == (
        (tmp_path / "a/omnitensor-plugin.json").resolve(),
        (tmp_path / "z/omnitensor-plugin.json").resolve(),
    )


def test_entry_point_enumeration_failure_is_redacted(tmp_path):
    def fail(**_selection):
        raise RuntimeError("private environment")

    with pytest.raises(PluginDiscoveryError) as excinfo:
        discover_plugin_metadata(bundled_root=tmp_path, entry_points_provider=fail)
    assert str(excinfo.value) == "entry-point enumeration failed: RuntimeError"


def test_candidate_count_is_bounded(tmp_path):
    candidates = [
        FakeEntryPoint(f"plugin-{index}", "module:Plugin")
        for index in range(MAX_PLUGIN_CANDIDATES + 1)
    ]
    with pytest.raises(PluginDiscoveryError, match="more than 256 plugin candidates"):
        discover_plugin_metadata(
            bundled_root=tmp_path / "bundled",
            entry_points_provider=lambda **_selection: candidates,
        )


def test_candidate_count_accepts_the_exact_bound(tmp_path):
    candidates = [
        FakeEntryPoint(f"plugin-{index}", "module:Plugin")
        for index in range(MAX_PLUGIN_CANDIDATES)
    ]

    discovered = discover_plugin_metadata(
        bundled_root=tmp_path / "bundled",
        entry_points_provider=lambda **_selection: candidates,
    )

    assert len(discovered) == MAX_PLUGIN_CANDIDATES


def test_distribution_file_scan_is_bounded_and_isolates_the_candidate(tmp_path):
    files = (Path(f"file-{index}") for index in range(MAX_DISTRIBUTION_FILES + 1))
    distribution = FakeDistribution(tmp_path, files=files)
    discovered = discover_plugin_metadata(
        bundled_root=tmp_path / "bundled",
        entry_points_provider=lambda **_selection: [
            FakeEntryPoint("large-dist", "module:Plugin", distribution)
        ],
    )
    assert discovered[0].manifest_paths == ()
    assert discovered[0].metadata_error == (
        "distribution metadata unavailable: PluginDiscoveryError"
    )
