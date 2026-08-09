"""Executable documentation inventory for the bundled Cinnamon use cases."""

from pathlib import Path

from omnitensor.registry import bundled_workloads_path, load_workloads

ROOT = Path(__file__).resolve().parents[1]
USE_CASES = ROOT / "docs/use-cases.md"


def test_use_case_guide_covers_every_bundled_profile():
    guide = USE_CASES.read_text(encoding="utf-8")
    for workload_id in load_workloads(bundled_workloads_path()):
        assert f"`{workload_id}`" in guide


def test_readme_links_the_use_case_guide():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "[Cinnamon workload use cases](docs/use-cases.md)" in readme
    assert "[cinnamon-tpuwlm](../cinnamon-tpuwlm)" in readme


def test_extension_guide_covers_the_publishing_lifecycle():
    """The guide is the end-to-end route for a plugin author (OMNI-0083)."""
    guide = (ROOT / "docs/extension-guide.md").read_text(encoding="utf-8")
    for topic in (
        "## 1. Packaging",
        "## 2. Discovery and identity",
        "## 3. Schemas and configuration",
        "## 4. Permissions",
        "## 5. Artifacts",
        "## 6. Testing locally",
        "## 7. Installing",
        "## 8. Upgrading and rolling back",
        "## 9. Compatibility policy",
    ):
        assert topic in guide
    for reference in (
        "plugin-discovery.md",
        "plugin-identity.md",
        "plugin-sdk.md",
        "plugin-conformance.md",
        "plugin-workers.md",
        "local-plugin-runner.md",
    ):
        assert reference in guide, f"the guide must route to {reference}"


def test_readme_links_the_extension_guide():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "[extension guide](docs/extension-guide.md)" in readme
