"""Executable documentation inventory for the bundled Cinnamon use cases."""

from pathlib import Path

from omnitensor.registry import bundled_workloads_path, load_workloads

ROOT = Path(__file__).resolve().parents[1]
USE_CASES = ROOT / "docs/use-cases.md"
LOCAL_TRAINING = ROOT / "docs/local-training.md"


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


def test_local_training_guide_names_every_installed_feature_semantic():
    guide = LOCAL_TRAINING.read_text(encoding="utf-8")
    for term in (
        "`featureContract`",
        "recipe",
        "ordered feature names",
        "target",
        "window",
        "observation-count horizon",
        "oldest-first",
        "observations-then-features",
        "same-shaped inline tensor",
    ):
        assert term in guide


def test_installation_guide_documents_every_verifier_check():
    """The guide must explain what each automated check failing means."""
    from omnitensor.acceptance import REQUIRED_SCHEMAS

    guide = (ROOT / "docs/installation.md").read_text(encoding="utf-8")
    for check in (
        "executable",
        "service",
        "schemas",
        "workloads",
        "discovery",
        "isolation",
        "dbus",
        "snapshot",
        "applet",
    ):
        assert f"`{check}`" in guide, f"the guide must explain the {check} check"
    assert "omnitensor-verify-install" in guide
    assert REQUIRED_SCHEMAS  # the verifier checks a non-empty schema set


def test_installation_guide_states_conversion_parity_and_evidence_limits():
    guide = (ROOT / "docs/installation.md").read_text(encoding="utf-8")
    normalized = " ".join(guide.split())

    for term in (
        "### Producing accelerator artifacts",
        "[convert]",
        "[convert-npu]",
        "--format ncnn",
        "--format openvino",
        "edgetpu_compiler",
        "does not prove inference correctness",
        "no target NPU needed to convert",
    ):
        assert term in normalized
