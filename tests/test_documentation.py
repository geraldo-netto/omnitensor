"""Executable documentation inventory for the bundled Cinnamon use cases."""

import json
import tomllib
from pathlib import Path

from omnitensor.registry import bundled_workloads_path, load_workloads

ROOT = Path(__file__).resolve().parents[1]
USE_CASES = ROOT / "docs/use-cases.md"
LOCAL_TRAINING = ROOT / "docs/local-training.md"


def test_use_case_guide_covers_every_bundled_profile():
    guide = USE_CASES.read_text(encoding="utf-8")
    for workload_id in load_workloads(bundled_workloads_path()):
        assert f"`{workload_id}`" in guide


def test_use_case_guide_distinguishes_delivered_and_local_models():
    guide = USE_CASES.read_text(encoding="utf-8")

    resource_row = next(line for line in guide.splitlines() if "`resource-scheduler`" in line)
    visual_row = next(line for line in guide.splitlines() if "`visual-library`" in line)
    assert "opt-in local recipe" in resource_row
    assert "restricted binding" in resource_row
    assert "pinned ncnn classifier" in visual_row
    assert "Embedding/search remains" in visual_row
    assert "other eight manifests" not in guide


def test_readme_links_the_use_case_guide():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "[Cinnamon workload use cases](docs/use-cases.md)" in readme
    assert "[cinnamon-xpuwlm](../cinnamon-xpuwlm)" in readme


def test_xpu_rename_is_coordinated_across_service_defaults_and_installation():
    from omnitensor.acceptance import DEFAULT_APPLET_ROOT
    from omnitensor.service import DEFAULT_STATE_PATH
    from omnitensor.training.cli import DEFAULT_SNAPSHOT_PATH

    guide = (ROOT / "docs/installation.md").read_text(encoding="utf-8")
    normalized = " ".join(guide.split())
    assert DEFAULT_APPLET_ROOT == "~/.local/share/cinnamon/applets/cinnamon-xpuwlm@geraldo-netto"
    assert DEFAULT_STATE_PATH == DEFAULT_SNAPSHOT_PATH
    assert DEFAULT_STATE_PATH == "~/.local/state/xpu-workload-manager/state.json"
    assert "remove the old `cinnamon-tpuwm@geraldo-netto` panel instance" in normalized
    assert "deploy the matched service" in normalized


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


def test_revocation_guidance_uses_only_the_public_sdk():
    sdk_guide = (ROOT / "docs/plugin-sdk.md").read_text(encoding="utf-8")
    extension_guide = (ROOT / "docs/extension-guide.md").read_text(encoding="utf-8")
    guidance = sdk_guide + extension_guide

    assert "`PermissionView.require()`" in sdk_guide
    assert "installed host monitors the live" in extension_guide
    assert "service-side grant monitors" in sdk_guide
    assert "LiveGrantView" not in guidance
    assert "ConsentGuard" not in guidance


def test_readme_links_the_extension_guide():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "[extension guide](docs/extension-guide.md)" in readme


def test_generation_guide_keeps_models_and_transports_replaceable():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs/generation-providers.md").read_text(encoding="utf-8")

    assert "[generation provider contract](docs/generation-providers.md)" in readme
    for boundary in (
        "task- and provider-neutral",
        "only opaque, bounded content references",
        "must not enter the public runtime snapshot, control payloads, logs",
        "outside the core service process",
        "default preference is exactly `gpu`",
        "only when NPU admission or model loading fails before generation",
        "CPU providers are rejected",
    ):
        assert boundary in guide


def test_event_result_guide_keeps_evidence_private_and_writes_confirmed():
    generation = (ROOT / "docs/generation-providers.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs/event-results.md").read_text(encoding="utf-8")
    prose = " ".join(guide.split())

    assert "[grounded event result\ncontract](event-results.md)" in generation
    for boundary in (
        "opaque `private:` source",
        "Public snapshots, logs, and general control",
        "explicitly `succeeded`, `partial`, or `refused`",
        # Version 2 states the rule the domain actually enforces: an offset
        # depends on the date, so `events._validate_when` refuses one that
        # arrives without it. The version 1 guarantee this line used to name —
        # refusing an offset that disagrees with the zone's local wall time —
        # is not implemented, and asserting the prose kept it made the guide
        # promise something the code stopped doing.
        "a UTC offset is accepted only alongside a known date",
        "cannot mark its own output confirmed or rejected",
        "one explicit human decision for every candidate",
        "calendar sink is invoked only for confirmed candidates",
    ):
        assert boundary in prose


def test_qwen_guide_requires_pinned_artifacts_and_real_accelerator_evidence():
    generation = (ROOT / "docs/generation-providers.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs/qwen-event-providers.md").read_text(encoding="utf-8")
    prose = " ".join(guide.split())

    assert "[Qwen event provider guide](qwen-event-providers.md)" in generation
    for contract in (
        "Vulkan",
        "partial offload is a",
        "Detecting an NPU does not enable",
        "no CPU fallback",
        "precision and recall are each at least 0.90",
        "prompt-injection probe",
        "p95 generation latency",
        "cancellation completes within 2 seconds",
        "token streams need not be byte-identical",
        "does not mutate or bless",
    ):
        assert contract in prose


def test_event_workload_guide_covers_dependencies_privacy_and_legacy_transition():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    generation = (ROOT / "docs/generation-providers.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs/event-extraction.md").read_text(encoding="utf-8")
    prose = " ".join(guide.split())
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "[event extraction guide](docs/event-extraction.md)" in readme
    assert "[event extraction workload guide](event-extraction.md)" in generation
    assert project["project"]["optional-dependencies"]["events"] == ["pymupdf>=1.24"]
    assert project["project"]["scripts"]["omnitensor-import-events"] == (
        "omnitensor.event_cli:main"
    )
    for boundary in (
        "never watches or scans",
        "pip install 'omnitensor[events]'",
        "needs Tesseract language data",
        "There is no CPU lane",
        "1–32 absolute paths",
        "packaging template, not a bundled live profile",
        'workerState: "ready"',
        "per-request broker directory",
        "never writes, moves, renames, or deletes",
        "must be a new absolute path",
        "`lostutils/import_events.py` compatibility",
    ):
        assert boundary in prose


def test_document_question_guide_freezes_manual_privacy_and_model_boundaries():
    guide = " ".join((ROOT / "docs/document-questions.md").read_text(encoding="utf-8").split())
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for required in (
        "1–16",
        "opset 11",
        "opset 13",
        "There is no CPU lane",
        "files:read-selected",
        "at most 512 spans",
        "at most 2,048 characters",
        "selected-file-N",
        "file name, page, and character span",
        "Qwen3-8B-Q4_K_M",
        "all 37 model layers",
        "1.0 citation integrity",
        "does not claim quality for arbitrary private corpora",
        "omnitensor-qualify-document-questions",
    ):
        assert required in guide
    assert "[Ask selected files](docs/document-questions.md)" in readme


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


def test_local_training_guide_keeps_snapshot_collection_explicit_and_bounded():
    guide = LOCAL_TRAINING.read_text(encoding="utf-8")

    assert "omnitensor-record-runtime-snapshot" in guide
    assert "queueDepth" in guide
    assert "runningProfiles" in guide
    assert "Persistent=false" in guide
    assert "No timer is installed or enabled" in guide
    assert "1 MiB" in guide
    assert "record-runtime-snapshot" not in (ROOT / "systemd/omnitensor.service").read_text(
        encoding="utf-8"
    )


def test_local_training_examples_share_snapshot_feature_names():
    guide = LOCAL_TRAINING.read_text(encoding="utf-8")

    assert "--features queueDepth,runningProfiles" in guide
    assert "--target queueDepth" in guide
    assert "--feature queueDepth=3" in guide
    assert "--feature runningProfiles=1" in guide


def test_model_tooling_guide_covers_every_reviewed_license_and_target_stage():
    guide = LOCAL_TRAINING.read_text(encoding="utf-8")
    normalized = " ".join(guide.split())
    recipes = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((ROOT / "model-recipes").glob("*.json"))
    ]

    assert recipes
    for recipe in recipes:
        assert f"`{recipe['id']}`" in guide
        assert recipe["license"]["spdx"] in guide
        assert recipe["license"]["termsUri"] in guide
    for command in (
        "omnitensor-list-model-recipes",
        "omnitensor-fetch-model-source",
        "available_targets",
        "omnitensor-verify-install",
        "omnitensor-install-trained-model",
    ):
        assert command in guide
    for boundary in (
        "Compiler availability is not device availability",
        "Vulkan GPU",
        "Intel NPU",
        "Coral TPU",
        "CPU fallback",
        "100% Edge TPU operation mapping",
        "specific to `forecast-v1`",
        "Do not publish an operator's raw corpus",
    ):
        assert boundary in normalized


def test_document_model_guide_keeps_opsets_one_contract_and_native_gate_explicit():
    project = tomllib.loads(
        (ROOT / "packaging" / "omnitensor-training" / "pyproject.toml").read_text(encoding="utf-8")
    )
    guide = LOCAL_TRAINING.read_text(encoding="utf-8")
    normalized = " ".join(guide.split())

    assert project["project"]["scripts"]["omnitensor-install-document-model"] == (
        "omnitensor.training.document_model:main"
    )
    assert set(project["project"]["optional-dependencies"]["document-producers"]) >= {
        "ncnn>=1.0.20260526",
        "onnxruntime>=1.17",
        "pnnx>=20260526",
        "tokenizers>=0.22",
        "transformers>=4.40",
    }
    for boundary in (
        "ONNX opset 11",
        "MiniLM mean-pooling wrapper legitimately uses opset 13",
        "no second service API",
        "refuses software Vulkan devices and CPU fallback",
        "cosine parity of at least `0.999`",
        "maximum absolute error at most `0.001`",
        "remains disabled by default",
        "other model-less profiles are untouched",
        "NPU needs a separately gated OpenVINO artifact",
    ):
        assert boundary in normalized


def test_use_case_guide_records_source_only_profile_boundaries():
    guide = USE_CASES.read_text(encoding="utf-8")
    expected = {
        "hardware-health": "labelled sensor windows",
        "storage-intelligence": "Backblaze Drive Stats",
        "build-advisor": "approved-metadata producer",
        "network-peripherals": "normal-only aggregate producer",
        "desktop-context": "content-free producer",
        "document-intelligence": "MiniLM and BGE",
        "low-light-enhancement": "Retinexformer",
    }
    for profile_id, boundary in expected.items():
        row = next(line for line in guide.splitlines() if f"`{profile_id}`" in line)
        assert boundary in row
        assert "remain" in row


def test_selected_text_guide_keeps_capture_private_manual_and_accelerated():
    guide = (ROOT / "docs/selected-text-tools.md").read_text(encoding="utf-8")
    normalized = " ".join(guide.split())
    for boundary in (
        "explicit user action",
        "never monitors the clipboard",
        "clipboard:read-once",
        "explain",
        "summarize",
        "rewrite",
        "translate",
        "extract-tasks",
        "GPU",
        "NPU",
        "No CPU",
        "No additional base dependency",
        "selectionSha256",
    ):
        assert boundary in normalized


def test_file_organizer_guide_keeps_selection_explicit_and_plan_review_only():
    guide = " ".join((ROOT / "docs/file-organizer.md").read_text().split())
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for boundary in (
        "one to sixteen explicit absolute file paths",
        "files:read-selected",
        "at most two bounded spans per file",
        "full source SHA-256",
        "never absolute paths or source text",
        "no apply action",
        "no CPU fallback",
        "GPU",
        "NPU",
    ):
        assert boundary in guide
    assert "[File organizer](docs/file-organizer.md)" in readme


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
        "control",
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


def test_plugin_admission_guide_documents_the_pool_and_how_to_tune_it():
    """A knob nobody can find is a knob nobody has."""
    from omnitensor.plugin_admission import DEFAULT_MAX_CONCURRENT, MAX_CONCURRENT_LIMIT

    guide = (ROOT / "docs/plugin-admission.md").read_text(encoding="utf-8")
    assert "OMNITENSOR_PLUGIN_SLOTS" in guide
    assert f"| {DEFAULT_MAX_CONCURRENT} | {MAX_CONCURRENT_LIMIT} |" in guide
    assert "1 / weight" in guide
    assert "[Plugin admission](docs/plugin-admission.md)" in (ROOT / "README.md").read_text(
        encoding="utf-8"
    )


def test_boundary_guide_splits_capability_from_intent_and_records_the_refusals():
    """The rule a new feature is placed by, and the two rejected shortcuts.

    Both rejections cost something real if re-derived: a socket verb that stops
    the service strands every watching workload, and renaming the published
    snapshot path costs three consumers a migration for tidiness.
    """
    guide = (ROOT / "docs/runtime-client-boundary.md").read_text(encoding="utf-8")

    assert "capability against intent" in guide
    assert "correct with no client attached" in guide
    assert "pause is service state" in guide.lower()
    assert "one writer" in guide
    assert "No socket verb that stops the service" in guide
    assert "No rename of the published snapshot path" in guide


def test_readme_links_the_boundary_guide():
    assert "docs/runtime-client-boundary.md" in (ROOT / "README.md").read_text(encoding="utf-8")


def qwen_distributions() -> dict[str, str]:
    """Every Qwen-family provider distribution in the tree, by workload id.

    Derived rather than listed. The installation guide named four wheels in
    five places — a table, a build loop, a wheel install, a grant block and an
    uninstall — and a fifth distribution changed none of them, so a person
    following the guide installed a workload the service then reported as
    missing.
    """
    found: dict[str, str] = {}
    for path in sorted((ROOT / "providers").glob("*/pyproject.toml")):
        project = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
        if not project["name"].startswith("omnitensor-qwen-"):
            continue
        [workload_id] = project["entry-points"]["omnitensor.workloads"]
        found[workload_id] = project["name"]
    return found


def test_installation_guide_names_every_qwen_distribution_it_installs():
    guide = (ROOT / "docs/qwen-workload-installation.md").read_text(encoding="utf-8")

    assert len(qwen_distributions()) == 4
    for workload_id, name in qwen_distributions().items():
        wheel = name.replace("-", "_")
        assert f"`{name}`" in guide, f"{name} is not in the wheel table"
        assert f"{wheel}-0.2.0-*.whl" in guide, f"{name} is never installed"
        assert f"grant {workload_id} accelerator:gpu" in guide
        assert f"revoke {workload_id} accelerator:gpu" in guide
        assert "providers/$provider" in guide
        loop = guide.split("for provider in", 1)[1].split("; do", 1)[0]
        assert workload_id in loop, f"{workload_id} is never built"


# Which guide answers "what does this workload do?" for each shipped provider
# distribution. Not derivable from the filename: `ask-selected-files` is
# documented in `document-questions.md`, because the guide is named for the
# question a person has and the workload for what the runtime calls it.
WORKLOAD_GUIDES = {
    "ask-selected-files": "docs/document-questions.md",
    "event-extraction": "docs/event-extraction.md",
    "file-organizer": "docs/file-organizer.md",
    "media-transcription": "docs/media-transcription.md",
    "media-similarity": "docs/media-similarity.md",
    "selected-text-tools": "docs/selected-text-tools.md",
}


def shipped_workloads() -> set[str]:
    """Every workload this repository ships a provider distribution for."""
    return {
        path.parent.name
        for path in (ROOT / "providers").glob("*/pyproject.toml")
        if "omnitensor.workloads"
        in tomllib.loads(path.read_text(encoding="utf-8"))["project"].get("entry-points", {})
    }


def test_every_shipped_workload_has_a_guide_the_readme_routes_to():
    """OMNI-0540. A workload nobody can read about is a workload nobody uses.

    Derived from the tree, so shipping a sixth distribution fails here rather
    than in the silence of a README that never mentioned it.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert set(WORKLOAD_GUIDES) == shipped_workloads()
    for workload_id, guide in WORKLOAD_GUIDES.items():
        assert (ROOT / guide).is_file(), f"{workload_id} has no guide"
        assert f"({guide})" in readme, f"the README never routes to {guide}"


def test_every_shipped_guide_is_reachable_from_another_document():
    """The other direction: a guide nothing links to is written and unread."""
    guides = sorted((ROOT / "docs").glob("*.md"))
    corpus = [ROOT / "README.md", *guides]

    unreachable = [
        guide.name
        for guide in guides
        if not any(
            guide.name in other.read_text(encoding="utf-8") for other in corpus if other != guide
        )
    ]
    assert unreachable == []


def test_every_command_a_person_must_run_is_written_down():
    """OMNI-0545. An installed command nobody documented cannot be run.

    `test_ci_supply_chain.py` proves each console script resolves to a real
    callable; a callable nothing names is still unreachable for the person who
    has to run it.
    """
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "scripts"
    ]
    written = (ROOT / "README.md").read_text(encoding="utf-8") + "".join(
        guide.read_text(encoding="utf-8") for guide in sorted((ROOT / "docs").glob("*.md"))
    )

    undocumented = sorted(name for name in scripts if name not in written)
    assert undocumented == []


def test_the_installation_guide_names_every_provider_wheel_a_person_must_build():
    """OMNI-0556: the runtime wheels are not derivable from the workload ids.

    `qwen_distributions()` finds a distribution by its workload entry point,
    and the two runtime wheels publish none — so nothing in the guide's own
    gates would have noticed that a person following it never built the
    embedder, and `ask-selected-files` would fail to import at worker start.
    """
    guide = (ROOT / "docs/qwen-workload-installation.md").read_text(encoding="utf-8")
    runtime_wheels = {
        path.parent.name: tomllib.loads(path.read_text(encoding="utf-8"))["project"]["name"]
        for path in sorted((ROOT / "providers").glob("*/pyproject.toml"))
        if "entry-points" not in tomllib.loads(path.read_text(encoding="utf-8"))["project"]
    }

    assert set(runtime_wheels) == {"vulkan-runtime", "ncnn-embeddings"}
    loop = guide.split("for provider in", 1)[1].split("; do", 1)[0]
    for directory, name in runtime_wheels.items():
        wheel = name.replace("-", "_")
        assert f"`{name}`" in guide, f"{name} is not in the wheel table"
        assert f"{wheel}-0.2.0-*.whl" in guide, f"{name} is never installed"
        assert directory in loop, f"{directory} is never built"
