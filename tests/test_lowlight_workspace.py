"""Low-light folder configuration and non-destructive publication contracts."""

from __future__ import annotations

import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

import omnitensor.lowlight as lowlight
from omnitensor.lowlight import (
    LOW_LIGHT_CONFIGURATION_SPEC,
    ConfiguredLowLightWorkspace,
    LowLightWorkspace,
    LowLightWorkspaceError,
    configure_low_light_workspace,
    configure_main,
    load_low_light_workspace,
    publish_low_light_output,
    validate_low_light_workspace,
)
from omnitensor.plugins.settings import PluginSettingsStore


def _workspace(tmp_path: Path) -> tuple[LowLightWorkspace, Path]:
    source_root = tmp_path / "inputs"
    output_root = tmp_path / "outputs"
    source_root.mkdir()
    output_root.mkdir()
    source = source_root / "dark.jpg"
    source.write_bytes(b"original image")
    return validate_low_light_workspace(source_root, output_root), source


def test_configuration_persists_canonical_folders_and_reloads(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    source_alias = tmp_path / "source-alias"
    output_alias = tmp_path / "output-alias"
    source_alias.symlink_to(source, target_is_directory=True)
    output_alias.symlink_to(output, target_is_directory=True)
    store = PluginSettingsStore(tmp_path / "settings")

    configured = configure_low_light_workspace(store, source_alias, output_alias)

    assert configured == ConfiguredLowLightWorkspace(
        LowLightWorkspace(source.resolve(), output.resolve()),
        1,
    )
    assert load_low_light_workspace(store) == configured
    document = json.loads((tmp_path / "settings/low-light-enhancement.json").read_text())
    assert document["configuration"] == {
        "inputFolder": str(source.resolve()),
        "outputFolder": str(output.resolve()),
    }


def test_unconfigured_store_is_explicit(tmp_path):
    assert load_low_light_workspace(PluginSettingsStore(tmp_path)) is None


def test_semantically_incomplete_or_noncanonical_persisted_settings_fail_closed(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    detour = source / "detour"
    source.mkdir()
    output.mkdir()
    detour.mkdir()
    store = PluginSettingsStore(tmp_path / "settings")

    store.update(
        LOW_LIGHT_CONFIGURATION_SPEC,
        expected_revision=0,
        configuration={"inputFolder": str(source), "outputFolder": None},
    )
    with pytest.raises(LowLightWorkspaceError) as incomplete:
        load_low_light_workspace(store)
    assert incomplete.value.code == "configuration-incomplete"

    store.update(
        LOW_LIGHT_CONFIGURATION_SPEC,
        expected_revision=1,
        configuration={
            "inputFolder": str(detour / ".."),
            "outputFolder": str(output),
        },
    )
    with pytest.raises(LowLightWorkspaceError) as noncanonical:
        load_low_light_workspace(store)
    assert noncanonical.value.code == "configuration-not-canonical"


@pytest.mark.parametrize("relationship", ["equal", "output-inside", "input-inside"])
def test_equal_or_nested_folders_are_rejected_to_prevent_feedback(tmp_path, relationship):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    if relationship == "equal":
        source, output = first, first
    elif relationship == "output-inside":
        output = first / "results"
        output.mkdir()
        source = first
    else:
        source = second / "incoming"
        source.mkdir()
        output = second

    with pytest.raises(LowLightWorkspaceError) as excinfo:
        validate_low_light_workspace(source, output)

    assert excinfo.value.code == "folders-overlap"
    assert excinfo.value.detail == (
        "input and output folders must be disjoint; neither may contain the other"
    )


@pytest.mark.parametrize("label", ["input", "output"])
def test_folder_must_exist_and_be_a_directory(tmp_path, label):
    good = tmp_path / "good"
    good.mkdir()
    invalid = tmp_path / "not-a-directory"
    invalid.write_text("file")
    source, output = (invalid, good) if label == "input" else (good, invalid)

    with pytest.raises(LowLightWorkspaceError) as excinfo:
        validate_low_light_workspace(source, output)

    assert excinfo.value.code == f"{label}-folder-invalid"
    assert excinfo.value.detail == f"{label} folder is not a directory"


def test_missing_overlong_and_non_path_folders_have_stable_errors(tmp_path):
    good = tmp_path / "good"
    good.mkdir()
    with pytest.raises(LowLightWorkspaceError) as missing:
        validate_low_light_workspace(tmp_path / "missing", good)
    assert missing.value.code == "input-folder-invalid"

    with pytest.raises(LowLightWorkspaceError) as overlong:
        validate_low_light_workspace("/" + "a" * 4097, good)
    assert overlong.value.detail == "input folder path exceeds 4096 characters"

    with pytest.raises(LowLightWorkspaceError) as wrong_type:
        validate_low_light_workspace(3, good)  # type: ignore[arg-type]
    assert wrong_type.value.detail == "input folder is invalid"


@pytest.mark.parametrize("denied", ["input", "output"])
def test_configured_folder_permissions_are_checked(tmp_path, monkeypatch, denied):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    denied_path = source.resolve() if denied == "input" else output.resolve()
    real_access = os.access
    monkeypatch.setattr(
        "omnitensor.lowlight.os.access",
        lambda path, mode: False if path == denied_path else real_access(path, mode),
    )

    with pytest.raises(LowLightWorkspaceError) as excinfo:
        validate_low_light_workspace(source, output)

    assert excinfo.value.code == f"{denied}-folder-unavailable"
    expected = "readable and searchable" if denied == "input" else "writable and searchable"
    assert excinfo.value.detail == f"{denied} folder must be {expected}"


def test_publication_is_atomic_and_preserves_the_original_exactly(tmp_path):
    workspace, source = _workspace(tmp_path)
    before = source.stat()
    original = source.read_bytes()

    published = publish_low_light_output(workspace, source, b"enhanced image")

    after = source.stat()
    assert published == workspace.output_folder / source.name
    assert published.read_bytes() == b"enhanced image"
    assert source.read_bytes() == original
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    assert list(workspace.output_folder.glob(".low-light-output-*")) == []


def test_existing_output_or_symlink_is_never_overwritten(tmp_path):
    workspace, source = _workspace(tmp_path)
    destination = workspace.output_folder / source.name
    destination.write_bytes(b"previous result")
    original = source.read_bytes()

    with pytest.raises(LowLightWorkspaceError) as collision:
        publish_low_light_output(workspace, source, b"new result")
    assert collision.value.code == "output-exists"
    assert destination.read_bytes() == b"previous result"
    assert source.read_bytes() == original

    destination.unlink()
    outside = tmp_path / "outside-result"
    outside.write_bytes(b"outside")
    destination.symlink_to(outside)
    with pytest.raises(LowLightWorkspaceError) as symlink_collision:
        publish_low_light_output(workspace, source, b"new result")
    assert symlink_collision.value.code == "output-exists"
    assert outside.read_bytes() == b"outside"
    assert source.read_bytes() == original


def test_source_must_be_a_regular_file_inside_input_root(tmp_path):
    workspace, source = _workspace(tmp_path)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"private")
    escaping_link = workspace.input_folder / "escape.jpg"
    escaping_link.symlink_to(outside)

    for invalid in (outside, escaping_link, workspace.input_folder):
        with pytest.raises(LowLightWorkspaceError) as excinfo:
            publish_low_light_output(workspace, invalid, b"result")
        assert excinfo.value.code == "input-invalid"
        assert excinfo.value.detail == (
            "input must be a regular file contained by the configured input folder"
        )
    with pytest.raises(LowLightWorkspaceError) as missing:
        publish_low_light_output(workspace, tmp_path / "missing", b"result")
    assert missing.value.code == "input-invalid"
    assert missing.value.detail == "input does not resolve to an existing regular file"
    with pytest.raises(LowLightWorkspaceError) as wrong_type:
        publish_low_light_output(workspace, 3, b"result")  # type: ignore[arg-type]
    assert wrong_type.value.code == "input-invalid"
    assert wrong_type.value.detail == "input path is invalid"
    assert source.read_bytes() == b"original image"
    assert not any(workspace.output_folder.iterdir())


@pytest.mark.parametrize("name", ["", ".", "..", "nested/result.jpg", "\x00bad", "a" * 256])
def test_output_name_cannot_escape_or_be_ambiguous(tmp_path, name):
    workspace, source = _workspace(tmp_path)
    with pytest.raises(LowLightWorkspaceError) as excinfo:
        publish_low_light_output(workspace, source, b"result", output_name=name)
    assert excinfo.value.code == "output-name-invalid"
    assert excinfo.value.detail == "output name must be one non-empty filename"
    assert source.read_bytes() == b"original image"
    assert not any(workspace.output_folder.iterdir())


@pytest.mark.parametrize(
    ("payload", "limit", "code"),
    [
        (b"", 1, "output-invalid"),
        (bytearray(b"x"), 1, "output-invalid"),
        (b"xx", 1, "output-too-large"),
    ],
)
def test_output_bytes_are_typed_and_bounded(tmp_path, payload, limit, code):
    workspace, source = _workspace(tmp_path)
    with pytest.raises(LowLightWorkspaceError) as excinfo:
        publish_low_light_output(workspace, source, payload, max_output_bytes=limit)
    assert excinfo.value.code == code
    expected = (
        "encoded output must be non-empty bytes"
        if code == "output-invalid"
        else f"encoded output exceeds the {limit} byte limit"
    )
    assert excinfo.value.detail == expected


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5])
def test_output_limit_must_be_a_positive_integer(tmp_path, limit):
    workspace, source = _workspace(tmp_path)
    with pytest.raises(ValueError, match="max_output_bytes must be a positive integer"):
        publish_low_light_output(workspace, source, b"result", max_output_bytes=limit)


def test_output_size_and_name_limits_are_inclusive(tmp_path):
    workspace, source = _workspace(tmp_path)
    name = "x" * 255
    published = publish_low_light_output(
        workspace,
        source,
        b"exact",
        output_name=name,
        max_output_bytes=5,
    )
    assert published.name == name
    assert published.read_bytes() == b"exact"


def test_staging_and_link_options_keep_atomic_publish_inside_output_root(
    tmp_path, monkeypatch
):
    workspace, source = _workspace(tmp_path)
    real_mkstemp = tempfile.mkstemp
    real_link = os.link
    stages = []
    links = []

    def tracked_mkstemp(*, dir, prefix):
        stages.append((dir, prefix))
        return real_mkstemp(dir=dir, prefix=prefix)

    def tracked_link(source_path, destination_path, *, follow_symlinks):
        links.append((Path(source_path), Path(destination_path), follow_symlinks))
        return real_link(source_path, destination_path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr("omnitensor.lowlight.tempfile.mkstemp", tracked_mkstemp)
    monkeypatch.setattr("omnitensor.lowlight.os.link", tracked_link)

    destination = publish_low_light_output(workspace, source, b"result")

    assert stages == [(workspace.output_folder, ".low-light-output-")]
    assert len(links) == 1
    staged, linked_destination, follows = links[0]
    assert staged.parent == workspace.output_folder
    assert linked_destination == destination
    assert follows is False


def test_directory_sync_uses_read_only_directory_descriptor_and_closes(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "omnitensor.lowlight.os.open",
        lambda directory, flags: calls.append(("open", directory, flags)) or 17,
    )
    monkeypatch.setattr("omnitensor.lowlight.os.fsync", lambda fd: calls.append(("fsync", fd)))
    monkeypatch.setattr("omnitensor.lowlight.os.close", lambda fd: calls.append(("close", fd)))

    lowlight._fsync_directory(Path("/configured/output"))

    assert calls == [
        (
            "open",
            Path("/configured/output"),
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        ),
        ("fsync", 17),
        ("close", 17),
    ]


def test_directory_sync_is_best_effort(monkeypatch):
    monkeypatch.setattr(
        "omnitensor.lowlight.os.open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unsupported")),
    )
    assert lowlight._fsync_directory(Path("/configured/output")) is None


def test_publish_failure_cleans_only_its_stage(tmp_path, monkeypatch):
    workspace, source = _workspace(tmp_path)
    original = source.read_bytes()
    monkeypatch.setattr(
        "omnitensor.lowlight.os.link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("filesystem refused link")),
    )

    with pytest.raises(LowLightWorkspaceError) as excinfo:
        publish_low_light_output(workspace, source, b"result")

    assert excinfo.value.code == "output-unavailable"
    assert excinfo.value.detail == f"cannot publish output in {workspace.output_folder}"
    assert source.read_bytes() == original
    assert list(workspace.output_folder.iterdir()) == []


def test_concurrent_publishers_create_exactly_one_result(tmp_path):
    workspace, source = _workspace(tmp_path)

    def publish(payload: bytes):
        try:
            return publish_low_light_output(workspace, source, payload)
        except LowLightWorkspaceError as error:
            return error.code

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, (b"first", b"second")))

    assert sum(isinstance(outcome, Path) for outcome in outcomes) == 1
    assert outcomes.count("output-exists") == 1
    assert (workspace.output_folder / source.name).read_bytes() in {b"first", b"second"}
    assert source.read_bytes() == b"original image"
    assert list(workspace.output_folder.glob(".low-light-output-*")) == []


def test_cli_configuration_integrates_with_store_and_publisher(tmp_path, capsys):
    source = tmp_path / "incoming"
    output = tmp_path / "enhanced"
    settings = tmp_path / "settings"
    source.mkdir()
    output.mkdir()
    image = source / "night.png"
    image.write_bytes(b"original")

    assert configure_main(
        [
            "--input-folder",
            str(source),
            "--output-folder",
            str(output),
            "--settings-root",
            str(settings),
        ]
    ) == 0
    expected_response = {
        "workloadId": "low-light-enhancement",
        "revision": 1,
        "inputFolder": str(source.resolve()),
        "outputFolder": str(output.resolve()),
        "originalsPreserved": True,
        "existingOutputsOverwritten": False,
    }
    captured = capsys.readouterr().out
    assert captured == json.dumps(expected_response, indent=2) + "\n"
    response = json.loads(captured)
    assert response == expected_response
    configured = load_low_light_workspace(PluginSettingsStore(settings))
    assert configured is not None
    assert publish_low_light_output(
        configured.workspace,
        image,
        b"enhanced",
        output_name="night-enhanced.png",
    ).read_bytes() == b"enhanced"
    assert image.read_bytes() == b"original"


def test_cli_reports_configuration_refusal_without_traceback(tmp_path, capsys):
    folder = tmp_path / "same"
    folder.mkdir()
    assert configure_main(
        [
            "--input-folder",
            str(folder),
            "--output-folder",
            str(folder),
            "--settings-root",
            str(tmp_path / "settings"),
        ]
    ) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("low-light configuration failed: folders-overlap:")
    assert "Traceback" not in captured.err


def test_cli_help_and_required_arguments_are_a_stable_contract(capsys):
    with pytest.raises(SystemExit) as help_exit:
        configure_main(["--help"])
    assert help_exit.value.code == 0
    assert capsys.readouterr().out == (
        "usage: omnitensor-configure-low-light [-h] --input-folder INPUT_FOLDER\n"
        "                                      --output-folder OUTPUT_FOLDER\n"
        "                                      [--settings-root SETTINGS_ROOT]\n"
        "\n"
        "Configure disjoint input and output folders for low-light enhancement\n"
        "\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
        "  --input-folder INPUT_FOLDER\n"
        "  --output-folder OUTPUT_FOLDER\n"
        "  --settings-root SETTINGS_ROOT\n"
    )

    for arguments, missing in (
        (["--output-folder", "/output"], "--input-folder"),
        (["--input-folder", "/input"], "--output-folder"),
    ):
        with pytest.raises(SystemExit) as required_exit:
            configure_main(arguments)
        assert required_exit.value.code == 2
        assert missing in capsys.readouterr().err


def test_cli_uses_the_documented_default_settings_root(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source"
    output = tmp_path / "output"
    settings = tmp_path / "default-settings"
    source.mkdir()
    output.mkdir()
    monkeypatch.setattr(
        "omnitensor.lowlight.DEFAULT_LOW_LIGHT_SETTINGS_ROOT",
        str(settings),
    )

    assert configure_main(
        ["--input-folder", str(source), "--output-folder", str(output)]
    ) == 0
    capsys.readouterr()
    assert load_low_light_workspace(PluginSettingsStore(settings)) is not None


@given(
    payload=st.binary(min_size=1, max_size=512),
    tail=st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789._-", max_size=40),
)
def test_property_safe_names_publish_exact_bytes_without_touching_source(payload, tail):
    with tempfile.TemporaryDirectory() as directory:
        workspace, source = _workspace(Path(directory))
        original = source.read_bytes()
        name = f"result-{tail}.bin"

        published = publish_low_light_output(
            workspace,
            source,
            payload,
            output_name=name,
            max_output_bytes=512,
        )

        assert published.name == name
        assert published.read_bytes() == payload
        assert source.read_bytes() == original


def test_manifest_and_guide_freeze_non_destructive_folder_contract():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "workloads/low-light-enhancement/manifest.json").read_text(encoding="utf-8")
    )
    responsibilities = manifest["pipeline"]["hostResponsibilities"]
    assert any(
        "configured input folder" in item and "originals" in item
        for item in responsibilities
    )
    assert any("disjoint configured output folder" in item for item in responsibilities)
    guide = " ".join((root / "docs/use-cases.md").read_text(encoding="utf-8").split())
    for contract in (
        "omnitensor-configure-low-light",
        "cannot be equal or nested",
        "never renames, overwrites, or deletes the source image",
        "does not claim",
    ):
        assert contract in guide
