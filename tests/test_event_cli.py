from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from omnitensor import event_cli
from omnitensor.plugins.generation_contracts import ModelCatalog, ModelEvaluation, ModelSource


def result_document():
    return {
        "version": 2,
        "requestId": "request-1",
        "outcome": "succeeded",
        "code": "events-found",
        "detail": "",
        "duplicatePolicy": "keep-first-label-date-time-place",
        "confirmationState": "pending",
        "events": [
            {
                "candidateId": "event-1",
                "label": {"text": "Review", "source": "stated"},
                "when": {
                    "date": "2026-08-12",
                    "time": "10:00",
                    "timezone": {"name": "Europe/Rome", "utcOffset": "+02:00", "source": "stated"},
                    "needs": [],
                },
                "where": {"kind": "physical", "venue": "Room 1", "source": "stated"},
                "confirmation": "pending",
                "evidence": [
                    {
                        "sourceRef": "private:request-1:source:1:page:1",
                        "sourceSha256": "a" * 64,
                        "page": 1,
                        "span": {"start": 0, "end": 6},
                        "textSha256": "b" * 64,
                        "readAs": "text",
                    }
                ],
            }
        ],
    }


def test_preview_contains_evidence_count_but_no_private_material():
    preview = event_cli.preview_document(result_document())
    assert preview == {
        "requestId": "request-1",
        "outcome": "succeeded",
        "confirmationState": "pending",
        "duplicatesDropped": 0,
        "events": [
            {
                "candidateId": "event-1",
                "label": "Review",
                "date": "2026-08-12",
                "time": "10:00",
                "timezone": "Europe/Rome",
                "place": "Room 1",
                "needs": [],
                "status": "scheduled",
                "evidenceCount": 1,
            }
        ],
    }
    assert "private:" not in json.dumps(preview)


def test_preview_and_confirmed_export_cli(tmp_path, capsys):
    source = tmp_path / "result.json"
    source.write_text(json.dumps(result_document()))
    assert event_cli.main(["preview", str(source)]) == 0
    assert json.loads(capsys.readouterr().out)["confirmationState"] == "pending"

    output = tmp_path / "events.ics"
    assert (
        event_cli.main(["export", str(source), "--confirm", "event-1", "--output", str(output)])
        == 0
    )
    assert "SUMMARY:Review" in output.read_text()
    assert output.stat().st_mode & 0o777 == 0o600
    assert source.exists()

    assert (
        event_cli.main(["export", str(source), "--confirm", "event-1", "--output", str(output)])
        == 2
    )
    assert json.loads(capsys.readouterr().err)["code"] == "client-failed"


def test_export_requires_every_explicit_decision(tmp_path, capsys):
    source = tmp_path / "result.json"
    source.write_text(json.dumps(result_document()))
    output = tmp_path / "events.ics"
    assert event_cli.main(["export", str(source), "--output", str(output)]) == 2
    assert json.loads(capsys.readouterr().err)["code"] == "confirmation-invalid"
    assert not output.exists()


def test_legacy_folder_is_explicit_bounded_and_deterministic(tmp_path):
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "a.ics").write_text("a")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.md").write_text("c")
    assert event_cli.legacy_selected_files(tmp_path) == (
        str(tmp_path / "a.ics"),
        str(tmp_path / "b.txt"),
    )
    assert event_cli.legacy_selected_files(tmp_path, recursive=True) == (
        str(tmp_path / "a.ics"),
        str(tmp_path / "b.txt"),
        str(nested / "c.md"),
    )
    with pytest.raises(event_cli.EventClientError):
        event_cli.legacy_selected_files(Path("relative"))
    link = tmp_path.parent / "folder-link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(event_cli.EventClientError) as linked:
        event_cli.legacy_selected_files(link)
    assert str(linked.value) == ("directory-invalid: input must be an existing absolute directory")


def test_model_check_never_downloads_and_requires_every_digest(tmp_path, monkeypatch, capsys):
    model = tmp_path / "model.gguf"
    companion = tmp_path / "projector.gguf"
    model.write_bytes(b"model")
    companion.write_bytes(b"projector")
    sources = (
        ModelSource("model", "https://example.invalid/m", "a" * 40, model.name, "1" * 64, 5),
        ModelSource(
            "projector", "https://example.invalid/p", "a" * 40, companion.name, "2" * 64, 9
        ),
    )
    catalog = ModelCatalog(
        "qwen",
        "1",
        "a" * 40,
        "Apache-2.0",
        sources,
        (),
        ModelEvaluation("x", object()),
    )
    monkeypatch.setattr(event_cli, "load_generation_catalog", lambda: catalog)
    digests = {model: "1" * 64, companion: "2" * 64}
    monkeypatch.setattr(event_cli, "file_digest", lambda path: digests[path])

    assert event_cli.main(["check-model", str(tmp_path)]) == 0
    ready = json.loads(capsys.readouterr().out)
    assert ready == {
        "artifacts": [
            {"filename": "model.gguf", "ready": True, "role": "model", "sha256": "1" * 64},
            {
                "filename": "projector.gguf",
                "ready": True,
                "role": "projector",
                "sha256": "2" * 64,
            },
        ],
        "license": "Apache-2.0",
        "modelId": "qwen",
        "ready": True,
        "version": "1",
    }
    companion.unlink()
    assert event_cli.main(["check-model", str(tmp_path)]) == 1
    unavailable = json.loads(capsys.readouterr().out)
    assert unavailable["ready"] is False
    assert unavailable["artifacts"][1]["ready"] is False


@pytest.mark.parametrize("content", ["[]", "{", "null"])
def test_client_rejects_malformed_result_without_traceback(tmp_path, capsys, content):
    source = tmp_path / "result.json"
    source.write_text(content)
    assert event_cli.main(["preview", str(source)]) == 2
    assert json.loads(capsys.readouterr().err)["ok"] is False


def test_export_requires_absolute_path_and_removes_partial_output(tmp_path, monkeypatch):
    with pytest.raises(event_cli.EventClientError) as relative:
        event_cli._write_new(Path("events.ics"), b"calendar")
    assert relative.value.code == "output-invalid"

    output = tmp_path / "partial.ics"

    def interrupted(*_args, **_kwargs):
        raise OSError("sync")

    monkeypatch.setattr(event_cli, "_write_bytes_atomic", interrupted)
    with pytest.raises(OSError, match="sync"):
        event_cli._write_new(output, b"calendar")
    assert not output.exists()

    monkeypatch.undo()
    nested = tmp_path / "new" / "parents" / "event.ics"
    event_cli._write_new(nested, b"calendar")
    assert nested.read_bytes() == b"calendar"
    assert nested.stat().st_mode & 0o777 == 0o600


def test_client_error_result_reader_and_json_writer_contracts(tmp_path, capsys, monkeypatch):
    error = event_cli.EventClientError("stable-code", "stable detail")
    assert (error.code, error.detail, str(error)) == (
        "stable-code",
        "stable detail",
        "stable-code: stable detail",
    )

    missing = tmp_path / "missing.json"
    with pytest.raises(event_cli.EventClientError) as unreadable:
        event_cli._read_result(missing)
    assert str(unreadable.value) == "result-unreadable: result is not readable JSON"

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (event_cli.MAX_RESULT_BYTES + 1))
    with pytest.raises(event_cli.EventClientError) as too_large:
        event_cli._read_result(oversized)
    assert str(too_large.value) == "result-too-large: result exceeds its byte limit"

    event_cli._print_json({"b": "é", "a": 1})
    assert capsys.readouterr().out == '{"a":1,"b":"\\u00e9"}\n'
    with pytest.raises(ValueError):
        event_cli._print_json({"value": float("nan")})

    monkeypatch.setattr(event_cli, "file_digest", lambda _path: "wrong")
    checked = event_cli.verify_event_artifacts(tmp_path)
    assert checked["ready"] is False
    assert [item["ready"] for item in checked["artifacts"]] == [False, False]


def test_parser_help_and_argument_contract_is_stable():
    parser = event_cli._parser()
    assert parser.format_help() == (
        "usage: omnitensor-import-events [-h]\n"
        "                                {preview,export,check-model,legacy-files} ...\n\n"
        "Preview or explicitly export a private grounded event result.\n\n"
        "positional arguments:\n"
        "  {preview,export,check-model,legacy-files}\n"
        "    preview             show pending candidates\n"
        "    export              write confirmed candidates as ICS\n"
        "    check-model         verify pinned local model artifacts\n"
        "    legacy-files        list an explicitly selected legacy folder\n\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
    )
    assert vars(parser.parse_args(["preview", "/tmp/result.json"])) == {
        "command": "preview",
        "result": Path("/tmp/result.json"),
    }
    assert vars(
        parser.parse_args(
            [
                "export",
                "/tmp/result.json",
                "--confirm",
                "one",
                "--reject",
                "two",
                "--output",
                "/tmp/out.ics",
            ]
        )
    ) == {
        "command": "export",
        "result": Path("/tmp/result.json"),
        "confirm": ["one"],
        "reject": ["two"],
        "output": Path("/tmp/out.ics"),
    }


def test_legacy_selection_count_boundaries(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for index in range(32):
            (root / f"{index:02}.txt").write_text("event")
        assert len(event_cli.legacy_selected_files(root)) == 32
        (root / "32.txt").write_text("event")
        with pytest.raises(event_cli.EventWorkloadError) as error:
            event_cli.legacy_selected_files(root)
        assert error.value.detail == "select 1-32 source files"


def test_main_legacy_and_export_outputs_are_exact(tmp_path, capsys):
    source_file = tmp_path / "event.txt"
    source_file.write_text("event")
    assert event_cli.main(["legacy-files", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"sources": [str(source_file)]}

    result = tmp_path / "result.json"
    result.write_text(json.dumps(result_document()))
    output = tmp_path / "nested" / "events.ics"
    assert (
        event_cli.main(["export", str(result), "--confirm", "event-1", "--output", str(output)])
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "confirmed": 1,
        "output": str(output),
        "written": True,
    }

    missing = tmp_path / "missing"
    assert event_cli.main(["legacy-files", str(missing)]) == 2
    assert json.loads(capsys.readouterr().err) == {
        "code": "directory-invalid",
        "ok": False,
    }
