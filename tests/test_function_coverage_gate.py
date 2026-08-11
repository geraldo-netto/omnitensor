from __future__ import annotations

import json

import pytest

from omnitensor.mutation_quality import main as mutation_main
from omnitensor.mutation_quality import mutation_failures, parse_results
from omnitensor.quality import (
    _file_reports,
    _function_percentage,
    _function_reports,
    coverage_failures,
    main,
    source_files,
    validate_threshold,
)


def report(percent=80.0, *, name="Subject.method", filename="src/omnitensor/subject.py"):
    return {
        "files": {
            str(filename): {
                "functions": {
                    name: {
                        "start_line": 7,
                        "summary": {"percent_statements_covered": percent},
                    },
                    "": {
                        "start_line": 1,
                        "summary": {"percent_statements_covered": 0.0},
                    },
                }
            }
        }
    }


def test_per_function_gate_accepts_the_floor_and_ignores_module_totals():
    assert coverage_failures(report(), 80.0, ["src/omnitensor/subject.py"]) == ()


def test_quality_threshold_accepts_both_percentage_boundaries():
    assert validate_threshold(0) == 0.0
    assert validate_threshold(100) == 100.0


def test_per_function_gate_names_every_callable_below_the_floor():
    assert coverage_failures(report(79.9), 80.0, ["src/omnitensor/subject.py"]) == (
        "src/omnitensor/subject.py:7 Subject.method 79.9%",
    )


@pytest.mark.parametrize("document", [None, {}, {"files": {"x.py": {}}}])
def test_per_function_gate_rejects_malformed_coverage_reports(document):
    with pytest.raises(ValueError, match="coverage report"):
        coverage_failures(document, 80.0, ["src/omnitensor/subject.py"])


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (None, "coverage report has no files object"),
        ({"files": {}}, "coverage report contains no measured files"),
    ],
)
def test_coverage_file_parser_preserves_exact_fail_closed_diagnostics(document, message):
    with pytest.raises(ValueError) as caught:
        _file_reports(document)
    assert str(caught.value) == message


def test_coverage_file_parser_returns_the_report_mapping_unchanged():
    files = {"subject.py": {"functions": {}}}
    assert _file_reports({"files": files}) is files


def test_per_function_gate_rejects_empty_and_partial_source_inventories():
    with pytest.raises(ValueError) as empty:
        coverage_failures(report(), 80.0, [])
    assert str(empty.value) == "coverage source inventory is empty"
    with pytest.raises(ValueError) as partial:
        coverage_failures(
            report(),
            80.0,
            ["src/omnitensor/subject.py", "src/omnitensor/missing.py"],
        )
    assert "coverage report omits source file:" in str(partial.value)
    assert str(partial.value).endswith("src/omnitensor/missing.py")


@pytest.mark.parametrize("percent", [None, True, -1, 101, float("inf"), float("nan")])
def test_per_function_gate_rejects_invalid_percentages(percent):
    with pytest.raises(ValueError, match="valid percentage"):
        coverage_failures(report(percent), 80.0, ["src/omnitensor/subject.py"])


@pytest.mark.parametrize("threshold", [None, True, -1, 101, float("inf"), float("nan")])
def test_quality_threshold_rejects_fail_open_values(threshold):
    with pytest.raises(ValueError) as caught:
        validate_threshold(threshold)
    expected = (
        "quality threshold must be a finite percentage"
        if threshold is None or threshold is True
        else "quality threshold must be between 0 and 100"
    )
    assert str(caught.value) == expected


def test_function_report_helpers_accept_valid_data_and_name_malformed_fields():
    functions = report()["files"]["src/omnitensor/subject.py"]["functions"]
    assert _function_reports("subject.py", {"functions": functions}) is functions
    assert _function_percentage("subject.py", "Subject.method", functions["Subject.method"]) == 80

    with pytest.raises(ValueError) as missing_functions:
        _function_reports("subject.py", {})
    assert str(missing_functions.value) == (
        "coverage report has no functions object for subject.py"
    )
    with pytest.raises(ValueError) as missing_percentage:
        _function_percentage("subject.py", "Subject.method", {})
    assert str(missing_percentage.value) == (
        "coverage report has no valid percentage for subject.py:Subject.method"
    )


def test_per_function_gate_rejects_reports_without_named_functions():
    document = report(name="")
    with pytest.raises(ValueError) as caught:
        coverage_failures(document, 80, ["src/omnitensor/subject.py"])
    assert str(caught.value) == "coverage report contains no measured functions"


def test_per_function_gate_reports_unknown_start_lines_exactly():
    document = report(79)
    function = document["files"]["src/omnitensor/subject.py"]["functions"]["Subject.method"]
    del function["start_line"]
    assert coverage_failures(document, 80, ["src/omnitensor/subject.py"]) == (
        "src/omnitensor/subject.py:? Subject.method 79.0%",
    )


def test_source_inventory_is_complete_and_refuses_empty_roots(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "a.py").write_text("pass\n", encoding="utf-8")
    nested = package / "nested"
    nested.mkdir()
    (nested / "b.py").write_text("pass\n", encoding="utf-8")

    assert source_files(package) == (
        (package / "a.py").resolve(),
        (nested / "b.py").resolve(),
    )
    with pytest.raises(ValueError, match="not a directory"):
        source_files(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError) as caught:
        source_files(empty)
    assert str(caught.value) == f"coverage source root contains no Python files: {empty}"


def test_per_function_gate_cli_returns_nonzero_for_a_gap(tmp_path, capsys):
    path = tmp_path / "coverage.json"
    source = tmp_path / "src/omnitensor"
    source.mkdir(parents=True)
    subject = source / "subject.py"
    subject.write_text("def method():\n    return 1\n", encoding="utf-8")
    path.write_text(json.dumps(report(50.0, filename=subject)), encoding="utf-8")

    assert main([str(path), "--source-root", str(source)]) == 1
    assert capsys.readouterr().out == (
        "Per-function coverage below 80%:\n"
        f"{subject}:7 Subject.method 50.0%\n"
    )


def test_per_function_gate_cli_reports_success_and_invalid_input(tmp_path, capsys):
    source = tmp_path / "src/omnitensor"
    source.mkdir(parents=True)
    subject = source / "subject.py"
    subject.write_text("def method():\n    return 1\n", encoding="utf-8")
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(report(filename=subject)), encoding="utf-8")

    assert main([str(path), "--source-root", str(source)]) == 0
    assert capsys.readouterr().out == "per-function coverage: 1 functions at or above 80%\n"
    assert main([str(tmp_path / "missing.json"), "--source-root", str(source)]) == 2
    assert capsys.readouterr().out.startswith("coverage gate failed: [Errno 2]")
    with pytest.raises(SystemExit, match="0"):
        main(["--help"])
    output = capsys.readouterr().out
    assert "Repository quality gates shared by local development and CI." in output
    assert "coverage.py JSON report" in output
    assert "--threshold THRESHOLD" in output
    assert "--source-root SOURCE_ROOT" in output


def mutation_report(*statuses):
    return "\n".join(
        f"    omnitensor.subject.x_method__mutmut_{index}: {status}"
        for index, status in enumerate(statuses, 1)
    )


def test_mutation_gate_accepts_floor_and_counts_timeouts_as_detected():
    text = mutation_report(*(["killed"] * 7), "timeout", "survived", "survived")
    assert mutation_failures(text, ["omnitensor.subject.x_method"], 80) == ()


def test_mutation_gate_names_each_callable_below_floor():
    assert mutation_failures(
        mutation_report("killed", "survived"),
        ["omnitensor.subject.x_method"],
        80,
    ) == ("omnitensor.subject.x_method 50.0% (1/2 detected)",)


@pytest.mark.parametrize(
    ("text", "selectors", "match"),
    [
        ("noise", ["omnitensor.subject.x_method"], "no results"),
        (mutation_report("killed"), [], "exact callables"),
        (mutation_report("killed"), "selector", "collection"),
        (mutation_report("killed"), ["omnitensor.subject.*"], "without wildcards"),
        (
            mutation_report("killed"),
            ["omnitensor.subject.x_method", "omnitensor.subject.x_method"],
            "unique",
        ),
        (mutation_report("killed"), ["omnitensor.missing.x_method"], "omits callable"),
        (mutation_report("not checked"), ["omnitensor.subject.x_method"], "untested"),
    ],
)
def test_mutation_gate_rejects_incomplete_or_ambiguous_scope(text, selectors, match):
    with pytest.raises(ValueError, match=match):
        mutation_failures(text, selectors, 80)


def test_mutation_result_parser_ignores_progress_noise():
    assert parse_results("progress\n" + mutation_report("survived")) == {
        "omnitensor.subject.x_method__mutmut_1": "survived"
    }


def test_mutation_gate_cli_returns_nonzero_for_a_gap(tmp_path, capsys):
    path = tmp_path / "mutmut.txt"
    path.write_text(mutation_report("killed", "survived"), encoding="utf-8")

    assert mutation_main(
        [str(path), "--selector", "omnitensor.subject.x_method"]
    ) == 1
    assert capsys.readouterr().out == (
        "Per-callable mutation score below 80%:\n"
        "omnitensor.subject.x_method 50.0% (1/2 detected)\n"
    )


def test_mutation_gate_cli_reports_success_and_invalid_input(tmp_path, capsys):
    path = tmp_path / "mutmut.txt"
    path.write_text(mutation_report("killed"), encoding="utf-8")
    arguments = [str(path), "--selector", "omnitensor.subject.x_method"]

    assert mutation_main(arguments) == 0
    assert capsys.readouterr().out == (
        "per-callable mutation score: 1 callables at or above 80%\n"
    )
    assert mutation_main([str(tmp_path / "missing"), "--selector", "x"]) == 2
    assert capsys.readouterr().out.startswith("mutation gate failed: [Errno 2]")
    with pytest.raises(SystemExit, match="0"):
        mutation_main(["--help"])
    output = capsys.readouterr().out
    assert "Fail-closed per-callable mutation-score gate for an exact mutmut" in output
    assert "output from mutmut results --all true" in output
    assert "--selector SELECTOR" in output

    with pytest.raises(SystemExit, match="2"):
        mutation_main(["report.txt"])
    assert "the following arguments are required: --selector" in capsys.readouterr().err
