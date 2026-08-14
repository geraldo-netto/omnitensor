from __future__ import annotations

import pytest

from omnitensor.mutation_quality import main as mutation_main
from omnitensor.mutation_quality import mutation_failures, parse_results
from omnitensor.quality import validate_threshold


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
        (mutation_report("killed"), ["omnitensor.subject.x_method?"], "without wildcards"),
        (mutation_report("killed"), ["omnitensor.subject.x_method[1]"], "without wildcards"),
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


@pytest.mark.parametrize("threshold", [None, True, -1, 101, float("inf"), float("nan")])
def test_mutation_gate_rejects_fail_open_thresholds(threshold):
    with pytest.raises(ValueError):
        validate_threshold(threshold)


def test_mutation_result_parser_ignores_progress_noise():
    assert parse_results("progress\n" + mutation_report("survived")) == {
        "omnitensor.subject.x_method__mutmut_1": "survived"
    }
    with pytest.raises(ValueError, match="repeats result"):
        parse_results(mutation_report("killed") + "\n" + mutation_report("survived"))


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
    assert "--selector-file SELECTOR_FILE" in output
    assert "--shard SHARD" in output

    with pytest.raises(SystemExit, match="2"):
        mutation_main(["report.txt"])
    assert "one of the arguments --selector --selector-file is required" in (
        capsys.readouterr().err
    )
