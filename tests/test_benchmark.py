"""What counts as a right answer, and what a benchmark must never reward.

The scoring is where a benchmark becomes honest or useless, so it is pinned
here without a GPU: every rule is pure, and every rule is checked only when a
case asks for it. Two properties matter more than the rest and are tested from
several directions — that an invented citation fails, and that a refusal is
right or wrong depending on what the case actually contains.

Nothing here rewards a shorter answer. That is deliberate and enforced: this
codebase does not cap answers, and a benchmark that scored brevity would push
in exactly that direction.
"""

from __future__ import annotations

import json

import pytest

from omnitensor.benchmark import cases as case_files
from omnitensor.benchmark.cases import Case, CaseError
from omnitensor.benchmark.harness import Outcome, Run, Timing
from omnitensor.benchmark.report import as_document, as_markdown, write_results
from omnitensor.benchmark.scoring import Judgement, Tally, judge

SOURCES = ("case:one:0", "case:one:1")


def case(**overrides) -> Case:
    base = {
        "id": "example",
        "workload": "ask-selected-files",
        "sources": ("What is the total?", "Total due: 4,235.00 EUR"),
        "expect": {},
    }
    base.update(overrides)
    return Case(**base)


class TestJudgingAnAnswer:
    def test_an_answer_carrying_the_facts_passes(self):
        judgement = judge(
            case(expect={"contains": ["4,235"]}),
            {"answer": "The total due is 4,235.00 EUR."},
            SOURCES,
        )

        assert judgement.correct is True

    def test_a_fact_the_answer_lost_fails(self):
        """Names, numbers and dates are what a translation or a summary is not
        allowed to drop."""
        judgement = judge(
            case(expect={"contains": ["4,235"]}),
            {"answer": "The total is stated in the invoice."},
            SOURCES,
        )

        assert judgement.correct is False
        assert judgement.failed == ("contains",)

    def test_nothing_parseable_is_a_failure_and_not_a_skip(self):
        """A model that cannot hold its own output contract is exactly what
        this is meant to catch."""
        judgement = judge(case(), None, SOURCES)

        assert judgement.correct is False
        assert judgement.failed == ("produced-a-usable-document",)

    def test_a_rule_the_case_does_not_ask_for_is_not_applied(self):
        judgement = judge(case(expect={}), {"answer": "anything at all"}, SOURCES)

        assert judgement.correct is True
        assert "contains" not in judgement.passed

    def test_the_score_says_which_rules_it_is_made_of(self):
        """A number that cannot be taken apart decides nothing."""
        judgement = judge(
            case(expect={"contains": ["4,235"], "absent": ["4,235"]}),
            {"answer": "4,235.00 EUR"},
            SOURCES,
        )

        assert judgement.passed == ("answers-rather-than-refusing", "contains")
        assert judgement.failed == ("absent",)
        assert judgement.score == pytest.approx(2 / 3)


class TestCitations:
    """The failure that matters most in a grounded workload: a confident
    answer citing a document that does not exist."""

    def test_citing_a_supplied_source_passes(self):
        judgement = judge(
            case(expect={"cites_supplied_sources": True}),
            {"answer": "4,235.00 EUR", "citations": [{"sourceRef": SOURCES[1]}]},
            SOURCES,
        )

        assert judgement.correct is True

    def test_citing_something_that_was_never_supplied_fails(self):
        judgement = judge(
            case(expect={"cites_supplied_sources": True}),
            {"answer": "4,235.00 EUR", "citations": [{"sourceRef": "invoice-2025.pdf"}]},
            SOURCES,
        )

        assert judgement.failed == ("cites_supplied_sources",)

    def test_citing_nothing_at_all_fails_when_the_case_asks_for_citations(self):
        judgement = judge(
            case(expect={"cites_supplied_sources": True}),
            {"answer": "4,235.00 EUR", "citations": []},
            SOURCES,
        )

        assert judgement.failed == ("cites_supplied_sources",)

    def test_evidence_inside_an_event_is_checked_too(self):
        """Event extraction puts its references one level down, and an
        unchecked reference is an unchecked claim."""
        judgement = judge(
            case(expect={"cites_supplied_sources": True}),
            {"events": [{"start": "2026-09-03T14:00", "evidence": [{"sourceRef": "elsewhere"}]}]},
            SOURCES,
        )

        assert judgement.failed == ("cites_supplied_sources",)


class TestRefusals:
    def test_a_case_with_no_answer_in_it_expects_a_refusal(self):
        """Inventing a VAT number is worse than saying the document has none."""
        judgement = judge(
            case(expect={"refuses": True}),
            {"outcome": "refused", "code": "no-event-supported"},
            SOURCES,
        )

        assert judgement.correct is True

    def test_answering_a_question_the_document_cannot_answer_fails(self):
        judgement = judge(
            case(expect={"refuses": True}),
            {"answer": "The VAT number is PT501234567."},
            SOURCES,
        )

        assert judgement.correct is False

    def test_refusing_a_case_that_was_answerable_fails(self):
        """The live defect this exists to measure: event-extraction declining
        documents that meet its own stated bar."""
        judgement = judge(
            case(expect={"contains": ["4,235"]}),
            {"confirmationState": "refused", "detail": "no supported content"},
            SOURCES,
        )

        assert judgement.correct is False
        assert "answers-rather-than-refusing" in judgement.failed

    def test_a_refusal_case_is_not_also_judged_on_content(self):
        """It refused, so there is no content to judge, and reporting a second
        failure for the missing text would double-count one behaviour."""
        judgement = judge(
            case(expect={"refuses": True, "contains": ["never mind"]}),
            {"outcome": "refused"},
            SOURCES,
        )

        assert judgement.failed == ()


class TestTheOtherRules:
    def test_hebrew_is_checked_by_script_rather_than_by_claim(self):
        """A model that answers "this is Hebrew" in English has not
        translated anything."""
        hebrew = judge(case(expect={"hebrew": True}), {"result": "הפגישה תתקיים בליסבון"}, SOURCES)
        english = judge(
            case(expect={"hebrew": True}), {"result": "The meeting is in Lisbon"}, SOURCES
        )

        assert hebrew.correct is True
        assert english.correct is False

    def test_an_operation_must_be_the_one_that_was_asked_for(self):
        judgement = judge(
            case(expect={"operation": "translate"}),
            {"operation": "summarize", "result": "short"},
            SOURCES,
        )

        assert judgement.failed == ("operation",)

    def test_a_collection_can_be_required_to_hold_enough_items(self):
        three = {"tasks": [{"t": 1}, {"t": 2}, {"t": 3}], "operation": "extract-tasks"}
        one = {"tasks": [{"t": 1}], "operation": "extract-tasks"}

        assert judge(case(expect={"at_least": {"tasks": 3}}), three, SOURCES).correct
        assert not judge(case(expect={"at_least": {"tasks": 3}}), one, SOURCES).correct

    def test_an_event_is_checked_at_its_stated_start(self):
        found = {"events": [{"when": {"date": "2026-09-03", "time": "14:00", "needs": []}}]}

        assert judge(case(expect={"event_starts": ["2026-09-03T14:00"]}), found, SOURCES).correct

    def test_an_event_at_the_wrong_time_is_not_the_event(self):
        found = {"events": [{"when": {"date": "2026-09-03", "time": "09:00", "needs": []}}]}

        assert not judge(
            case(expect={"event_starts": ["2026-09-03T14:00"]}), found, SOURCES
        ).correct

    def test_a_source_that_states_only_a_date_can_be_asked_for_by_date(self):
        # Version 2 keeps an event whose source named no time, so a case has to
        # be able to expect exactly that rather than a full datetime.
        found = {"events": [{"when": {"date": "2026-12-27", "needs": ["time"]}}]}

        assert judge(case(expect={"event_starts": ["2026-12-27"]}), found, SOURCES).correct

    def test_an_extraction_that_invents_the_missing_part_is_caught(self):
        # The rule that would have caught the original defect from the other
        # side: silently completing a partial source passes `event_starts` and
        # still puts an event where nobody chose.
        invented = {"events": [{"when": {"date": "2026-12-27", "time": "09:00", "needs": []}}]}

        assert not judge(case(expect={"event_needs": [["time"]]}), invented, SOURCES).correct
        honest = {"events": [{"when": {"date": "2026-12-27", "needs": ["time"]}}]}
        assert judge(case(expect={"event_needs": [["time"]]}), honest, SOURCES).correct

    def test_a_longer_answer_is_never_scored_below_a_shorter_one(self):
        """Enforced rather than assumed: a benchmark that rewarded brevity
        would push toward exactly the capping this codebase refuses to do."""
        wanted = case(expect={"contains": ["4,235"]})
        short = judge(wanted, {"answer": "4,235.00 EUR"}, SOURCES)
        long = judge(
            wanted,
            {
                "answer": "The invoice from Nordwind Lda states a total due of 4,235.00 EUR, "
                "made up of a 3,500.00 EUR subtotal and 735.00 EUR of VAT."
            },
            SOURCES,
        )

        assert short.score == long.score


class TestATallyOfManyCases:
    def judged(self, *outcomes: bool) -> Tally:
        return Tally(
            tuple(
                Judgement(f"case-{index}", ("rule",) if ok else (), () if ok else ("rule",))
                for index, ok in enumerate(outcomes)
            )
        )

    def test_accuracy_counts_whole_answers(self):
        """Strict on purpose: an answer with a good summary and an invented
        citation is worse than no answer, because somebody will believe it."""
        assert self.judged(True, True, False, False).accuracy == 0.5

    def test_the_partial_view_is_kept_beside_the_strict_one(self):
        tally = Tally((Judgement("a", ("one", "two"), ("three",)),))

        assert tally.accuracy == 0.0
        assert tally.rule_score == pytest.approx(2 / 3)

    def test_the_failures_are_reported_most_common_first(self):
        tally = Tally(
            (
                Judgement("a", (), ("cites_supplied_sources",)),
                Judgement("b", (), ("cites_supplied_sources",)),
                Judgement("c", (), ("contains",)),
            )
        )

        assert list(tally.failures()) == ["cites_supplied_sources", "contains"]

    def test_an_empty_tally_does_not_divide_by_zero(self):
        assert Tally(()).accuracy == 0.0
        assert Tally(()).rule_score == 0.0


class TestTheCaseFiles:
    def test_every_shipped_case_file_loads(self):
        available = case_files.available()

        assert set(available) >= {
            "ask-selected-files",
            "event-extraction",
            "file-organizer",
            "selected-text-tools",
        }
        for workload in available:
            assert case_files.load(workload)

    def test_every_expectation_names_a_rule_that_exists(self):
        """An expectation nobody checks is a comment pretending to be data."""
        from omnitensor.benchmark.scoring import RULES

        for workload in case_files.available():
            for loaded in case_files.load(workload):
                unknown = set(loaded.expect) - set(RULES) - {"refuses"}
                assert not unknown, f"{loaded.id} states unchecked expectations: {unknown}"

    def test_every_grounded_workload_is_asked_something_it_cannot_answer(self):
        """A benchmark of only answerable cases rewards a model that answers
        everything, which is the failure mode of a grounded workload.

        How that is checked differs by contract, and both forms count.
        `event-extraction` can refuse, so its cases say so. `ask-selected-files`
        has no refusal state at all — it holds an answer and citations — so its
        unanswerable cases check that nothing was invented instead."""
        for workload in ("ask-selected-files", "event-extraction"):
            loaded = case_files.load(workload)
            unanswerable = [
                item
                for item in loaded
                if not item.answerable
                or "no_invented_numbers" in item.expect
                or "no_invented_names" in item.expect
            ]

            assert unanswerable, f"{workload} is only asked what it can answer"
            assert any(item.answerable for item in loaded)

    def test_a_case_without_sources_is_refused(self, tmp_path):
        (tmp_path / "broken.json").write_text(json.dumps({"cases": [{"id": "x", "sources": []}]}))

        with pytest.raises(CaseError):
            case_files.load("broken", tmp_path)

    def test_a_file_that_is_not_cases_is_refused(self, tmp_path):
        (tmp_path / "broken.json").write_text("[]")

        with pytest.raises(CaseError):
            case_files.load("broken", tmp_path)

    def test_a_missing_file_says_which_one(self, tmp_path):
        with pytest.raises(CaseError, match="no case file"):
            case_files.load("absent", tmp_path)


def run(**overrides) -> Run:
    base = {
        "workload": "ask-selected-files",
        "model_id": "qwen3-8b-q4-k-m",
        "device": "vulkan-0",
        "outcomes": (
            Outcome("a", Judgement("a", ("contains",), ()), Timing(12.0, 600), {"answer": "x"}),
            Outcome("b", Judgement("b", (), ("contains",)), Timing(8.0, 200), {"answer": "y"}),
        ),
        "load_seconds": 30.0,
        "layers_offloaded": 37,
        "context_tokens": 32_768,
        "cache": "q8_0",
    }
    base.update(overrides)
    return Run(**base)


class TestWritingItDown:
    def test_the_document_holds_every_case_so_a_surprise_can_be_taken_apart(self):
        document = as_document([run()])

        assert document["runs"][0]["cases"] == 2
        assert [outcome["caseId"] for outcome in document["runs"][0]["outcomes"]] == ["a", "b"]
        assert document["runs"][0]["failures"] == {"contains": 1}

    def test_the_document_records_how_the_model_was_run(self):
        """Layers, context and cache: the same numbers a qualification receipt
        holds, so a score can be tied to the conditions that produced it."""
        recorded = as_document([run()])["runs"][0]

        assert (recorded["layersOffloaded"], recorded["contextTokens"], recorded["cache"]) == (
            37,
            32_768,
            "q8_0",
        )

    def test_the_table_puts_the_two_models_side_by_side(self):
        printed = as_markdown([run(), run(model_id="qwen3-4b-q4-k-m")])

        assert "qwen3-8b-q4-k-m" in printed
        assert "qwen3-4b-q4-k-m" in printed
        assert printed.count("| ask-selected-files |") == 2

    def test_results_are_written_where_they_can_be_read_later(self, tmp_path):
        written = write_results([run()], tmp_path)

        assert written.exists()
        assert json.loads((tmp_path / "benchmark.json").read_text())["runs"]

    def test_writing_twice_replaces_rather_than_accumulates(self, tmp_path):
        """Written after every model, so an interrupted run leaves findings
        behind — which only works if the file is the whole story each time."""
        write_results([run()], tmp_path)
        write_results([run(), run(model_id="qwen3-4b-q4-k-m")], tmp_path)

        assert len(json.loads((tmp_path / "benchmark.json").read_text())["runs"]) == 2


class TestCatchingAnInvention:
    """What replaces "did it refuse" for a workload whose contract has no
    refusal state.

    `document-question-result` holds an answer and citations and nothing else,
    so asking whether it refused is asking about a field that does not exist —
    a flaw in the first version of these cases, found when the 8B "failed" two
    of them by answering perfectly reasonably. What a grounded workload may not
    do is produce a fact that was never there, and that is checkable.
    """

    def sources(self):
        return (
            "What is the supplier's VAT identification number?",
            "INVOICE 2026-0417\nSupplier: Nordwind Lda\nTotal due: 4,235.00 EUR",
        )

    def judged(self, answer, expect):
        return judge(
            case(expect=expect),
            {"answer": answer},
            ["case:x:0", "case:x:1"],
            self.sources(),
        )

    def test_saying_the_document_does_not_carry_it_passes(self):
        """At any length it likes: this measures whether the answer is right,
        never whether it is short."""
        judgement = self.judged(
            "The invoice does not state a VAT identification number for Nordwind "
            "Lda. It carries the invoice number 2026-0417 and a total of 4,235.00 "
            "EUR, but no tax identifier appears anywhere in the document.",
            {"no_invented_numbers": True},
        )

        assert judgement.correct is True

    def test_a_plausible_invented_number_fails(self):
        judgement = self.judged(
            "The supplier's VAT number is PT501234567.", {"no_invented_numbers": True}
        )

        assert judgement.failed == ("no_invented_numbers",)

    def test_numbers_that_were_in_the_document_are_not_inventions(self):
        judgement = self.judged(
            "Invoice 2026-0417 totals 4,235.00 EUR.", {"no_invented_numbers": True}
        )

        assert judgement.correct is True

    def test_a_name_that_was_never_supplied_fails(self):
        judgement = judge(
            case(expect={"no_invented_names": True}),
            {"answer": "The signature was witnessed by Carlos Pereira."},
            ["case:x:0"],
            ("SERVICE AGREEMENT signed in Lisbon by Marta Oliveira.",),
        )

        assert judgement.failed == ("no_invented_names",)

    def test_a_name_from_the_document_is_fine(self):
        judgement = judge(
            case(expect={"no_invented_names": True}),
            {"answer": "The document names only Marta Oliveira, who signed it."},
            ["case:x:0"],
            ("SERVICE AGREEMENT signed in Lisbon by Marta Oliveira.",),
        )

        assert judgement.correct is True

    def test_a_sentence_starting_with_a_capital_is_not_a_name(self):
        """Capitalisation there is grammar, and treating it as invention would
        fail every well-formed answer."""
        judgement = judge(
            case(expect={"no_invented_names": True}),
            {
                "answer": "Nothing in the document names a witness. The agreement "
                "records only the signatory."
            },
            ["case:x:0"],
            ("SERVICE AGREEMENT signed in Lisbon by Marta Oliveira.",),
        )

        assert judgement.correct is True

    def test_the_citation_rule_still_reads_references_not_text(self):
        """Both halves travel together now; each rule takes the half it needs."""
        judgement = judge(
            case(expect={"cites_supplied_sources": True}),
            {"answer": "x", "citations": [{"sourceRef": "case:x:1"}]},
            ["case:x:0", "case:x:1"],
            self.sources(),
        )

        assert judgement.correct is True


class TestAskingForWhatTheModelActuallyEmits:
    """A case must name the shape the model produces, not the one a plugin
    maps it to afterwards.

    Both mistakes in this file's first version were this: a refusal state that
    only the plugin result has, and `plan` when the answer schema says
    `suggestions`. Neither could ever pass, and each cost a workload's worth of
    GPU time to discover. This pins the pairing so the next one is caught here.
    """

    # Workload, the schema its task declares, and the collections that schema
    # holds. The names differ from the workload ids, which is part of why this
    # went wrong twice.
    ANSWER_SCHEMAS = {
        "ask-selected-files": ("document-question-answer", {"answer", "citations"}),
        "file-organizer": ("file-organizer-answer", {"suggestions"}),
        "selected-text-tools": (
            "selected-text-answer",
            {"operation", "result", "tasks", "evidence"},
        ),
    }

    def collections_named_by(self, workload):
        named = set()
        for loaded in case_files.load(workload):
            named |= set(loaded.expect.get("at_least", {}))
        return named

    @pytest.mark.parametrize("workload", sorted(ANSWER_SCHEMAS))
    def test_every_collection_a_case_counts_is_one_the_answer_holds(self, workload):
        _schema, keys = self.ANSWER_SCHEMAS[workload]
        named = self.collections_named_by(workload)

        assert named <= keys, (
            f"{workload} cases count {named - keys}, which the task's answer schema does not hold"
        )

    def test_the_answer_shapes_here_are_the_ones_the_schemas_declare(self):
        """So this test fails if a schema moves, rather than quietly agreeing
        with a stale copy of it."""
        from omnitensor.registry import load_schema

        for workload, (name, expected) in self.ANSWER_SCHEMAS.items():
            declared = set(load_schema(f"{name}.schema.json").get("properties", {}))
            declared -= {"version", "requestId"}

            assert expected <= declared, f"{workload}: {expected - declared} is not declared"

    def test_a_rule_reads_both_the_answer_shape_and_the_result_shape(self):
        """The same rule is useful on either document, and which one it gets
        depends on whether a plugin has run."""
        answer = judge(
            case(expect={"contains": ["deploy.sh"]}),
            {"suggestions": [{"fileName": "deploy.sh"}]},
            SOURCES,
        )
        result = judge(
            case(expect={"contains": ["deploy.sh"]}),
            {"plan": [{"fileName": "deploy.sh"}]},
            SOURCES,
        )

        assert answer.correct is True
        assert result.correct is True


class TestScriptLeaks:
    """A Hebrew translation that is partly Arabic.

    Found by reading output these rules had already passed: the 8B produced
    "אם הדוח متأخر" — the Arabic for "late" inside a Hebrew sentence — and the
    Hebrew rule saw Hebrew and said yes. The shipped Hebrew runtime refuses
    exactly this, so a benchmark that accepts it is measuring a workload nobody
    ships.
    """

    def judged(self, text):
        return judge(
            case(expect={"hebrew": True, "no_foreign_script": True}),
            {"result": text, "operation": "translate"},
            SOURCES,
        )

    def test_clean_hebrew_passes(self):
        assert self.judged("הפגישה תתקיים בליסבון ב-3 בספטמבר 2026.").correct is True

    def test_an_arabic_word_inside_hebrew_fails(self):
        judgement = self.judged("אם הדוח متأخر, ריבית של 3.5 אחוז לשנה תחול.")

        assert judgement.failed == ("no_foreign_script",)

    def test_cyrillic_fails_the_same_way(self):
        assert self.judged("הפגישה תתקיים במוסקבה Москва.").failed == ("no_foreign_script",)

    def test_latin_is_not_a_leak(self):
        """Numbers, currency codes and proper names in Latin script are normal
        in Hebrew text; refusing them would fail correct translations."""
        assert self.judged("מארטה אישרה 42,000 EUR לשלב השני.").correct is True

    def test_the_rule_matches_what_production_refuses(self):
        """Kept in step deliberately: this is the same character range the
        shipped Hebrew runtime rejects."""
        from omnitensor.benchmark.scoring import FOREIGN_SCRIPT

        for character in ("Ѐ", "ԯ", "؀", "ۿ"):
            assert FOREIGN_SCRIPT.search(character)
        for character in ("א", "A", "1", "€"):
            assert not FOREIGN_SCRIPT.search(character)


class TestARuleThatDoesNotExist:
    """A case naming a rule the code lacks must stop the run, not be skipped.

    The judge applies the rules it has; an expectation with no rule behind it
    was simply not checked, and the answer scored as though the case had never
    asked. That is not hypothetical — a benchmark process holding an older
    module scored a Hebrew translation containing Cyrillic as 10/10 against a
    case that forbade exactly that.
    """

    def test_a_case_naming_an_unknown_rule_is_refused_at_load(self, tmp_path):
        (tmp_path / "made-up.json").write_text(
            json.dumps({"cases": [{"id": "x", "sources": ["text"], "expect": {"vibes": True}}]})
        )

        with pytest.raises(CaseError, match="does not have"):
            case_files.load("made-up", tmp_path)

    def test_the_refusal_names_the_rule_so_it_can_be_fixed(self, tmp_path):
        (tmp_path / "made-up.json").write_text(
            json.dumps(
                {"cases": [{"id": "x", "sources": ["t"], "expect": {"no_foreign_scrpt": True}}]}
            )
        )

        with pytest.raises(CaseError, match="no_foreign_scrpt"):
            case_files.load("made-up", tmp_path)

    def test_refuses_is_allowed_although_it_is_not_in_the_rule_table(self, tmp_path):
        """It is answered before the rules run, so it has no entry there."""
        (tmp_path / "fine.json").write_text(
            json.dumps({"cases": [{"id": "x", "sources": ["t"], "expect": {"refuses": True}}]})
        )

        assert case_files.load("fine", tmp_path)
