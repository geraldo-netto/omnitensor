"""What counts as a correct answer, per workload, as checkable rules.

Every rule here is named, pure, and applied only when a case asks for it, so a
score can always be read back as "it failed *this*". A single number that
cannot be taken apart is a number nobody can act on — the point of this whole
exercise is choosing between models, and "0.71 versus 0.68" decides nothing if
neither can be explained.

Nothing here rewards brevity or punishes length. A model that answers fully is
not scored below one that answers shortly: this measures whether the answer is
right, and capping or trimming answers is not something this codebase does.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

HEBREW = re.compile(r"[֐-׿]")
# Cyrillic and Arabic, the two scripts a Hebrew translation leaks into. Kept
# identical to the check the shipped Hebrew runtime already applies
# (`omnitensor_qwen_runtime.hebrew._DISALLOWED_SCRIPT`): a benchmark that
# accepts what production refuses is measuring a workload nobody ships.
FOREIGN_SCRIPT = re.compile(r"[\u0400-\u052f\u0600-\u06ff]")


@dataclass(frozen=True, slots=True)
class Given:
    """What the model was handed: the reference names, and the text behind them.

    Both, because the rules need different halves — a citation rule checks the
    names, an invention rule checks the words — and two lists that must stay in
    step is a worse bargain than one value holding both.
    """

    references: tuple[str, ...] = ()
    texts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Judgement:
    """How one answer did, and on which rules."""

    case_id: str
    passed: tuple[str, ...]
    failed: tuple[str, ...]

    @property
    def score(self) -> float:
        checked = len(self.passed) + len(self.failed)
        return len(self.passed) / checked if checked else 0.0

    @property
    def correct(self) -> bool:
        """Every rule the case asked for. A partly-right answer is not right —
        an answer with a good summary and an invented citation is worse than
        no answer, because somebody will believe it."""
        return not self.failed


def judge(
    case,
    document: dict | None,
    sources: Sequence[str] | Given,
    texts: Sequence[str] = (),
) -> Judgement:
    """Apply every rule this case asks for to what the model returned.

    ``sources`` are the references a citation may name *and* the text those
    references held: the invention rules need the words, the citation rule
    needs the names, and passing two lists that must stay in step would be a
    worse bargain than passing both forms of the same thing.
    """
    given = sources if isinstance(sources, Given) else Given(tuple(sources), tuple(texts))
    if document is None:
        # Nothing parseable came back. Recorded as a failed rule rather than
        # skipped, because a model that cannot hold its own output contract is
        # exactly what this is meant to catch.
        return Judgement(case.id, (), ("produced-a-usable-document",))
    passed: list[str] = []
    failed: list[str] = []
    refused = _refused(document)
    if case.expect.get("refuses") is True:
        (passed if refused else failed).append("refuses")
        return Judgement(case.id, tuple(passed), tuple(failed))
    (failed if refused else passed).append("answers-rather-than-refusing")
    for name, rule in RULES.items():
        if name not in case.expect:
            continue
        (passed if rule(document, case.expect[name], given) else failed).append(name)
    return Judgement(case.id, tuple(passed), tuple(failed))


def _refused(document: dict) -> bool:
    state = str(document.get("confirmationState", "")).lower()
    return state == "refused" or document.get("outcome") == "refused"


def _text_of(document: dict) -> str:
    """Everything the answer says, wherever this workload puts it."""
    parts = [
        document.get("answer"),
        document.get("result"),
        document.get("summary"),
        document.get("detail"),
    ]
    # Both the answer shape a model emits and the result shape a plugin maps it
    # to: a rule should work on whichever document it is handed.
    for key in ("tasks", "events", "suggestions", "plan", "citations"):
        value = document.get(key)
        if isinstance(value, list):
            parts.extend(_flatten(item) for item in value)
    return " ".join(str(part) for part in parts if part)


def _flatten(item: object) -> str:
    if isinstance(item, dict):
        return " ".join(str(value) for value in item.values())
    return str(item)


def _contains_all(document: dict, wanted: object, _sources) -> bool:
    """Every one of these survives into the answer.

    Names, numbers, dates: the things a translation or a summary is not
    allowed to lose. Matched case-insensitively because casing is a rendering
    choice, not a fact.
    """
    text = _text_of(document).lower()
    return all(str(word).lower() in text for word in _as_list(wanted))


def _contains_none(document: dict, unwanted: object, _sources) -> bool:
    """None of these appear. For inventions a workload is prone to."""
    text = _text_of(document).lower()
    return not any(str(word).lower() in text for word in _as_list(unwanted))


def _at_least(document: dict, wanted: object, _sources) -> bool:
    """At least this many items in the named collection."""
    for key, minimum in dict(wanted).items():
        value = document.get(key)
        if not isinstance(value, list) or len(value) < int(minimum):
            return False
    return True


def _cites_only_supplied_sources(document: dict, wanted: object, given: Given) -> bool:
    """Every citation names a source that was actually given.

    The failure this catches is the one that matters most in a grounded
    workload: a confident answer citing a document that does not exist.
    """
    references = _references(document)
    if wanted is True and not references:
        return False
    supplied = set(given.references)
    return all(reference in supplied for reference in references)


def _references(document: dict) -> list[str]:
    found: list[str] = []
    for key in ("citations", "evidence"):
        value = document.get(key)
        entries = value if isinstance(value, list) else [value] if value else []
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("sourceRef"), str):
                found.append(entry["sourceRef"])
    for entry in document.get("events", []) or []:
        if isinstance(entry, dict):
            for item in entry.get("evidence", []) or []:
                if isinstance(item, dict) and isinstance(item.get("sourceRef"), str):
                    found.append(item["sourceRef"])
    return found


_NUMBER = re.compile(r"\d[\d.,/-]*")
_NAME = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b", re.MULTILINE)


def _digits(value: str) -> set[str]:
    return {run.strip(".,/-") for run in _NUMBER.findall(value) if run.strip(".,/-")}


def _no_invented_numbers(document: dict, _wanted, given: Given) -> bool:
    """Every number in the answer appears in what the model was given.

    The check that replaces "did it refuse" for a workload whose contract has
    no refusal state. Asked for a VAT number the invoice does not carry, a
    grounded workload may say so at any length it likes — what it may not do is
    produce a plausible one, and a fabricated number is the form that does the
    most damage.
    """
    supplied = set()
    for text in given.texts:
        supplied |= _digits(text)
    return _digits(_text_of(document)) <= supplied


def _no_invented_names(document: dict, _wanted, given: Given) -> bool:
    """Every capitalised name in the answer was in the sources.

    "Who witnessed the signature" against a document naming no witness: the
    failure is a name that was never there. Sentence-initial words are ignored,
    since capitalisation there is grammar rather than a name.
    """
    supplied = " ".join(given.texts).lower()
    return all(name.lower() in supplied for name in _NAME.findall(_text_of(document)))


def _in_hebrew(document: dict, wanted: object, _sources) -> bool:
    """The Hebrew case, which is why model choice matters at all here."""
    text = str(document.get("result") or document.get("answer") or "")
    return bool(HEBREW.search(text)) is bool(wanted)


def _no_foreign_script(document: dict, _wanted, _given) -> bool:
    """No Cyrillic or Arabic in the answer.

    Found by reading output the rules had passed: the 8B translated "if the
    report is late" into Hebrew with the Arabic word متأخر sitting in the
    middle of the sentence. The Hebrew script rule saw Hebrew and said yes.
    Production would have refused that answer outright, so this refuses it too.
    """
    return not FOREIGN_SCRIPT.search(_text_of(document))


def _operation_is(document: dict, wanted: object, _sources) -> bool:
    return str(document.get("operation", "")) == str(wanted)


def _field_equals(document: dict, wanted: object, _sources) -> bool:
    for path, expected in dict(wanted).items():
        if _dig(document, path.split(".")) != expected:
            return False
    return True


def _event_at(document: dict, wanted: object, _sources) -> bool:
    """An event that starts when the case says, however many others were found.

    A wanted moment is `YYYY-MM-DD`, `YYYY-MM-DDTHH:MM`, or `THH:MM` for a case
    whose source states a time and no date — which version 2 records rather
    than refusing, so the cases have to be able to ask for it.
    """
    found = set()
    for entry in document.get("events", []) or []:
        if not isinstance(entry, dict):
            continue
        when = entry.get("when") or {}
        date = str(when.get("date") or "")
        time = str(when.get("time") or "")[:5]
        found.update({f"{date}T{time}", date, f"T{time}"} - {"", "T"})
    return all(str(moment)[:16] in found for moment in _as_list(wanted))


def _event_needs(document: dict, wanted: object, _sources) -> bool:
    """Every event reports exactly the gaps the case says its source has.

    The rule that would have caught the original defect: an extraction that
    silently completes a partial source passes `event_starts` and is still
    wrong, because a calendar will place it somewhere nobody chose.
    """
    wanted_sets = [set(_as_list(item)) for item in _as_list(wanted)]
    events = [entry for entry in document.get("events", []) or [] if isinstance(entry, dict)]
    if len(events) != len(wanted_sets):
        return False
    reported = [set((entry.get("when") or {}).get("needs") or ()) for entry in events]
    return all(any(item == expected for item in reported) for expected in wanted_sets)


def _dig(document: object, path: Sequence[str]):
    current = document
    for step in path:
        if not isinstance(current, dict) or step not in current:
            return None
        current = current[step]
    return current


def _as_list(value: object) -> list:
    return list(value) if isinstance(value, (list, tuple)) else [value]


RULES: dict[str, Callable[[dict, object, Given], bool]] = {
    "contains": _contains_all,
    "absent": _contains_none,
    "at_least": _at_least,
    "cites_supplied_sources": _cites_only_supplied_sources,
    "hebrew": _in_hebrew,
    "no_foreign_script": _no_foreign_script,
    "no_invented_numbers": _no_invented_numbers,
    "no_invented_names": _no_invented_names,
    "operation": _operation_is,
    "fields": _field_equals,
    "event_starts": _event_at,
    "event_needs": _event_needs,
}


@dataclass(frozen=True, slots=True)
class Tally:
    """A whole run of cases, and where it went wrong."""

    judgements: tuple[Judgement, ...]

    @property
    def correct(self) -> int:
        return sum(1 for judgement in self.judgements if judgement.correct)

    @property
    def total(self) -> int:
        return len(self.judgements)

    @property
    def accuracy(self) -> float:
        """Whole answers that were right. Deliberately strict."""
        return self.correct / self.total if self.total else 0.0

    @property
    def rule_score(self) -> float:
        """Rules passed across every case: the partial-credit view, kept
        beside the strict one because they disagree in a way that is
        informative — a model that is 90% of the way there on every case is a
        different animal from one that is perfect on 60% and lost on the
        rest."""
        checked = sum(len(j.passed) + len(j.failed) for j in self.judgements)
        passed = sum(len(j.passed) for j in self.judgements)
        return passed / checked if checked else 0.0

    def failures(self) -> dict[str, int]:
        """Which rules failed, most common first. This is what to read."""
        counted: dict[str, int] = {}
        for judgement in self.judgements:
            for name in judgement.failed:
                counted[name] = counted.get(name, 0) + 1
        return dict(sorted(counted.items(), key=lambda item: (-item[1], item[0])))


__all__ = ["HEBREW", "Given", "Judgement", "RULES", "Tally", "judge"]
