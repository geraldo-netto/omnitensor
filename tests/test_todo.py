"""The finding ledger's own rules, which nothing enforced.

`AGENTS.md` states the schema of `TODO.md` exactly — three tables, one column
list, which statuses belong in which table, stable `OMNI-0001` ids, no
duplicates — and no test read the file. Six rows had drifted into the wrong
table by 2026-08-20: `blocked` findings sitting in `Open`, where they read as
work somebody could pick up, which is precisely what moving them to
`Blocked / Deferred` exists to prevent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TODO = ROOT / "TODO.md"
AGENTS = ROOT / "AGENTS.md"

SCHEMA = "| id | status | severity | effort | description |"
STATUSES = {
    "Open": {"open", "in_progress"},
    "Blocked / Deferred": {"blocked", "deferred"},
    "Rejected / Won't fix": {"rejected", "wont_fix"},
}
SEVERITIES = {"critical", "high", "medium", "low"}
EFFORTS = {"xs", "s", "m", "l", "xl"}
IDENTIFIER = re.compile(r"^OMNI-\d{4}$")


def tables() -> dict[str, list[list[str]]]:
    """Each table's data rows, by heading, as split cells."""
    found: dict[str, list[list[str]]] = {}
    heading = None
    for line in TODO.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            heading = line.removeprefix("## ").strip()
            continue
        if not line.startswith("|") or heading is None:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells[0] in ("id", "---"):
            # The heading is recorded from its schema line rather than from a
            # row, so a table with nothing in it is still a table. Every open
            # finding being resolved is the state this ledger exists to reach,
            # and it used to read here as a missing table.
            found.setdefault(heading, [])
            continue
        found.setdefault(heading, []).append(cells)
    return found


def rows() -> list[tuple[str, list[str]]]:
    return [(heading, row) for heading, table in tables().items() for row in table]


def test_the_ledger_has_exactly_the_three_tables_the_rules_name():
    assert set(tables()) == set(STATUSES)


@pytest.mark.parametrize("heading", sorted(STATUSES))
def test_every_table_declares_the_one_schema(heading):
    """All three use exactly this schema; a fourth column is a column nothing reads."""
    body = TODO.read_text(encoding="utf-8").split(f"## {heading}", 1)[1]
    assert body.lstrip().startswith(SCHEMA)
    assert SCHEMA in AGENTS.read_text(encoding="utf-8")


def test_every_row_has_the_five_cells_and_nothing_else():
    wrong = [(heading, row[0], len(row)) for heading, row in rows() if len(row) != 5]

    assert wrong == []


def test_a_status_belongs_to_the_table_it_is_written_in():
    """A `blocked` finding in `Open` reads as work somebody could pick up."""
    misplaced = [
        f"{heading}:{row[0]}:{row[1]}" for heading, row in rows() if row[1] not in STATUSES[heading]
    ]

    assert misplaced == []


def test_severity_and_effort_are_from_the_declared_vocabularies():
    wrong = [
        f"{row[0]}:{row[2]}/{row[3]}"
        for _heading, row in rows()
        if row[2] not in SEVERITIES or row[3] not in EFFORTS
    ]

    assert wrong == []


def test_every_id_is_unique_and_in_the_stable_form():
    identifiers = [row[0] for _heading, row in rows()]
    duplicated = sorted({name for name in identifiers if identifiers.count(name) > 1})
    malformed = sorted(name for name in identifiers if IDENTIFIER.fullmatch(name) is None)

    assert duplicated == []
    assert malformed == []
    assert len(identifiers) > 50


def test_every_row_says_something():
    """A row with an empty description is an id nobody can act on."""
    empty = [row[0] for _heading, row in rows() if len(row[4]) < 20]

    assert empty == []


def test_no_row_is_left_in_the_transitional_status():
    """`done` is transitional only: a resolved row is removed in the same commit."""
    left = [row[0] for _heading, row in rows() if row[1] == "done"]

    assert left == []
