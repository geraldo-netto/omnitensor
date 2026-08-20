"""One refusal shape, and every class that carries it."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from omnitensor.plugins.collection import CollectionError
from omnitensor.plugins.generation import GenerationError
from omnitensor.stable_error import StableError

ROOT = Path(__file__).parents[1] / "src" / "omnitensor"


@pytest.mark.parametrize("error_type", [CollectionError, GenerationError])
def test_the_shape_is_a_code_a_detail_and_the_two_of_them_as_text(error_type):
    error = error_type("source-invalid", "the source said nothing")

    assert error.code == "source-invalid"
    assert error.detail == "the source said nothing"
    assert str(error) == "source-invalid: the source said nothing"


def test_each_class_keeps_the_builtin_base_its_callers_catch():
    # `except ValueError` around a parser and `except RuntimeError` around a
    # worker call are what the call sites were written against.
    assert issubclass(CollectionError, ValueError)
    assert issubclass(GenerationError, RuntimeError)
    assert not issubclass(StableError, Exception)


def test_no_module_writes_the_shape_out_for_itself_again():
    """OMNI-0524: forty-nine classes each declared the same three lines."""
    offenders = []
    for path in ROOT.rglob("*.py"):
        if "__pycache__" in path.parts or path.name == "stable_error.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ClassDef):
                continue
            inits = [
                n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
            ]
            if not inits:
                continue
            body = [
                statement
                for statement in inits[0].body
                if not (
                    isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
                )
            ]
            dumped = ast.dump(ast.Module(body=body, type_ignores=[]), annotate_fields=False)
            if len(body) == 3 and "'code'" in dumped and "'detail'" in dumped and "super" in dumped:
                offenders.append(f"{path}:{node.lineno} {node.name}")
    # The one exception carries a typed rejection code rather than a string.
    assert [name.rsplit(" ", 1)[-1] for name in offenders] == ["_CandidateRejectedError"]
