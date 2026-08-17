"""llama.cpp JSON-grammar projection kept separate from native execution."""

from __future__ import annotations

import copy
from collections.abc import Mapping


def grammar_schema(schema: Mapping[str, object]) -> dict:
    """Project unsupported large string bounds for grammar only.

    The canonical task schema remains unchanged and validates the final reply.
    """
    projected = copy.deepcopy(dict(schema))
    definitions = projected.get("$defs", {})
    if not isinstance(definitions, dict):
        raise ValueError("generation schema definitions are invalid")
    projected = _inline_local_refs(projected, definitions, ())
    projected.pop("$defs", None)
    _project_grammar_keywords(projected)
    return projected


def _project_grammar_keywords(value) -> None:
    if isinstance(value, list):
        for child in value:
            _project_grammar_keywords(child)
        return
    if not isinstance(value, dict):
        return
    value.pop("maxLength", None)
    # A pattern written with `\d` compiles to literal garbage — llama.cpp emits
    # `"\d\d\d\d" "-\d-\d"` rather than a digit class — which is why these
    # were stripped wholesale. A pattern written as a character class compiles
    # correctly and is enforced, and enforcing the shape of a date is the whole
    # point of requiring one: a model that must emit the key will otherwise
    # displace a neighbouring value into it.
    pattern = value.get("pattern")
    if isinstance(pattern, str) and "\\d" in pattern:
        value.pop("pattern", None)
    properties = value.get("properties")
    if isinstance(properties, dict):
        confirmation = properties.get("confirmation")
        if isinstance(confirmation, dict) and "pending" in confirmation.get("enum", ()):
            properties["confirmation"] = {"const": "pending"}
    for child in value.values():
        _project_grammar_keywords(child)


def _inline_local_refs(value, definitions: Mapping[str, object], stack: tuple[str, ...]):
    """Inline bounded local definitions unsupported by llama.cpp grammars."""
    if isinstance(value, list):
        return [_inline_local_refs(child, definitions, stack) for child in value]
    if not isinstance(value, dict):
        return value
    reference = value.get("$ref")
    if reference is not None:
        prefix = "#/$defs/"
        if not isinstance(reference, str) or not reference.startswith(prefix):
            raise ValueError("generation schema uses an unsupported reference")
        name = reference.removeprefix(prefix)
        if name in stack or name not in definitions:
            raise ValueError("generation schema reference is invalid")
        target = definitions[name]
        if not isinstance(target, dict):
            raise ValueError("generation schema definition is invalid")
        merged = copy.deepcopy(target)
        merged.update({key: child for key, child in value.items() if key != "$ref"})
        return _inline_local_refs(merged, definitions, (*stack, name))
    return {key: _inline_local_refs(child, definitions, stack) for key, child in value.items()}
