from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from omnitensor.plugins.target_language import (
    MAX_LANGUAGE_CHARACTERS,
    TARGET_LANGUAGE_PATTERN,
    valid_target_language,
)

ROOT = Path(__file__).parents[1]
MANIFESTS = ROOT / "plugin-manifests"
TRANSLATING = {
    "selected-text-tools": ("language",),
    "ask-selected-files": ("language",),
}


@pytest.mark.parametrize(
    "language",
    [
        "Português",
        "中文",
        "עברית",
        "Ελληνικά",
        "русский",
        "zh-CN",
        "Brazilian Portuguese",
        "K'iche'",
        " Português ",
        "English",
    ],
)
def test_a_person_may_name_their_own_language_in_their_own_script(language):
    """The old rule refused most of these, and refused them after the job ran."""
    assert valid_target_language(language)


@pytest.mark.parametrize(
    "value",
    ["", "   ", "1abc", "-fr", "'fr'", "fr\nnext", "x" * (MAX_LANGUAGE_CHARACTERS + 1)],
)
def test_what_is_refused_is_a_defect_rather_than_a_language(value):
    assert not valid_target_language(value)


@pytest.mark.parametrize("value", [None, 3, True, ["fr"], {"language": "fr"}])
def test_a_value_that_is_not_text_is_not_a_language(value):
    assert not valid_target_language(value)


@pytest.mark.parametrize(("workload_id", "fields"), sorted(TRANSLATING.items()))
def test_the_manifest_and_the_module_state_one_rule(workload_id, fields):
    """G5: two statements of one rule that disagreed cost a whole generation.

    The manifest refused what the module accepted, so a target the schema
    rejects arrived as `language-invalid` after the run rather than before it.
    """
    document = json.loads((MANIFESTS / f"{workload_id}.json").read_text(encoding="utf-8"))
    properties = document["plugin"]["schemas"]["input"]["properties"]
    for field in fields:
        assert properties[field]["pattern"] == TARGET_LANGUAGE_PATTERN
        assert properties[field]["maxLength"] == MAX_LANGUAGE_CHARACTERS


@pytest.mark.parametrize(("workload_id", "fields"), sorted(TRANSLATING.items()))
def test_every_name_the_module_accepts_the_manifest_accepts_too(workload_id, fields):
    document = json.loads((MANIFESTS / f"{workload_id}.json").read_text(encoding="utf-8"))
    properties = document["plugin"]["schemas"]["input"]["properties"]
    for field in fields:
        validator = jsonschema.Draft202012Validator(properties[field])
        for language in ("Português", "中文", "zh-CN", " Português "):
            assert valid_target_language(language)
            validator.validate(language)
        for value in ("1abc", "-fr", ""):
            assert not valid_target_language(value)
            with pytest.raises(jsonschema.ValidationError):
                validator.validate(value)
