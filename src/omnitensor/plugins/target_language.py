"""What a person may name as a target language, stated once.

Two workloads translate — `selected-text-tools` into a language somebody
types, `document-translation` into one it requires — and each stated the rule
twice: in its manifest's input contract and again in its own module. Four
copies of one sentence.

The rule itself used to be `^[A-Za-z][A-Za-z -]*$`, which refuses the name
most people would write for their own language: "Português" is a refusal, and
so are "中文", "עברית" and every tag with a digit in it. The refusal arrives
after the job has run, as `language-invalid`, so the cost of it is a whole
generation. A target language is a person's own words for what they want back;
what the model needs is a name, not an ASCII name.

Two things are still refused, because both are defects rather than languages:
a value whose first non-blank character is not a letter — a digit, a quotation
mark somebody's editor inserted — and anything carrying a control character,
which is not a name in any script. Surrounding blanks are allowed and then
trimmed: the manifest validates what was sent and the module validates the
same string, so a pasted " Português " cannot pass one and fail the other,
which is how a refusal used to arrive after the job had already run.
"""

from __future__ import annotations

import re

MAX_LANGUAGE_CHARACTERS = 64
# `\w` is Unicode-aware in Python's `re`, which is what validates every
# document that crosses this boundary, so this admits any script's letters.
TARGET_LANGUAGE_PATTERN = r"^\s*[^\W\d_][\w '\-]*\s*$"
TARGET_LANGUAGE = re.compile(TARGET_LANGUAGE_PATTERN)


def valid_target_language(value: object) -> bool:
    """Whether this is a name a workload may be asked to translate into."""
    return (
        isinstance(value, str)
        and len(value) <= MAX_LANGUAGE_CHARACTERS
        and TARGET_LANGUAGE.fullmatch(value) is not None
    )


__all__ = [
    "MAX_LANGUAGE_CHARACTERS",
    "TARGET_LANGUAGE",
    "TARGET_LANGUAGE_PATTERN",
    "valid_target_language",
]
