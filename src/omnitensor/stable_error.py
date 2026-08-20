"""One refusal shape, instead of forty-nine copies of it.

A refusal a caller can act on is a stable machine-readable ``code`` and a
sentence of ``detail`` beside it. Forty-nine exception classes across this
package each declared that with the same three lines, over three different
bases chosen inconsistently for one concept, so the shape could drift and
nothing named it.

The mixin carries the shape; each class keeps the built-in base it already had,
because ``except ValueError`` around a parser and ``except RuntimeError``
around a worker call are what the callers were written against::

    class GrantError(StableError, ValueError):
        \"\"\"Why a grant was refused.\"\"\"
"""

from __future__ import annotations


class StableError:
    """Mixin: a refusal carrying a machine-readable code and a sentence.

    Not an exception on its own — it is always mixed with the built-in base a
    caller catches, and never raised by itself.
    """

    __slots__ = ()

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


__all__ = ["StableError"]
