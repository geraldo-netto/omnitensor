"""Run the pinned mutation engine with OmniTensor's behavioral operators."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


def disable_string_literal_mutations() -> None:
    """Remove only mutmut's prose-oriented string-literal operator."""
    from mutmut.mutation import mutators  # noqa: PLC0415

    mutators.mutation_operators[:] = [
        entry
        for entry in mutators.mutation_operators
        if entry[1] is not mutators.operator_string
    ]


@contextmanager
def without_string_literal_mutations() -> Iterator[None]:
    """Temporarily apply the project operator policy during source inspection."""
    from mutmut.mutation import mutators  # noqa: PLC0415

    original = list(mutators.mutation_operators)
    disable_string_literal_mutations()
    try:
        yield
    finally:
        mutators.mutation_operators[:] = original


def main() -> None:
    """Invoke mutmut after applying the project operator policy."""
    disable_string_literal_mutations()
    from mutmut.__main__ import cli  # noqa: PLC0415

    cli()


if __name__ == "__main__":
    main()
