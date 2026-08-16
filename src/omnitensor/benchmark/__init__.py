"""Measuring what a model is worth on this machine: accuracy beside speed.

Split the way the questions split. :mod:`cases` is the labelled work,
:mod:`scoring` is what counts as right, :mod:`harness` runs it through the real
generation path, and :mod:`report` writes down what happened so a decision
taken next month can be checked against it.
"""

from .cases import Case, CaseError, load
from .harness import Outcome, Run, Timing
from .scoring import Judgement, Tally, judge

__all__ = [
    "Case",
    "CaseError",
    "Judgement",
    "Outcome",
    "Run",
    "Tally",
    "Timing",
    "judge",
    "load",
]
