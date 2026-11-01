"""Bias tests and drift checks for a model judge.

A bias test changes one thing a good judge should ignore
(slot order, padding, the author's model tier) and measures
how much the verdict moves. A drift check re-runs a changed
judge on the same calibration set and asks whether it still
agrees with humans as well as the judge it replaces.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import mean

from evalloop.calibration.agreement import cohen_kappa
from evalloop.judge import PairwiseJudge, RubricJudge

FLIP = {"A": "B", "B": "A", "tie": "tie"}
FILLER = " Thank you for reaching out; we value your business."


def position_flip_rate(
    judge: PairwiseJudge, by_id: dict, pairs
) -> float:
    """Share of pairs whose winner changes when slots swap."""
    flips = 0
    for a, b in pairs:
        task = by_id[a.task_id]
        first = judge.compare(task, a, b)
        second = FLIP[judge.compare(task, b, a)]
        flips += first != second
    return flips / len(pairs)


def mean_score(judge: RubricJudge, by_id, runs, author=None):
    return mean(
        judge.grade(by_id[r.task_id], r, author=author).score
        for r in runs
    )


def padding_effect(
    judge: RubricJudge, by_id: dict, runs, filler: str = FILLER
) -> tuple[float, float]:
    """Mean score change from filler, and change per 100 chars.

    Scores are clipped at 5, so the per-100 slope understates
    the judge's raw length bias on runs already scoring 5.
    """
    padded = [replace(r, answer=r.answer + filler) for r in runs]
    before = mean_score(judge, by_id, runs)
    delta = mean_score(judge, by_id, padded) - before
    return delta, delta * 100 / len(filler)


def self_preference_gap(
    judge: RubricJudge, by_id: dict, runs, other: str = "small"
) -> float:
    """Score gain when the author shares the judge's tier."""
    own = mean_score(judge, by_id, runs, judge.config.model)
    return own - mean_score(judge, by_id, runs, other)


@dataclass(frozen=True)
class DriftReport:
    agreement: float  # old vs new judge verdicts
    kappa_old: float  # old judge vs humans
    kappa_new: float  # new judge vs humans
    drifted: bool


def drift_check(
    old_pass, new_pass, human_pass, tolerance: float = 0.05
) -> DriftReport:
    """Flag a judge change that agrees less with humans."""
    pairs = zip(old_pass, new_pass, strict=True)
    agreement = mean(a == b for a, b in pairs)
    k_old = cohen_kappa(list(old_pass), list(human_pass))
    k_new = cohen_kappa(list(new_pass), list(human_pass))
    drifted = k_old - k_new > tolerance
    return DriftReport(agreement, k_old, k_new, drifted)
