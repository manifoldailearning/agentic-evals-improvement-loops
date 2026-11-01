"""Agreement between a judge and humans, beyond raw accuracy.

A judge is only useful if it agrees with the people whose
judgment it replaces. Here: simulated human labels (noisy,
and worse on subtle faults), the confusion matrix with
"fail" as the positive class, Cohen's kappa to discount
chance agreement, and a pass threshold chosen on held data.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from evalloop.core import Trajectory, rng_for, stable_hash

SUBTLE = {"hallucinated_fact"}  # faults people often miss


def human_label(
    traj: Trajectory,
    error: float = 0.04,
    subtle_miss: float = 0.3,
) -> bool:
    """A simulated reviewer's pass/fail, from ground truth.

    Clean runs are failed by mistake with ``error``; a run
    whose only faults are subtle passes with ``subtle_miss``.
    """
    rng = rng_for(
        "human", traj.task_id, traj.config_key, traj.seed
    )
    if not traj.faults:
        return rng.random() >= error
    miss = subtle_miss if set(traj.faults) <= SUBTLE else error
    return rng.random() < miss


@dataclass(frozen=True)
class Confusion:
    """Counts with "fail" as positive: the judge's job."""

    tp: int  # judge fail, human fail
    fp: int  # judge fail, human pass
    fn: int  # judge pass, human fail
    tn: int  # judge pass, human pass

    @property
    def precision(self) -> float:
        flagged = self.tp + self.fp
        return self.tp / flagged if flagged else 1.0

    @property
    def recall(self) -> float:
        failing = self.tp + self.fn
        return self.tp / failing if failing else 1.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / (
            self.tp + self.fp + self.fn + self.tn
        )


def confusion(
    judge_pass: Sequence[bool], human_pass: Sequence[bool]
) -> Confusion:
    pairs = Counter(zip(judge_pass, human_pass, strict=True))
    return Confusion(
        tp=pairs[False, False],
        fp=pairs[False, True],
        fn=pairs[True, False],
        tn=pairs[True, True],
    )


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """Agreement corrected for chance: 1 perfect, 0 chance."""
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    count_a, count_b = Counter(a), Counter(b)
    expected = sum(
        count_a[k] * count_b[k] for k in count_a.keys() | count_b
    ) / (n * n)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Linear correlation; 0 when either side is constant."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum(
        (x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)
    )
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else 0.0


def split_half(items: Sequence, key, salt: str = "calib"):
    """Stable split into (calibration, held-out) by item key."""
    calib, held = [], []
    for item in items:
        bucket = int(stable_hash(salt, key(item))[:8], 16) % 2
        (calib if bucket == 0 else held).append(item)
    return calib, held


def kappa_by_threshold(
    scores: Sequence[int],
    human_pass: Sequence[bool],
    thresholds: Sequence[int] = (2, 3, 4, 5),
) -> dict[int, float]:
    """Kappa against humans if a score >= t counted as pass."""
    return {
        t: cohen_kappa([s >= t for s in scores], list(human_pass))
        for t in thresholds
    }


def choose_threshold(scores, human_pass, thresholds=(2, 3, 4, 5)):
    """The pass threshold with the best kappa (ties: lower t)."""
    table = kappa_by_threshold(scores, human_pass, thresholds)
    return max(table, key=lambda t: (round(table[t], 6), -t))
