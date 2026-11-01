"""Confidence intervals: how sure are we about a pass rate?

A pass rate from 30 tasks is a guess with wide error bars.
Wilson intervals stay honest near 0% and 100%, where the
textbook normal interval breaks; the bootstrap works for
any statistic; and the paired bootstrap keeps each task
matched with itself when two configs run the same tasks.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist, fmean

from evalloop.core import rng_for


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float

    @property
    def width(self) -> float:
        return self.high - self.low

    def excludes(self, value: float) -> bool:
        """True when ``value`` lies outside the interval."""
        return not self.low <= value <= self.high

    def __str__(self) -> str:
        return (
            f"{self.estimate:.1%}"
            f" [{self.low:.1%}, {self.high:.1%}]"
        )


def z_score(confidence: float) -> float:
    """Two-sided critical value: 1.96 for 95% confidence."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    return NormalDist().inv_cdf(0.5 + confidence / 2)


def wilson_interval(
    successes: int, n: int, confidence: float = 0.95
) -> Interval:
    """Wilson score interval for a binomial proportion.

    Assumes the n outcomes are independent draws with one
    shared success rate. Unlike the normal (Wald) interval,
    it never leaves [0, 1] and is not zero-width at 0 or n.
    """
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("need n > 0 and 0 <= successes <= n")
    z = z_score(confidence)
    p = successes / n
    shrink = 1 + z * z / n
    center = (p + z * z / (2 * n)) / shrink
    spread = p * (1 - p) / n + z * z / (4 * n * n)
    half = z * math.sqrt(spread) / shrink
    return Interval(p, center - half, center + half)


def difference_interval(
    s_a: int, n_a: int, s_b: int, n_b: int, confidence=0.95
) -> Interval:
    """Interval for rate_b - rate_a from independent groups.

    Newcombe's hybrid method combines the two Wilson
    intervals. Use it only when the groups are separate
    samples (an A/B test), never for the same tasks.
    """
    a = wilson_interval(s_a, n_a, confidence)
    b = wilson_interval(s_b, n_b, confidence)
    d = b.estimate - a.estimate
    down = math.hypot(b.estimate - b.low, a.high - a.estimate)
    up = math.hypot(b.high - b.estimate, a.estimate - a.low)
    return Interval(d, d - down, d + up)


def bootstrap_interval(
    values: Sequence[float],
    stat: Callable[[Sequence[float]], float] = fmean,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval for any statistic.

    Resamples the values with replacement, so it assumes
    they are independent and that the sample resembles the
    population. Seeded, so the interval is reproducible.
    """
    if not values:
        raise ValueError("need at least one value")
    rng = rng_for("bootstrap", seed)
    k = len(values)
    draws = sorted(
        stat(rng.choices(values, k=k)) for _ in range(resamples)
    )
    cut = round((1 - confidence) / 2 * (resamples - 1))
    low, high = draws[cut], draws[resamples - 1 - cut]
    return Interval(stat(values), low, high)


def paired_differences(
    a: Mapping[str, float], b: Mapping[str, float]
) -> list[float]:
    """Per-task score differences b - a, in task-id order.

    A paired comparison is only valid on the same tasks, so
    mismatched task sets are an error, not a silent join.
    """
    if a.keys() != b.keys():
        raise ValueError("paired data needs the same task ids")
    return [float(b[t]) - float(a[t]) for t in sorted(a)]


def paired_bootstrap(
    a: Mapping[str, float],
    b: Mapping[str, float],
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Bootstrap interval for the mean per-task gain b - a.

    Resampling whole tasks keeps each pair together, so the
    shared difficulty of a task cancels out of the width.
    """
    diffs = paired_differences(a, b)
    return bootstrap_interval(
        diffs, confidence=confidence, seed=seed
    )
