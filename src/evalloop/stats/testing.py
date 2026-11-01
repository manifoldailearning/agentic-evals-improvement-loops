"""Hypothesis tests: is v2 really better than v1?

Two configs that run the same tasks give paired outcomes,
and only the tasks where they disagree carry information.
McNemar's test uses exactly those; the unpaired z-test is
here for separate samples and to show what pairing buys.
Holm's method keeps many comparisons from crying wolf.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist

from evalloop.stats.intervals import Interval, paired_bootstrap


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial p-value under a 50/50 null."""
    n = wins + losses
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / 2**n)


def normal_p_value(z: float) -> float:
    """Two-sided p-value for a standard normal statistic."""
    return 2 * (1 - NormalDist().cdf(abs(z)))


@dataclass(frozen=True)
class McNemar:
    fixed: int  # failed under a, passed under b
    broken: int  # passed under a, failed under b
    p_value: float
    exact: bool


def mcnemar(
    a: Mapping[str, bool],
    b: Mapping[str, bool],
    exact_below: int = 50,
) -> McNemar:
    """McNemar's test on paired pass/fail outcomes.

    Assumes tasks are independent of each other; the pairs
    within a task may be as correlated as they like. Under
    the null, each discordant task is a fair coin flip, so
    small counts get the exact binomial p-value and larger
    ones the continuity-corrected chi-square (1 df).
    """
    if a.keys() != b.keys():
        raise ValueError("paired data needs the same task ids")
    fixed = sum(b[t] and not a[t] for t in a)
    broken = sum(a[t] and not b[t] for t in a)
    n = fixed + broken
    if n < exact_below:
        return McNemar(
            fixed, broken, sign_test(fixed, broken), True
        )
    chi2 = (abs(fixed - broken) - 1) ** 2 / n
    p = normal_p_value(math.sqrt(chi2))
    return McNemar(fixed, broken, p, False)


def two_proportion_test(
    s_a: int, n_a: int, s_b: int, n_b: int
) -> float:
    """Pooled two-proportion z-test p-value (unpaired).

    Valid for independent groups. On paired data it ignores
    the pairing and loses most of its power.
    """
    pooled = (s_a + s_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 1.0
    return normal_p_value((s_b / n_b - s_a / n_a) / se)


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (family-wise error control).

    Sort ascending, multiply the i-th smallest by (m - i),
    and keep the sequence non-decreasing. Compare adjusted
    values with alpha as usual; no independence assumed.
    """
    order = sorted(p_values, key=p_values.__getitem__)
    m = len(order)
    adjusted, running = {}, 0.0
    for i, name in enumerate(order):
        running = max(running, min(1.0, (m - i) * p_values[name]))
        adjusted[name] = running
    return adjusted


def cohens_h(p_a: float, p_b: float) -> float:
    """Effect size for two rates: 0.2 small, 0.5 medium."""
    angle_a = math.asin(math.sqrt(p_a))
    angle_b = math.asin(math.sqrt(p_b))
    return 2 * (angle_b - angle_a)


@dataclass(frozen=True)
class Comparison:
    n: int
    rate_a: float
    rate_b: float
    gain: Interval  # paired bootstrap of b - a
    test: McNemar
    verdict: str  # better | worse | inconclusive


def compare(
    a: Mapping[str, bool],
    b: Mapping[str, bool],
    alpha: float = 0.05,
    seed: int = 0,
) -> Comparison:
    """Decide whether b beats a on the same tasks.

    The verdict comes from McNemar's test at ``alpha``; the
    gain interval says how big the difference might be.
    "inconclusive" means the eval cannot tell, not "equal".
    """
    gain = paired_bootstrap(a, b, 1 - alpha, seed)
    test = mcnemar(a, b)
    verdict = "inconclusive"
    if test.p_value < alpha:
        verdict = (
            "better" if test.fixed > test.broken else "worse"
        )
    n = len(a)
    rate_a, rate_b = sum(a.values()) / n, sum(b.values()) / n
    return Comparison(n, rate_a, rate_b, gain, test, verdict)
