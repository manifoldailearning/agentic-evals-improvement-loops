"""Sequential monitoring: peeking at a live test without lying.

Checking an A/B test daily at p < 0.05 and stopping at the
first win inflates false positives far above 5%. An alpha
spending function budgets the error across looks; here the
O'Brien-Fleming-type spending of Lan and DeMets, applied in
a deliberately conservative way that is easy to verify.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

from evalloop.core import rng_for
from evalloop.online.traffic import split

Z = NormalDist()


def obf_spending(t: float, alpha: float = 0.05) -> float:
    """Alpha spent by information fraction t (0 < t <= 1).

    Spends almost nothing early and the bulk at the end,
    so the final look keeps nearly the full alpha.
    """
    if t <= 0:
        return 0.0
    z = Z.inv_cdf(1 - alpha / 2)
    return 2 * (1 - Z.cdf(z / math.sqrt(min(t, 1.0))))


def boundaries(
    fractions: Sequence[float], alpha: float = 0.05
) -> list[float]:
    """|z| needed to stop at each look.

    Each look is tested alone at the alpha spent since the
    previous look. By the union bound the chance of any
    false stop is at most alpha, whatever the correlation
    between looks. Exact boundaries account for that
    correlation and are a little lower; these are safe.
    """
    spent, bounds = 0.0, []
    for t in fractions:
        total = obf_spending(t, alpha)
        bounds.append(Z.inv_cdf(1 - (total - spent) / 2))
        spent = total
    return bounds


def z_statistic(sessions, metric: str = "resolved") -> float:
    """Pooled two-proportion z: treatment minus control."""
    a, b = split(sessions, metric)
    pooled = (sum(a) + sum(b)) / (len(a) + len(b))
    se = math.sqrt(
        pooled * (1 - pooled) * (1 / len(a) + 1 / len(b))
    )
    return (sum(b) / len(b) - sum(a) / len(a)) / se if se else 0.0


@dataclass(frozen=True)
class Look:
    sessions: int
    z: float
    boundary: float

    @property
    def stop(self) -> bool:
        return abs(self.z) >= self.boundary


def monitor(
    sessions,
    fractions: Sequence[float],
    metric: str = "resolved",
    alpha: float = 0.05,
) -> list[Look]:
    """Look at the planned fractions; stop at the first cross.

    ``sessions`` must be in arrival order and hold the full
    planned sample; each look sees only what had arrived.
    The alpha guarantee needs the looks fixed in advance.
    Two-sided: a large negative z means the variant hurts.
    """
    looks = []
    for t, bound in zip(
        fractions, boundaries(fractions, alpha), strict=True
    ):
        seen = sessions[: round(t * len(sessions))]
        looks.append(
            Look(len(seen), z_statistic(seen, metric), bound)
        )
        if looks[-1].stop:
            break
    return looks


def false_alarm_rate(
    bounds: Sequence[float],
    fractions: Sequence[float],
    trials: int = 20000,
    seed: int = 0,
) -> float:
    """Share of A/A tests that ever cross a boundary.

    Under no true difference the z path is a scaled Brownian
    motion, so it can be simulated directly from normal steps.
    """
    rng = rng_for("aa", seed)
    alarms = 0
    for _ in range(trials):
        walk, last = 0.0, 0.0
        for t, bound in zip(fractions, bounds, strict=True):
            walk += rng.gauss(0.0, math.sqrt(t - last))
            last = t
            if abs(walk / math.sqrt(t)) >= bound:
                alarms += 1
                break
    return alarms / trials
