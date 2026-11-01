"""Planning: how many tasks does an eval need?

Before running an eval, decide what difference matters and
check that the eval can see it. These are the standard
normal-approximation formulas for two rates, unpaired and
paired, at a significance level alpha and a target power.
"""

from __future__ import annotations

import math
from statistics import NormalDist

Z = NormalDist()


def sample_size(
    p_a: float, p_b: float, alpha=0.05, power=0.8
) -> int:
    """Tasks per config to detect p_a vs p_b, unpaired.

    Two independent samples, two-sided test at ``alpha``.
    """
    if p_a == p_b:
        raise ValueError("rates must differ")
    z_a, z_b = Z.inv_cdf(1 - alpha / 2), Z.inv_cdf(power)
    p = (p_a + p_b) / 2
    null_sd = math.sqrt(2 * p * (1 - p))
    alt_sd = math.sqrt(p_a * (1 - p_a) + p_b * (1 - p_b))
    n = (z_a * null_sd + z_b * alt_sd) ** 2 / (p_a - p_b) ** 2
    return math.ceil(n)


def power_unpaired(
    p_a: float, p_b: float, n: int, alpha: float = 0.05
) -> float:
    """Chance an unpaired test with n per config detects it."""
    z_a = Z.inv_cdf(1 - alpha / 2)
    p = (p_a + p_b) / 2
    null_sd = math.sqrt(2 * p * (1 - p))
    alt_sd = math.sqrt(p_a * (1 - p_a) + p_b * (1 - p_b))
    signal = abs(p_a - p_b) * math.sqrt(n)
    return Z.cdf((signal - z_a * null_sd) / alt_sd)


def paired_sample_size(
    gain: float, discordance: float, alpha=0.05, power=0.8
) -> int:
    """Tasks needed for McNemar's test (Connor, 1987).

    ``discordance`` is the share of tasks where the configs
    disagree; ``gain`` is fixed share minus broken share.
    Similar configs disagree rarely, so pairing needs far
    fewer tasks than two separate samples would.
    """
    if not 0 < abs(gain) <= discordance <= 1:
        raise ValueError("need 0 < |gain| <= discordance <= 1")
    z_a, z_b = Z.inv_cdf(1 - alpha / 2), Z.inv_cdf(power)
    root = math.sqrt(discordance - gain * gain)
    n = (z_a * math.sqrt(discordance) + z_b * root) ** 2
    return math.ceil(n / gain**2)


def min_detectable_effect(
    p: float, n: int, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Smallest unpaired gain over rate p that n tasks see.

    Approximates both configs' variance by p's variance.
    """
    z_a, z_b = Z.inv_cdf(1 - alpha / 2), Z.inv_cdf(power)
    return (z_a + z_b) * math.sqrt(2 * p * (1 - p) / n)
