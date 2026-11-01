from __future__ import annotations

import math

import pytest

from evalloop.stats import (
    bootstrap_interval,
    cohens_h,
    compare,
    difference_interval,
    holm,
    mcnemar,
    min_detectable_effect,
    paired_bootstrap,
    paired_sample_size,
    power_unpaired,
    sample_size,
    sign_test,
    two_proportion_test,
    wilson_interval,
)


def by_id(values: list[bool]) -> dict[str, bool]:
    return {f"T-{i:04d}": v for i, v in enumerate(values)}


def pairs(fixed: int, broken: int, same: int = 0):
    """Paired outcomes with the given discordant counts."""
    a = [False] * fixed + [True] * broken + [True] * same
    b = [True] * fixed + [False] * broken + [True] * same
    return by_id(a), by_id(b)


def test_wilson_matches_newcombe_1998():
    ci = wilson_interval(81, 263)
    assert ci.low == pytest.approx(0.2553, abs=1e-4)
    assert ci.high == pytest.approx(0.3662, abs=1e-4)
    zero = wilson_interval(0, 10)
    assert zero.low == pytest.approx(0.0, abs=1e-12)
    z2 = 1.959964**2
    assert zero.high == pytest.approx(z2 / (10 + z2), abs=1e-6)
    with pytest.raises(ValueError):
        wilson_interval(3, 0)


def test_difference_interval_newcombe_method_10():
    ci = difference_interval(48, 80, 56, 70)
    assert ci.estimate == pytest.approx(0.2)
    assert ci.low == pytest.approx(0.0524, abs=1e-4)
    assert ci.high == pytest.approx(0.3339, abs=1e-4)


def test_mcnemar_exact_and_chi_square():
    a, b = pairs(fixed=10, broken=2, same=20)
    exact = mcnemar(a, b)
    assert (exact.fixed, exact.broken, exact.exact) == (
        10,
        2,
        True,
    )
    # 2 * (C(12,0) + C(12,1) + C(12,2)) / 2**12
    assert exact.p_value == pytest.approx(158 / 4096)
    approx = mcnemar(a, b, exact_below=5)
    chi2 = (10 - 2 - 1) ** 2 / 12
    p = math.erfc(math.sqrt(chi2 / 2))  # chi-square, 1 df
    assert approx.p_value == pytest.approx(p)
    assert sign_test(0, 0) == 1.0
    with pytest.raises(ValueError):
        mcnemar(a, {"other": True})


def test_two_proportion_test_by_hand():
    pooled = 104 / 150
    se = math.sqrt(pooled * (1 - pooled) * (1 / 80 + 1 / 70))
    z = (56 / 70 - 48 / 80) / se
    expected = math.erfc(z / math.sqrt(2))
    assert two_proportion_test(48, 80, 56, 70) == pytest.approx(
        expected
    )
    assert expected == pytest.approx(0.00805, abs=1e-5)


def test_holm_textbook_example():
    raw = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}
    adj = holm(raw)
    assert adj["d"] == pytest.approx(0.02)
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)  # kept monotone
    assert holm({"x": 0.6, "y": 0.7}) == {"x": 1.0, "y": 1.0}


def test_sample_size_and_power_agree():
    # Fleiss, no continuity correction: 387.3 -> 388
    assert sample_size(0.5, 0.6) == 388
    assert power_unpaired(0.5, 0.6, 388) >= 0.8
    assert power_unpaired(0.5, 0.6, 387) < 0.8
    # Connor: (1.96*sqrt(.2) + .8416*sqrt(.19))^2 / .01
    assert paired_sample_size(0.1, 0.2) == 155
    with pytest.raises(ValueError):
        paired_sample_size(0.3, 0.2)
    with pytest.raises(ValueError):
        sample_size(0.5, 0.5)


def test_effect_size_and_detectable_effect():
    assert cohens_h(0.5, 0.6) == pytest.approx(0.2014, abs=1e-4)
    assert cohens_h(0.6, 0.5) == pytest.approx(-0.2014, abs=1e-4)
    mde = min_detectable_effect(0.5, 100)
    assert mde == pytest.approx(2.8016 * math.sqrt(0.005), 1e-3)


def test_bootstrap_is_seeded_and_paired_cancels():
    values = [1.0] * 60 + [0.0] * 40
    one = bootstrap_interval(values, seed=3)
    assert one == bootstrap_interval(values, seed=3)
    assert one.low < 0.6 < one.high
    a, b = pairs(fixed=0, broken=0, same=50)
    same = paired_bootstrap(a, b)
    assert (same.low, same.high) == (0.0, 0.0)


def test_compare_verdicts():
    better = compare(*pairs(fixed=10, broken=0, same=30))
    assert better.verdict == "better"
    assert better.gain.estimate == pytest.approx(0.25)
    worse = compare(*pairs(fixed=1, broken=9, same=30))
    assert worse.verdict == "worse"
    unsure = compare(*pairs(fixed=3, broken=1, same=30))
    assert unsure.verdict == "inconclusive"


def test_example_main(capsys):
    from ch13_statistics import main

    main()
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert len(lines) <= 25
    assert max(map(len, lines)) <= 64
    assert "   30 " in lines[2] and "inconclusive" in lines[2]
    assert lines[4].endswith("better")
    assert "unpaired z-test:  p = 0.204" in out
