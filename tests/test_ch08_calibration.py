from __future__ import annotations

import pytest

from evalloop.calibration import (
    choose_threshold,
    cohen_kappa,
    confusion,
    drift_check,
    human_label,
    padding_effect,
    pearson,
    position_flip_rate,
    self_preference_gap,
    split_half,
)
from evalloop.core import BASELINE, DEMO_TASKS, SimAgent
from evalloop.judge import JudgeConfig, PairwiseJudge, RubricJudge

BY_ID = {t.id: t for t in DEMO_TASKS}
RUNS = [SimAgent(BASELINE).run(t) for t in DEMO_TASKS]


def test_human_labels_follow_truth_with_error():
    exact = [human_label(r, error=0, subtle_miss=0) for r in RUNS]
    assert exact == [r.ok for r in RUNS]
    sloppy = [
        human_label(r, error=1, subtle_miss=1) for r in RUNS
    ]
    assert sloppy == [not r.ok for r in RUNS]
    noisy = [human_label(r) for r in RUNS]
    assert noisy == [human_label(r) for r in RUNS]


def test_confusion_treats_fail_as_positive():
    judge = [False, False, True, True, True]
    human = [False, True, False, True, True]
    c = confusion(judge, human)
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 2)
    assert c.precision == 0.5 and c.recall == 0.5
    assert c.accuracy == 0.6
    none_flagged = confusion([True, True], [True, True])
    assert none_flagged.precision == 1.0


def test_cohen_kappa_known_values():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == 0.0
    a = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    b = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0


def test_pearson_edges():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert pearson([1, 2, 3], [5, 5, 5]) == 0.0


def test_split_half_is_stable_and_disjoint():
    ids = [r.task_id for r in RUNS]
    calib, held = split_half(ids, key=str)
    assert sorted(calib + held) == sorted(ids)
    assert not set(calib) & set(held)
    assert split_half(ids, key=str) == (calib, held)


def test_choose_threshold_maximizes_kappa():
    scores = [5, 5, 4, 4, 3, 2, 5, 4]
    human = [True, True, False, False, False, False, True, False]
    assert choose_threshold(scores, human) == 5
    assert choose_threshold([5, 1], [True, False]) == 2


def test_position_flip_rate_needs_position_bias():
    pairs = [(r, r) for r in RUNS[:20]]
    fair = JudgeConfig(position_bias=0.0, noise=0.0)
    assert (
        position_flip_rate(PairwiseJudge(fair), BY_ID, pairs) == 0
    )
    biased = JudgeConfig(noise=0.0)
    rate = position_flip_rate(PairwiseJudge(biased), BY_ID, pairs)
    assert rate == 1.0  # slot A always wins identical runs


def test_padding_and_self_preference_are_measurable():
    judge = RubricJudge()
    delta, slope = padding_effect(judge, BY_ID, RUNS)
    assert delta > 0 and slope > 0
    flat = RubricJudge(JudgeConfig(length_bias=0.0))
    assert padding_effect(flat, BY_ID, RUNS)[0] == 0
    assert self_preference_gap(judge, BY_ID, RUNS) > 0
    modest = RubricJudge(JudgeConfig(self_preference=0.0))
    assert self_preference_gap(modest, BY_ID, RUNS) == 0


def test_drift_check_flags_only_real_drops():
    human = [True] * 6 + [False] * 4
    old = list(human)
    worse = [True] * 9 + [False]
    report = drift_check(old, worse, human)
    assert report.drifted and report.kappa_old == 1.0
    assert report.agreement == 0.7
    assert not drift_check(old, list(old), human).drifted


def test_example_reports_kappa_and_drift(capsys):
    from ch08_calibration import main

    main()
    out = capsys.readouterr().out
    assert "agreement 88.0%, kappa 0.73" in out
    assert "chosen on calibration half: pass_at=5" in out
    assert "drifted: True" in out
