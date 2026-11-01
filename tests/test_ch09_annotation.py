from __future__ import annotations

from collections import Counter

import pytest

from evalloop.annotation import (
    GUIDE_V1,
    GUIDE_V2,
    POOL,
    Annotator,
    adjudicate,
    annotate,
    assign,
    contested,
    cost_report,
    flagged,
    fleiss_kappa,
    gold_scores,
)
from evalloop.core import BASELINE, DEMO_TASKS, MODES, SimAgent

RUNS = [SimAgent(BASELINE).run(t) for t in DEMO_TASKS]
T, F = True, False


def test_guidelines_are_data_with_gaps():
    assert all(GUIDE_V2.covers(m) for m in MODES)
    silent = [m for m in MODES if not GUIDE_V1.covers(m)]
    assert silent == [
        "no_citation",
        "premature_answer",
        "bad_tone",
    ]


def test_uncovered_modes_fall_back_on_opinion():
    lax = Annotator("lax", 0.9, 30, 30.0, strictness=0.0)
    strict = Annotator("strict", 0.9, 30, 30.0, strictness=1.0)
    assert lax.counts_as_failure("loop", GUIDE_V1)
    assert not lax.counts_as_failure("bad_tone", GUIDE_V1)
    assert strict.counts_as_failure("bad_tone", GUIDE_V1)


def test_perfect_rater_reproduces_ground_truth():
    ideal = Annotator("ideal", 1.0, 30, 30.0, strictness=1.0)
    labels = [annotate(ideal, r, GUIDE_V1) for r in RUNS]
    assert labels == [r.ok for r in RUNS]
    again = [annotate(POOL[4], r, GUIDE_V2) for r in RUNS]
    assert again == [annotate(POOL[4], r, GUIDE_V2) for r in RUNS]


def test_assign_spreads_load_with_distinct_raters():
    items = [f"I-{i}" for i in range(23)]
    plan = assign(items, POOL, overlap=3)
    assert all(len(set(raters)) == 3 for raters in plan.values())
    load = Counter(r for raters in plan.values() for r in raters)
    assert max(load.values()) - min(load.values()) <= 1
    with pytest.raises(ValueError):
        assign(items, POOL, overlap=6)


def test_fleiss_kappa_known_values():
    assert fleiss_kappa([[T, T, T], [F, F, F]]) == 1.0
    mixed = [[T, T, T], [F, F, F], [T, T, F], [T, F, F]]
    assert fleiss_kappa(mixed) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        fleiss_kappa([[T, T], [T, F, F]])


def test_adjudication_sends_only_contested_items():
    labels = {
        "a": {"r1": T, "r2": T, "r3": T},
        "b": {"r1": T, "r2": F, "r3": F},
    }
    asked = []

    def expert(item):
        asked.append(item)
        return True

    assert contested(labels) == ["b"]
    assert adjudicate(labels, expert) == {"a": True, "b": True}
    assert asked == ["b"]


def test_gold_questions_flag_weak_raters():
    labels = {
        "g1": {"r1": T, "r2": F},
        "g2": {"r1": F, "r2": F},
        "x": {"r1": T, "r2": T},
    }
    scores = gold_scores(labels, {"g1": T, "g2": F})
    assert scores == {"r1": (2, 2), "r2": (1, 2)}
    assert flagged(scores, min_accuracy=0.8) == ["r2"]


def test_cost_report_counts_hours_and_money():
    pool = [
        Annotator("r1", 0.9, 10, 20.0, 0.5),
        Annotator("r2", 0.9, 20, 10.0, 0.5),
    ]
    expert = Annotator("e", 1.0, 5, 50.0, 0.5)
    labels = {f"i{k}": {"r1": T, "r2": T} for k in range(10)}
    bill = cost_report(labels, pool, expert, escalated=5)
    assert bill.labels == 25
    assert bill.hours == 2.5  # 1.0 + 0.5 + 1.0 expert
    assert bill.cost == 75.0  # 20 + 5 + 50
    assert bill.per_final_label == 7.5


def test_example_shows_guideline_effect(capsys):
    from ch09_annotation import main

    main()
    out = capsys.readouterr().out
    assert "Fleiss' kappa 0.60, contested 29" in out
    assert "final vs truth 99.2%" in out
    assert (
        "rater-5 skill 0.65: 25/30  $0.30/label  flagged" in out
    )
