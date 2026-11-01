from __future__ import annotations

import math
import random
from statistics import fmean, variance

import pytest

from evalloop.core import BASELINE, Step, Trajectory, make_tasks
from evalloop.online import (
    Guardrail,
    Session,
    assign,
    boundaries,
    check_guardrail,
    credit,
    false_alarm_rate,
    interleaving_result,
    mean_difference,
    monitor,
    novelty_check,
    obf_spending,
    resolved,
    simulate_traffic,
    team_draft,
    z_statistic,
)


def session(i: int, variant: str, ok: bool, day=0, cost=1.0):
    return Session(f"U-{i}", variant, day, ok, ok, cost, 100)


def arms(n: int, rate_c: float, rate_t: float, day=0) -> list:
    """Interleaved sessions with exact success counts."""
    out = []
    for i in range(n):
        out.append(session(2 * i, "control", i < rate_c * n, day))
        out.append(
            session(2 * i + 1, "treatment", i < rate_t * n, day)
        )
    return out


def test_assign_is_stable_and_respects_share():
    users = [f"U-{i}" for i in range(10000)]
    first = [assign(u, "exp", 0.3) for u in users]
    assert first == [assign(u, "exp", 0.3) for u in users]
    share = first.count("treatment") / len(users)
    assert share == pytest.approx(0.3, abs=0.015)
    other = [assign(u, "exp-2", 0.3) for u in users]
    assert other != first  # experiments bucket independently


def test_resolved_reads_answer_and_side_effects():
    task = next(
        t
        for t in make_tasks(50, seed=7)
        if t.expected["must_not"] == ["issue_refund"]
    )
    facts = " ".join(task.expected["answer_contains"])

    def run(steps, answer):
        return Trajectory(
            "T", "v", "k", 0, steps, answer, 1, 1, 1, 0.0
        )

    good = run([Step("tool", "escalate")], f"I have {facts}.")
    assert resolved(task, good)
    refund = run(
        [Step("tool", "issue_refund")], f"I have {facts}."
    )
    assert not resolved(task, refund)
    assert not resolved(task, run([], ""))


def test_mean_difference_and_guardrail():
    a, b = [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 6.0]
    d = mean_difference(a, b)
    se = math.sqrt(variance(a) / 4 + variance(b) / 4)
    assert d.estimate == pytest.approx(1.25)
    assert d.high - d.estimate == pytest.approx(
        1.959964 * se, 1e-5
    )
    cost = random.Random(1).uniform
    variants = ("control", "treatment")
    flat = [
        session(i, variants[i % 2], True, cost=cost(0.9, 1.1))
        for i in range(2000)
    ]
    ok, rel = check_guardrail(flat, Guardrail("cost", 0.05))
    assert ok and abs(rel.estimate) < 0.02
    ok, _ = check_guardrail(flat, Guardrail("cost", -0.05))
    assert not ok


def test_spending_and_boundaries_by_hand():
    assert obf_spending(1.0) == pytest.approx(0.05)
    assert obf_spending(0.0) == 0.0
    assert obf_spending(0.25) < obf_spending(0.5) < 0.05
    assert boundaries([1.0])[0] == pytest.approx(1.959964, 1e-6)
    bounds = boundaries([0.25, 0.5, 0.75, 1.0])
    # the first look spends 2(1 - Phi(z/sqrt t)): bound = z/sqrt t
    assert bounds[0] == pytest.approx(1.959964 / 0.5, 1e-6)
    assert bounds == sorted(bounds, reverse=True)
    assert bounds[-1] > 1.96  # conservative final look


def test_false_alarm_rates_match_theory():
    looks = [0.2, 0.4, 0.6, 0.8, 1.0]
    # Armitage, McPherson and Rowe (1969): 14.2% for 5 looks
    naive = false_alarm_rate([1.96] * 5, looks, trials=20000)
    assert naive == pytest.approx(0.142, abs=0.01)
    safe = false_alarm_rate(
        boundaries(looks), looks, trials=20000
    )
    assert safe <= 0.05


def test_monitor_stops_at_first_crossing():
    strong = arms(1000, 0.50, 0.65)
    assert z_statistic(strong) > 0
    looks = monitor(strong, [0.25, 0.5, 0.75, 1.0])
    assert looks[-1].stop and not any(x.stop for x in looks[:-1])
    assert len(looks) < 4
    null = arms(1000, 0.5, 0.5)
    assert z_statistic(null) == 0.0
    assert len(monitor(null, [0.5, 1.0])) == 2


def test_team_draft_is_fair_and_complete():
    a = ["x", "y", "z", "w"]
    b = ["z", "x", "v", "y"]
    merged, team = team_draft(a, b, random.Random(3))
    assert sorted(merged) == sorted(set(a) | set(b))
    picks = list(team.values())
    assert abs(picks.count("A") - picks.count("B")) <= 1
    same, _ = team_draft(a, a, random.Random(5))
    assert same == a
    assert credit(["x"], {"x": "B"}) == "B"
    assert credit([], {"x": "B"}) == "tie"
    wins = ["B"] * 10 + ["A"] * 2 + ["tie"] * 5
    assert interleaving_result(wins) == (2, 10, 5, 158 / 4096)


def test_novelty_check_flags_a_fading_lift():
    early = arms(500, 0.50, 0.70, day=1)
    late = arms(500, 0.50, 0.51, day=9)
    result = novelty_check(early + late, "resolved", split_day=5)
    assert result.early.estimate == pytest.approx(0.20)
    assert result.late.estimate == pytest.approx(0.01)
    assert result.p_value < 0.001


def test_simulate_traffic_is_deterministic():
    tasks = make_tasks(20, seed=7)
    configs = {"control": BASELINE, "treatment": BASELINE}
    one = simulate_traffic(configs, tasks, users=200)
    assert one == simulate_traffic(configs, tasks, users=200)
    assert len({s.user for s in one}) == 200
    assert [s.day for s in one] == sorted(s.day for s in one)
    assert fmean(s.resolved for s in one) > 0.5


def test_example_main(capsys):
    from ch15_online import main

    main()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) <= 25
    assert max(map(len, lines)) <= 64
    out = "\n".join(lines)
    assert "stop: v2 wins" in out
    assert "decision: hold: a guardrail failed" in out
    assert "naive p < 0.05 14.0%" in out
