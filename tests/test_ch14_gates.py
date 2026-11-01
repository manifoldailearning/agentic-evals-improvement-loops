from __future__ import annotations

import pytest

from evalloop.core import IMPROVED, SimAgent, make_tasks
from evalloop.gates import (
    Baseline,
    GatePolicy,
    Quarantine,
    Tolerance,
    apply_quarantine,
    check_critical,
    check_metric,
    check_regressions,
    collect,
    is_flaky,
    load_baseline,
    pass_at_k,
    pass_hat_k,
    run_gate,
    save_baseline,
    seed_spread,
    summarize,
)

T, F = True, False
OUTCOMES = {
    "T-1": [T, T, T, T],
    "T-2": [T, F, T, F],
    "T-3": [F, F, F, F],
    "T-4": [T, T, T, F],
}


def test_pass_at_k_and_pass_hat_k_by_hand():
    # n=10, c=3, k=2: 1 - C(7,2)/C(10,2) and C(3,2)/C(10,2)
    assert pass_at_k(10, 3, 2) == pytest.approx(1 - 21 / 45)
    assert pass_hat_k(10, 3, 2) == pytest.approx(3 / 45)
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)
    assert pass_hat_k(10, 3, 1) == pytest.approx(0.3)
    assert pass_at_k(5, 0, 5) == 0.0
    assert pass_hat_k(5, 5, 5) == 1.0
    with pytest.raises(ValueError):
        pass_at_k(3, 4, 1)


def test_summarize_weights_tasks_equally():
    m = summarize(OUTCOMES, 4)
    assert m["pass_rate"] == pytest.approx((4 + 2 + 0 + 3) / 16)
    assert m["pass@k"] == pytest.approx(3 / 4)
    assert m["pass^k"] == pytest.approx(1 / 4)
    assert m["flaky_rate"] == pytest.approx(2 / 4)
    assert is_flaky([T, F]) and not is_flaky([F, F])
    assert seed_spread(OUTCOMES) == (0.25, 0.75)


def test_quarantine_expires_on_its_date():
    q = Quarantine("T-2", "flaky", "2026-12-01")
    assert q.active("2026-11-30")
    assert not q.active("2026-12-01")
    old = Quarantine("T-4", "flaky", "2026-01-01")
    gated, held, expired = apply_quarantine(
        OUTCOMES, [q, old], "2026-11-15"
    )
    assert held == ["T-2"] and expired == ["T-4"]
    assert set(gated) == {"T-1", "T-3", "T-4"}


def test_check_metric_direction_and_boundary():
    rate = Tolerance("pass_rate", 0.03)
    assert check_metric(rate, 0.90, 0.87).passed  # at the limit
    assert not check_metric(rate, 0.90, 0.86).passed
    cost = Tolerance("cost", 0.001, higher_is_better=False)
    assert check_metric(cost, 0.010, 0.005).passed
    assert not check_metric(cost, 0.010, 0.012).passed


def test_check_critical_reports_failing_and_missing():
    ok = check_critical(OUTCOMES, frozenset({"T-1"}))
    assert ok.passed and ok.detail.startswith("1/1")
    bad = check_critical(
        OUTCOMES, frozenset({"T-1", "T-4", "T-9"})
    )
    assert not bad.passed
    assert "T-4" in bad.detail and "T-9" in bad.detail


def test_check_regressions_needs_significant_breaks():
    ids = [f"T-{i}" for i in range(20)]
    base = Baseline("b", 2, {t: [T, T] for t in ids}, {})
    some = {
        t: [F, F] if i < 2 else [T, T] for i, t in enumerate(ids)
    }
    assert check_regressions(base, some, 0.05).passed
    many = {
        t: [F, F] if i < 8 else [T, T] for i, t in enumerate(ids)
    }
    result = check_regressions(base, many, 0.05)
    assert not result.passed
    assert result.detail.startswith("8 broke, 0 fixed")


def test_baseline_round_trip_and_subset(tmp_path):
    base = Baseline("v1", 4, OUTCOMES, {"cost": 0.002})
    path = tmp_path / "baseline.json"
    save_baseline(path, base)
    loaded = load_baseline(path)
    assert loaded == base
    sub = loaded.metrics(["T-1", "T-3"])
    assert sub["pass_rate"] == pytest.approx(0.5)
    assert sub["cost"] == 0.002


def test_run_gate_exit_codes_and_critical_quarantine():
    base = Baseline("v1", 4, OUTCOMES, {})
    policy = GatePolicy(
        (Tolerance("pass_rate", 0.05),),
        critical=frozenset({"T-1"}),
        flaky_budget=0.75,
    )
    same = run_gate("same", OUTCOMES, base, policy)
    assert same.passed and same.exit_code == 0
    worse = {**OUTCOMES, "T-1": [T, F, T, T]}
    report = run_gate("worse", worse, base, policy)
    assert report.exit_code == 1
    failed = [c.name for c in report.checks if not c.passed]
    assert failed == ["pass_rate", "critical"]
    hide = [Quarantine("T-1", "flaky", "2027-01-01")]
    held = run_gate(
        "hide", worse, base, policy, hide, "2026-11-15"
    )
    names = [c.name for c in held.checks if not c.passed]
    assert names == ["quarantine"]


def test_collect_is_deterministic_per_seed():
    tasks = make_tasks(10, seed=7)
    agent = SimAgent(IMPROVED)
    # ground truth stands in for a real grader's verdict
    first = collect(agent, tasks, range(3), lambda t, r: r.ok)
    again = collect(agent, tasks, range(3), lambda t, r: r.ok)
    assert first == again
    assert all(len(r) == 3 for r in first.values())


def test_example_main(capsys):
    from ch14_regression_gates import main

    main()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) <= 25
    assert max(map(len, lines)) <= 64
    assert "Gate v3 +stop_after_answer: PASS (exit 0)" in lines
    assert "Gate v4 -check_refund_policy: FAIL (exit 1)" in lines
    assert any("FAIL critical" in line for line in lines)
    assert lines[-1].endswith("retries hide it")
