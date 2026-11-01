import ch01_vibes_vs_evals as ch01
from evalloop.core import BASELINE, IMPROVED, make_tasks

TASKS = make_tasks(200, seed=7)
DEMO = [t for t in TASKS if t.id in ch01.DEMO_IDS]


def test_demo_tasks_pass_for_both_configs():
    assert len(DEMO) == 3
    assert ch01.vibe_check(BASELINE, DEMO) == ["PASS"] * 3
    assert ch01.vibe_check(IMPROVED, DEMO) == ["PASS"] * 3


def test_vibe_check_reports_failures_too():
    refunds = [t for t in TASKS if t.category == "refund"]
    assert "FAIL" in ch01.vibe_check(BASELINE, refunds[:10])


def test_full_eval_separates_the_configs():
    base = ch01.success_by_category(BASELINE, TASKS)
    best = ch01.success_by_category(IMPROVED, TASKS)
    assert base["overall"] == 0.635
    assert best["overall"] == 0.895
    assert all(best[c] > base[c] for c in base)


def test_breakdown_finds_refund_as_weakest_baseline_category():
    base = ch01.success_by_category(BASELINE, TASKS)
    cats = {k: v for k, v in base.items() if k != "overall"}
    assert min(cats, key=cats.get) == "refund"


def test_success_by_category_handles_a_single_task():
    rates = ch01.success_by_category(BASELINE, TASKS[:1])
    assert set(rates) == {TASKS[0].category, "overall"}


def test_example_main_shows_the_hidden_gap(capsys):
    ch01.main()
    out = capsys.readouterr().out
    assert "Verdict: both look ready to ship." in out
    assert "Gap: 26.0 points" in out
    lines = out.splitlines()
    assert len(lines) <= 25
    assert max(len(line) for line in lines) <= 64
