from collections import Counter

import pytest

from evalloop.core import (
    BASELINE,
    IMPROVED,
    RULES,
    AgentConfig,
    SimAgent,
    Trajectory,
    World,
    failure_probs,
    load_tasks,
    make_tasks,
    run_suite,
    save_tasks,
)
from evalloop.core.failures import FAILURE_MODES, RULE_CUT

TASKS = make_tasks(200, seed=7)


def rate(config, tasks=TASKS):
    runs = run_suite(SimAgent(config), tasks)
    return sum(r.ok for r in runs) / len(runs)


def faults(config, mode):
    runs = run_suite(SimAgent(config), TASKS)
    return {r.task_id for r in runs if mode in r.faults}


def test_tasks_are_seeded_varied_and_roundtrip(tmp_path):
    assert make_tasks(20, seed=7) == TASKS[:20]
    assert make_tasks(20, seed=8) != TASKS[:20]
    assert TASKS[0].id == "T-0001"
    assert set(Counter(t.category for t in TASKS).values()) == {
        40
    }
    tags = {tag for t in TASKS for tag in t.tags}
    assert {"ambiguous", "adversarial", "long_input"} <= tags
    path = tmp_path / "tasks.jsonl"
    save_tasks(TASKS, path)
    assert load_tasks(path) == TASKS


def test_config_key_ignores_version_and_validates():
    a = AgentConfig("a", rules=frozenset({"cite_kb"}))
    assert a.key() == a.evolve(version="b").key()
    assert a.key() != a.evolve(few_shots=1).key()
    with pytest.raises(ValueError):
        AgentConfig(model="huge")
    with pytest.raises(ValueError):
        AgentConfig(rules=frozenset({"be_nice"}))


def test_run_is_deterministic_and_serializable():
    task = TASKS[11]
    one = SimAgent(BASELINE).run(task, seed=3)
    two = SimAgent(BASELINE).run(task, seed=3)
    assert one == two
    assert Trajectory.from_dict(one.to_dict()) == one


def test_temperature_controls_seed_disagreement():
    def flips(config):
        agent = SimAgent(config)
        seeds = range(5)
        return sum(
            len({agent.run(t, s).ok for s in seeds}) > 1
            for t in TASKS[:60]
        )

    assert flips(BASELINE.evolve(temperature=0.0)) == 0
    assert flips(BASELINE) > 10


def test_failure_probs_respond_to_each_config_feature():
    task = next(
        t
        for t in TASKS
        if t.expected["eligible"] is False
        and "adversarial" not in t.tags
    )
    base = failure_probs(task, BASELINE)
    ruled = BASELINE.evolve(
        rules=frozenset({"check_refund_policy"})
    )
    cut = failure_probs(task, ruled)["policy_violation"]
    assert cut == pytest.approx(
        base["policy_violation"] * RULE_CUT, abs=1e-3
    )
    large = failure_probs(task, BASELINE.evolve(model="large"))
    assert all(large[m] <= base[m] for m in base)
    shots = failure_probs(task, BASELINE.evolve(few_shots=3))
    assert shots["wrong_tool"] < base["wrong_tool"]
    assert shots["no_citation"] == base["no_citation"]


def test_each_rule_removes_most_of_its_failure_mode():
    for mode in FAILURE_MODES:
        if not mode.rule:
            continue
        before = faults(BASELINE, mode.name)
        fixed = BASELINE.evolve(rules=frozenset({mode.rule}))
        after = faults(fixed, mode.name)
        assert after <= before, mode.name
        assert len(after) <= len(before) / 2, mode.name


def test_baseline_and_improved_success_rates():
    assert 0.55 <= rate(BASELINE) <= 0.65
    assert 0.85 <= rate(IMPROVED) <= 0.92
    assert set(RULES[:4]) == IMPROVED.rules


def test_fault_free_runs_behave_correctly():
    runs = run_suite(SimAgent(BASELINE), TASKS)
    for run, task in zip(runs, TASKS, strict=True):
        if not run.ok:
            continue
        exp = task.expected
        assert [s.tool for s in run.tool_calls] == exp["tools"]
        assert all(
            f in run.answer for f in exp["answer_contains"]
        )
        if exp["kb"]:
            assert f"[kb:{exp['kb']}]" in run.answer


def test_policy_violation_is_a_recorded_side_effect():
    world = World()
    agent = SimAgent(BASELINE, world)
    bad = [
        r
        for r in run_suite(agent, TASKS)
        if "policy_violation" in r.faults
    ]
    assert bad
    assert any(e.get("ok") is False for e in world.effects)
    assert all(
        r.tool_calls[-1].tool == "issue_refund" for r in bad
    )


def test_step_limit_produces_loop_without_answer():
    task = next(t for t in TASKS if len(t.expected["tools"]) == 3)
    run = SimAgent(IMPROVED.evolve(max_steps=2)).run(task)
    assert "loop" in run.faults
    assert run.answer == "" and len(run.steps) == 2


def test_world_tools_report_errors():
    w = World()
    assert "error" in w.call("lookup_order", {"order_id": "Z-1"})
    assert "error" in w.call("lookup_order", {})
    assert "error" in w.call("search_kb", {"query": "pizza"})
    assert "error" in w.call("delete_db", {})
    assert (
        w.call("search_kb", {"query": "warranty"})["article"]
        == "warranty"
    )


def test_bigger_model_costs_more():
    task = TASKS[0]
    small = SimAgent(BASELINE).run(task)
    large = SimAgent(BASELINE.evolve(model="large")).run(task)
    assert 0 < small.cost < large.cost
    assert 0 < small.latency_ms < large.latency_ms


def test_example_main_prints_suite_summary(capsys):
    from ch02_anatomy import main

    main()
    out = capsys.readouterr().out
    assert "Faults (ground truth): policy_violation" in out
    assert (
        "Suite (200 tasks): baseline 63.5% -> improved 89.5%"
        in out
    )
    assert max(len(line) for line in out.splitlines()) <= 64
