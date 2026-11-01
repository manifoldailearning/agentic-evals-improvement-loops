import json

import pytest

from evalloop.core import (
    BASELINE,
    IMPROVED,
    SimAgent,
    make_tasks,
    run_suite,
)
from evalloop.loop import (
    ExperimentLog,
    Loop,
    apply_change,
    dataset_version,
    evidence,
    gate,
    passed,
    split,
    triage,
)

TASKS = make_tasks(200, seed=7)


def test_checker_agrees_with_ground_truth():
    for config in (BASELINE, IMPROVED):
        runs = run_suite(SimAgent(config), TASKS)
        agree = sum(
            passed(t, r) == r.ok
            for t, r in zip(TASKS, runs, strict=True)
        )
        assert agree / len(runs) >= 0.97


def test_evidence_names_the_visible_problem():
    agent = SimAgent(BASELINE)
    seen = {}
    for t in TASKS:
        run = agent.run(t)
        for fault in run.faults:
            seen.setdefault(fault, set()).update(evidence(t, run))
    assert "missing_citation" in seen["no_citation"]
    assert "forbidden_action" in seen["policy_violation"]
    assert "guessed_order" in seen["ignored_ambiguity"]
    assert seen["loop"] == {"no_answer"}


def test_triage_ranks_untried_changes_by_count():
    runs = run_suite(SimAgent(BASELINE), TASKS)
    ideas = triage(TASKS, runs)
    counts = [h.count for h in ideas]
    assert counts == sorted(counts, reverse=True)
    top = ideas[0].change
    later = triage(TASKS, runs, frozenset({top}))
    assert top not in {h.change for h in later}


def test_apply_change_edits_one_field():
    c = apply_change(BASELINE, "rule:cite_kb")
    assert c.rules == {"cite_kb"}
    assert apply_change(c, "few_shots:3").few_shots == 3
    assert apply_change(c, "model:medium").model == "medium"
    assert apply_change(c, "temperature:0.2").temperature == 0.2
    with pytest.raises(ValueError):
        apply_change(c, "max_tokens:9")


def test_gate_needs_dev_gain_confirmed_on_test():
    assert gate((0.6, 0.7), (0.6, 0.65), 0.01)[0] == "accept"
    assert gate((0.6, 0.6), (0.6, 0.7), 0.01)[1] == "no dev gain"
    decision, why = gate((0.6, 0.7), (0.6, 0.59), 0.01)
    assert decision == "reject" and "test" in why


def test_split_is_stable_disjoint_and_versioned():
    dev, test = split(TASKS)
    assert split(TASKS) == (dev, test)
    assert not {t.id for t in dev} & {t.id for t in test}
    assert len(dev) + len(test) == len(TASKS)
    assert dataset_version(dev, test) != dataset_version(test)


def test_loop_raises_test_success_and_logs(tmp_path):
    dev, test = split(TASKS)
    loop = Loop(dev, test)
    config = loop.run(BASELINE, 4)
    kept = loop.log.accepted()
    assert kept and kept[-1].version == config.version
    assert kept[-1].test[1] > loop.log.records[0].test[0] + 0.1
    path = tmp_path / "log.jsonl"
    loop.log.save(path)
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert len(rows) == 4 and rows[0]["config_key"]


def test_rejected_change_keeps_incumbent():
    dev, test = split(TASKS)
    loop = Loop(dev, test, min_gain=1.0)  # nothing can pass
    assert loop.step(BASELINE) is BASELINE
    assert loop.log.records[0].decision == "reject"
    assert ExperimentLog().accepted() == []


def test_example_main(capsys):
    from ch16_improvement_loop import main

    main()
    out = capsys.readouterr().out
    assert "baseline: dev 58.7%" in out
    assert "sweep of 9" in out and "reject" in out
    lines = out.splitlines()
    assert len(lines) <= 25 and max(map(len, lines)) <= 64
