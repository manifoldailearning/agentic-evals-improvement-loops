from ch10_trajectory import main
from evalloop.core import (
    BASELINE,
    CATEGORIES,
    IMPROVED,
    SimAgent,
    Step,
    Trajectory,
    World,
    make_tasks,
    run_suite,
)
from evalloop.trajectory import (
    arg_accuracy,
    args_match,
    blame_counts,
    by_category,
    first_bad_step,
    order_violations,
    passes,
    redundant_calls,
    score_runs,
    summarize,
    tool_precision_recall,
)

TASKS = make_tasks(200, seed=7)
BY_ID = {t.id: t for t in TASKS}


def call(tool, **args):
    return Step("tool", tool, args, World().call(tool, args))


def traj(*steps, answer=""):
    steps = list(steps)
    if answer:
        steps.append(Step("answer", result=answer))
    return Trajectory(
        "T", "v", "k", 0, steps, answer, 0, 0, 0, 0.0
    )


def test_precision_and_recall_count_repeats_and_misses():
    task = BY_ID["T-0002"]  # lookup, search_kb, issue_refund
    look = call("lookup_order", order_id="A-1004")
    run = traj(look, look)
    assert tool_precision_recall(run, task) == (0.5, 1 / 3)
    clarify = BY_ID["T-0007"]  # expects no tool calls
    assert tool_precision_recall(traj(), clarify) == (1.0, 1.0)
    assert tool_precision_recall(traj(look), clarify)[0] == 0


def test_args_match_normalizes_text_and_money():
    task = BY_ID["T-0002"]
    amount = task.expected["args"]["amount"]
    assert args_match(
        call("search_kb", query=" Refund POLICY"), task
    )
    exact = call("issue_refund", order_id="A-1004", amount=amount)
    assert args_match(exact, task)
    wrong = call("lookup_order", order_id="A-1005")
    assert arg_accuracy(traj(wrong, exact), task) == 0.5


def test_order_violation_only_when_first_tool_expected():
    refund = BY_ID["T-0002"]
    pay = call("issue_refund", order_id="A-1004", amount=10.0)
    look = call("lookup_order", order_id="A-1004")
    assert ("lookup_order", "issue_refund") in order_violations(
        traj(pay), refund
    )
    assert order_violations(traj(look, pay), refund) == [
        ("search_kb", "issue_refund")
    ]
    faq = BY_ID["T-0003"]
    assert (
        order_violations(traj(call("escalate", reason="x")), faq)
        == []
    )


def test_redundant_calls_need_identical_args():
    a = call("lookup_order", order_id="A-1004")
    b = call("lookup_order", order_id="A-1005")
    assert redundant_calls(traj(a, b)) == 0
    assert redundant_calls(traj(a, b, a, a)) == 2


def test_passes_agrees_with_ground_truth():
    for config in (BASELINE, IMPROVED):
        runs = run_suite(SimAgent(config), TASKS, seeds=(0, 1))
        for r in runs:
            assert passes(r, BY_ID[r.task_id]) == r.ok


def test_first_bad_step_localizes_known_faults():
    task = BY_ID["T-0012"]  # over the limit: must escalate
    run = SimAgent(BASELINE).run(task)
    assert run.faults == ["policy_violation"]  # ground truth
    blame = first_bad_step(run, task)
    assert (blame.step, blame.kind) == (3, "forbidden")
    fine = SimAgent(IMPROVED).run(task)
    assert fine.ok and first_bad_step(fine, task) is None
    status = BY_ID["T-0001"]
    lie = call("lookup_order", order_id="A-1008")
    blame = first_bad_step(
        traj(lie, answer="It is lost."), status
    )
    assert (blame.step, blame.kind) == (2, "answer")


def test_summaries_show_improvement_and_its_cost():
    base = summarize(
        score_runs(run_suite(SimAgent(BASELINE), TASKS), TASKS)
    )
    best = summarize(
        score_runs(run_suite(SimAgent(IMPROVED), TASKS), TASKS)
    )
    assert best["success"] - base["success"] > 0.2
    assert best["tool_precision"] > base["tool_precision"]
    assert best["tool_recall"] > base["tool_recall"]
    ratio = best["cost_per_success"] / base["cost_per_success"]
    assert ratio > 5


def test_category_rollup_and_blame_cover_every_run():
    scores = score_runs(
        run_suite(SimAgent(BASELINE), TASKS), TASKS
    )
    cats = by_category(scores)
    assert set(cats) == set(CATEGORIES)
    assert (
        cats["refund"]["success"]
        < cats["order_status"]["success"]
    )
    failed = sum(not s.passed for s in scores)
    assert sum(blame_counts(scores).values()) == failed
    none_pass = [s for s in scores if not s.passed]
    assert summarize(none_pass)["cost_per_success"] == 0.0


def test_example_prints_localized_failures(capsys):
    main()
    out = capsys.readouterr().out
    assert "success                    63.5%     89.5%" in out
    assert "T-0012 refund, step 3: called issue_refund" in out
    assert all(len(line) <= 64 for line in out.splitlines())
    assert len(out.splitlines()) <= 25
