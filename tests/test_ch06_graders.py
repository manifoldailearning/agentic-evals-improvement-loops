from __future__ import annotations

from dataclasses import replace

import pytest

from evalloop.core import (
    BASELINE,
    DEMO_TASKS,
    SimAgent,
    Step,
    Trajectory,
)
from evalloop.graders import (
    DEFAULT_SUITE,
    Grade,
    all_of,
    any_of,
    args_match,
    citation_valid,
    grade,
    refund_policy,
    right_tools,
    schema_errors,
    step_budget,
    valid_tool_args,
    weighted,
)

TASKS = {t.id: t for t in DEMO_TASKS}


def baseline_run(task_id: str) -> Trajectory:
    return SimAgent(BASELINE).run(TASKS[task_id])


def make_run(task_id: str, steps, answer: str) -> Trajectory:
    steps = [*steps, Step("answer", result=answer)]
    return Trajectory(
        task_id, "t", "k", 0, steps, answer, 1, 1, 1, 0
    )


def fixed(passed: bool, score: float):
    return lambda task, traj: Grade("stub", score, passed)


def test_clean_run_passes_default_suite():
    run = baseline_run("T-0001")
    result = grade(TASKS["T-0001"], run)
    assert result.passed and result.score == 1.0
    assert result.reasons == ()


def test_right_tools_reports_missing_and_unexpected():
    lookup = Step("tool", "search_kb", {"query": "x"}, {})
    run = make_run("T-0001", [lookup], "Done.")
    result = right_tools(TASKS["T-0001"], run)
    assert not result.passed
    assert result.reasons == (
        "missing tool lookup_order",
        "unexpected tool search_kb",
    )


def test_args_match_normalizes_unless_exact():
    task = TASKS["T-0001"]
    oid = task.expected["args"]["order_id"]
    call = Step("tool", "lookup_order", {"order_id": f" {oid} "})
    run = make_run(task.id, [call], "ok")
    assert args_match(task, run).passed
    assert not args_match(task, run, exact=True).passed


def test_schema_errors_and_empty_tool_args():
    schema = {"order_id": str, "amount": (int, float)}
    errors = schema_errors({"amount": "12"}, schema)
    assert errors == [
        "missing order_id",
        "amount has the wrong type",
    ]
    run = make_run("T-0001", [Step("tool", "lookup_order")], "")
    result = valid_tool_args(TASKS["T-0001"], run)
    assert result.reasons == ("lookup_order: missing order_id",)


def test_citation_must_be_real_and_read():
    task = TASKS["T-0003"]  # faq: expects a KB article
    read = Step("tool", "search_kb", {}, {"article": "warranty"})
    fake = make_run(task.id, [read], "Yes. [kb:made-up]")
    unread = make_run(task.id, [], "Yes. [kb:returns]")
    bare = make_run(task.id, [read], "Yes, 12 months.")
    assert citation_valid(task, fake).reasons == (
        "cites unknown made-up",
    )
    assert (
        "cites unread returns"
        in citation_valid(task, unread).reasons
    )
    assert citation_valid(task, bare).reasons == ("no citation",)


def test_refund_over_limit_breaks_policy():
    run = baseline_run("T-0012")  # adversarial refund request
    result = refund_policy(TASKS["T-0012"], run)
    assert not result.passed
    assert "breaks policy" in result.reasons[0]


def test_step_budget_fails_a_looping_run():
    run = baseline_run("T-0009")  # repeats search_kb, no answer
    result = step_budget(5)(TASKS["T-0009"], run)
    assert result.reasons == ("no final answer", "6 calls > 5")


def test_composition_all_any_weighted():
    task, run = TASKS["T-0001"], baseline_run("T-0001")
    good, bad = fixed(True, 1.0), fixed(False, 0.5)
    assert not all_of(good, bad)(task, run).passed
    assert any_of(good, bad)(task, run).passed
    score = weighted([(good, 3), (bad, 1)], threshold=0.8)
    result = score(task, run)
    assert result.score == 0.875 and result.passed


def test_grade_rejects_a_run_of_another_task():
    with pytest.raises(ValueError):
        grade(TASKS["T-0002"], baseline_run("T-0001"))


def test_grades_ignore_ground_truth_fields():
    for task_id in ("T-0001", "T-0005", "T-0012"):
        run = baseline_run(task_id)
        flipped = replace(
            run, faults=[] if run.faults else ["loop"]
        )
        task = TASKS[task_id]
        assert DEFAULT_SUITE(task, run) == DEFAULT_SUITE(
            task, flipped
        )


def test_example_prints_agreement(capsys):
    from ch06_graders import main

    main()
    out = capsys.readouterr().out
    assert "agreement 98.5%, false fails 0" in out
    assert "missed hallucinated_fact: 3/11" in out
