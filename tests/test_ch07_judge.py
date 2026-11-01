from __future__ import annotations

from dataclasses import replace
from statistics import mean

from evalloop.core import (
    BASELINE,
    DEMO_TASKS,
    SimAgent,
    Step,
    Trajectory,
)
from evalloop.judge import (
    RUBRIC,
    JudgeConfig,
    PairwiseJudge,
    ReferenceJudge,
    RubricJudge,
    compare_both_orders,
    find_issues,
    length_corrected,
    majority_vote,
    render_prompt,
    ungrounded,
)

TASKS = {t.id: t for t in DEMO_TASKS}
RUNS = [SimAgent(BASELINE).run(t) for t in DEMO_TASKS]


def run_of(task_id: str) -> Trajectory:
    return next(r for r in RUNS if r.task_id == task_id)


def mean_score(judge, runs, author=None) -> float:
    return mean(
        judge.grade(TASKS[r.task_id], r, author=author).score
        for r in runs
    )


def test_prompt_renders_rubric_and_transcript():
    run = run_of("T-0001")
    prompt = render_prompt(RUBRIC, TASKS["T-0001"], run)
    for criterion in RUBRIC:
        assert (
            f"- {criterion.name}: {criterion.question}" in prompt
        )
    assert prompt.endswith(f"Reply: {run.answer}")


def test_issues_empty_for_clean_run_and_grounding_caught():
    clean = find_issues(TASKS["T-0001"], run_of("T-0001"))
    assert not any(clean.values())
    slip = find_issues(TASKS["T-0005"], run_of("T-0005"))
    assert slip["grounded"] == [
        "says A-1041 is shipped, tools say delayed"
    ]


def test_ungrounded_refund_amount():
    call = Step("tool", "issue_refund", {"amount": 50.0}, {})
    answer = "I have refunded 75.00."
    steps = [call, Step("answer", result=answer)]
    run = Trajectory(
        "T-0002", "v", "k", 0, steps, answer, 1, 1, 1, 0
    )
    assert ungrounded(run) == [
        "says refunded 75.00, refund was other"
    ]


def test_tone_flags_dismissive_and_cold_replies():
    run = run_of("T-0010")  # angry customer, polite reply
    task = TASKS["T-0010"]
    assert find_issues(task, run)["tone"] == []
    rude = replace(
        run, answer=run.answer.replace("I'm sorry.", "Relax.")
    )
    cold = replace(
        run, answer=run.answer.replace("I'm sorry. ", "")
    )
    assert (
        find_issues(task, rude)["tone"][0] == "dismissive 'relax'"
    )
    assert find_issues(task, cold)["tone"] == [
        "no empathy for an upset customer"
    ]


def test_judge_is_deterministic_but_samples_vary():
    judge = RubricJudge()
    task, run = TASKS["T-0012"], run_of("T-0012")
    assert judge.grade(task, run) == judge.grade(task, run)
    scores = {
        judge.grade(TASKS[r.task_id], r, sample=s).score
        for r in RUNS[:10]
        for s in range(5)
    }
    assert len(scores) > 2


def test_leniency_length_and_self_preference_raise_scores():
    strict = RubricJudge(JudgeConfig(leniency=0.0))
    judge = RubricJudge()
    assert mean_score(judge, RUNS) > mean_score(strict, RUNS)
    padded = [
        replace(r, answer=r.answer + " " * 80) for r in RUNS
    ]
    assert mean_score(judge, padded) > mean_score(judge, RUNS)
    same_tier = mean_score(judge, RUNS, author="medium")
    assert same_tier > mean_score(judge, RUNS, author="small")


def test_length_correction_undoes_the_slope():
    answer = "x" * 160  # 100 chars over the pivot
    assert length_corrected(4.0, answer, slope=1.0) == 3.0


def test_position_bias_and_swapped_mitigation():
    pair = PairwiseJudge(JudgeConfig(noise=0.0))
    task, run = TASKS["T-0001"], run_of("T-0001")
    assert pair.compare(task, run, run) == "A"
    assert compare_both_orders(pair, task, run, run) == "tie"
    fair = PairwiseJudge(JudgeConfig(noise=0.0, position_bias=0))
    assert fair.compare(task, run, run) == "tie"


def test_majority_vote_and_reference_blind_spot():
    judge = RubricJudge()
    task, slip = TASKS["T-0005"], run_of("T-0005")
    assert not majority_vote(judge, task, slip, samples=5).passed
    reference = ReferenceJudge().grade(task, slip)
    assert reference.passed  # the wrong status is not in the ref


def test_judges_ignore_ground_truth_fields():
    judge = RubricJudge()
    for run in RUNS[:15]:
        flipped = replace(
            run, faults=[] if run.faults else ["loop"]
        )
        task = TASKS[run.task_id]
        assert judge.grade(task, run) == judge.grade(
            task, flipped
        )


def test_example_shows_judge_catches(capsys):
    from ch07_judges import main

    main()
    out = capsys.readouterr().out
    assert "grounded: says A-1041 is shipped" in out
    assert "rubric judge, 5 samples  93.5%" in out
