"""Chapter 7: model-graded evaluation and its biases.

The rubric judge catches what chapter 6's checks missed: an
order status that contradicts the lookup, and a rude reply
in new words. It is also noisier, and its habits move
scores: padding, a familiar author, the first slot of a
pair. Sampling and swapping positions are the cheap fixes.
"""

from __future__ import annotations

from dataclasses import replace
from statistics import mean

from ch06_graders import reword
from evalloop.core import (
    BASELINE,
    IMPROVED,
    SimAgent,
    make_tasks,
    run_suite,
)
from evalloop.graders import grade
from evalloop.judge import (
    PairwiseJudge,
    ReferenceJudge,
    RubricJudge,
    compare_both_orders,
    majority_vote,
)

FILLER = " Thank you for reaching out; we value your business."


def accuracy(verdicts, runs) -> str:
    hits = sum(
        v == r.ok for v, r in zip(verdicts, runs, strict=True)
    )
    return f"{hits / len(runs):.1%}"


def show_catches(judge, by_id, runs) -> None:
    slip = next(r for r in runs if r.task_id == "T-0005")
    calm = next(r for r in runs if r.task_id == "T-0010")
    rude = reword(calm, "I'm sorry.", "Relax, nobody cares.")
    for label, run in (
        ("T-0005", slip),
        ("T-0010 reworded", rude),
    ):
        task = by_id[run.task_id]
        suite = grade(task, run).passed
        v = judge.grade(task, run)
        print(f"  {label}: suite pass={suite}, judge {v.score}")
        print(f"    {v.rationale[:58]}")


def show_accuracy(judge, by_id, runs) -> None:
    def verdicts(fn):
        return [fn(by_id[r.task_id], r).passed for r in runs]

    rows = [
        ("deterministic suite", verdicts(grade)),
        ("rubric judge, 1 sample", verdicts(judge.grade)),
        (
            "rubric judge, 5 samples",
            verdicts(lambda t, r: majority_vote(judge, t, r)),
        ),
        ("reference judge", verdicts(ReferenceJudge().grade)),
    ]
    print(
        f"Agreement with ground truth, {len(runs)} BASELINE runs:"
    )
    for label, passed in rows:
        print(f"  {label:<24} {accuracy(passed, runs)}")


def mean_score(judge, by_id, runs, author=None) -> float:
    return mean(
        judge.grade(by_id[r.task_id], r, author=author).score
        for r in runs
    )


def show_biases(judge, by_id, runs) -> None:
    base = mean_score(judge, by_id, runs)
    padded = [replace(r, answer=r.answer + FILLER) for r in runs]
    print("Biases (same runs, one thing changed):")
    print(
        f"  mean score {base:.2f}; padded with filler "
        f"{mean_score(judge, by_id, padded):.2f}"
    )
    same = mean_score(judge, by_id, runs, author="medium")
    print(f"  author on the judge's tier: {same:.2f}")


def show_position(by_id, runs, better) -> None:
    pair = PairwiseJudge()
    wins = {"slot A": 0, "slot B": 0, "both orders": 0}
    for old, new in zip(runs, better, strict=True):
        task = by_id[old.task_id]
        wins["slot A"] += pair.compare(task, new, old) == "A"
        wins["slot B"] += pair.compare(task, old, new) == "B"
        both = compare_both_orders(pair, task, new, old)
        wins["both orders"] += both == "A"
    print("Pairwise: improved beats baseline when it sits in")
    print(
        "  "
        + ", ".join(
            f"{k} {v / len(runs):.0%}" for k, v in wins.items()
        )
    )


def main() -> None:
    tasks = make_tasks(200, seed=7)
    by_id = {t.id: t for t in tasks}
    runs = run_suite(SimAgent(BASELINE), tasks)
    judge = RubricJudge()
    print("Rubric judge (medium tier, pass at 4 of 5):")
    show_catches(judge, by_id, runs)
    show_accuracy(judge, by_id, runs)
    show_biases(judge, by_id, runs)
    better = run_suite(SimAgent(IMPROVED), tasks)
    show_position(by_id, runs, better)
    print("Measure these biases before trusting scores (ch08).")


if __name__ == "__main__":
    main()
