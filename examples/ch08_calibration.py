"""Chapter 8: calibrating a judge against human labels.

Simulated reviewers label 200 baseline runs. The judge is
scored against them with a confusion matrix and Cohen's
kappa, its biases are measured one at a time, its pass
threshold is chosen on one half and checked on the other,
and a cheaper replacement judge is caught drifting.
"""

from __future__ import annotations

from evalloop.calibration import (
    choose_threshold,
    cohen_kappa,
    confusion,
    drift_check,
    human_label,
    kappa_by_threshold,
    padding_effect,
    pearson,
    position_flip_rate,
    self_preference_gap,
    split_half,
)
from evalloop.core import (
    BASELINE,
    IMPROVED,
    SimAgent,
    make_tasks,
    run_suite,
)
from evalloop.judge import (
    JudgeConfig,
    PairwiseJudge,
    RubricJudge,
    majority_vote,
)


def verdicts(judge, by_id, runs) -> list:
    return [judge.grade(by_id[r.task_id], r) for r in runs]


def show_agreement(judge_pass, human, runs) -> None:
    truth = [r.ok for r in runs]
    same = sum(h == t for h, t in zip(human, truth, strict=True))
    print(f"Calibration: {len(runs)} BASELINE runs, human labels")
    print(
        f"  humans vs ground truth: {same / len(runs):.1%} agree,"
        f" kappa {cohen_kappa(human, truth):.2f}"
    )
    c = confusion(judge_pass, human)
    print("Judge v1 (1 sample, pass at 4) vs humans:")
    print(
        f"  agreement {c.accuracy:.1%},"
        f" kappa {cohen_kappa(judge_pass, human):.2f}"
    )
    counts = f"TP {c.tp} FP {c.fp} FN {c.fn} TN {c.tn}"
    print(f"  fail=positive: {counts}")
    rates = f"precision {c.precision:.2f}, recall {c.recall:.2f}"
    print(f"  fail {rates}")


def show_biases(judge, by_id, runs, results, human) -> None:
    better = run_suite(SimAgent(IMPROVED), by_id.values())
    pairs = list(zip(better, runs, strict=True))
    flips = position_flip_rate(PairwiseJudge(), by_id, pairs)
    passes = [i for i, h in enumerate(human) if h]
    r = pearson(
        [len(runs[i].answer) for i in passes],
        [results[i].score for i in passes],
    )
    delta, _ = padding_effect(judge, by_id, runs)
    print("Bias tests:")
    print(
        f"  position: {flips:.0%} of pairs flip when slots swap"
    )
    print(
        f"  length: r={r:.2f} on human passes; pad {delta:+.2f}"
    )
    gap = self_preference_gap(judge, by_id, runs)
    print(f"  self-preference: {gap:+.2f} for a same-tier author")


def kappa_row(label, scores, human, rows) -> None:
    table = kappa_by_threshold(
        [scores[i] for i in rows], [human[i] for i in rows]
    )
    cells = "  ".join(f"t={t} {k:.2f}" for t, k in table.items())
    print(f"  {label:<16}{cells}")


def show_threshold(judge, by_id, runs, human) -> None:
    scores = [
        majority_vote(judge, by_id[r.task_id], r).score
        for r in runs
    ]
    index = list(range(len(runs)))
    calib, held = split_half(index, key=lambda i: runs[i].task_id)
    print("Pass threshold, 5-sample median scores (kappa):")
    kappa_row("calibration", scores, human, calib)
    kappa_row("held-out", scores, human, held)
    best = choose_threshold(
        [scores[i] for i in calib], [human[i] for i in calib]
    )
    print(f"  chosen on calibration half: pass_at={best}")


def main() -> None:
    tasks = make_tasks(200, seed=7)
    by_id = {t.id: t for t in tasks}
    runs = run_suite(SimAgent(BASELINE), tasks)
    human = [human_label(r) for r in runs]
    judge = RubricJudge()
    results = verdicts(judge, by_id, runs)
    v1_pass = [v.passed for v in results]
    show_agreement(v1_pass, human, runs)
    show_biases(judge, by_id, runs, results, human)
    show_threshold(judge, by_id, runs, human)
    cheap = JudgeConfig("judge-v2", "small", leniency=0.8)
    v2 = verdicts(RubricJudge(cheap), by_id, runs)
    report = drift_check(v1_pass, [v.passed for v in v2], human)
    print("Drift: v2 = small tier, leniency 0.8")
    print(
        f"  verdicts agree {report.agreement:.1%} with v1;"
        f" kappa {report.kappa_old:.2f} -> {report.kappa_new:.2f}"
    )
    print(
        f"  drifted: {report.drifted}; recalibrate before a swap"
    )


if __name__ == "__main__":
    main()
