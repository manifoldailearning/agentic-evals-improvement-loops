"""Chapter 9: running annotation as an operation.

Five raters label 120 baseline runs with overlap three.
A guideline that is silent on some failure types leaves
raters to their own opinions, so agreement and final label
quality drop. Gold questions expose the weakest rater, and
the bill shows what each final label really costs.
"""

from __future__ import annotations

from evalloop.annotation import (
    EXPERT,
    GUIDE_V1,
    GUIDE_V2,
    POOL,
    adjudicate,
    annotate,
    assign,
    collect,
    contested,
    cost_report,
    coverage,
    flagged,
    fleiss_kappa,
    gold_scores,
)
from evalloop.core import (
    BASELINE,
    SimAgent,
    make_tasks,
    run_suite,
)


def label_round(runs, assignment, guide):
    """Collect labels, adjudicate, and report agreement."""
    labels = collect(runs, assignment, POOL, guide)
    kappa = fleiss_kappa(
        [list(v.values()) for v in labels.values()]
    )
    by_id = {r.task_id: r for r in runs}
    final = adjudicate(
        labels, lambda item: annotate(EXPERT, by_id[item], guide)
    )
    right = sum(final[r.task_id] == r.ok for r in runs)
    print(f"Guideline {guide.version} ({coverage(guide)}):")
    print(
        f"  Fleiss' kappa {kappa:.2f}, contested"
        f" {len(contested(labels))}, final vs truth"
        f" {right / len(runs):.1%}"
    )
    return labels


def show_gold(labels, runs) -> None:
    gold = {r.task_id: r.ok for r in runs[:50]}
    scores = gold_scores(labels, gold)
    weak = flagged(scores, min_accuracy=0.85)
    print(
        f"Gold questions ({len(gold)} items with known labels):"
    )
    for rater in POOL:
        right, seen = scores[rater.id]
        price = rater.hourly_rate / rater.items_per_hour
        note = "  flagged" if rater.id in weak else ""
        who = f"{rater.id} skill {rater.skill:.2f}"
        print(
            f"  {who}: {right}/{seen}  ${price:.2f}/label{note}"
        )


def main() -> None:
    runs = run_suite(SimAgent(BASELINE), make_tasks(120, seed=7))
    items = [r.task_id for r in runs]
    assignment = assign(items, POOL, overlap=3)
    print(
        f"Labeling {len(runs)} BASELINE runs: 5 raters, overlap 3"
    )
    label_round(runs, assignment, GUIDE_V1)
    labels = label_round(runs, assignment, GUIDE_V2)
    show_gold(labels, runs)
    disputed = len(contested(labels))
    bill = cost_report(labels, POOL, EXPERT, disputed)
    print(
        f"Cost ({GUIDE_V2.version}): {bill.labels} labels,"
        f" {bill.hours} h, ${bill.cost:.2f}"
    )
    per = bill.per_final_label
    print(
        f"  ${per:.2f} per final label; {disputed} went to expert"
    )


if __name__ == "__main__":
    main()
