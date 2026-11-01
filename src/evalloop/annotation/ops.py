"""Annotation operations: assign, agree, adjudicate, audit, pay.

Labels at scale need process. Items go to several raters
with balanced overlap; Fleiss' kappa measures agreement
among more than two raters; contested items go to an expert;
gold questions with known answers catch weak raters; and
the cost of every final label is accounted for.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from evalloop.annotation.raters import (
    Annotator,
    Guideline,
    annotate,
)
from evalloop.core import rng_for

Labels = dict[str, dict[str, bool]]  # item -> rater -> pass


def assign(
    items: Sequence[str],
    pool: Sequence[Annotator],
    overlap: int = 3,
    seed: int = 0,
) -> dict[str, tuple[str, ...]]:
    """Each item to ``overlap`` raters, load spread evenly."""
    if not 1 <= overlap <= len(pool):
        raise ValueError(
            "overlap must be between 1 and pool size"
        )
    order = [r.id for r in pool]
    rng_for("assign", seed).shuffle(order)
    n = len(order)
    return {
        item: tuple(
            order[(i * overlap + k) % n] for k in range(overlap)
        )
        for i, item in enumerate(items)
    }


def collect(runs, assignment, pool, guide: Guideline) -> Labels:
    """Each assigned rater labels each run (item = task id)."""
    by_id = {r.id: r for r in pool}
    return {
        run.task_id: {
            rid: annotate(by_id[rid], run, guide)
            for rid in assignment[run.task_id]
        }
        for run in runs
    }


def fleiss_kappa(ratings: Sequence[Sequence]) -> float:
    """Agreement of many raters per item, chance-corrected.

    Every item must have the same number of ratings (>= 2).
    """
    n = len(ratings[0])
    if n < 2 or any(len(r) != n for r in ratings):
        raise ValueError("need the same number (>= 2) per item")
    per_item = [Counter(r) for r in ratings]
    p_item = [
        (sum(c * c for c in counts.values()) - n) / (n * (n - 1))
        for counts in per_item
    ]
    totals = sum(per_item, Counter())
    shares = [v / (n * len(ratings)) for v in totals.values()]
    p_bar = sum(p_item) / len(p_item)
    p_chance = sum(s * s for s in shares)
    if p_chance == 1:
        return 1.0
    return (p_bar - p_chance) / (1 - p_chance)


def contested(labels: Labels) -> list[str]:
    """Items whose raters did not all agree."""
    return [
        i for i, v in labels.items() if len(set(v.values())) > 1
    ]


def adjudicate(labels: Labels, expert_label) -> dict[str, bool]:
    """Unanimous labels stand; the expert settles the rest.

    ``expert_label(item)`` returns the expert's pass label.
    """
    disputed = set(contested(labels))
    return {
        item: expert_label(item)
        if item in disputed
        else next(iter(votes.values()))
        for item, votes in labels.items()
    }


def gold_scores(
    labels: Labels, gold: dict[str, bool]
) -> dict[str, tuple[int, int]]:
    """Per rater: (correct, answered) on gold questions."""
    scores: dict[str, list[int]] = {}
    for item, answer in gold.items():
        for rater, label in labels.get(item, {}).items():
            right, seen = scores.setdefault(rater, [0, 0])
            scores[rater] = [right + (label == answer), seen + 1]
    return {r: (c, n) for r, (c, n) in sorted(scores.items())}


def flagged(scores, min_accuracy: float = 0.8) -> list[str]:
    """Raters whose gold accuracy is below ``min_accuracy``."""
    return [
        r for r, (c, n) in scores.items() if c / n < min_accuracy
    ]


@dataclass(frozen=True)
class CostReport:
    labels: int
    hours: float
    cost: float
    per_final_label: float


def cost_report(
    labels: Labels, pool, expert: Annotator, escalated: int
) -> CostReport:
    """Hours and money for all labels, expert reviews included."""
    by_id = {r.id: r for r in pool}
    load = Counter(r for votes in labels.values() for r in votes)
    hours = {
        r: k / by_id[r].items_per_hour for r, k in load.items()
    }
    cost = sum(h * by_id[r].hourly_rate for r, h in hours.items())
    expert_hours = escalated / expert.items_per_hour
    cost += expert_hours * expert.hourly_rate
    total_hours = sum(hours.values()) + expert_hours
    per = cost / len(labels) if labels else 0.0
    return CostReport(
        sum(load.values()) + escalated,
        round(total_hours, 1),
        round(cost, 2),
        round(per, 2),
    )
