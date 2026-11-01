"""Chapter 3: designing the eval set before trusting a score.

Starting from 200 generated tasks, we remove duplicates,
read the coverage matrix, compare random and stratified
samples, pick a golden set, split dev/test without leakage,
find the weakest slice and version the result by content.
"""

from __future__ import annotations

from dataclasses import replace

from evalloop.core import (
    BASELINE,
    SimAgent,
    make_tasks,
    run_suite,
)
from evalloop.datasets import (
    TAGS,
    DatasetVersion,
    all_cells,
    coverage,
    dedup,
    diff,
    golden_set,
    grouped_split,
    leaks,
    random_sample,
    random_split,
    slice_scores,
    slices,
    stratified_sample,
    thin_cells,
)

SPLITS = {"Random": random_split, "Grouped": grouped_split}
SHORT = ("plain", "ambig", "advers", "angry", "long", "multi")


def print_matrix(matrix) -> None:
    """Category rows, tag columns; '.' marks an empty cell."""
    print("  " + " " * 12 + "".join(f"{s:>7}" for s in SHORT))
    for cat in sorted({c for c, _ in matrix}):
        counts = [matrix[(cat, tag)] or "." for tag in TAGS]
        row = "".join(f"{n:>7}" for n in counts)
        print(f"  {cat:<12}{row}")


def success_by_task(tasks) -> dict[str, float]:
    """Ground-truth success of the baseline, per task id."""
    runs = run_suite(SimAgent(BASELINE), tasks)
    return {r.task_id: float(r.ok) for r in runs}


def mean(score: dict[str, float], tasks) -> float:
    return sum(score[t.id] for t in tasks) / len(tasks)


def main() -> None:
    raw = make_tasks(200, seed=7)
    tasks, dropped = dedup(raw)
    print(f"Raw: {len(raw)} tasks; after dedup: {len(tasks)}")
    print(f"  ({len(dropped)} were repeats of a kept task)")
    matrix = coverage(tasks)
    print("Coverage, category x tag (deduped):")
    print_matrix(matrix)
    thin = thin_cells(matrix)
    empty = thin_cells(matrix, 1, all_cells())
    print(f"Thin cells (<3 tasks): {len(thin)} of {len(matrix)}")
    print(f"Empty cells: {len(empty)} of {len(all_cells())}")
    rnd = coverage(random_sample(tasks, 30))
    strat = coverage(stratified_sample(tasks, 30))
    print(
        f"Sample of 30 covers: random {len(rnd)}/{len(matrix)}"
        f" cells, stratified {len(strat)}/{len(matrix)}"
    )
    gold = golden_set(tasks)
    print(f"Golden set: {len(gold)} tasks, hardest per cell")
    for name, split in SPLITS.items():
        dev, test = split(tasks)
        n = len(leaks(dev, test))
        print(f"{name} split: {n} of {len(test)} test tasks leak")
    ok = success_by_task(raw)
    named = slices(tasks)
    scores = slice_scores(named, ok)
    worst = min(scores, key=scores.get)
    print(
        f"Baseline success: raw {mean(ok, raw):.1%},"
        f" deduped {mean(ok, tasks):.1%}"
    )
    size = len(named[worst])
    print(f"Worst slice: {worst} {scores[worst]:.0%} (n={size})")
    v1 = DatasetVersion("support-raw", tuple(raw))
    fixed = replace(tasks[0], difficulty=0.6)  # re-rated
    v2 = DatasetVersion("support-v2", (fixed, *tasks[1:]))
    print(f"Versions: {v1.content_hash} -> {v2.content_hash}")
    print(f"  diff: {diff(v1, v2).summary()}")


if __name__ == "__main__":
    main()
