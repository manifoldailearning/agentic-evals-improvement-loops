"""Chapter 4: error analysis, from raw traces to ranked fixes.

A simulated reviewer reads 200 baseline trajectories and
writes free-text notes. A codebook groups the notes into a
taxonomy, the Pareto table ranks it, saturation says when to
stop, and ground-truth faults check the taxonomy itself.
"""

from __future__ import annotations

from evalloop.analysis import (
    apply_fix,
    category_agreement,
    category_counts,
    code_notes,
    describe_fix,
    detection_agreement,
    pareto,
    review,
    saturation,
)
from evalloop.core import (
    BASELINE,
    SimAgent,
    make_tasks,
    run_suite,
)


def success(config, tasks) -> float:
    """Ground-truth success rate, used only to check fixes."""
    runs = run_suite(SimAgent(config), tasks)
    return sum(r.ok for r in runs) / len(runs)


def print_sample(notes, n: int = 3) -> None:
    print("Sample notes (open coding):")
    seen = set()
    for note in notes:
        if note.task_id not in seen and len(seen) < n:
            seen.add(note.task_id)
            print(f"  {note.task_id}  {note.text}")


def print_validation(coded, runs) -> None:
    print("Check against ground-truth faults:")
    found = detection_agreement(coded, runs)
    p, r = found.precision, found.recall
    print(f"  traces flagged: precision {p:.0%}, recall {r:.0%}")
    per = category_agreement(coded, runs)
    low_p = min(per, key=lambda c: per[c].precision)
    low_r = min(per, key=lambda c: per[c].recall)
    print(
        f"  lowest precision: {low_p} {per[low_p].precision:.0%}"
    )
    print(f"  lowest recall: {low_r} {per[low_r].recall:.0%}")


def main() -> None:
    tasks = make_tasks(200, seed=7)
    runs = run_suite(SimAgent(BASELINE), tasks)
    notes = review(tasks, runs)
    coded = code_notes(notes)
    print(
        f"Reviewed {len(runs)} baseline runs:"
        f" {len(notes)} notes on {len(coded)} traces"
    )
    print_sample(notes)
    print("Taxonomy (axial coding): traces, cumulative share")
    table = pareto(category_counts(coded))
    for category, n, share in table:
        fix = describe_fix(category)
        print(f"  {category:<17}{n:>3} {share:>4.0%}  {fix}")
    curve = " ".join(map(str, saturation(coded, batch=5)))
    print(f"New categories per 5 traces: {curve}")
    print_validation(coded, runs)
    config = BASELINE
    for category, _, _ in table[:3]:
        config = apply_fix(config, category)
    before = success(BASELINE, tasks)
    after = success(config, tasks)
    print(f"Top 3 fixes: success {before:.1%} -> {after:.1%}")


if __name__ == "__main__":
    main()
