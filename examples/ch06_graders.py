"""Chapter 6: deterministic graders against ground truth.

The default suite grades 200 baseline runs from trajectory
evidence alone. Comparing its verdicts with the simulator's
hidden faults shows how far hard checks go, and where they
go blind: facts the reference does not list, and rudeness
phrased in words no check anticipated.
"""

from __future__ import annotations

import textwrap
from collections import Counter
from dataclasses import replace

from evalloop.core import (
    BASELINE,
    SimAgent,
    Step,
    Trajectory,
    make_tasks,
    run_suite,
)
from evalloop.graders import (
    DEFAULT_SUITE,
    failing_checks,
    grade,
    grade_all,
)


def misses(trajs, grades) -> Counter:
    """Faults present in runs the suite passed (ground truth)."""
    found: Counter = Counter()
    for traj, g in zip(trajs, grades, strict=True):
        if g.passed:
            found.update(traj.faults)
    return found


def reword(traj: Trajectory, old: str, new: str) -> Trajectory:
    """The same run with one phrase of its answer replaced."""
    answer = traj.answer.replace(old, new)
    steps = [*traj.steps[:-1], Step("answer", result=answer)]
    return replace(traj, steps=steps, answer=answer)


def report_agreement(runs, grades) -> None:
    pairs = list(zip(runs, grades, strict=True))
    n = len(pairs)
    passed = sum(g.passed for _, g in pairs) / n
    truth = sum(r.ok for r, _ in pairs) / n
    agree = sum(g.passed == r.ok for r, g in pairs) / n
    false_fails = sum(r.ok and not g.passed for r, g in pairs)
    print(f"Deterministic suite on {n} BASELINE runs")
    print(f"  suite pass {passed:.1%} | ground truth {truth:.1%}")
    print(f"  agreement {agree:.1%}, false fails {false_fails}")
    print("Runs failed by each check:")
    for name, count in failing_checks(grades).most_common():
        print(f"  {name:<17} {count:>3}")


def report_misses(runs, grades) -> None:
    faults = Counter(f for r in runs for f in r.faults)
    missed = misses(runs, grades)
    caught = sum(faults.values()) - sum(missed.values())
    print(f"Faults caught: {caught}/{sum(faults.values())}")
    for mode, count in missed.items():
        print(f"  missed {mode}: {count}/{faults[mode]}")
    pairs = zip(runs, grades, strict=True)
    slip = next(r for r, g in pairs if g.passed and r.faults)
    print(f"  e.g. {slip.task_id} passed with this answer:")
    for line in textwrap.wrap(slip.answer, 52):
        print(f"    {line}")
    status = slip.tool_calls[0].result["status"]
    print(f"  but lookup_order said the order is {status}.")


def main() -> None:
    tasks = make_tasks(200, seed=7)
    runs = run_suite(SimAgent(BASELINE), tasks)
    grades = grade_all(tasks, runs)
    report_agreement(runs, grades)
    report_misses(runs, grades)
    by_id = {t.id: t for t in tasks}
    calm = next(r for r in runs if r.ok and "sorry" in r.answer)
    rude = reword(calm, "I'm sorry.", "Relax, nobody cares.")
    result = grade(by_id[rude.task_id], rude, DEFAULT_SUITE)
    print(
        f"Reworded rude reply on {rude.task_id}: {result.passed=}"
    )
    print("Unlisted facts and tone need a judge (chapter 7).")


if __name__ == "__main__":
    main()
