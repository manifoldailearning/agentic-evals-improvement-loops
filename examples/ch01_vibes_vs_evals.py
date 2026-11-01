"""Chapter 1: vibes versus evals.

Three hand-picked demo tasks make two agent configs look
equally good. Run both on 200 tasks and the difference is
plain, and the per-category breakdown shows where the
weaker config fails. Success here is ground truth.
"""

from __future__ import annotations

from collections import defaultdict

from evalloop.core import (
    BASELINE,
    CATEGORIES,
    IMPROVED,
    SimAgent,
    make_tasks,
    run_suite,
)

DEMO_IDS = ("T-0001", "T-0003", "T-0004")  # the happy path
CONFIGS = {"baseline": BASELINE, "improved": IMPROVED}


def vibe_check(config, tasks) -> list[str]:
    """What a demo shows: PASS or FAIL per hand-picked task."""
    runs = run_suite(SimAgent(config), tasks)
    return ["PASS" if r.ok else "FAIL" for r in runs]


def success_by_category(config, tasks) -> dict[str, float]:
    """Share of successful runs per category, plus 'overall'."""
    runs = run_suite(SimAgent(config), tasks)
    groups = defaultdict(list)
    for task, run in zip(tasks, runs, strict=True):
        groups[task.category].append(run.ok)
        groups["overall"].append(run.ok)
    return {k: sum(v) / len(v) for k, v in groups.items()}


def print_vibes(demo) -> None:
    print(f"Vibe check: {len(demo)} hand-picked demo tasks")
    checks = {n: vibe_check(c, demo) for n, c in CONFIGS.items()}
    for i, task in enumerate(demo):
        marks = [f"{name} {v[i]}" for name, v in checks.items()]
        print(f"  {task.id} {task.category:<12}", *marks)
    print("  Verdict: both look ready to ship.")


def print_eval(tasks) -> dict[str, float]:
    """Per-category table; returns overall success by name."""
    print(f"Eval: {len(tasks)} tasks, 40 per category")
    table = {
        n: success_by_category(c, tasks)
        for n, c in CONFIGS.items()
    }
    print(
        f"  {'category':<13}"
        + "".join(f"{n:>10}" for n in CONFIGS)
    )
    for cat in (*CATEGORIES, "overall"):
        cols = [f"{table[n][cat]:>10.1%}" for n in CONFIGS]
        print(f"  {cat:<13}" + "".join(cols))
    return {n: table[n]["overall"] for n in CONFIGS}


def main() -> None:
    tasks = make_tasks(200, seed=7)
    demo = [t for t in tasks if t.id in DEMO_IDS]
    print_vibes(demo)
    overall = print_eval(tasks)
    gap = overall["improved"] - overall["baseline"]
    print(
        f"  Gap: {gap * 100:.1f} points the demo could not see."
    )
    print("Chance a random 3-task demo is all PASS:")
    for name, rate in overall.items():
        print(f"  {name}: {rate ** len(demo):.0%}")


if __name__ == "__main__":
    main()
