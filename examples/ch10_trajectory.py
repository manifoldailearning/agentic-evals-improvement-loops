"""Chapter 10: evaluate the path an agent took, not just the end.

Both configs run the same 200 tasks. Trajectory metrics show
which steps improved, what each success costs, which
categories lag, and where failed runs first went wrong.
"""

from __future__ import annotations

from evalloop.core import (
    BASELINE,
    IMPROVED,
    SimAgent,
    make_tasks,
    run_suite,
)
from evalloop.trajectory import (
    blame_counts,
    by_category,
    score_runs,
    summarize,
)

ROWS = (  # (label, metric key, format)
    ("success", "success", "{:.1%}"),
    ("tool precision", "tool_precision", "{:.1%}"),
    ("tool recall", "tool_recall", "{:.1%}"),
    ("arg accuracy", "arg_accuracy", "{:.1%}"),
    ("checked before acting", "in_order", "{:.1%}"),
    ("runs with repeats", "has_repeats", "{:.1%}"),
    ("mean steps", "mean_steps", "{:.2f}"),
    ("mean tokens", "mean_tokens", "{:.0f}"),
    ("cost per success", "cost_per_success", "${:.5f}"),
)


def compare(base: dict, best: dict) -> None:
    print(f"{'metric':<22}{'baseline':>10}{'improved':>10}")
    for label, key, fmt in ROWS:
        a, b = fmt.format(base[key]), fmt.format(best[key])
        print(f"{label:<22}{a:>10}{b:>10}")


def main() -> None:
    tasks = make_tasks(200, seed=7)
    base = score_runs(run_suite(SimAgent(BASELINE), tasks), tasks)
    best = score_runs(run_suite(SimAgent(IMPROVED), tasks), tasks)
    print("Trajectory metrics, 200 tasks, seed 0")
    compare(summarize(base), summarize(best))
    print("Baseline by category: success, recall, args")
    for cat, m in by_category(base).items():
        print(
            f"  {cat:<13}{m['success']:>7.1%}"
            f"{m['tool_recall']:>8.1%}{m['arg_accuracy']:>8.1%}"
        )
    kinds = blame_counts(base).most_common()
    print("First bad step, baseline failures:")
    print("  " + ", ".join(f"{k} {n}" for k, n in kinds))
    for kind in ("forbidden", "answer"):
        s = next(
            s for s in base if s.blame and s.blame.kind == kind
        )
        where = f"{s.task_id} {s.category}, step {s.blame.step}"
        print(f"  e.g. {where}: {s.blame.detail}")


if __name__ == "__main__":
    main()
