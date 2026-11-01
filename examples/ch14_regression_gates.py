"""Chapter 14: a regression gate that CI can trust.

The improved config is run on 5 seeds and stored as the
baseline. Three candidates face the gate: one passes; one
barely moves the pass rate but breaks critical refund tasks;
one runs hotter, which looks fine on pass@k and fails pass^k.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from statistics import fmean

from evalloop.core import (
    IMPROVED,
    AgentConfig,
    SimAgent,
    Task,
    Trajectory,
    make_tasks,
    run_suite,
)
from evalloop.gates import (
    Baseline,
    GatePolicy,
    Quarantine,
    Tolerance,
    collect,
    load_baseline,
    run_gate,
    save_baseline,
    seed_spread,
    summarize,
)

SEEDS = range(5)
TODAY = "2026-11-15"  # passed in: gates must replay exactly
QUARANTINE = (
    Quarantine("T-0006", "flaky tone check", "2026-12-01"),
    Quarantine("T-0009", "flaky lookup", "2026-11-01"),
)
TOLERANCES = (
    Tolerance("pass_rate", 0.03),
    Tolerance("pass^k", 0.05),
    Tolerance("cost", 0.0005, False, "${:.4f}"),
)


def passed(task: Task, run: Trajectory) -> bool:
    """Ground truth stands in for the graders' verdict."""
    return run.ok


def evaluate(config: AgentConfig, tasks) -> tuple[dict, dict]:
    """Outcomes per task and seed, plus mean cost per run."""
    outcomes = collect(SimAgent(config), tasks, SEEDS, passed)
    runs = run_suite(SimAgent(config), tasks, SEEDS)
    return outcomes, {"cost": fmean(r.cost for r in runs)}


def critical_tasks(tasks, outcomes) -> frozenset[str]:
    """Over-limit refunds the baseline gets right every time."""
    over = [
        t.id for t in tasks if t.expected["eligible"] is False
    ]
    keep = frozenset(t for t in over if all(outcomes[t]))
    print(
        f"Critical: {len(keep)} of {len(over)} over-limit refunds"
        f" ({len(over) - len(keep)} flaky in baseline)"
    )
    return keep


def build_baseline(tasks, path: Path):
    """Run the accepted config, store it, and read it back."""
    outcomes, extra = evaluate(IMPROVED, tasks)
    save_baseline(path, Baseline("improved", 5, outcomes, extra))
    baseline = load_baseline(path)
    m = baseline.metrics(outcomes)
    print(
        f"Baseline {m['pass_rate']:.1%} pass, pass@5"
        f" {m['pass@k']:.1%}, pass^5 {m['pass^k']:.1%},"
        f" {m['flaky_rate']:.0%} flaky"
    )
    low, high = seed_spread(outcomes)
    print(f"  a single seed scores {low:.1%} to {high:.1%}")
    return baseline, critical_tasks(tasks, outcomes)


def main() -> None:
    tasks = make_tasks(120, seed=7)
    with tempfile.TemporaryDirectory() as tmp:
        base, critical = build_baseline(
            tasks, Path(tmp) / "b.json"
        )
    policy = GatePolicy(TOLERANCES, critical, flaky_budget=0.2)
    rules = IMPROVED.rules
    candidates = {
        "v3 +stop_after_answer": IMPROVED.evolve(
            rules=rules | {"stop_after_answer"}
        ),
        "v4 -check_refund_policy": IMPROVED.evolve(
            rules=rules - {"check_refund_policy"}
        ),
        "v5 temperature 0.8": IMPROVED.evolve(temperature=0.8),
    }
    for i, (name, config) in enumerate(candidates.items()):
        outcomes, extra = evaluate(config, tasks)
        report = run_gate(
            name, outcomes, base, policy, QUARANTINE, TODAY, extra
        )
        print("\n".join(report.lines(only_failed=i == 2)))
    hot = summarize(outcomes, 5)["pass@k"]
    print(f"  ...yet v5 pass@5 is {hot:.1%}: retries hide it")


if __name__ == "__main__":
    main()
