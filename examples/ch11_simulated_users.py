"""Chapter 11: simulated users and multi-turn evaluation.

Four personas talk to each config over the same 200 tasks.
Turn-level pass rate, single-turn success and conversation
goal completion tell different stories; frustration and
give-ups show who the agent loses, and one transcript shows
a clarifying question doing its job.
"""

from __future__ import annotations

import textwrap

from evalloop.core import (
    BASELINE,
    IMPROVED,
    SimAgent,
    make_tasks,
    run_suite,
)
from evalloop.simulate import (
    by_persona,
    conversation_metrics,
    goal_met,
    simulate_suite,
    turn_pass_rate,
)
from evalloop.trajectory import passes

ROWS = (  # (label, metric key, format)
    ("single-turn success", "single", "{:.1%}"),
    ("turn-level pass rate", "turns", "{:.1%}"),
    ("conversation goal met", "goal_met", "{:.1%}"),
    ("turns to success", "turns_to_success", "{:.2f}"),
    ("frustrated users", "frustrated", "{:.1%}"),
    ("users who gave up", "gave_up", "{:.1%}"),
    ("happy but goal missed", "silent_failures", "{}"),
)


def evaluate(config, tasks) -> tuple[dict, list]:
    """Single-turn and multi-turn metrics for one config."""
    agent = SimAgent(config)
    runs = run_suite(agent, tasks)
    by_id = {t.id: t for t in tasks}
    single = sum(passes(r, by_id[r.task_id]) for r in runs)
    convs = simulate_suite(agent, tasks)
    m = conversation_metrics(convs)
    m["single"] = single / len(tasks)
    m["turns"] = turn_pass_rate(convs)
    return m, convs


def show(conv) -> None:
    for turn in conv.turns:
        said = turn.run.answer or "(no reply)"
        print("  user:  " + textwrap.shorten(turn.task.input, 54))
        print("  agent: " + textwrap.shorten(said, 54))
    met = "goal met" if goal_met(conv) else "goal missed"
    n = len(conv.turns)
    print(f"  -> {conv.outcome} after {n} turns, {met}")


def compare(base: dict, best: dict) -> None:
    print(f"{'metric':<24}{'baseline':>10}{'improved':>10}")
    for label, key, fmt in ROWS:
        a, b = fmt.format(base[key]), fmt.format(best[key])
        print(f"{label:<24}{a:>10}{b:>10}")


def main() -> None:
    tasks = make_tasks(200, seed=7)
    base, b_convs = evaluate(BASELINE, tasks)
    best, i_convs = evaluate(IMPROVED, tasks)
    print("Simulated users: 200 tasks, 4 personas, seed 0")
    compare(base, best)
    print("Goal met by persona")
    worse, better = by_persona(b_convs), by_persona(i_convs)
    for name, m in worse.items():
        a, b = m["goal_met"], better[name]["goal_met"]
        print(f"  {name:<22}{a:>10.1%}{b:>10.1%}")
    calm = IMPROVED.evolve(
        rules=IMPROVED.rules | {"escalate_angry_customers"}
    )
    angry = by_persona(evaluate(calm, tasks)[1])["angry"]
    print(
        "Improved + escalate_angry_customers, angry users:"
        f" {angry['goal_met']:.1%}"
    )
    conv = next(
        c for c in b_convs if c.persona == "confused"
        and len(c.turns) == 3 and goal_met(c)
    )  # fmt: skip
    print(f"Baseline, confused user, goal {conv.goal.id}:")
    show(conv)


if __name__ == "__main__":
    main()
