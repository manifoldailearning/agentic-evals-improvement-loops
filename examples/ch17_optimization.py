"""Chapter 17: what each prompt part buys, and at what cost.

Leave-one-out ablation prices every rule; few-shot picks
are compared by the tool errors they prevent; greedy search
spends a budget of dev evals and is checked for optimism on
test; the Pareto frontier shows which model tiers pay off.
"""

from __future__ import annotations

from evalloop.core import (
    BASELINE,
    IMPROVED,
    MODELS,
    RULES,
    make_tasks,
)
from evalloop.loop import evaluate, split
from evalloop.optimize import (
    ablate,
    compare_shots,
    frontier,
    greedy_search,
    optimism,
)

TASKS = make_tasks(200, seed=7)
ALL_RULES = BASELINE.evolve(rules=frozenset(RULES))
PROMPT = dict(rules=IMPROVED.rules, few_shots=3, temperature=0.2)


def show_ablation() -> None:
    loss = ablate(ALL_RULES, TASKS)
    print("Ablation (small model, all 6 rules): pts lost if cut")
    items = [f"{r:<24} {100 * d:4.1f}" for r, d in loss.items()]
    for i in range(0, len(items), 2):
        print("  " + "  ".join(items[i : i + 2]))


def show_shots() -> None:
    pool = make_tasks(60, seed=99)
    scores = compare_shots(ALL_RULES, TASKS, pool, k=6)
    print("Six few-shot examples: tool errors and success")
    cells = [
        f"{k:<8}{e:>2} errors {s:6.1%}"
        for k, (s, e) in scores.items()
    ]
    for i in range(0, len(cells), 2):
        print("  " + "   ".join(cells[i : i + 2]))


def show_search() -> None:
    print("Greedy search from baseline, budget 60 dev evals")
    for share in (0.2, 0.4):
        dev, test = split(TASKS, share)
        found = greedy_search(BASELINE, dev, budget=60)
        gap = optimism(BASELINE, found.config, dev, test)
        flag = "overfit warning" if gap > 0.03 else "ok"
        held = evaluate(found.config, test)[0]
        print(
            f"  dev {len(dev)}: {len(found.path)} moves in"
            f" {found.evals} evals, dev {found.score:.1%},"
            f" test {held:.1%}"
        )
        print(f"    optimism {100 * gap:+.1f} pts: {flag}")


def show_frontier() -> None:
    configs = {}
    for model in MODELS:
        bare = BASELINE.evolve(model=model)
        configs[f"{model} bare"] = bare
        configs[f"{model} +prompt"] = bare.evolve(**PROMPT)
    points, front = frontier(configs, TASKS)
    print("Cost vs quality ($ per 1k tasks; * = on frontier)")
    for p in points:
        mark = "*" if p in front else " "
        money = f"${1000 * p.cost:.2f}"
        print(
            f"  {mark} {p.label:<16}{money:>7}  {p.success:.1%}"
        )
    for p in points:
        if p not in front:
            better = max(
                (q for q in front if q.cost < p.cost),
                key=lambda q: q.success,
            )
            print(f"{p.label} is dominated by {better.label}")


def main() -> None:
    show_ablation()
    show_shots()
    show_search()
    show_frontier()


if __name__ == "__main__":
    main()
