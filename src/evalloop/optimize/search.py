"""Config search under a budget, an overfitting guard, and cost.

Searching many prompt variants against one dev set finds
the dev set's quirks as well as real gains. Greedy search
here spends a fixed budget of dev evals, the guard compares
dev gain with test gain, and the Pareto frontier shows which
model tiers buy quality at a price worth paying.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evalloop.core import RULES, AgentConfig
from evalloop.loop import apply_change, evaluate


def moves(c: AgentConfig) -> list[str]:
    """Single prompt or context edits reachable from ``c``."""
    out = [f"rule:{r}" for r in RULES if r not in c.rules]
    if c.few_shots < 6:
        out.append(f"few_shots:{c.few_shots + 2}")
    out += [
        f"temperature:{t}"
        for t in (0.0, 0.2, 0.7)
        if t != c.temperature
    ]
    return out


@dataclass
class SearchResult:
    config: AgentConfig
    score: float  # dev success of the final config
    evals: int  # dev evaluations spent
    path: list[tuple[str, float]] = field(default_factory=list)


def greedy_search(
    start: AgentConfig, dev, budget: int, min_gain=0.005
) -> SearchResult:
    """Take the best move each round until stuck or broke."""
    best = evaluate(start, dev)[0]
    result = SearchResult(start, best, 1)
    while result.evals < budget:
        tried = []
        for m in moves(result.config)[: budget - result.evals]:
            cand = apply_change(result.config, m)
            tried.append((evaluate(cand, dev)[0], m, cand))
            result.evals += 1
        score, move, cand = max(tried, key=lambda x: x[0])
        if score - result.score < min_gain:
            break
        result.config, result.score = cand, score
        result.path.append((move, score))
    return result


def optimism(start, end, dev, test) -> float:
    """Dev gain minus test gain: how much search fooled itself."""
    dev_gain = evaluate(end, dev)[0] - evaluate(start, dev)[0]
    test_gain = evaluate(end, test)[0] - evaluate(start, test)[0]
    return dev_gain - test_gain


@dataclass(frozen=True)
class Point:
    label: str
    cost: float  # mean $ per task
    success: float


def measure(label: str, config: AgentConfig, tasks) -> Point:
    """Success and mean cost per task from the trajectories."""
    rate, runs = evaluate(config, tasks)
    cost = sum(r.cost for r in runs) / len(runs)
    return Point(label, cost, rate)


def pareto(points) -> list[Point]:
    """Points no other point beats on both cost and success."""
    front = []
    for p in sorted(points, key=lambda p: (p.cost, -p.success)):
        if not front or p.success > front[-1].success:
            front.append(p)
    return front


def frontier(configs: dict[str, AgentConfig], tasks) -> tuple:
    """All measured points and the cost-quality frontier."""
    points = [measure(k, c, tasks) for k, c in configs.items()]
    return points, pareto(points)
