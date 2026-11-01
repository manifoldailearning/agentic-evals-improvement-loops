"""Ablation and few-shot selection: which parts of a prompt pay?

Prompts accrete rules and examples that nobody dares remove.
Leave-one-out ablation measures what each rule contributes.
Few-shot selection compares random, diverse and per-task
similar examples; the simulator counts only examples that
use the same tools as the task, a stand-in for relevance.
"""

from __future__ import annotations

import re

from evalloop.core import AgentConfig, SimAgent, Task, rng_for
from evalloop.loop import evaluate, evidence

TOOL_ERRORS = {"wrong_tool", "bad_args"}  # what examples fix


def ablate(config: AgentConfig, tasks) -> dict[str, float]:
    """Success lost when each rule is removed on its own."""
    full = evaluate(config, tasks)[0]
    loss = {}
    for rule in sorted(config.rules):
        without = config.evolve(rules=config.rules - {rule})
        loss[rule] = full - evaluate(without, tasks)[0]
    return dict(sorted(loss.items(), key=lambda kv: -kv[1]))


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def similarity(a: Task, b: Task) -> float:
    """Token Jaccard similarity of two task inputs."""
    x, y = words(a.input), words(b.input)
    return len(x & y) / len(x | y) if x | y else 0.0


def similar_shots(task: Task, pool, k: int) -> list[Task]:
    """The k pool examples closest to this task."""
    ranked = sorted(pool, key=lambda p: -similarity(task, p))
    return ranked[:k]


def diverse_shots(pool, k: int) -> list[Task]:
    """Farthest-point picks: each new shot least like the rest."""
    chosen = [pool[0]]
    while len(chosen) < k:
        nxt = max(
            (p for p in pool if p not in chosen),
            key=lambda p: -max(similarity(p, c) for c in chosen),
        )
        chosen.append(nxt)
    return chosen


def random_shots(pool, k: int, seed: int = 0) -> list[Task]:
    return rng_for("shots", seed).sample(list(pool), k)


def relevant(task: Task, shots) -> int:
    """Shots that demonstrate this task's exact tool sequence."""
    want = task.expected["tools"]
    return sum(s.expected["tools"] == want for s in shots)


def shot_scores(config, tasks, pick) -> tuple[float, int]:
    """Success and tool-use errors when ``pick`` picks shots."""
    wins = errors = 0
    for task in tasks:
        n = relevant(task, pick(task))
        run = SimAgent(config.evolve(few_shots=n)).run(task)
        found = evidence(task, run)
        wins += not found
        errors += bool(TOOL_ERRORS & set(found))
    return wins / len(tasks), errors


def compare_shots(config, tasks, pool, k: int) -> dict:
    """Random vs diverse (both fixed) vs similar (per task)."""
    fixed_r = random_shots(pool, k)
    fixed_d = diverse_shots(pool, k)
    picks = {
        "none": lambda t: [],
        "random": lambda t: fixed_r,
        "diverse": lambda t: fixed_d,
        "similar": lambda t: similar_shots(t, pool, k),
    }
    return {
        name: shot_scores(config, tasks, pick)
        for name, pick in picks.items()
    }
