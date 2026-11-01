"""Repeated runs: measuring an agent that does not repeat itself.

A sampled agent can pass a task on one seed and fail it on
the next, so one run per task makes a noisy gate. Running k
seeds gives pass@k (solved at least once) and pass^k (solved
every time), shows which tasks are flaky, and lets a team
quarantine known-flaky tasks until a fixed expiry date.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from statistics import fmean

from evalloop.core import SimAgent, Task, Trajectory, run_suite

Outcomes = dict[str, list[bool]]  # task id -> pass per seed
Verdict = Callable[[Task, Trajectory], bool]


def collect(
    agent: SimAgent,
    tasks: Sequence[Task],
    seeds: Iterable[int],
    passed: Verdict,
) -> Outcomes:
    """Run every task on every seed; record pass or fail."""
    by_id = {t.id: t for t in tasks}
    outcomes: Outcomes = {t.id: [] for t in tasks}
    for run in run_suite(agent, tasks, tuple(seeds)):
        task = by_id[run.task_id]
        outcomes[run.task_id].append(passed(task, run))
    return outcomes


def pass_at_k(n: int, c: int, k: int) -> float:
    """Chance at least one of k runs passes, from c of n.

    The unbiased estimator 1 - C(n-c, k) / C(n, k): draw k
    of the n recorded runs without replacement.
    """
    if not 0 <= c <= n or not 0 < k <= n:
        raise ValueError("need 0 <= c <= n and 0 < k <= n")
    return 1 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Chance all k runs pass, from c passes in n: C(c,k)/C(n,k).

    pass^k is the reliability a user sees on repeat visits;
    it falls fast as k grows unless the task always passes.
    """
    if not 0 <= c <= n or not 0 < k <= n:
        raise ValueError("need 0 <= c <= n and 0 < k <= n")
    return math.comb(c, k) / math.comb(n, k)


def is_flaky(results: Sequence[bool]) -> bool:
    """Passed on some seeds and failed on others."""
    return 0 < sum(results) < len(results)


def summarize(outcomes: Outcomes, k: int) -> dict[str, float]:
    """Gate metrics over tasks, each task weighted equally."""
    rows = [(len(r), sum(r)) for r in outcomes.values()]
    return {
        "pass_rate": fmean(c / n for n, c in rows),
        "pass@k": fmean(pass_at_k(n, c, k) for n, c in rows),
        "pass^k": fmean(pass_hat_k(n, c, k) for n, c in rows),
        "flaky_rate": fmean(
            is_flaky(r) for r in outcomes.values()
        ),
    }


def seed_spread(outcomes: Outcomes) -> tuple[float, float]:
    """Lowest and highest pass rate of any single seed.

    A tolerance tighter than this spread fails good changes
    by chance; it is the gate's nondeterminism floor.
    """
    runs = len(next(iter(outcomes.values())))
    rates = [
        fmean(r[s] for r in outcomes.values())
        for s in range(runs)
    ]
    return min(rates), max(rates)


@dataclass(frozen=True)
class Quarantine:
    task_id: str
    reason: str
    expires: str  # ISO date; the entry lapses on this day

    def active(self, today: str) -> bool:
        """Dates are passed in, so gates replay identically."""
        end = date.fromisoformat(self.expires)
        return date.fromisoformat(today) < end


def apply_quarantine(
    outcomes: Outcomes,
    entries: Iterable[Quarantine],
    today: str,
) -> tuple[Outcomes, list[str], list[str]]:
    """Drop actively quarantined tasks from the gate.

    Returns the gated outcomes, the held task ids, and the
    ids whose quarantine expired (they count again).
    """
    held, expired = [], []
    for q in entries:
        (held if q.active(today) else expired).append(q.task_id)
    gated = {t: r for t, r in outcomes.items() if t not in held}
    return gated, sorted(held), sorted(expired)
