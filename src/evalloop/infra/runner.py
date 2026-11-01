"""Running evals at scale: cache, parallel runner, cost estimate.

Eval suites get rerun constantly, and every rerun costs
money and time. Results are cached under the key (task,
config, grader, seed) so unchanged work is never paid for
twice; jobs run in a thread pool but come back in input
order; a pilot sample estimates spend before a full run.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from itertools import zip_longest
from pathlib import Path

from evalloop.core import AgentConfig, SimAgent, Task
from evalloop.loop import evidence

GRADER = "checker-v1"  # bump when the grading logic changes


@dataclass(frozen=True)
class Result:
    task_id: str
    seed: int
    passed: bool
    evidence: tuple[str, ...]
    cost: float
    latency_ms: int


def run_one(config: AgentConfig, task: Task, seed: int) -> Result:
    """Run and grade one (task, seed) pair."""
    run = SimAgent(config).run(task, seed)
    found = tuple(evidence(task, run))
    return Result(
        task.id, seed, not found, found, run.cost, run.latency_ms
    )


def cache_key(task_id, config_key, grader, seed) -> str:
    return f"{task_id}|{config_key}|{grader}|{seed}"


@dataclass
class ResultCache:
    entries: dict[str, Result] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get(self, key: str) -> Result | None:
        found = self.entries.get(key)
        if found is None:
            self.misses += 1
        else:
            self.hits += 1
        return found

    def put(self, key: str, result: Result) -> None:
        self.entries[key] = result

    def save(self, path: str | Path) -> None:
        rows = [
            {"key": k, **asdict(r)}
            for k, r in self.entries.items()
        ]
        Path(path).write_text("\n".join(map(json.dumps, rows)))

    @classmethod
    def load(cls, path: str | Path) -> ResultCache:
        cache = cls()
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            key = row.pop("key")
            row["evidence"] = tuple(row["evidence"])
            cache.put(key, Result(**row))
        return cache


def run_suite_parallel(
    config, tasks, seeds=(0,), cache=None, workers=4
) -> tuple[list[Result], float]:
    """Results in task-major input order, plus the new spend."""
    cache = cache if cache is not None else ResultCache()
    jobs = [(t, s) for t in tasks for s in seeds]
    keys = [
        cache_key(t.id, config.key(), GRADER, s) for t, s in jobs
    ]
    found = [cache.get(k) for k in keys]
    todo = [i for i, r in enumerate(found) if r is None]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fresh = pool.map(
            lambda i: run_one(config, *jobs[i]), todo
        )
        for i, result in zip(todo, fresh, strict=True):
            cache.put(keys[i], result)
            found[i] = result
    spent = sum(found[i].cost for i in todo)
    return found, spent


@dataclass(frozen=True)
class Estimate:
    runs: int
    per_run: float  # mean $ per run in the pilot
    total: float  # projected $ with the safety margin
    pilot_cost: float


def pilot_sample(tasks, n: int) -> list[Task]:
    """Up to n tasks, taken round-robin across categories."""
    groups: dict[str, list[Task]] = {}
    for t in tasks:
        groups.setdefault(t.category, []).append(t)
    rows = zip_longest(*groups.values())
    flat = [t for row in rows for t in row if t is not None]
    return flat[:n]


def estimate_cost(config, tasks, seeds, pilot=10, margin=0.2):
    """Project spend from a small pilot before the full run."""
    sample = pilot_sample(tasks, pilot)
    costs = [run_one(config, t, seeds[0]).cost for t in sample]
    per_run = sum(costs) / len(costs)
    runs = len(tasks) * len(seeds)
    total = per_run * runs * (1 + margin)
    return Estimate(runs, per_run, total, sum(costs))
