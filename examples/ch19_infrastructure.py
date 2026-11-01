"""Chapter 19: run evals cheaply, in parallel, and reproducibly.

A pilot estimates spend before the full run; a cache keyed
by (task, config, grader, seed) makes reruns free; the
parallel runner returns results in input order whatever the
worker count; manifests and a JSONL store let two runs be
compared input by input and task by task.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from evalloop.core import IMPROVED, make_tasks
from evalloop.infra import (
    ResultCache,
    ResultStore,
    compare,
    estimate_cost,
    make_manifest,
    run_suite_parallel,
)

TASKS = make_tasks(200, seed=7)
SEEDS = (0, 1, 2)
BUDGET = 6.00


def plan(config) -> None:
    est = estimate_cost(config, TASKS, SEEDS, pilot=20)
    n, k = len(TASKS), len(SEEDS)
    print(f"Plan: {n} tasks x {k} seeds = {est.runs} runs")
    print(
        f"  pilot ${est.pilot_cost:.2f}: ${est.per_run:.4f}/run"
        f" -> ${est.total:.2f} with 20% margin"
    )
    verdict = "go" if est.total <= BUDGET else "stop"
    print(f"  budget ${BUDGET:.2f}: {verdict}")


def run(store, cache, config, label, workers=8):
    results, spent = run_suite_parallel(
        config, TASKS, SEEDS, cache, workers
    )
    manifest = make_manifest(config, TASKS, SEEDS)
    store.save(manifest, results)
    print(
        f"{label} {manifest.run_id}: {cache.hits} hits,"
        f" {cache.misses} misses, ${spent:.2f} spent"
    )
    cache.hits = cache.misses = 0
    return manifest, results


def main() -> None:
    plan(IMPROVED)
    with tempfile.TemporaryDirectory() as tmp:
        store, cache = ResultStore(Path(tmp)), ResultCache()
        a, first = run(store, cache, IMPROVED, "Run A")
        print(f"  {a.dataset} cfg {a.config} {a.grader} {a.code}")
        run(store, cache, IMPROVED, "Rerun", workers=2)
        solo, _ = run_suite_parallel(IMPROVED, TASKS, SEEDS)
        print(
            f"  4 workers, no cache, same as A: {solo == first}"
        )
        rule = IMPROVED.rules | {"escalate_angry_customers"}
        v2 = IMPROVED.evolve(version="v2", rules=rule)
        b, _ = run(store, cache, v2, "Run B")
        diff = compare(store.load(a.run_id), store.load(b.run_id))
        print(f"A -> B changed: {', '.join(diff.changed)}")
        print(
            f"  success {diff.before:.1%} -> {diff.after:.1%};"
            f" fixed {len(diff.fixed)}, broken {len(diff.broken)}"
        )
        print(f"Store holds {len(store.runs())} runs")


if __name__ == "__main__":
    main()
