"""Chapter 19: eval infrastructure and cost.

A result cache keyed by (task, config, grader, seed), a
parallel runner with deterministic ordering, cost estimates
from a pilot before a full run, run manifests for
reproducibility, a JSONL result store, and run comparison.
"""

from evalloop.infra.runner import (
    GRADER,
    Estimate,
    Result,
    ResultCache,
    cache_key,
    estimate_cost,
    run_one,
    run_suite_parallel,
)
from evalloop.infra.store import (
    Comparison,
    ResultStore,
    RunManifest,
    code_version,
    compare,
    dataset_hash,
    make_manifest,
)

__all__ = [
    "GRADER",
    "Comparison",
    "Estimate",
    "Result",
    "ResultCache",
    "ResultStore",
    "RunManifest",
    "cache_key",
    "code_version",
    "compare",
    "dataset_hash",
    "estimate_cost",
    "make_manifest",
    "run_one",
    "run_suite_parallel",
]
