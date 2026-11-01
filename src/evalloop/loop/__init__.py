"""Chapter 16: the improvement loop.

Triage failed dev runs into evidence, map the evidence to a
candidate config change, evaluate it on dev, gate it on a
held-out test set, and record every decision in a log so
the path from baseline to today's config can be replayed.
"""

from evalloop.loop.loop import (
    ExperimentLog,
    ExperimentRecord,
    Loop,
    apply_change,
    dataset_version,
    evaluate,
    gate,
    split,
)
from evalloop.loop.triage import (
    FIXES,
    Hypothesis,
    evidence,
    passed,
    root_cause,
    triage,
)

__all__ = [
    "FIXES",
    "ExperimentLog",
    "ExperimentRecord",
    "Hypothesis",
    "Loop",
    "apply_change",
    "dataset_version",
    "evaluate",
    "evidence",
    "gate",
    "passed",
    "root_cause",
    "split",
    "triage",
]
