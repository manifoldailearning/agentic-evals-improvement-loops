"""Validation: does the taxonomy agree with what really happened?

A taxonomy built from notes is a hypothesis. In this
simulated world the ground-truth faults are known, so we
can measure how often the annotator flagged the right
traces and how well each category matches its failure mode.
This is the only place in the analysis that reads faults.
"""

from __future__ import annotations

from dataclasses import dataclass

from evalloop.core import Trajectory

CATEGORY_MODE = {  # taxonomy category -> ground-truth mode
    "no answer": "loop",
    "ignored ambiguity": "ignored_ambiguity",
    "policy breach": "policy_violation",
    "wrong tool": "wrong_tool",
    "bad arguments": "missing_arg",
    "made-up facts": "hallucinated_fact",
    "missing citation": "no_citation",
    "stopped early": "premature_answer",
    "bad tone": "bad_tone",
}


@dataclass(frozen=True)
class Agreement:
    precision: float  # flagged traces that truly failed
    recall: float  # failed traces that were flagged
    tp: int
    fp: int
    fn: int


def agreement(flagged: set[str], truth: set[str]) -> Agreement:
    """Compare two sets of trace ids: flagged vs truly failed."""
    tp = len(flagged & truth)
    fp, fn = len(flagged - truth), len(truth - flagged)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return Agreement(precision, recall, tp, fp, fn)


def detection_agreement(
    coded: dict[str, set[str]], runs: list[Trajectory]
) -> Agreement:
    """Did the annotator flag exactly the traces with faults?"""
    truth = {r.task_id for r in runs if r.faults}
    return agreement(set(coded), truth)


def category_agreement(
    coded: dict[str, set[str]], runs: list[Trajectory]
) -> dict[str, Agreement]:
    """Per category: coded traces vs traces with its mode."""
    result = {}
    for category, mode in CATEGORY_MODE.items():
        flagged = {
            t for t, cats in coded.items() if category in cats
        }
        truth = {r.task_id for r in runs if mode in r.faults}
        if flagged or truth:
            result[category] = agreement(flagged, truth)
    return result
