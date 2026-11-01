"""Trajectory scores per run, rolled up by category.

One pass rate hides how an agent succeeds and how it fails.
Scoring every run on tool choice, arguments, order, repeats,
cost and the first bad step, then aggregating by category,
shows where to look and what each success really costs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean

from evalloop.core import Task, Trajectory
from evalloop.trajectory.credit import (
    Blame,
    first_bad_step,
    passes,
)
from evalloop.trajectory.path import (
    arg_accuracy,
    order_violations,
    redundant_calls,
    tool_precision_recall,
)


@dataclass(frozen=True)
class TrajectoryScore:
    task_id: str
    category: str
    precision: float
    recall: float
    arg_accuracy: float
    in_order: bool
    redundant: int
    steps: int
    tokens: int
    cost: float
    passed: bool
    blame: Blame | None


def score(traj: Trajectory, task: Task) -> TrajectoryScore:
    """Every trajectory metric for one run of one task."""
    precision, recall = tool_precision_recall(traj, task)
    return TrajectoryScore(
        task_id=task.id,
        category=task.category,
        precision=precision,
        recall=recall,
        arg_accuracy=arg_accuracy(traj, task),
        in_order=not order_violations(traj, task),
        redundant=redundant_calls(traj),
        steps=len(traj.steps),
        tokens=traj.tokens_in + traj.tokens_out,
        cost=traj.cost,
        passed=passes(traj, task),
        blame=first_bad_step(traj, task),
    )


def score_runs(runs, tasks) -> list[TrajectoryScore]:
    """Score each run against the task it was given."""
    by_id = {t.id: t for t in tasks}
    return [score(r, by_id[r.task_id]) for r in runs]


def summarize(scores: list[TrajectoryScore]) -> dict:
    """Suite-level trajectory metrics, success included."""
    wins = sum(s.passed for s in scores)
    spent = sum(s.cost for s in scores)
    return {
        "success": wins / len(scores),
        "tool_precision": mean(s.precision for s in scores),
        "tool_recall": mean(s.recall for s in scores),
        "arg_accuracy": mean(s.arg_accuracy for s in scores),
        "in_order": mean(s.in_order for s in scores),
        "has_repeats": mean(s.redundant > 0 for s in scores),
        "mean_steps": mean(s.steps for s in scores),
        "mean_tokens": mean(s.tokens for s in scores),
        "cost_per_success": spent / wins if wins else 0.0,
    }


def by_category(scores: list[TrajectoryScore]) -> dict:
    """``summarize`` per task category, in first-seen order."""
    groups: dict[str, list] = defaultdict(list)
    for s in scores:
        groups[s.category].append(s)
    return {c: summarize(g) for c, g in groups.items()}


def blame_counts(scores: list[TrajectoryScore]) -> Counter:
    """How often each kind of step was the first to go wrong."""
    return Counter(s.blame.kind for s in scores if s.blame)
