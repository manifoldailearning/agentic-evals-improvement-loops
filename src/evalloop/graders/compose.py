"""Composing checks into a grading suite for whole trajectories.

One check answers one narrow question; a verdict on a run
needs several. ``all_of`` demands every check, ``any_of``
accepts one, and ``weighted`` turns partial credit into a
score with a threshold. The default suite is what the rest
of the book means by "the deterministic graders".
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import replace

from evalloop.core import Task, Trajectory
from evalloop.graders.checks import (
    Grade,
    Grader,
    answer_contains,
    args_match,
    avoids_forbidden,
    citation_valid,
    refund_policy,
    right_tools,
    step_budget,
    valid_tool_args,
)


def merge(name: str, grades: list[Grade], passed: bool) -> Grade:
    """One Grade from many: mean score, reasons kept by check."""
    score = sum(g.score for g in grades) / len(grades)
    reasons = tuple(
        f"{g.name}: {r}" for g in grades for r in g.reasons
    )
    return Grade(name, round(score, 4), passed, reasons)


def all_of(*graders: Grader, name: str = "all_of") -> Grader:
    """Passes only if every grader passes."""

    def combined(task: Task, traj: Trajectory) -> Grade:
        grades = [g(task, traj) for g in graders]
        return merge(name, grades, all(g.passed for g in grades))

    return combined


def any_of(*graders: Grader, name: str = "any_of") -> Grader:
    """Passes if at least one grader passes."""

    def combined(task: Task, traj: Trajectory) -> Grade:
        grades = [g(task, traj) for g in graders]
        return merge(name, grades, any(g.passed for g in grades))

    return combined


def weighted(
    pairs: Sequence[tuple[Grader, float]],
    threshold: float = 0.8,
    name: str = "weighted",
) -> Grader:
    """Weighted mean of scores; passes at ``threshold``."""
    weights = [w for _, w in pairs]

    def combined(task: Task, traj: Trajectory) -> Grade:
        grades = [g(task, traj) for g, _ in pairs]
        parts = zip(grades, weights, strict=True)
        score = sum(g.score * w for g, w in parts) / sum(weights)
        merged = merge(name, grades, score >= threshold)
        return replace(merged, score=round(score, 4))

    return combined


DEFAULT_CHECKS = (
    right_tools,
    args_match,
    valid_tool_args,
    answer_contains,
    avoids_forbidden,
    citation_valid,
    refund_policy,
    step_budget(5),
)
DEFAULT_SUITE = all_of(*DEFAULT_CHECKS, name="suite")


def grade(
    task: Task, traj: Trajectory, grader: Grader = DEFAULT_SUITE
) -> Grade:
    """Grade one trajectory against its task."""
    if task.id != traj.task_id:
        raise ValueError(
            f"{traj.task_id} is not a run of {task.id}"
        )
    return grader(task, traj)


def grade_all(
    tasks: Iterable[Task],
    trajs: Iterable[Trajectory],
    grader: Grader = DEFAULT_SUITE,
) -> list[Grade]:
    """Grade every trajectory, matching runs to tasks by id."""
    by_id = {t.id: t for t in tasks}
    return [grade(by_id[r.task_id], r, grader) for r in trajs]


def failing_checks(grades: Iterable[Grade]) -> Counter:
    """How many grades each check failed (once per grade)."""
    counts: Counter = Counter()
    for g in grades:
        counts.update({r.split(":")[0] for r in g.reasons})
    return counts
