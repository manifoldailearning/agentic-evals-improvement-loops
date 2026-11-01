"""Trajectory evaluation: judge the path, not only the answer.

Tool precision and recall, argument accuracy, order checks,
repeated calls, efficiency per success, and credit assignment
to the first bad step, all read from trajectory evidence and
rolled up by category.
"""

from evalloop.trajectory.credit import (
    Blame,
    answer_problems,
    first_bad_step,
    passes,
    problems,
    unsupported,
)
from evalloop.trajectory.path import (
    arg_accuracy,
    args_match,
    forbidden_calls,
    order_violations,
    redundant_calls,
    tool_precision_recall,
)
from evalloop.trajectory.report import (
    TrajectoryScore,
    blame_counts,
    by_category,
    score,
    score_runs,
    summarize,
)

__all__ = [
    "Blame",
    "TrajectoryScore",
    "answer_problems",
    "arg_accuracy",
    "args_match",
    "blame_counts",
    "by_category",
    "first_bad_step",
    "forbidden_calls",
    "order_violations",
    "passes",
    "problems",
    "redundant_calls",
    "score",
    "score_runs",
    "summarize",
    "tool_precision_recall",
    "unsupported",
]
