"""Path metrics: did the agent take the right route?

An answer can be right for the wrong reasons, or wrong after
a perfect start. These metrics read only the trajectory and
the task's expectations: which tools were called, with what
arguments, in what order, and how often a call was repeated.
"""

from __future__ import annotations

import json
from collections import Counter

from evalloop.core import Step, Task, Trajectory

ARG_KEYS = {  # the arguments that decide a call is right
    "lookup_order": ("order_id",),
    "search_kb": ("query",),
    "issue_refund": ("order_id", "amount"),
    "escalate": ("order_id",),
}
PRECEDES = (  # (first, then): look before you act
    ("lookup_order", "issue_refund"),
    ("lookup_order", "escalate"),
    ("search_kb", "issue_refund"),
)


def called(traj: Trajectory) -> list[str]:
    """Tool names in call order."""
    return [s.tool for s in traj.tool_calls]


def tool_precision_recall(
    traj: Trajectory, task: Task
) -> tuple[float, float]:
    """Share of calls that were expected, and vice versa.

    Counts are multisets, so a repeated call lowers precision.
    """
    got = Counter(called(traj))
    want = Counter(task.expected["tools"])
    hits = sum((got & want).values())
    precision = hits / sum(got.values()) if got else 1.0
    recall = hits / sum(want.values()) if want else 1.0
    return precision, recall


def normalize(value: object) -> object:
    """Text ignores case and spacing; money rounds to cents."""
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, str):
        return " ".join(value.lower().split())
    return value


def args_match(step: Step, task: Task) -> bool:
    """Every argument that matters equals the expected one."""
    want = task.expected["args"]
    return all(
        normalize(step.args.get(k)) == normalize(want.get(k))
        for k in ARG_KEYS.get(step.tool, ())
    )


def arg_accuracy(traj: Trajectory, task: Task) -> float:
    """Share of calls to expected tools with correct args."""
    want = set(task.expected["tools"])
    checked = [s for s in traj.tool_calls if s.tool in want]
    if not checked:
        return 1.0
    good = sum(args_match(s, task) for s in checked)
    return good / len(checked)


def forbidden_calls(traj: Trajectory, task: Task) -> list[str]:
    """Calls to tools the task says must never be used."""
    banned = set(task.expected["must_not"])
    return [t for t in called(traj) if t in banned]


def order_violations(
    traj: Trajectory, task: Task
) -> list[tuple[str, str]]:
    """Actions taken before the check that should precede them.

    A pair applies only when the task expects its first tool.
    """
    tools = called(traj)
    want = task.expected["tools"]
    bad = []
    for first, then in PRECEDES:
        if first in want and then in tools:
            before = tools[: tools.index(then)]
            if first not in before:
                bad.append((first, then))
    return bad


def redundant_calls(traj: Trajectory) -> int:
    """Calls identical (tool and args) to an earlier call."""
    seen: set[tuple[str, str]] = set()
    repeats = 0
    for step in traj.tool_calls:
        key = (step.tool, json.dumps(step.args, sort_keys=True))
        repeats += key in seen
        seen.add(key)
    return repeats
