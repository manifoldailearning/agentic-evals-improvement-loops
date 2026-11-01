"""Outcome checks and credit assignment for one trajectory.

A failed run is only actionable once you know where it went
wrong. These checks judge the final answer from evidence in
the trajectory (expected facts, citations, agreement with
tool results) and walk the steps to blame the first bad one.
"""

from __future__ import annotations

from dataclasses import dataclass

from evalloop.core import Step, Task, Trajectory
from evalloop.trajectory.path import (
    ARG_KEYS,
    arg_accuracy,
    args_match,
    forbidden_calls,
    tool_precision_recall,
)


@dataclass(frozen=True)
class Blame:
    step: int  # 1-based position in traj.steps
    kind: str  # tool | args | forbidden | extra | early | answer
    detail: str


def restated(step: Step) -> str:
    """The fact a faithful answer repeats from a tool result."""
    r = step.result
    if step.tool == "lookup_order":
        return r["status"]
    if step.tool == "search_kb":
        return r["text"]
    if step.tool == "issue_refund":
        return f"{r['amount']:.2f}"
    return r["ticket"]


def unsupported(traj: Trajectory) -> list[str]:
    """Tool facts that the final answer fails to repeat.

    The latest successful result per tool is what counts.
    """
    ok = [s for s in traj.tool_calls if "error" not in s.result]
    latest = {s.tool: s for s in ok}
    facts = [restated(s) for s in latest.values()]
    return [f for f in facts if f not in traj.answer]


def answer_problems(traj: Trajectory, task: Task) -> list[str]:
    """What is wrong with the final answer, if anything."""
    if not traj.answer:
        return ["no final answer"]
    exp, text = task.expected, traj.answer.lower()
    out = [
        f"missing {f!r}"
        for f in exp["answer_contains"]
        if f.lower() not in text
    ]
    phrases = [p for p in exp["must_not"] if p not in ARG_KEYS]
    out += [f"says {p!r}" for p in phrases if p.lower() in text]
    if exp["kb"] and f"[kb:{exp['kb']}]" not in traj.answer:
        out.append("no citation")
    out += [f"unsupported {f!r}" for f in unsupported(traj)]
    return out


def problems(traj: Trajectory, task: Task) -> list[str]:
    """Every reason this run fails its task; empty if none."""
    precision, recall = tool_precision_recall(traj, task)
    out = []
    if recall < 1:
        out.append("missed an expected tool")
    if precision < 1:
        out.append("made an unexpected call")
    if arg_accuracy(traj, task) < 1:
        out.append("wrong arguments")
    out += [f"called {t}" for t in forbidden_calls(traj, task)]
    return out + answer_problems(traj, task)


def passes(traj: Trajectory, task: Task) -> bool:
    """Pass/fail from trajectory evidence alone."""
    return not problems(traj, task)


def step_blame(step: Step, n: int, task: Task) -> Blame | None:
    """Judge the n-th tool call (1-based) against the plan."""
    want = task.expected["tools"]
    if step.tool in task.expected["must_not"]:
        return Blame(n, "forbidden", f"called {step.tool}")
    if n > len(want):
        return Blame(n, "extra", f"unneeded {step.tool}")
    if step.tool != want[n - 1]:
        why = f"{step.tool} instead of {want[n - 1]}"
        return Blame(n, "tool", why)
    if not args_match(step, task):
        return Blame(n, "args", f"bad args to {step.tool}")
    return None


def first_bad_step(traj: Trajectory, task: Task) -> Blame | None:
    """Credit assignment: the earliest step that went wrong."""
    calls = traj.tool_calls
    for n, step in enumerate(calls, 1):
        if blame := step_blame(step, n, task):
            return blame
    want, at = task.expected["tools"], len(calls) + 1
    if traj.answer and len(calls) < len(want):
        return Blame(
            at, "early", f"answered before {want[at - 1]}"
        )
    if issues := answer_problems(traj, task):
        return Blame(at, "answer", issues[0])
    return None
