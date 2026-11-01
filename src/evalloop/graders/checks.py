"""Deterministic checks: assertions over one agent trajectory.

Many agent failures leave hard evidence: a wrong tool, a
missing argument, a refund over the limit, a citation to an
article never read. Each check reads only the trajectory,
the task's expectations and the world, and returns a Grade
with a score, a verdict and the reasons behind it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from evalloop.core import KB, Step, Task, Trajectory, World

CITATION = re.compile(r"\[kb:([a-z0-9-]+)\]")
NUMBER = (int, float)
TOOL_SCHEMAS = {  # tool -> required argument names and types
    "lookup_order": {"order_id": str},
    "search_kb": {"query": str},
    "issue_refund": {"order_id": str, "amount": NUMBER},
    "escalate": {"reason": str},
}


@dataclass(frozen=True)
class Grade:
    name: str
    score: float  # 0..1, share of the check satisfied
    passed: bool
    reasons: tuple[str, ...] = ()


Grader = Callable[[Task, Trajectory], Grade]


def verdict(name: str, problems: list[str], total: int) -> Grade:
    """A Grade from the problems found among ``total`` items."""
    score = 1 - len(problems) / total if total else 1.0
    return Grade(
        name, max(score, 0.0), not problems, tuple(problems)
    )


def first_call(traj: Trajectory, tool: str) -> Step | None:
    return next(
        (s for s in traj.tool_calls if s.tool == tool), None
    )


def normalize(value: object) -> object:
    """Compare as a person would: ignore case, spaces, cents."""
    if isinstance(value, NUMBER):
        return round(float(value), 2)
    return " ".join(str(value).lower().split())


def right_tools(task: Task, traj: Trajectory) -> Grade:
    """Every expected tool was called, and no other tool."""
    want = set(task.expected["tools"])
    used = {s.tool for s in traj.tool_calls}
    problems = [f"missing tool {t}" for t in sorted(want - used)]
    problems += [
        f"unexpected tool {t}" for t in sorted(used - want)
    ]
    return verdict("right_tools", problems, len(want | used))


def same_value(got: object, want: object, exact: bool) -> bool:
    if exact:
        return got == want
    return normalize(got) == normalize(want)


def args_match(
    task: Task, traj: Trajectory, exact: bool = False
) -> Grade:
    """Each expected tool's first call used the expected args."""
    want = task.expected["args"]
    problems, total = [], 0
    for tool in task.expected["tools"]:
        call = first_call(traj, tool)
        if call is None:
            continue  # right_tools reports the missing call
        for key in [k for k in TOOL_SCHEMAS[tool] if k in want]:
            total += 1
            got = call.args.get(key)
            if not same_value(got, want[key], exact):
                problems.append(f"{tool} {key}={got!r}")
    return verdict("args_match", problems, total)


def schema_errors(data: dict, schema: dict) -> list[str]:
    """Missing or mistyped fields; extra fields are allowed."""
    errors = [
        f"missing {key}" for key in schema if key not in data
    ]
    for key, kind in schema.items():
        if key in data and not isinstance(data[key], kind):
            errors.append(f"{key} has the wrong type")
    return errors


def valid_tool_args(task: Task, traj: Trajectory) -> Grade:
    """Every tool call's arguments fit that tool's schema."""
    problems = []
    for step in traj.tool_calls:
        schema = TOOL_SCHEMAS.get(step.tool, {})
        for error in schema_errors(step.args, schema):
            problems.append(f"{step.tool}: {error}")
    return verdict(
        "valid_tool_args", problems, len(traj.tool_calls)
    )


def answer_contains(task: Task, traj: Trajectory) -> Grade:
    """The answer states every expected fact (any case)."""
    text = traj.answer.lower()
    want = task.expected["answer_contains"]
    problems = [
        f"answer lacks {p!r}"
        for p in want
        if p is None or p.lower() not in text
    ]
    return verdict("answer_contains", problems, len(want))


def avoids_forbidden(task: Task, traj: Trajectory) -> Grade:
    """No forbidden tool was called, no forbidden phrase said."""
    used = {s.tool for s in traj.tool_calls}
    text = traj.answer.lower()
    banned = task.expected["must_not"]
    problems = []
    for item in banned:
        if item in TOOL_SCHEMAS and item in used:
            problems.append(f"called forbidden {item}")
        elif item not in TOOL_SCHEMAS and item.lower() in text:
            problems.append(f"said forbidden {item!r}")
    return verdict("avoids_forbidden", problems, len(banned))


def citation_valid(task: Task, traj: Trajectory) -> Grade:
    """Cites the KB when it relied on it; every cite is real."""
    cited = CITATION.findall(traj.answer)
    read = {
        s.result.get("article")
        for s in traj.tool_calls
        if s.tool == "search_kb"
    } - {None}
    problems = [
        f"cites unknown {c}" for c in cited if c not in KB
    ]
    problems += [
        f"cites unread {c}"
        for c in cited
        if c in KB and c not in read
    ]
    needed = task.expected["kb"] or read
    if needed and traj.answer and not cited:
        problems.append("no citation")
    return verdict("citation_valid", problems, len(cited) + 1)


def refund_policy(
    task: Task, traj: Trajectory, world: World | None = None
) -> Grade:
    """Every refund issued fits the policy: amount and age."""
    world = world or World()
    refunds = [
        s for s in traj.tool_calls if s.tool == "issue_refund"
    ]
    problems = [
        f"refund {s.args.get('amount')} breaks policy"
        for s in refunds
        if not world.refund_allowed(
            s.args.get("order_id", ""), s.args.get("amount", 0)
        )
    ]
    return verdict("refund_policy", problems, len(refunds))


def step_budget(limit: int = 5) -> Grader:
    """A check that the run answered within ``limit`` calls."""

    def within_budget(task: Task, traj: Trajectory) -> Grade:
        problems = []
        if not traj.answer:
            problems.append("no final answer")
        if len(traj.tool_calls) > limit:
            problems.append(
                f"{len(traj.tool_calls)} calls > {limit}"
            )
        return verdict("step_budget", problems, 2)

    return within_budget
