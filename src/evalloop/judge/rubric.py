"""Rubrics as data, and the evidence a judge reads to apply them.

A model-graded eval is a prompt: criteria, each a question
with a weight. Keeping the rubric as data lets it be
versioned, diffed and rendered. ``find_issues`` is what the
simulated judge "reads": tool results, answer and policy,
never the simulator's hidden fault list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from evalloop.core import Task, Trajectory
from evalloop.graders import (
    answer_contains,
    args_match,
    citation_valid,
    refund_policy,
    right_tools,
)

STATUS_CLAIM = re.compile(r"Order (\S+) is (\w+)\.")
REFUND_CLAIM = re.compile(r"refunded (\d+\.\d+)")
DISMISSIVE = ("calm down", "relax", "nobody cares", "not my")
EMPATHY = ("sorry", "apologi", "understand")


@dataclass(frozen=True)
class Criterion:
    name: str
    question: str
    weight: float  # points lost on a 1-5 scale when violated


RUBRIC = (
    Criterion("finished", "Did the agent give an answer?", 4),
    Criterion("resolved", "Did it take the needed steps?", 3),
    Criterion("grounded", "Do facts match tool results?", 3),
    Criterion("policy", "Did it respect refund policy?", 4),
    Criterion("cited", "Does it cite articles it used?", 1.5),
    Criterion("tone", "Is it courteous to the customer?", 3),
)


def render_prompt(rubric, task: Task, traj: Trajectory) -> str:
    """The judge prompt: rubric questions, request, transcript."""
    lines = ["Score the support reply from 1 to 5.", "Criteria:"]
    lines += [f"- {c.name}: {c.question}" for c in rubric]
    lines.append(f"Request: {task.input}")
    for step in traj.tool_calls:
        lines.append(
            f"Tool {step.tool}{step.args} -> {step.result}"
        )
    lines.append(f"Reply: {traj.answer or '(none)'}")
    return "\n".join(lines)


def ungrounded(traj: Trajectory) -> list[str]:
    """Stated facts that no tool result in the run supports."""
    looked_up = {
        s.result.get("id"): s.result.get("status")
        for s in traj.tool_calls
        if s.tool == "lookup_order"
    }
    refunds = {
        round(float(s.args.get("amount", 0)), 2)
        for s in traj.tool_calls
        if s.tool == "issue_refund"
    }
    problems = [
        f"says {oid} is {said}, tools say {looked_up.get(oid)}"
        for oid, said in STATUS_CLAIM.findall(traj.answer)
        if looked_up.get(oid) != said
    ]
    problems += [
        f"says refunded {amount}, refund was other"
        for amount in REFUND_CLAIM.findall(traj.answer)
        if float(amount) not in refunds
    ]
    return problems + misquoted(traj)


def misquoted(traj: Trajectory) -> list[str]:
    """Articles the agent read but did not repeat faithfully."""
    return [
        f"misquotes {s.result['article']}"
        for s in traj.tool_calls
        if s.tool == "search_kb"
        and "text" in s.result
        and traj.answer
        and s.result["text"] not in traj.answer
    ]


def upset(text: str) -> bool:
    """Shouting (a word in capitals) or an exclamation mark."""
    words = re.findall(r"[A-Za-z]{3,}", text)
    shouting = any(w.isupper() for w in words)
    return shouting or "!" in text


def tone_issues(task: Task, traj: Trajectory) -> list[str]:
    reply = traj.answer.lower()
    problems = [
        f"dismissive '{p}'" for p in DISMISSIVE if p in reply
    ]
    caring = any(word in reply for word in EMPATHY)
    if traj.answer and upset(task.input) and not caring:
        problems.append("no empathy for an upset customer")
    return problems


def reasons(check, task: Task, traj: Trajectory) -> list[str]:
    return list(check(task, traj).reasons)


def find_issues(task: Task, traj: Trajectory) -> dict[str, list]:
    """Evidence per criterion; empty lists mean no problem."""
    resolved = []
    for check in (right_tools, args_match, answer_contains):
        resolved += reasons(check, task, traj)
    return {
        "finished": [] if traj.answer else ["no final answer"],
        "resolved": resolved,
        "grounded": ungrounded(traj),
        "policy": reasons(refund_policy, task, traj),
        "cited": reasons(citation_valid, task, traj),
        "tone": tone_issues(task, traj),
    }
