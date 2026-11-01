"""Open coding: read each trajectory and write what went wrong.

Error analysis starts with a person reading traces and
jotting free-text notes. This simulated annotator does the
same from evidence alone: tool calls and arguments, the
answer text, citations, and the records the answer claims
to report. It never sees the ground-truth fault labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from evalloop.core import KB, Task, Trajectory, World, rng_for

Evidence = tuple[str, dict]  # (observation, details)
NOTICE = {  # chance a reviewer spots each kind of evidence
    "loop": 1.0,
    "tone": 0.95,
    "tool": 0.9,
    "args": 0.9,
    "policy": 0.9,
    "vague": 0.9,
    "early": 0.85,
    "status": 0.8,
    "amount": 0.8,
    "cite": 0.75,
    "quote": 0.7,
}
PHRASES = {  # several ways a reviewer might put the same thing
    "tool": (
        "used {got} where {want} was needed",
        "picked the wrong tool: {got} instead of {want}",
        "went straight to {got}; should have started with {want}",
    ),
    "args": (
        "{tool} called without its arguments, tool errored",
        "forgot the order id in {tool}",
        "empty args on {tool} -> error",
    ),
    "status": (
        "says order {oid} is {said}; records show {true}",
        "order status contradicts lookup: {said} vs {true}",
    ),
    "quote": (
        "answer misquotes the {kb} article",
        "help-center fact contradicts {kb}",
    ),
    "amount": (
        "claims a refund of {said}; tool refunded {true}",
        "refund amount in reply contradicts tool ({said})",
    ),
    "policy": (
        "refunded {amount} on {oid}, which policy forbids",
        "issued a refund outside the refund policy",
        "refund of {amount} should have been escalated",
    ),
    "cite": (
        "no [kb:] citation for the article it used",
        "uses help-center text but cites nothing",
    ),
    "loop": (
        "repeated {tool} {n} times and never replied",
        "stuck calling {tool}; no answer",
        "ran out of steps without answering",
    ),
    "early": (
        "replied before calling {tool}",
        "stopped early, {tool} never called",
        "answered without finishing: no {tool}",
    ),
    "vague": (
        "guessed order {oid} instead of asking which order",
        "request was vague but it never asked for the order",
        "assumed an order; should have asked",
    ),
    "tone": (
        "told the customer to '{phrase}'",
        "dismissive tone with an upset customer",
        "rude opener: '{phrase}'",
    ),
}


@dataclass(frozen=True)
class Note:
    task_id: str
    text: str


def check_tools(task: Task, run: Trajectory) -> list[Evidence]:
    """Tool choice, arguments, looping and stopping early."""
    found: list[Evidence] = []
    got = [s.tool for s in run.tool_calls]
    want = task.expected["tools"]
    if got and want and got[0] != want[0]:
        found.append(("tool", {"got": got[0], "want": want[0]}))
    for step in run.tool_calls:
        if not step.args:
            found.append(("args", {"tool": step.tool}))
    if not run.answer:
        tool = got[-1] if got else "nothing"
        found.append(
            ("loop", {"tool": tool, "n": got.count(tool)})
        )
    elif len(got) < len(want) and want[: len(got)] == got:
        found.append(("early", {"tool": want[len(got)]}))
    return found


def check_status(answer: str) -> list[Evidence]:
    """Order statuses in the answer versus the order book."""
    found: list[Evidence] = []
    pattern = r"Order (A-\d+) is (\w+)\."
    for oid, said in re.findall(pattern, answer):
        true = World().lookup_order(oid).get("status")
        if said != true:
            found.append(("status", claim(oid, said, true)))
    return found


def check_quotes(run: Trajectory) -> list[Evidence]:
    """Articles cited or retrieved must be quoted faithfully."""
    if not run.answer:
        return []
    cited = re.findall(r"\[kb:([\w-]+)\]", run.answer)
    used = [s.result["article"] for s in searches(run)]
    return [
        ("quote", {"kb": kb})
        for kb in dict.fromkeys(cited + used)
        if KB.get(kb, "") not in run.answer
    ]


def check_amounts(run: Trajectory) -> list[Evidence]:
    """A refund amount in the answer must match the tool call."""
    issued = [s.args.get("amount") for s in refund_calls(run)]
    said = re.findall(r"refunded (\d+\.\d\d)", run.answer)
    if issued and said and float(said[0]) != issued[-1]:
        return [("amount", claim("", said[0], issued[-1]))]
    return []


def claim(oid: str, said: str, true: object) -> dict:
    return {"oid": oid, "said": said, "true": true}


def searches(run: Trajectory) -> list:
    return [
        s
        for s in run.tool_calls
        if s.tool == "search_kb" and "article" in s.result
    ]


def refund_calls(run: Trajectory) -> list:
    return [s for s in run.tool_calls if s.tool == "issue_refund"]


def check_conduct(task: Task, run: Trajectory) -> list[Evidence]:
    """Policy, citations, ambiguity handling and tone."""
    found: list[Evidence] = []
    for s in refund_calls(run):
        oid, amount = s.args.get("order_id"), s.args.get("amount")
        if not World().refund_allowed(oid, amount):
            found.append(
                ("policy", {"oid": oid, "amount": amount})
            )
            break
    if run.answer and searches(run) and "[kb:" not in run.answer:
        found.append(("cite", {}))
    if task.expected["clarify"] and run.tool_calls:
        oid = run.tool_calls[0].args.get("order_id", "?")
        found.append(("vague", {"oid": oid}))
    for phrase in task.expected["must_not"]:
        if phrase in run.answer.lower():
            found.append(("tone", {"phrase": phrase}))
    return found


def observe(task: Task, run: Trajectory) -> list[Evidence]:
    """Everything a careful reviewer could point to."""
    return (
        check_tools(task, run)
        + check_status(run.answer)
        + check_quotes(run)
        + check_amounts(run)
        + check_conduct(task, run)
    )


def noticed(annotator: str, task_id: str, code: str) -> bool:
    """Reviewers miss subtle evidence more often than loud."""
    draw = rng_for(annotator, task_id, code, "notice").random()
    return draw < NOTICE[code]


def write_note(
    annotator: str, task_id: str, code: str, details: dict
) -> str:
    """Phrase one observation the way this reviewer would."""
    rng = rng_for(annotator, task_id, code, "phrase")
    return rng.choice(PHRASES[code]).format(**details)


def open_code(
    task: Task, run: Trajectory, annotator: str = "ann-1"
) -> list[Note]:
    """Notes on one trajectory; empty if nothing was noticed."""
    return [
        Note(task.id, write_note(annotator, task.id, code, d))
        for code, d in observe(task, run)
        if noticed(annotator, task.id, code)
    ]


def review(
    tasks: list[Task], runs: list[Trajectory], annotator="ann-1"
) -> list[Note]:
    """Open-code every run; all notes, in run order."""
    by_id = {t.id: t for t in tasks}
    return [
        note
        for run in runs
        for note in open_code(by_id[run.task_id], run, annotator)
    ]
