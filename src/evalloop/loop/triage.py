"""Triage: turn what failed runs show into changes worth trying.

A loop needs a verdict per run and a reason per failure. The
local checker reads only trajectory evidence (tools, results,
answer) and names what it sees; a production loop would plug
in its full grader suite here. Triage counts the evidence
and maps each kind to candidate config changes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from evalloop.core import Task, Trajectory

FIXES = {  # evidence -> candidate changes; most specific first
    "rude_reply": ("rule:escalate_angry_customers",),
    "forbidden_action": ("rule:check_refund_policy",),
    "guessed_order": ("rule:ask_when_ambiguous",),
    "bad_args": ("rule:verify_order_id",),
    "wrong_tool": ("few_shots:3",),
    "no_answer": ("rule:stop_after_answer",),
    "missing_citation": ("rule:cite_kb",),
    "ungrounded": ("temperature:0.2", "model:medium"),
    "skipped_step": ("model:medium",),
}


def claims(step) -> str:
    """The text a faithful answer must repeat from a tool."""
    r = step.result
    if step.tool == "lookup_order":
        return f"is {r['status']}"
    if step.tool == "search_kb":
        return r["text"]
    if step.tool == "issue_refund":
        return f"{r['amount']:.2f}"
    return r.get("ticket", "")


def tool_evidence(task: Task, run: Trajectory) -> list[str]:
    """Problems visible in which tools were called, and how."""
    exp = task.expected
    called = [s.tool for s in run.tool_calls]
    found = []
    if exp["clarify"] and called:
        found.append("guessed_order")
    if any(
        "error" in s.result and s.tool in exp["tools"]
        for s in run.tool_calls
    ):
        found.append("bad_args")
    allowed = set(exp["tools"]) | set(exp["must_not"])
    if not exp["clarify"] and set(called) - allowed:
        found.append("wrong_tool")
    if set(exp["must_not"]) & set(called):
        found.append("forbidden_action")
    elif set(exp["tools"]) - set(called):
        found.append("skipped_step")
    return found


def answer_evidence(task: Task, run: Trajectory) -> list[str]:
    """Problems visible in the final answer text."""
    exp, text = task.expected, run.answer
    found = []
    ok = [s for s in run.tool_calls if "error" not in s.result]
    if any(claims(s) not in text for s in ok):
        found.append("ungrounded")
    if "search_kb" in {s.tool for s in ok} and "[kb:" not in text:
        found.append("missing_citation")
    banned = [w for w in exp["must_not"] if " " in w]
    if any(w in text.lower() for w in banned):
        found.append("rude_reply")
    if any(w not in text for w in exp["answer_contains"]):
        found.append("missing_fact")
    return found


def evidence(task: Task, run: Trajectory) -> list[str]:
    """Problems visible in the run; empty means it passed."""
    if not any(s.kind == "answer" for s in run.steps):
        return ["no_answer"]
    return tool_evidence(task, run) + answer_evidence(task, run)


def passed(task: Task, run: Trajectory) -> bool:
    return not evidence(task, run)


@dataclass(frozen=True)
class Hypothesis:
    evidence: str  # what the failed runs showed
    count: int  # how many failed runs showed it
    change: str  # the config change expected to fix it


def root_cause(found: list[str]) -> str:
    """The most specific evidence explains the failure."""
    known = [kind for kind in FIXES if kind in found]
    return known[0] if known else found[0]


def triage(tasks, runs, tried=frozenset()) -> list[Hypothesis]:
    """Rank untried changes by how many failures they target."""
    by_id = {t.id: t for t in tasks}
    found = [evidence(by_id[r.task_id], r) for r in runs]
    seen = Counter(root_cause(f) for f in found if f)
    ranked = []
    for kind, n in seen.most_common():
        todo = [c for c in FIXES.get(kind, ()) if c not in tried]
        if todo:
            ranked.append(Hypothesis(kind, n, todo[0]))
    return ranked
