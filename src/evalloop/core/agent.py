"""The simulated agent: a config goes in, a trajectory comes out.

Improvement loops are only real if behavior responds to the
configuration. ``SimAgent`` samples failure modes from
``failure_probs``, plays out tool calls against the world,
and writes an answer from what the tools returned. The same
task, config and seed always give the same trajectory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from evalloop.core.failures import MODES, sample_faults
from evalloop.core.types import (
    RULES,
    AgentConfig,
    Step,
    Task,
    Trajectory,
    rng_for,
)
from evalloop.core.world import FACTS, KB, ORDERS, STATUSES, World

BASELINE = AgentConfig("baseline", "small", temperature=0.5)
IMPROVED = AgentConfig(  # first four rules, medium model
    "improved", "medium", frozenset(RULES[:4]), 3, 6, 0.2
)
MODEL_COST = {  # $ per 1k tokens in, out; ms per turn, token
    "small": (0.0005, 0.0015, 350, 8),
    "small-ft": (0.0008, 0.0024, 350, 8),
    "medium": (0.003, 0.015, 700, 15),
    "large": (0.015, 0.075, 1400, 30),
}
CLARIFY = "Could you share your order number so I can check?"
WRONG_TOOL = {
    "lookup_order": "search_kb",
    "search_kb": "escalate",
}
Call = tuple[str, dict]


def args_for(tool: str, task: Task) -> dict:
    a = task.expected["args"]
    if tool == "lookup_order":
        return {"order_id": a["order_id"]}
    if tool == "search_kb":
        return {"query": a["query"]}
    if tool == "issue_refund":
        return {"order_id": a["order_id"], "amount": a["amount"]}
    why = "refund outside policy"
    if task.category != "refund":
        why = "customer asked for a manager"
    return {"reason": why, "order_id": a.get("order_id", "")}


def swap_tool(task: Task, call: Call) -> Call:
    tool = WRONG_TOOL[call[0]]
    if tool == "search_kb":
        return tool, {"query": task.input[-60:]}
    return tool, {"reason": "could not answer"}


def resolve(faults: list[str]) -> set[str]:
    """Drop faults that another fault hides from view."""
    f = set(faults)
    if "loop" in f:  # no answer, so answer faults never show
        f -= {"hallucinated_fact", "no_citation", "bad_tone"}
        f.discard("premature_answer")
    if f & {"premature_answer", "bad_tone"}:
        f.discard("policy_violation")  # final action skipped
    if "wrong_tool" in f:
        f.discard("missing_arg")  # the bad call replaced it
    return f


def plan_calls(task: Task, faults: set[str]) -> list[Call]:
    """The tool calls the agent makes, faults included."""
    if task.expected["clarify"]:
        if "ignored_ambiguity" not in faults:
            return []
        guess = rng_for(task.id, "guess").choice(list(ORDERS))
        return [("lookup_order", {"order_id": guess})]
    tools = task.expected["tools"]
    calls = [(t, args_for(t, task)) for t in tools]
    if "policy_violation" in faults:
        calls[-1] = (
            "issue_refund",
            args_for("issue_refund", task),
        )
    if faults & {"premature_answer", "bad_tone"}:
        calls = calls[:-1]
    if "missing_arg" in faults:
        calls[0] = (calls[0][0], {})
    if "wrong_tool" in faults:
        calls[0] = swap_tool(task, calls[0])
    return calls


def loop_calls(calls: list[Call], max_steps: int) -> list[Call]:
    """Repeat the last call until the step limit cuts it off."""
    extra = [calls[-1]] * max(0, max_steps - len(calls))
    return (calls + extra)[:max_steps]


def corrupt(task: Task, facts: dict[str, dict]) -> dict:
    """Replace one stated fact with a plausible false one."""
    facts = {k: dict(v) for k, v in facts.items()}
    kb = facts.get("search_kb", {}).get(
        "article", task.expected["kb"]
    )
    if "issue_refund" in facts:
        facts["issue_refund"]["amount"] *= 1.5
    elif kb:
        true, false = FACTS[kb]
        text = KB[kb].replace(true, false)
        facts["search_kb"] = {"article": kb, "text": text}
    else:
        oid = task.expected["args"]["order_id"]
        o = facts.get("lookup_order", vars(ORDERS[oid]))
        i = (STATUSES.index(o["status"]) + 2) % len(STATUSES)
        facts["lookup_order"] = {**o, "status": STATUSES[i]}
    return facts


def recall(task: Task, steps: list[Step], faults: set) -> dict:
    """What the agent believes: tool results, maybe corrupted."""
    facts = {
        s.tool: s.result
        for s in steps
        if s.kind == "tool" and "error" not in s.result
    }
    if "hallucinated_fact" in faults:
        facts = corrupt(task, facts)
    return facts


def compose_answer(task: Task, facts: dict, faults: set) -> str:
    """Write the reply from the facts the agent holds."""
    if task.expected["clarify"] and not facts:
        return CLARIFY
    lines = []
    if "angry" in task.tags:
        rude = "bad_tone" in faults
        lines.append(
            "Please calm down." if rude else "I'm sorry."
        )
    if o := facts.get("lookup_order"):
        lines.append(f"Order {o['id']} is {o['status']}.")
    if a := facts.get("search_kb"):
        lines.append(a["text"])
    if r := facts.get("issue_refund"):
        lines.append(f"I have refunded {r['amount']:.2f}.")
    if e := facts.get("escalate"):
        lines.append(f"I have escalated this ({e['ticket']}).")
    if not facts:
        lines.append("I could not find the details right now.")
    if a and "no_citation" not in faults:
        lines.append(f"[kb:{a['article']}]")
    return " ".join(lines)


def tokens(text: str) -> int:
    return max(1, len(text) // 4)


def usage(c: AgentConfig, task: Task, steps: list[Step]) -> tuple:
    """Tokens in and out: each turn re-reads the context."""
    ctx = 300 + 45 * len(c.rules) + 110 * c.few_shots
    ctx += tokens(task.input)
    t_in = t_out = 0
    for s in steps:
        out = 25 if s.kind == "tool" else tokens(s.result)
        t_in, t_out = t_in + ctx, t_out + out
        ctx += out + tokens(json.dumps(s.result))
    return t_in, t_out


def measure(
    c: AgentConfig, task: Task, steps, seed: int
) -> tuple:
    """Tokens in, tokens out, latency (ms) and cost ($)."""
    t_in, t_out = usage(c, task, steps)
    p_in, p_out, turn_ms, token_ms = MODEL_COST[c.model]
    ms = len(steps) * (turn_ms + 60) + t_out * token_ms
    ms *= rng_for(task.id, c.key(), seed, "ms").uniform(
        0.85, 1.35
    )
    cost = (t_in * p_in + t_out * p_out) / 1000
    return t_in, t_out, int(ms), round(cost, 6)


@dataclass
class SimAgent:
    config: AgentConfig
    world: World = field(default_factory=World)

    def run(self, task: Task, seed: int = 0) -> Trajectory:
        """Sample faults, call tools, then write the answer."""
        c = self.config
        faults = resolve(sample_faults(task, c, seed))
        calls = plan_calls(task, faults)
        if "loop" in faults or len(calls) > c.max_steps:
            faults.add("loop")
            calls = loop_calls(calls, c.max_steps)
        steps = [
            Step("tool", t, a, self.world.call(t, a))
            for t, a in calls
        ]
        facts = recall(task, steps, faults)
        if "search_kb" not in facts:
            faults.discard("no_citation")
        answer = ""
        if "loop" not in faults:
            answer = compose_answer(task, facts, faults)
            steps.append(Step("answer", result=answer))
        head = (task.id, c.version, c.key(), seed, steps, answer)
        found = [m for m in MODES if m in faults]
        return Trajectory(
            *head, *measure(c, task, steps, seed), found
        )


def run_suite(agent: SimAgent, tasks, seeds=(0,)) -> list:
    """Run every task once per seed, task-major order."""
    return [agent.run(t, s) for t in tasks for s in seeds]
