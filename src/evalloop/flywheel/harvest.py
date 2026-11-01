"""Harvest: turn logged production runs into training examples.

Production traffic is the best source of training data, and
the messiest. Simulated logs here carry user-written noise:
greetings, repeats and contact details. Passing runs become
examples as-is; failed runs are corrected by a stronger
teacher config and kept only if the checker then passes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from evalloop.core import (
    AgentConfig,
    SimAgent,
    Task,
    Trajectory,
    make_tasks,
    rng_for,
)
from evalloop.loop import passed

OPENERS = ("", "Hi, ", "Hello team. ", "Quick question: ")
REPHRASE = {  # how users say the same thing differently
    "How long does": "How many days does",
    "What is your": "What's your",
    "Is there a": "Do you offer a",
    "How do I": "What's the way to",
    "How can I": "Is it possible to",
    "Where is my package?": "Any sign of my package?",
    "I want a refund": "I need my money back",
}
CONTACT = (
    " You can reach me at {name}@example.com.",
    " Call me on +1 415 555 {num:04d} if easier.",
)


@dataclass(frozen=True)
class Example:
    id: str
    category: str
    input: str
    tools: tuple[str, ...]
    answer: str
    source: str  # success | corrected
    passed: bool = True  # local checker verdict on the run


def user_noise(task: Task, i: int) -> Task:
    """Wrap a task in the way real users write."""
    rng = rng_for("logs", task.id)
    text = task.input
    if rng.random() < 0.5:
        for old, new in REPHRASE.items():
            text = text.replace(old, new)
    text = rng.choice(OPENERS) + text
    if rng.random() < 0.3:
        tmpl = rng.choice(CONTACT)
        text += tmpl.format(name=f"user{i}", num=i)
    return replace(task, id=f"L-{i:04d}", input=text)


def production_logs(n: int, seed: int = 21) -> list[Task]:
    """Logged requests; about 1 in 10 is a user's retry."""
    rng = rng_for("retries", seed)
    logs = []
    for i, task in enumerate(make_tasks(n, seed=seed)):
        logs.append(user_noise(task, i))
        if rng.random() < 0.1:
            logs.append(replace(logs[-1], id=f"L-{i:04d}r"))
    return logs


def as_example(task: Task, run: Trajectory, source: str):
    tools = tuple(s.tool for s in run.tool_calls)
    ok = passed(task, run)
    return Example(
        task.id, task.category, task.input, tools, run.answer,
        source, ok,
    )  # fmt: skip


def harvest(tasks, runs, teacher: AgentConfig) -> list[Example]:
    """Keep passing runs; replace failures with a correction."""
    fixer = SimAgent(teacher)
    out = []
    for task, run in zip(tasks, runs, strict=True):
        if passed(task, run):
            out.append(as_example(task, run, "success"))
        else:
            fixed = fixer.run(task)
            out.append(as_example(task, fixed, "corrected"))
    return out


def quality(ex: Example) -> float:
    """A 0..1 score: checker verdict, answer present, length."""
    checks = (
        ex.passed,
        bool(ex.answer),
        len(ex.answer) <= 300,
        bool(ex.tools) or "order number" in ex.answer,
    )
    return sum(checks) / len(checks)
