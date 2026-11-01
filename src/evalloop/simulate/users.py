"""Simulated users: a persona, a hidden goal, and a reaction.

Multi-turn behavior needs someone on the other side. Each
simulated user has a persona (patient, terse, angry,
confused) and a concrete goal it knows in full, including
what a correct reply says. Given the agent's reply it either
leaves satisfied, or answers back, or runs out of patience.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from evalloop.core import Task, rng_for

UNUSUAL = frozenset({"ambiguous", "adversarial", "long_input"})


@dataclass(frozen=True)
class Persona:
    name: str
    patience: int  # frustration tolerated before giving up
    recall: float  # chance a reply includes the order number
    gives: str  # reply that supplies the order number
    vague: str  # reply that forgets to
    retry: str  # reply after a wrong or useless answer
    angry: bool = False  # the agent sees an angry customer


PERSONAS = (
    Persona(
        "patient", 3, 1.0, "Sure, it's order {oid}. Thanks!",
        "", "Hmm, that doesn't look right. Check again?",
    ),
    Persona("terse", 2, 1.0, "{oid}", "", "Wrong."),
    Persona(
        "angry", 1, 1.0, "{oid}. Hurry up.", "",
        "That is useless. Fix it now!", angry=True,
    ),
    Persona(
        "confused", 3, 0.5, "Oh, I think it's {oid}?",
        "Order number? It was the blue one, last week.",
        "Sorry, I don't follow. Can you check again?",
    ),
)  # fmt: skip


def goal_for(task: Task, pool) -> Task:
    """The concrete request behind a task.

    An ambiguous task hides a real order: the user means one
    ordinary task of the same category from the pool.
    """
    if not task.expected["clarify"]:
        return task
    plain = [
        t
        for t in pool
        if t.category == task.category and not t.tags & UNUSUAL
    ]
    if not plain:
        raise ValueError(f"no plain {task.category} task")
    return rng_for(task.id, "goal").choice(plain)


def in_character(task: Task, persona: Persona) -> Task:
    """The task as this persona sends it."""
    if persona.angry:
        return replace(task, tags=task.tags | {"angry"})
    return task


def satisfied(answer: str, goal: Task) -> bool:
    """The user's view: does the reply give what they wanted?"""
    exp, text = goal.expected, answer.lower()
    oid = exp["args"].get("order_id", "")
    wanted = [*exp["answer_contains"], oid]
    if not all(w.lower() in text for w in wanted):
        return False
    return not any(p.lower() in text for p in exp["must_not"])


def frustration(answer: str) -> int:
    """Frustration a reply causes: 0 for a fair question."""
    text = answer.lower()
    if "calm down" in text:
        return 2
    return 0 if "order number" in text else 1


def user_reply(
    asked: Task, goal: Task, persona: Persona, turn: int
) -> Task:
    """The user's next message, as the next task to run.

    While the request is still ambiguous the user supplies
    the order number (or forgets to); that turns the next
    task into the goal under a new id. Otherwise the user
    repeats the same request in their own words.
    """
    if not asked.expected["clarify"]:
        return replace(asked, input=persona.retry)
    rng = rng_for(asked.id, persona.name, turn, "recall")
    if rng.random() >= persona.recall:
        return replace(asked, input=persona.vague)
    oid = goal.expected["args"]["order_id"]
    text = persona.gives.format(oid=oid)
    follow = replace(
        goal, id=f"{asked.id}.{turn + 1}", input=text
    )
    return in_character(follow, persona)
