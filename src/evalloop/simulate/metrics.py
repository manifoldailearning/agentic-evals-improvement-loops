"""Conversation-level and turn-level metrics.

A conversation can succeed after a bad turn, and fail after
good ones. Turn-level pass rate judges each reply alone;
conversation metrics judge the whole exchange: was the goal
met, in how many turns, with how much frustration. The goal
check reads the final trajectory, not the user's opinion.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from evalloop.simulate.conversation import Conversation
from evalloop.trajectory import passes


def goal_met(conv: Conversation) -> bool:
    """The last turn resolved the real request correctly."""
    last = conv.turns[-1]
    if last.task.expected["clarify"]:
        return False  # still asking which order
    return passes(last.run, last.task)


def turn_pass_rate(convs: list[Conversation]) -> float:
    """Share of individual replies that pass their own turn."""
    turns = [t for c in convs for t in c.turns]
    return mean(passes(t.run, t.task) for t in turns)


def conversation_metrics(convs: list[Conversation]) -> dict:
    """Whole-conversation outcomes for a set of runs."""
    met = [c for c in convs if goal_met(c)]
    happy = [c for c in convs if c.outcome == "satisfied"]
    return {
        "goal_met": len(met) / len(convs),
        "turns_to_success": (
            mean(len(c.turns) for c in met) if met else 0.0
        ),
        "frustrated": mean(c.frustration > 0 for c in convs),
        "gave_up": mean(c.outcome == "gave_up" for c in convs),
        "silent_failures": sum(not goal_met(c) for c in happy),
    }


def by_persona(convs: list[Conversation]) -> dict:
    """``conversation_metrics`` per persona."""
    groups: dict[str, list] = defaultdict(list)
    for c in convs:
        groups[c.persona].append(c)
    return {p: conversation_metrics(g) for p, g in groups.items()}
