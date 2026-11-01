"""Interleaving: compare two rankers inside one result list.

For ranked results (help-center articles, say), showing each
user a blend of both rankings is far more sensitive than an
A/B split. Team-draft interleaving credits each shown item
to the ranker that picked it; clicks decide each query.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from random import Random

from evalloop.stats import sign_test


def team_draft(
    a: Sequence[str], b: Sequence[str], rng: Random
) -> tuple[list[str], dict[str, str]]:
    """Merge two rankings; return the list and each item's team.

    The team with fewer picks goes next; a coin breaks ties,
    so neither ranker owns the top slot.
    """
    merged: list[str] = []
    team: dict[str, str] = {}
    lists = {"A": list(a), "B": list(b)}
    picks = {"A": 0, "B": 0}
    while True:
        left = {
            k: [x for x in v if x not in team]
            for k, v in lists.items()
        }
        ready = [k for k in "AB" if left[k]]
        if not ready:
            return merged, team
        turn = min(ready, key=lambda k: (picks[k], rng.random()))
        item = left[turn][0]
        merged.append(item)
        team[item] = turn
        picks[turn] += 1


def credit(clicks: Iterable[str], team: dict[str, str]) -> str:
    """Which ranker won one query: A, B or tie."""
    votes = [team[c] for c in clicks if c in team]
    a, b = votes.count("A"), votes.count("B")
    return "A" if a > b else "B" if b > a else "tie"


def interleaving_result(
    outcomes: Sequence[str],
) -> tuple[int, int, int, float]:
    """Wins for A and B, ties, and a sign-test p-value.

    Ties carry no preference and are left out of the test;
    queries are assumed independent of each other.
    """
    a, b = outcomes.count("A"), outcomes.count("B")
    ties = len(outcomes) - a - b
    return a, b, ties, sign_test(a, b)
