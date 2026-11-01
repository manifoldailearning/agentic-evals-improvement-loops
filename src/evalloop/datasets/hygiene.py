"""Hygiene: duplicates and leakage between dev and test.

Duplicated tasks inflate a set's apparent size, and a test
task with a near twin in dev rewards memorizing dev. We
match exact duplicates on normalized text, near duplicates
on token Jaccard, and split by duplicate group so no group
straddles dev and test.
"""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations

from evalloop.core import Task, rng_for


def normalize(text: str) -> str:
    """Lowercase words and digits, single-spaced."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def token_set(text: str) -> frozenset[str]:
    return frozenset(normalize(text).split())


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Shared tokens over all tokens; 1.0 for two empties."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def find_twin(
    text: str, pool: list[Task], threshold: float
) -> str | None:
    """Id of the first task in ``pool`` similar to ``text``."""
    words = token_set(text)
    for other in pool:
        if jaccard(words, token_set(other.input)) >= threshold:
            return other.id
    return None


def dedup(
    tasks: list[Task], threshold: float = 0.9
) -> tuple[list[Task], dict[str, str]]:
    """Keep first occurrences; map each dropped id to its twin.

    Exact matches use normalized text. Near matches use token
    Jaccard at or above ``threshold``.
    """
    kept: list[Task] = []
    dropped: dict[str, str] = {}
    first: dict[str, str] = {}  # normalized text -> id
    for t in tasks:
        text = normalize(t.input)
        twin = first.get(text) or find_twin(
            t.input, kept, threshold
        )
        if twin:
            dropped[t.id] = twin
        else:
            first[text] = t.id
            kept.append(t)
    return kept, dropped


def duplicate_groups(
    tasks: list[Task], threshold: float = 0.8
) -> list[list[Task]]:
    """Connected groups of tasks linked by near duplication."""
    parent = list(range(len(tasks)))

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    words = [token_set(t.input) for t in tasks]
    for i, j in combinations(range(len(tasks)), 2):
        if jaccard(words[i], words[j]) >= threshold:
            parent[root(i)] = root(j)
    groups: dict[int, list[Task]] = defaultdict(list)
    for i, t in enumerate(tasks):
        groups[root(i)].append(t)
    return sorted(groups.values(), key=lambda g: g[0].id)


def by_id(tasks: list[Task]) -> list[Task]:
    return sorted(tasks, key=lambda t: t.id)


def random_split(
    tasks: list[Task], test_share: float = 0.3, seed: int = 0
) -> tuple[list[Task], list[Task]]:
    """Naive split: shuffle tasks, cut off the test share."""
    order = list(tasks)
    rng_for("split", seed).shuffle(order)
    cut = round(len(order) * test_share)
    return by_id(order[cut:]), by_id(order[:cut])


def grouped_split(
    tasks: list[Task],
    test_share: float = 0.3,
    seed: int = 0,
    threshold: float = 0.8,
) -> tuple[list[Task], list[Task]]:
    """Fill test with whole duplicate groups until it is full."""
    groups = duplicate_groups(tasks, threshold)
    rng_for("split", seed).shuffle(groups)
    dev, test = [], []
    for group in groups:
        full = len(test) >= len(tasks) * test_share
        (dev if full else test).extend(group)
    return by_id(dev), by_id(test)


def leaks(
    dev: list[Task], test: list[Task], threshold: float = 0.8
) -> list[tuple[str, str]]:
    """(test id, dev id) for each test task with a dev twin."""
    pairs = [
        (t.id, find_twin(t.input, dev, threshold)) for t in test
    ]
    return [(t, d) for t, d in pairs if d]
