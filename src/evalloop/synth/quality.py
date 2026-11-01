"""Quality: keep only synthetic tasks you can trust, then compare.

A synthetic task is useless if its answer key is wrong or
it repeats another task. Validity checks confirm each
expectation is derivable from the world; dedup removes
repeats; shift metrics show how far the synthetic set sits
from the real-like set it is meant to extend.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from evalloop.core import KB, Task, World
from evalloop.datasets import cells, dedup, token_set


def order_problems(task: Task, world: World) -> list[str]:
    """The order must exist, appear in the text, and match."""
    exp = task.expected
    oid = exp["args"].get("order_id")
    if oid is None:
        return []
    order = world.lookup_order(oid)
    if "error" in order:
        return ["unknown order"]
    if oid not in task.input:
        return ["order id not in input"]
    if task.category == "order_status":
        if exp["answer_contains"] != [order["status"]]:
            return ["wrong status"]
    if exp["eligible"] is not None:
        fits = world.refund_allowed(oid, order["total"])
        if exp["eligible"] != fits:
            return ["wrong eligibility"]
    return []


def kb_problems(task: Task, world: World) -> list[str]:
    """The article must exist and the query must retrieve it."""
    kb = task.expected["kb"]
    if not kb:
        return []
    if kb not in KB:
        return ["unknown article"]
    query = task.expected["args"].get("query", "")
    if world.search_kb(query).get("article") != kb:
        return ["query misses article"]
    return []


def clarify_problems(task: Task) -> list[str]:
    """An ambiguous task must really lack the order id."""
    exp = task.expected
    if exp["clarify"] and re.search(r"A-\d{4}", task.input):
        return ["ambiguous task names an order"]
    if exp["clarify"] and exp["tools"]:
        return ["ambiguous task expects tools"]
    return []


def problems(task: Task, world: World | None = None) -> list:
    """Reasons the task's answer key cannot be trusted."""
    world = world or World()
    return (
        order_problems(task, world)
        + kb_problems(task, world)
        + clarify_problems(task)
    )


def quality_filter(
    tasks: list[Task],
) -> tuple[list[Task], dict[str, str]]:
    """Valid, deduplicated tasks plus a reason per rejected id."""
    rejected, valid = {}, []
    for t in tasks:
        found = problems(t)
        if found:
            rejected[t.id] = found[0]
        else:
            valid.append(t)
    kept, repeats = dedup(valid)
    rejected.update({tid: "duplicate" for tid in repeats})
    return kept, rejected


def distribution(labels: list[str]) -> dict[str, float]:
    counts = Counter(labels)
    return {k: n / len(labels) for k, n in counts.items()}


def tvd(p: dict[str, float], q: dict[str, float]) -> float:
    """Total variation distance: 0 same, 1 disjoint."""
    keys = p.keys() | q.keys()
    return sum(abs(p.get(k, 0) - q.get(k, 0)) for k in keys) / 2


@dataclass(frozen=True)
class Shift:
    category_tvd: float
    tag_tvd: float
    length_ratio: float  # mean synthetic / mean real words
    vocab_overlap: float  # Jaccard of the two vocabularies


def mean_words(tasks: list[Task]) -> float:
    return sum(len(t.input.split()) for t in tasks) / len(tasks)


def vocabulary(tasks: list[Task]) -> frozenset[str]:
    return frozenset().union(*(token_set(t.input) for t in tasks))


def shift(real: list[Task], synth: list[Task]) -> Shift:
    """How far the synthetic set sits from the real-like set."""
    cats = [
        distribution([t.category for t in s])
        for s in (real, synth)
    ]
    tags = [
        distribution([g for t in s for _, g in cells(t)])
        for s in (real, synth)
    ]
    a, b = vocabulary(real), vocabulary(synth)
    return Shift(
        category_tvd=tvd(*cats),
        tag_tvd=tvd(*tags),
        length_ratio=mean_words(synth) / mean_words(real),
        vocab_overlap=len(a & b) / len(a | b),
    )
