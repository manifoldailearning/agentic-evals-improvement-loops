"""Axial coding: from messy notes to a taxonomy that drives fixes.

Free-text notes are grouped into a small set of failure
categories by a codebook of keyword rules. Counting traces
per category gives a Pareto chart; new categories per batch
show when review has saturated; and each category links to
the config change most likely to fix it.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable

from evalloop.analysis.annotate import Note
from evalloop.core import AgentConfig, rng_for

CODEBOOK = (  # first matching rule wins
    ("no answer", r"repeated|stuck|without answering"),
    ("ignored ambiguity", r"guessed|vague|should have asked"),
    ("policy breach", r"forbids|refund policy|been escalated"),
    ("wrong tool", r"wrong tool|instead of|was needed|started"),
    ("bad arguments", r"arguments|order id in|empty args"),
    ("made-up facts", r"records|contradicts|misquotes|claims"),
    ("missing citation", r"citation|cites nothing"),
    ("stopped early", r"before calling|early|without finishing"),
    ("bad tone", r"calm down|tone|rude"),
)
UNCODED = "uncoded"
FIXES = {  # category -> (config field, change)
    "no answer": ("rules", "stop_after_answer"),
    "ignored ambiguity": ("rules", "ask_when_ambiguous"),
    "policy breach": ("rules", "check_refund_policy"),
    "wrong tool": ("few_shots", 3),
    "bad arguments": ("rules", "verify_order_id"),
    "made-up facts": ("model", "medium"),
    "missing citation": ("rules", "cite_kb"),
    "stopped early": ("model", "medium"),
    "bad tone": ("rules", "escalate_angry_customers"),
}


def axial_code(text: str) -> str:
    """The category whose rule first matches the note."""
    for category, pattern in CODEBOOK:
        if re.search(pattern, text, re.IGNORECASE):
            return category
    return UNCODED


def code_notes(notes: Iterable[Note]) -> dict[str, set[str]]:
    """Task id -> the categories its notes were coded into."""
    coded: dict[str, set[str]] = defaultdict(set)
    for note in notes:
        coded[note.task_id].add(axial_code(note.text))
    return dict(coded)


def category_counts(coded: dict[str, set[str]]) -> Counter:
    """Number of traces showing each category."""
    return Counter(c for cats in coded.values() for c in cats)


def pareto(counts: Counter) -> list[tuple[str, int, float]]:
    """(category, count, cumulative share), largest first."""
    total = sum(counts.values())
    rows, running = [], 0
    ranked = sorted(
        counts.items(), key=lambda kv: (-kv[1], kv[0])
    )
    for category, n in ranked:
        running += n
        rows.append((category, n, running / total))
    return rows


def saturation(
    coded: dict[str, set[str]], batch: int = 5, seed: int = 0
) -> list[int]:
    """New categories found in each batch of reviewed traces.

    Traces are reviewed in a seeded random order; a run of
    zeros means further review is unlikely to add categories.
    """
    order = sorted(coded)
    rng_for("review-order", seed).shuffle(order)
    seen: set[str] = set()
    fresh = []
    for start in range(0, len(order), batch):
        ids = order[start : start + batch]
        found = set().union(*(coded[t] for t in ids))
        fresh.append(len(found - seen))
        seen |= found
    return fresh


def describe_fix(category: str) -> str:
    """Human-readable change for a category, or 'investigate'."""
    if category not in FIXES:
        return "investigate"
    field, change = FIXES[category]
    if field == "rules":
        return f"add rule {change}"
    return f"set {field}={change}"


def apply_fix(config: AgentConfig, category: str) -> AgentConfig:
    """The config with this category's fix applied."""
    field, change = FIXES[category]
    if field == "rules":
        return config.evolve(rules=config.rules | {change})
    return config.evolve(**{field: change})
