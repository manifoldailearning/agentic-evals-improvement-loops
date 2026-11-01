"""Curate: filter harvested examples into a dataset to tune on.

Raw examples repeat, leak contact details, and can copy the
very eval items that will judge the tuned model. Filters run
in order (quality, dedup, PII scrub, contamination check by
n-gram overlap, balance by category), each logged in a
funnel, and the result ships with a dataset card.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, replace

from evalloop.core import AgentConfig, stable_hash
from evalloop.flywheel.harvest import Example, quality

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"\+?\d[\d ]{8,}\d")


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9-]+", text.lower())


def dedup(examples) -> list[Example]:
    """Drop repeats of the same normalized request."""
    seen, out = set(), []
    for ex in examples:
        key = " ".join(tokens(ex.input))
        if key not in seen:
            seen.add(key)
            out.append(ex)
    return out


def scrub(text: str) -> str:
    """Mask emails and phone numbers."""
    return PHONE.sub("[phone]", EMAIL.sub("[email]", text))


def scrub_pii(examples) -> list[Example]:
    return [replace(ex, input=scrub(ex.input)) for ex in examples]


def windows(t: list[str], k: int) -> set[tuple[str, ...]]:
    return {tuple(t[i : i + k]) for i in range(len(t) - k + 1)}


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    """Word n-grams; a text shorter than n is one gram."""
    t = tokens(text)
    return windows(t, n) if len(t) > n else {tuple(t)}


def eval_index(eval_tasks, n: int = 8) -> list[set]:
    """The n-gram set of every eval input."""
    return [ngrams(t.input, n) for t in eval_tasks]


def contaminated(ex: Example, index, min_share=0.6) -> bool:
    """True if most of some eval item's n-grams are in ``ex``."""
    t = tokens(ex.input)
    sizes = {len(next(iter(g))) for g in index}
    mine = set().union(*(windows(t, k) for k in sizes))
    return any(
        len(grams & mine) >= min_share * len(grams)
        for grams in index
    )


def balance(examples, cap: int) -> list[Example]:
    """At most ``cap`` per category, picked by stable hash."""
    ranked = sorted(examples, key=lambda e: stable_hash(e.id))
    count, out = Counter(), []
    for ex in ranked:
        if count[ex.category] < cap:
            count[ex.category] += 1
            out.append(ex)
    return sorted(out, key=lambda e: e.id)


@dataclass
class Funnel:
    steps: list[tuple[str, int]] = field(default_factory=list)

    def log(self, step: str, examples) -> list:
        self.steps.append((step, len(examples)))
        return examples


def curate(examples, eval_tasks, cap=30, n=8) -> tuple:
    """Run every filter in order; return data and funnel."""
    f = Funnel()
    data = f.log("harvested", list(examples))
    data = f.log("quality", [e for e in data if quality(e) == 1])
    data = f.log("dedup", dedup(data))
    data = scrub_pii(data)
    index = eval_index(eval_tasks, n)
    clean = [e for e in data if not contaminated(e, index)]
    data = f.log("decontaminate", clean)
    data = f.log("balance", balance(data, cap))
    return data, f


def finetune(
    base: AgentConfig, data, min_size=100
) -> AgentConfig:
    """SIMULATED tuning: swaps in core's fixed small-ft model.

    small-ft scales every failure mode by 0.65; it does not
    read ``data``. Only the size check below uses the data.
    """
    if base.model != "small":
        raise ValueError("only the small model can be tuned")
    if len(data) < min_size:
        raise ValueError(f"need {min_size}+ examples")
    tag = base.version + "-ft"
    return base.evolve(model="small-ft", version=tag)
