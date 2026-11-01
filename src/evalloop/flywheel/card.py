"""Dataset cards: what a training set holds and how it was made.

A tuned model is only as auditable as its data. The card
records a content hash, size, category and source mix, the
filter funnel, and the caveats a reader must know before
trusting a model tuned on it or an eval run against it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from textwrap import wrap

from evalloop.core import stable_hash


@dataclass
class DatasetCard:
    name: str
    version: str  # hash of the examples' content
    size: int
    categories: dict[str, int]
    sources: dict[str, int]
    funnel: list[tuple[str, int]]
    pii_masked: int
    eval_set: (
        str  # version of the eval data it was checked against
    )
    caveats: list[str] = field(default_factory=list)

    def lines(self, width: int = 62) -> list[str]:
        """The card as short text lines for a terminal or page."""
        cats = " ".join(
            f"{k}={v}" for k, v in self.categories.items()
        )
        srcs = " ".join(
            f"{k}={v}" for k, v in self.sources.items()
        )
        head = f"{self.name} {self.version}: {self.size} examples"
        body = [
            f"categories: {cats}",
            f"sources: {srcs}; PII masked in {self.pii_masked}",
            f"decontaminated against eval set {self.eval_set}",
            *(f"caveat: {c}" for c in self.caveats),
        ]
        out = [head]
        for text in body:
            out += wrap(text, width, initial_indent="  ",
                        subsequent_indent="    ")  # fmt: skip
        return out


def content_hash(examples) -> str:
    parts = [(e.id, e.input, e.answer) for e in examples]
    return "sha-" + stable_hash(*parts)[:10]


def masked(ex) -> bool:
    return "[email]" in ex.input or "[phone]" in ex.input


def make_card(name, data, funnel, eval_set, caveats=()):
    """Describe a curated dataset from its rows and funnel."""
    cats = Counter(e.category for e in data)
    return DatasetCard(
        name,
        content_hash(data),
        len(data),
        dict(sorted(cats.items())),
        dict(Counter(e.source for e in data)),
        list(funnel.steps),
        sum(map(masked, data)),
        eval_set,
        list(caveats),
    )
