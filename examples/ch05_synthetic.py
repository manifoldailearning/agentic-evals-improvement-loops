"""Chapter 5: synthetic tasks, filtered and compared.

We generate 300 tasks from templates, personas and
perturbations, drop the ones whose answer key is broken or
repeated, measure how far the survivors sit from the
real-like set, and check whether both sets rank configs alike.
"""

from __future__ import annotations

import textwrap
from collections import Counter

from evalloop.core import (
    BASELINE,
    IMPROVED,
    SimAgent,
    make_tasks,
    run_suite,
)
from evalloop.datasets import slice_scores, slices
from evalloop.synth import (
    PERSONAS,
    PERTURBATIONS,
    TEMPLATES,
    make_synthetic,
    quality_filter,
    shift,
)

CONFIGS = {"Baseline": BASELINE, "Improved": IMPROVED}
SHOWN = ("S-0001", "S-0005", "S-0025")


def scores(config, tasks) -> dict[str, float]:
    """Ground-truth success per task id."""
    runs = run_suite(SimAgent(config), tasks)
    return {r.task_id: float(r.ok) for r in runs}


def rate(config, tasks) -> float:
    return sum(scores(config, tasks).values()) / len(tasks)


def print_samples(tasks) -> None:
    for t in (t for t in tasks if t.id in SHOWN):
        tags = ",".join(sorted(t.tags))
        print(f"  {t.id} {t.category} [{tags}]")
        text = textwrap.shorten(t.input, 56, placeholder="...")
        print(f'    "{text}"')


def print_tags(tasks) -> None:
    counts = Counter(g for t in tasks for g in t.tags)
    pairs = [f"{g} {n}" for g, n in counts.most_common()]
    text = "Tags: " + ", ".join(pairs)
    print(textwrap.fill(text, 62, subsequent_indent="  "))


def print_shift(real, synth) -> None:
    s = shift(real, synth)
    print(f"Shift vs real-like set ({len(real)} tasks):")
    print(
        f"  category TVD {s.category_tvd:.2f},"
        f" tag TVD {s.tag_tvd:.2f}"
    )
    print(
        f"  length ratio {s.length_ratio:.2f}x,"
        f" vocabulary overlap {s.vocab_overlap:.0%}"
    )


def print_success(real, synth) -> None:
    """Do both sets tell the same story about the configs?"""
    for name, config in CONFIGS.items():
        r, s = rate(config, real), rate(config, synth)
        print(f"{name} success: real {r:.1%}, synthetic {s:.1%}")
    named = slices(synth)
    by_slice = slice_scores(named, scores(BASELINE, synth))
    worst = min(by_slice, key=by_slice.get)
    print(
        f"Hardest synthetic slice: {worst}"
        f" {by_slice[worst]:.0%} (n={len(named[worst])})"
    )


def main() -> None:
    synth = make_synthetic(300, seed=11)
    print(
        f"Generated {len(synth)}: {len(TEMPLATES)} templates x"
        f" {len(PERSONAS)} personas x"
        f" {len(PERTURBATIONS)} perturbations"
    )
    print_samples(synth)
    print_tags(synth)
    kept, rejected = quality_filter(synth)
    print(f"Quality filter: {len(synth)} -> {len(kept)} kept")
    reasons = Counter(rejected.values()).most_common()
    print("  " + ", ".join(f"{why} {n}" for why, n in reasons))
    real = make_tasks(200, seed=7)
    print_shift(real, kept)
    print_success(real, kept)


if __name__ == "__main__":
    main()
