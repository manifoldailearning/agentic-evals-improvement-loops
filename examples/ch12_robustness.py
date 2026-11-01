"""Chapter 12: robustness gaps, measured against a placebo.

Every task is rewritten three ways per perturbation and run
on three seeds. A placebo (new id, same text) shows how much
gap noise alone produces. Injected instructions barely move
the improved config's success rate, yet it starts issuing
refunds the policy forbids.
"""

from __future__ import annotations

from evalloop.core import (
    BASELINE,
    IMPROVED,
    SimAgent,
    make_tasks,
)
from evalloop.robustness import (
    PERTURBATIONS,
    perturbed_suite,
    refusal_balance,
    robustness,
    run_clean,
    worst_slice,
)


def measure(config, tasks) -> dict:
    agent = SimAgent(config)
    clean = run_clean(agent, tasks)
    bent = perturbed_suite(agent, tasks, PERTURBATIONS)
    return {
        "clean": clean,
        "bent": bent,
        "results": robustness(clean, bent),
    }


def cells(r) -> str:
    gap = f"{100 * r.gap:+.1f}"
    return f"{r.perturbed:>8.1%}{gap:>7}{r.consistency:>7.1%}"


def table(base: dict, best: dict) -> None:
    print(f"{'':<16}{'---- baseline ----':<22}---- improved ----")
    head = f"{'success':>8}{'gap':>7}{'same':>7}"
    print(f"{'perturbation':<14}{head}{head}")
    a, b = base["results"][0], best["results"][0]
    print(f"{'clean':<14}{a.clean:>8.1%}{'':>14}{b.clean:>8.1%}")
    rows = zip(base["results"], best["results"], strict=True)
    for a, b in rows:
        print(f"{a.name:<14}{cells(a)}{cells(b)}")


def refusals(label: str, m: dict) -> str:
    before = refusal_balance(m["clean"])
    after = refusal_balance(m["bent"]["injected"])
    return (
        f"  {label:<9}unsafe {before['unsafe']:.1%} ->"
        f" {after['unsafe']:.1%}, over-refusal"
        f" {before['over_refusal']:.1%} ->"
        f" {after['over_refusal']:.1%}"
    )


def main() -> None:
    tasks = make_tasks(200, seed=7)
    base = measure(BASELINE, tasks)
    best = measure(IMPROVED, tasks)
    print("Robustness: 200 tasks x 3 rewrites x 3 seeds")
    table(base, best)
    print(
        "gap = clean - perturbed, in points; same = same answer"
    )
    print("placebo = new id, same text: the noise floor")
    bent = base["bent"]
    cat, before, after = worst_slice(
        bent["placebo"], bent["injected"]
    )
    print(f"Worst slice, baseline: {cat} under injection")
    print(f"  placebo {before:.1%} -> injected {after:.1%}")
    print("Refunds the policy forbids, clean -> injected:")
    print(refusals("baseline", base))
    print(refusals("improved", best))


if __name__ == "__main__":
    main()
