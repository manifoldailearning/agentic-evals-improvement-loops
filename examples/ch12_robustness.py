"""Chapter 12: robustness gaps, measured against a placebo.

Every task is rewritten three ways per perturbation and run
on three seeds. A placebo (new id, same text) shows how much
gap noise alone produces. Harm gets the same treatment:
forbidden refunds are compared with placebo runs, on many
rewrites of the few tasks where a refund is forbidden.
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
    run_perturbed,
    worst_slice,
)

HARM_VARIANTS = 40  # few tasks forbid refunds: sample more


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


def harm(label: str, config, tasks) -> str:
    """Forbidden refunds, placebo vs injected rewrites."""
    agent = SimAgent(config)
    risky = [
        t
        for t in tasks
        if "issue_refund" in t.expected["must_not"]
    ]
    rate = {}
    for p in PERTURBATIONS:
        if p.name in ("placebo", "injected"):
            runs = run_perturbed(agent, risky, p, HARM_VARIANTS)
            rate[p.name] = refusal_balance(runs)["unsafe"]
    return (
        f"  {label:<9}placebo {rate['placebo']:.1%} ->"
        f" injected {rate['injected']:.1%}"
        f" ({len(runs)} runs each)"
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
    print("Refunds the policy forbids, placebo -> injected:")
    print(harm("baseline", BASELINE, tasks))
    print(harm("improved", IMPROVED, tasks))


if __name__ == "__main__":
    main()
