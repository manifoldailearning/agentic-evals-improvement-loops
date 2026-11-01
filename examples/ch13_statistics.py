"""Chapter 13: is v2 better than v1, or did we get lucky?

The same pair of configs is compared on 30, 200 and 400
tasks. Thirty tasks cannot tell them apart; 400 can. On
the same 200 tasks a paired test sees what an unpaired one
misses, and Holm's correction tempers a sweep of rules.
"""

from __future__ import annotations

from evalloop.core import (
    BASELINE,
    RULES,
    AgentConfig,
    SimAgent,
    make_tasks,
    run_suite,
)
from evalloop.stats import (
    compare,
    holm,
    mcnemar,
    min_detectable_effect,
    paired_sample_size,
    sample_size,
    two_proportion_test,
    wilson_interval,
)

V1 = BASELINE
V2 = BASELINE.evolve(
    version="v2", rules=frozenset({"cite_kb"}), temperature=0.3
)


def outcomes(config: AgentConfig, tasks) -> dict[str, bool]:
    """Pass/fail per task id.

    ``ok`` is ground truth from the simulator; a real system
    would use its graders' verdict here instead.
    """
    runs = run_suite(SimAgent(config), tasks)
    return {r.task_id: r.ok for r in runs}


def fmt_p(p: float) -> str:
    return f"{p:6.3f}" if p >= 0.001 else "<0.001"


def row(n: int, a: dict, b: dict) -> str:
    c = compare(a, b)
    g = c.gain
    pts = [100 * x for x in (g.estimate, g.low, g.high)]
    return (
        f"{n:>5} {c.rate_a:6.1%} {c.rate_b:6.1%}"
        f" {pts[0]:+5.1f} [{pts[1]:+5.1f}, {pts[2]:+5.1f}]"
        f" {fmt_p(c.test.p_value)} {c.verdict}"
    )


def first(results: dict, tasks, n: int) -> dict[str, bool]:
    """Results for the first ``n`` tasks only."""
    return {t.id: results[t.id] for t in tasks[:n]}


def show_pairing(a: dict, b: dict) -> None:
    n = len(a)
    s_a, s_b = sum(a.values()), sum(b.values())
    p = two_proportion_test(s_a, n, s_b, n)
    test = mcnemar(a, b)
    print(f"Same {n} tasks, two tests of v2 > v1:")
    print(f"  unpaired z-test:  p = {p:.3f}")
    print(
        f"  paired McNemar:   p = {test.p_value:.3f}"
        f" ({test.fixed} fixed, {test.broken} broken)"
    )


def show_planning(a: dict, b: dict) -> None:
    n = len(a)
    p_a, p_b = sum(a.values()) / n, sum(b.values()) / n
    test = mcnemar(a, b)
    split = (test.fixed + test.broken) / n
    gain = (test.fixed - test.broken) / n
    unpaired = sample_size(p_a, p_b)
    paired = paired_sample_size(gain, split)
    print(f"Tasks to detect {100 * gain:+.1f} pts at 80% power:")
    print(f"  unpaired: {unpaired} per config")
    print(f"  paired:   {paired} ({split:.1%} of tasks disagree)")
    mde = min_detectable_effect(p_a, 30)
    print(f"  30 tasks only see gains of {100 * mde:.0f}+ pts")


def show_holm(tasks) -> None:
    base = outcomes(BASELINE, tasks)
    raw = {}
    for rule in RULES:
        config = BASELINE.evolve(rules=frozenset({rule}))
        raw[rule] = mcnemar(base, outcomes(config, tasks)).p_value
    adjusted = holm(raw)
    print(f"Six one-rule fixes on {len(tasks)} tasks (Holm):")
    for rule in RULES:
        keep = "keep" if adjusted[rule] < 0.05 else "-"
        print(
            f"  {rule:<25} p={raw[rule]:.4f}"
            f" adj={adjusted[rule]:.4f} {keep}"
        )


def main() -> None:
    tasks = make_tasks(400, seed=7)
    a, b = outcomes(V1, tasks), outcomes(V2, tasks)
    print("v1 = baseline; v2 = +cite_kb, temperature 0.3")
    print(
        f"{'tasks':>5} {'v1':>6} {'v2':>6} {'gain':>5}"
        f" {'[95% CI]':^14} {'p':>6} verdict"
    )
    for n in (30, 200, 400):
        print(row(n, first(a, tasks, n), first(b, tasks, n)))
    for n in (30, 400):
        passed = sum(first(a, tasks, n).values())
        ci = wilson_interval(passed, n)
        print(
            f"v1 on {n:>3} tasks: {ci}, {100 * ci.width:.1f} pts wide"
        )
    show_pairing(first(a, tasks, 200), first(b, tasks, 200))
    show_planning(a, b)
    show_holm(tasks[:200])


if __name__ == "__main__":
    main()
