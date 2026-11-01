"""Chapter 15: online experiments on simulated live traffic.

Four thousand users are split by stable hashing between the
baseline and a variant with two extra prompt rules. The
readout gives the lift with an interval, guardrails, and a
sequential monitor that stops early without inflating false
alarms; then a novelty check and a ranker interleaving test.
"""

from __future__ import annotations

import re

from evalloop.core import BASELINE, KB, make_tasks, rng_for
from evalloop.online import (
    Guardrail,
    boundaries,
    check_guardrail,
    credit,
    false_alarm_rate,
    interleaving_result,
    lift,
    monitor,
    novelty_check,
    simulate_traffic,
    team_draft,
)

TREATMENT = BASELINE.evolve(
    version="v2",
    rules=frozenset(
        {"check_refund_policy", "ask_when_ambiguous"}
    ),
)
GUARDRAILS = (
    Guardrail("cost", 0.10),
    Guardrail("latency_ms", 0.10),
)
LOOKS = (0.25, 0.5, 0.75, 1.0)


def pts(i) -> str:
    """An interval of rates as signed percentage points."""
    lo, hi = 100 * i.low, 100 * i.high
    return f"{100 * i.estimate:+.1f} pts [{lo:+.1f}, {hi:+.1f}]"


def show_readout(sessions) -> None:
    n_t = sum(s.variant == "treatment" for s in sessions)
    n_c = len(sessions) - n_t
    print(
        f"A/B: baseline vs v2 (+2 rules), {len(sessions)} users"
    )
    print(f"  {n_c} control / {n_t} treatment, by stable hash")
    print(f"  resolved lift {pts(lift(sessions))}")
    held = True
    for g in GUARDRAILS:
        ok, rel = check_guardrail(sessions, g)
        held = held and ok
        verdict = "ok" if ok else "FAIL"
        print(
            f"  {g.metric:<10} {rel.estimate:+.1%}"
            f" [{rel.low:+.1%}, {rel.high:+.1%}]"
            f" limit {g.max_increase:+.0%} {verdict}"
        )
    decision = "ship" if held else "hold: a guardrail failed"
    print(f"  decision: {decision}")


def show_sequential(sessions) -> None:
    print("Sequential looks, O'Brien-Fleming-type spending:")
    print("  look  sessions      z  bound  decision")
    for i, look in enumerate(monitor(sessions, LOOKS), 1):
        verdict = "stop: v2 wins" if look.stop else "continue"
        print(
            f"  {i:>4}  {look.sessions:>8} {look.z:>6.2f}"
            f" {look.boundary:>6.2f}  {verdict}"
        )
    five = (0.2, 0.4, 0.6, 0.8, 1.0)
    naive = false_alarm_rate([1.96] * 5, five)
    safe = false_alarm_rate(boundaries(five), five)
    print("A/A tests with 5 looks, share that ever 'win':")
    print(
        f"  naive p < 0.05 {naive:.1%}; alpha spending {safe:.1%}"
    )


def show_novelty(sessions) -> None:
    likes = novelty_check(sessions, "thumbs_up", split_day=4)
    steady = novelty_check(sessions, "resolved", split_day=4)
    print("Novelty check, thumbs-up lift before/after day 4:")
    print(f"  days 0-3  {pts(likes.early)}")
    print(
        f"  days 4-13 {pts(likes.late)}  (p={likes.p_value:.3f})"
    )
    print(f"  resolved lift unchanged (p={steady.p_value:.2f})")


def tokens(text: str, size: int = 0) -> set[str]:
    """Lowercase words, cut to ``size`` letters if given."""
    found = re.findall(r"[a-z0-9]+", text.lower())
    return {w[:size] if size else w for w in found}


def by_title(query: str) -> list[str]:
    """Ranker A: articles whose id shares words with the query."""
    q = tokens(query)
    return sorted(KB, key=lambda k: (-len(q & tokens(k)), k))


def by_stem(query: str) -> list[str]:
    """Ranker B: 5-letter stems of the id and the article text."""
    q = tokens(query, 5)
    score = {k: len(q & tokens(f"{k} {KB[k]}", 5)) for k in KB}
    return sorted(KB, key=lambda k: (-score[k], k))


def show_interleaving(tasks, searches: int = 2000) -> None:
    """Team-draft each search; the user clicks the answer."""
    outcomes = []
    for i in range(searches):
        rng = rng_for("search", i)
        task = rng.choice(tasks)
        a, b = by_title(task.input), by_stem(task.input)
        shown, team = team_draft(a, b, rng)
        rank = shown.index(task.expected["kb"])
        seen = rng.random() < 0.9**rank  # lower = less seen
        clicks = [task.expected["kb"]] if seen else []
        outcomes.append(credit(clicks, team))
    a_wins, b_wins, ties, p = interleaving_result(outcomes)
    print(
        f"Interleaving title vs stem ranker, {searches} searches:"
    )
    print(
        f"  stem wins {b_wins}, title wins {a_wins},"
        f" {ties} no click, p={p:.4f}"
    )


def main() -> None:
    tasks = make_tasks(200, seed=7)
    configs = {"control": BASELINE, "treatment": TREATMENT}
    sessions = simulate_traffic(configs, tasks, novelty=0.2)
    show_readout(sessions)
    show_sequential(sessions)
    show_novelty(sessions)
    help_center = ("faq", "account")
    questions = make_tasks(400, seed=7)
    show_interleaving(
        [t for t in questions if t.category in help_center]
    )


if __name__ == "__main__":
    main()
