"""Release signals, joined into one ship-or-hold decision.

Each signal answers one question with evidence: is quality
high enough once uncertainty is counted, does the regression
gate hold, is the candidate no worse than the current
release, does it resist injected instructions, and is it
affordable. One failing signal holds the release.
"""

from __future__ import annotations

from dataclasses import dataclass

from evalloop.gates import Baseline, GatePolicy, run_gate
from evalloop.program.review import (
    INJECTED,
    TOLERANCES,
    Candidate,
    Evidence,
    ReleasePolicy,
    as_baseline,
    critical_tasks,
    gather,
)
from evalloop.robustness import perturb
from evalloop.stats import compare, wilson_interval


@dataclass(frozen=True)
class Signal:
    name: str
    passed: bool
    detail: str


def quality(ev: Evidence, policy: ReleasePolicy) -> Signal:
    """Pass rate per task (majority of seeds), with its CI."""
    votes = ev.majority()
    ci = wilson_interval(sum(votes.values()), len(votes))
    detail = (
        f"{ci.estimate:.1%} [{ci.low:.1%}, {ci.high:.1%}]"
        f" floor {policy.min_quality:.0%}"
    )
    return Signal("quality", ci.low >= policy.min_quality, detail)


def gate(
    cand: Evidence, base: Baseline, policy: ReleasePolicy
) -> Signal:
    """The CI regression gate against a stored baseline."""
    rules = GatePolicy(
        TOLERANCES,
        frozenset(critical_tasks(policy, base)),
        policy.flaky_budget,
    )
    extra = {"cost": cand.cost}
    report = run_gate(
        cand.version, cand.outcomes, base, rules, extra=extra
    )
    failed = [c.name for c in report.checks if not c.passed]
    detail = "failed " + ", ".join(failed) if failed else (
        f"all {len(report.checks)} checks ok"
    )
    return Signal("gate", report.passed, detail)


def versus(cand: Evidence, current: Evidence) -> Signal:
    """Paired comparison on the same tasks: never worse."""
    c = compare(current.majority(), cand.majority())
    gain = c.gain
    detail = (
        f"{gain.estimate * 100:+.1f} pts"
        f" [{gain.low * 100:+.1f}, {gain.high * 100:+.1f}]"
        f" {c.verdict}"
    )
    return Signal("versus", c.verdict != "worse", detail)


def injection(cand: Candidate, policy: ReleasePolicy) -> Signal:
    """Refunds that broke policy when tasks carry injections.

    Harm is counted from side effects, not from the words:
    money that actually moved against policy.
    """
    probes = [
        perturb(t, INJECTED)
        for t in policy.task_set()
        if t.expected["eligible"] is False
    ]
    harmed = 0
    for probe in probes:
        for seed in policy.seeds:
            agent = cand.agent()  # a fresh world per run
            agent.run(probe, seed)
            harmed += any(
                e.get("ok") is False for e in agent.world.effects
            )
    total = len(probes) * len(policy.seeds)
    rate, limit = harmed / total, policy.max_injection_harm
    detail = (
        f"{harmed}/{total} probes paid out"
        f" (limit {limit:.0%})"
    )
    return Signal("injection", rate <= limit, detail)


def cost(ev: Evidence, policy: ReleasePolicy) -> Signal:
    """Dollars per successful run, against the budget."""
    runs = sum(len(r) for r in ev.outcomes.values())
    wins = sum(sum(r) for r in ev.outcomes.values())
    per_win = ev.cost * runs / max(wins, 1)
    limit = policy.max_cost_per_success
    detail = f"${per_win:.4f} per success, limit ${limit:.2f}"
    return Signal("cost", per_win <= limit, detail)


@dataclass(frozen=True)
class Review:
    candidate: str
    current: str
    signals: tuple[Signal, ...]

    @property
    def decision(self) -> str:
        ok = all(s.passed for s in self.signals)
        return "ship" if ok else "hold"

    def lines(self) -> list[str]:
        head = f"Review {self.candidate} vs {self.current}"
        rows = [
            f"  {'ok' if s.passed else 'FAIL':<4}"
            f" {s.name:<9} {s.detail}"
            for s in self.signals
        ]
        return [head, *rows, f"  decision: {self.decision}"]


def review(
    candidate: Candidate,
    current: Candidate,
    policy: ReleasePolicy | None = None,
) -> Review:
    """Every signal for a candidate, against the release."""
    policy = policy or ReleasePolicy()
    cand = gather(candidate, policy)
    now = gather(current, policy)
    signals = (
        quality(cand, policy),
        gate(cand, as_baseline(now, policy), policy),
        versus(cand, now),
        injection(candidate, policy),
        cost(cand, policy),
    )
    return Review(candidate.version, current.version, signals)
