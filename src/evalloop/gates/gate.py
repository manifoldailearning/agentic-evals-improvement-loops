"""The regression gate: a verdict a CI job can act on.

A change ships only if it holds the line set by a stored
baseline: each metric within its tolerance, every critical
task passing on every seed, flakiness within budget, and no
significant rise in tasks that used to pass and now fail.
The report ends in an exit code, so CI can block a merge.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from evalloop.gates.flaky import (
    Outcomes,
    Quarantine,
    apply_quarantine,
    is_flaky,
    summarize,
)
from evalloop.stats import mcnemar


@dataclass(frozen=True)
class Baseline:
    version: str
    k: int  # seeds per task
    outcomes: Outcomes  # the accepted run, per task and seed
    extra: dict[str, float]  # run-level metrics, e.g. cost

    def metrics(
        self, task_ids: Iterable[str]
    ) -> dict[str, float]:
        """Metrics over the given tasks only.

        Recomputing on the candidate's task set keeps the
        comparison like for like when tasks are quarantined.
        """
        subset = {
            t: self.outcomes[t]
            for t in task_ids
            if t in self.outcomes
        }
        return {**summarize(subset, self.k), **self.extra}


def save_baseline(path: str | Path, baseline: Baseline) -> None:
    text = json.dumps(asdict(baseline), indent=1, sort_keys=True)
    Path(path).write_text(text + "\n")


def load_baseline(path: str | Path) -> Baseline:
    return Baseline(**json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class Tolerance:
    metric: str
    max_drop: float  # allowed worsening, in metric units
    higher_is_better: bool = True
    fmt: str = "{:.1%}"


@dataclass(frozen=True)
class GatePolicy:
    tolerances: tuple[Tolerance, ...]
    critical: frozenset[str] = frozenset()
    flaky_budget: float = 0.10  # max share of flaky tasks
    alpha: float = 0.05  # for the regression test


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class GateReport:
    version: str
    checks: tuple[Check, ...] = ()

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1

    def lines(self, only_failed: bool = False) -> list[str]:
        """The report as text; CI logs can show just failures."""
        verdict = "PASS" if self.passed else "FAIL"
        head = f"Gate {self.version}: {verdict}"
        shown = [
            c
            for c in self.checks
            if not (only_failed and c.passed)
        ]
        rows = [
            f"  {'ok' if c.passed else 'FAIL':<4} {c.name:<11}"
            f" {c.detail}"
            for c in shown
        ]
        return [f"{head} (exit {self.exit_code})", *rows]


def check_metric(
    tol: Tolerance, before: float, after: float
) -> Check:
    """Fail when the metric worsens by more than allowed."""
    worse = before - after
    if not tol.higher_is_better:
        worse = -worse
    show = tol.fmt.format
    detail = (
        f"{show(before)} -> {show(after)}"
        f" (tolerance {show(tol.max_drop)})"
    )
    return Check(
        tol.metric, worse <= tol.max_drop + 1e-12, detail
    )


def check_flaky(outcomes: Outcomes, budget: float) -> Check:
    """The nondeterminism budget: cap the share of flaky tasks."""
    flaky = sum(is_flaky(r) for r in outcomes.values())
    share = flaky / len(outcomes)
    detail = f"{flaky} tasks, {share:.1%} (budget {budget:.1%})"
    return Check("flaky", share <= budget, detail)


def check_critical(
    outcomes: Outcomes, critical: frozenset[str]
) -> Check:
    """Every critical task must pass on every seed."""
    missing = sorted(critical - outcomes.keys())
    failing = sorted(
        t
        for t in critical & outcomes.keys()
        if not all(outcomes[t])
    )
    bad = missing + failing
    passing = len(critical) - len(bad)
    detail = f"{passing}/{len(critical)} pass every seed"
    if bad:
        detail += ": " + ", ".join(bad[:3])
    return Check("critical", not bad, detail)


def majority(results: list[bool]) -> bool:
    return 2 * sum(results) >= len(results)


def check_regressions(
    baseline: Baseline, outcomes: Outcomes, alpha: float
) -> Check:
    """McNemar on majority votes: did more tasks break than fix?

    Pairs each task with its own baseline result, so only
    tasks that changed count. Fails only on a significant
    excess of breaks; a single break is left to tolerances.
    """
    shared = outcomes.keys() & baseline.outcomes.keys()
    before = {t: majority(baseline.outcomes[t]) for t in shared}
    after = {t: majority(outcomes[t]) for t in shared}
    test = mcnemar(before, after)
    detail = (
        f"{test.broken} broke, {test.fixed} fixed"
        f" (p={test.p_value:.3f})"
    )
    worse = test.broken > test.fixed and test.p_value < alpha
    return Check("regressions", not worse, detail)


def check_quarantine(
    held: list[str], expired: list[str], critical: frozenset
) -> Check:
    """Quarantine is for flaky tasks, never critical ones."""
    detail = f"{len(held)} held, {len(expired)} expired"
    blocked = sorted(set(held) & critical)
    if blocked:
        detail += ": critical " + ", ".join(blocked)
    return Check("quarantine", not blocked, detail)


def run_gate(
    version: str,
    outcomes: Outcomes,
    baseline: Baseline,
    policy: GatePolicy,
    quarantine: Iterable[Quarantine] = (),
    today: str = "1970-01-01",
    extra: Mapping[str, float] | None = None,
) -> GateReport:
    """Run every check against the baseline; collect them."""
    gated, held, expired = apply_quarantine(
        outcomes, quarantine, today
    )
    before = baseline.metrics(gated)
    now = {**summarize(gated, baseline.k), **(extra or {})}
    checks = [
        check_metric(t, before[t.metric], now[t.metric])
        for t in policy.tolerances
    ]
    checks += [
        check_flaky(gated, policy.flaky_budget),
        check_critical(gated, policy.critical - set(held)),
        check_regressions(baseline, gated, policy.alpha),
        check_quarantine(held, expired, policy.critical),
    ]
    return GateReport(version, tuple(checks))
