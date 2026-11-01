"""Online experiments: real users, randomized, measured with CIs.

Offline evals predict; live traffic decides. Users are split
into variants by a stable hash so each always sees the same
one, sessions are simulated against the agent, and the A/B
readout reports the lift with an interval, guardrails on
cost and latency, and a check for a fading novelty effect.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean, variance

from evalloop.core import (
    AgentConfig,
    SimAgent,
    Task,
    Trajectory,
    rng_for,
    stable_hash,
)
from evalloop.stats import (
    Interval,
    difference_interval,
    normal_p_value,
    z_score,
)


def assign(user: str, experiment: str, share: float = 0.5) -> str:
    """Stable bucketing: a user keeps one variant for good."""
    bucket = int(stable_hash(experiment, user)[:8], 16) / 16**8
    return "treatment" if bucket < share else "control"


def resolved(task: Task, run: Trajectory) -> bool:
    """What a user notices: the facts, and nothing forbidden.

    Users never see citations or tool calls, only the reply
    and its side effects, so this is laxer than offline evals.
    """
    text = run.answer.lower()
    facts = task.expected["answer_contains"]
    called = {s.tool for s in run.tool_calls}
    bad = task.expected["must_not"]
    return bool(text) and (
        all(f.lower() in text for f in facts)
        and not any(b in called or b in text for b in bad)
    )


@dataclass(frozen=True)
class Session:
    user: str
    variant: str
    day: int
    resolved: bool
    thumbs_up: bool
    cost: float
    latency_ms: int


def feedback(rng, ok: bool, boost: float) -> bool:
    """Thumbs up: likely when resolved, plus any novelty."""
    return rng.random() < (0.7 if ok else 0.2) + boost


def simulate_traffic(
    configs: Mapping[str, AgentConfig],
    tasks: Sequence[Task],
    users: int = 4000,
    days: int = 14,
    novelty: float = 0.0,
    experiment: str = "exp",
) -> list[Session]:
    """One support session per user, in order of arrival.

    ``configs`` maps "control" and "treatment" to configs.
    One session per user keeps sessions independent, so the
    per-session tests below are valid. Feedback is generous
    for anything new: the treatment's thumbs-up rate gets a
    ``novelty`` boost that halves every two days.
    """
    agents = {v: SimAgent(c) for v, c in configs.items()}
    sessions = []
    for i in range(users):
        user = f"U-{i:05d}"
        variant = assign(user, experiment)
        rng = rng_for(experiment, user)
        day = i * days // users  # steady arrivals
        task = rng.choice(tasks)
        run = agents[variant].run(task, seed=rng.randrange(10**6))
        ok = resolved(task, run)
        new = variant == "treatment"
        boost = novelty * 0.5 ** (day / 2) if new else 0.0
        liked = feedback(rng, ok, boost)
        sessions.append(
            Session(
                user,
                variant,
                day,
                ok,
                liked,
                run.cost,
                run.latency_ms,
            )
        )
    return sessions


def split(sessions, metric: str) -> tuple[list, list]:
    """Metric values for control and for treatment."""
    groups: dict[str, list] = {"control": [], "treatment": []}
    for s in sessions:
        groups[s.variant].append(getattr(s, metric))
    return groups["control"], groups["treatment"]


def lift(sessions, metric: str = "resolved") -> Interval:
    """Treatment minus control rate, Newcombe interval."""
    a, b = split(sessions, metric)
    return difference_interval(sum(a), len(a), sum(b), len(b))


def mean_difference(
    a: Sequence[float], b: Sequence[float], confidence=0.95
) -> Interval:
    """Difference of means b - a, normal approximation.

    Uses Welch's standard error; fine for the large,
    independent samples of an online test, even if skewed.
    """
    se = math.sqrt(variance(a) / len(a) + variance(b) / len(b))
    d, half = fmean(b) - fmean(a), z_score(confidence) * se
    return Interval(d, d - half, d + half)


@dataclass(frozen=True)
class Guardrail:
    metric: str  # a numeric Session field, lower is better
    max_increase: float  # relative to control, e.g. 0.10


def check_guardrail(
    sessions, g: Guardrail
) -> tuple[bool, Interval]:
    """Pass only if we are confident the rise stays in bounds.

    Returns the relative change interval (control's mean is
    treated as known). The upper bound must sit below the
    limit: an underpowered test cannot pass a guardrail.
    """
    a, b = split(sessions, g.metric)
    d, base = mean_difference(a, b), fmean(a)
    rel = Interval(d.estimate / base, d.low / base, d.high / base)
    return rel.high <= g.max_increase, rel


@dataclass(frozen=True)
class Novelty:
    early: Interval
    late: Interval
    p_value: float  # do early and late lifts differ?


def novelty_check(
    sessions, metric: str, split_day: int
) -> Novelty:
    """Compare the lift before and after ``split_day``.

    A lift that is large early and shrinks later is a
    novelty effect: ship on the late number, not the early.
    """
    early = [s for s in sessions if s.day < split_day]
    late = [s for s in sessions if s.day >= split_day]
    before, after = lift(early, metric), lift(late, metric)
    se = math.hypot(wald_se(early, metric), wald_se(late, metric))
    p = normal_p_value((before.estimate - after.estimate) / se)
    return Novelty(before, after, p)


def wald_se(sessions, metric: str) -> float:
    """Standard error of a difference of two rates."""
    a, b = split(sessions, metric)
    p_a, p_b = fmean(a), fmean(b)
    return math.sqrt(
        p_a * (1 - p_a) / len(a) + p_b * (1 - p_b) / len(b)
    )
