"""The operating model: owners, rhythm and a maturity ladder.

Evals decay without owners and a cadence: datasets go stale,
judges drift, gates get bypassed. This module records the
rituals that keep an eval program alive, scores a team's
practices against a four-level ladder, and names the next
practices worth adopting, lowest level first.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Ritual:
    name: str
    cadence: str
    owner: str
    output: str


RITUALS = (
    Ritual("release gate", "every change", "CI", "ship or hold"),
    Ritual("error analysis", "weekly", "eng lead", "top 3 fixes"),
    Ritual("judge calibration", "monthly", "eval owner", "kappa"),
    Ritual("annotation audit", "monthly", "eval owner", "gold"),
    Ritual("dataset refresh", "quarterly", "eval owner", "v+1"),
    Ritual("online review", "per launch", "product", "decision"),
)


@dataclass(frozen=True)
class Practice:
    key: str
    level: int  # 1 = foundations ... 4 = flywheel
    text: str


LADDER = (
    Practice("eval_set", 1, "versioned eval set, deduped"),
    Practice("graders", 1, "deterministic graders per task"),
    Practice("error_analysis", 1, "read failures, keep taxonomy"),
    Practice("ci_gate", 2, "regression gate blocks merges"),
    Practice("seeds", 2, "repeated seeds, flaky budget"),
    Practice("stats", 2, "intervals and paired tests"),
    Practice("judge", 3, "model judge calibrated to humans"),
    Practice("adversarial", 3, "injection and robustness probes"),
    Practice("online", 3, "A/B with guardrails"),
    Practice("loop", 4, "experiment log, dev/test gating"),
    Practice("flywheel", 4, "production data feeds eval sets"),
    Practice("owners", 4, "named owners for every ritual"),
)


def maturity(done: set[str]) -> int:
    """Highest level whose practices, and all below, are done."""
    level = 0
    for n in (1, 2, 3, 4):
        needed = {p.key for p in LADDER if p.level == n}
        if not needed <= done:
            break
        level = n
    return level


def next_steps(done: set[str], limit: int = 3) -> list[Practice]:
    """Missing practices, lowest level first."""
    missing = [p for p in LADDER if p.key not in done]
    return sorted(missing, key=lambda p: p.level)[:limit]


def unknown(done: set[str]) -> set[str]:
    """Practice keys that are not on the ladder (typos)."""
    return done - {p.key for p in LADDER}
