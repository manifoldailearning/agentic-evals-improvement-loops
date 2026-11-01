"""Online evaluation: experiments on live (simulated) traffic.

Offline evals cannot see everything users do. This package
assigns users to variants by stable hashing, reads out A/B
lifts with intervals and guardrails, monitors a running test
with alpha spending, interleaves rankers, and checks whether
an early lift is only novelty.
"""

from evalloop.online.interleave import (
    credit,
    interleaving_result,
    team_draft,
)
from evalloop.online.sequential import (
    Look,
    boundaries,
    false_alarm_rate,
    monitor,
    obf_spending,
    z_statistic,
)
from evalloop.online.traffic import (
    Guardrail,
    Novelty,
    Session,
    assign,
    check_guardrail,
    lift,
    mean_difference,
    novelty_check,
    resolved,
    simulate_traffic,
    split,
)

__all__ = [
    "Guardrail",
    "Look",
    "Novelty",
    "Session",
    "assign",
    "boundaries",
    "check_guardrail",
    "credit",
    "false_alarm_rate",
    "interleaving_result",
    "lift",
    "mean_difference",
    "monitor",
    "novelty_check",
    "obf_spending",
    "resolved",
    "simulate_traffic",
    "split",
    "team_draft",
    "z_statistic",
]
