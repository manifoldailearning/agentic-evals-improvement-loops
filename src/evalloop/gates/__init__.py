"""Regression gates: stop a change that makes the agent worse.

A CI gate compares a candidate's eval results with a stored
baseline under explicit tolerances. Repeated seeds expose
flaky tasks, critical tasks must pass every time, and known
flaky tasks can be quarantined until a fixed expiry date.
"""

from evalloop.gates.flaky import (
    Outcomes,
    Quarantine,
    apply_quarantine,
    collect,
    is_flaky,
    pass_at_k,
    pass_hat_k,
    seed_spread,
    summarize,
)
from evalloop.gates.gate import (
    Baseline,
    Check,
    GatePolicy,
    GateReport,
    Tolerance,
    check_critical,
    check_flaky,
    check_metric,
    check_quarantine,
    check_regressions,
    load_baseline,
    run_gate,
    save_baseline,
)

__all__ = [
    "Baseline",
    "Check",
    "GatePolicy",
    "GateReport",
    "Outcomes",
    "Quarantine",
    "Tolerance",
    "apply_quarantine",
    "check_critical",
    "check_flaky",
    "check_metric",
    "check_quarantine",
    "check_regressions",
    "collect",
    "is_flaky",
    "load_baseline",
    "pass_at_k",
    "pass_hat_k",
    "run_gate",
    "save_baseline",
    "seed_spread",
    "summarize",
]
