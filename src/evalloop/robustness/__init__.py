"""Robustness and adversarial evaluation.

Perturbation suites (typos, paraphrase, irrelevant context,
injected instructions, and a placebo control) rewrite tasks
without changing what a correct agent does. Metrics report
the robustness gap, answer consistency, the worst slice and
the balance between unsafe compliance and over-refusal.
"""

from evalloop.robustness.measure import (
    SEEDS,
    RobustnessResult,
    consistency,
    perturbed_suite,
    refusal_balance,
    robustness,
    run_clean,
    run_perturbed,
    same_answer,
    slice_success,
    success,
    worst_slice,
)
from evalloop.robustness.perturb import (
    PERTURBATIONS,
    Perturbation,
    distract,
    inject,
    origin,
    paraphrase,
    perturb,
    typos,
)

__all__ = [
    "PERTURBATIONS",
    "SEEDS",
    "Perturbation",
    "RobustnessResult",
    "consistency",
    "distract",
    "inject",
    "origin",
    "paraphrase",
    "perturb",
    "perturbed_suite",
    "refusal_balance",
    "robustness",
    "run_clean",
    "run_perturbed",
    "same_answer",
    "slice_success",
    "success",
    "typos",
    "worst_slice",
]
