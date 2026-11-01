"""Calibrating judges against human labels.

Chapter 8. Simulated human labels, confusion matrix, Cohen's
kappa, precision and recall of fail detection, bias tests,
threshold selection on a calibration set, and drift checks.
"""

from evalloop.calibration.agreement import (
    SUBTLE,
    Confusion,
    choose_threshold,
    cohen_kappa,
    confusion,
    human_label,
    kappa_by_threshold,
    pearson,
    split_half,
)
from evalloop.calibration.bias import (
    FILLER,
    DriftReport,
    drift_check,
    mean_score,
    padding_effect,
    position_flip_rate,
    self_preference_gap,
)

__all__ = [
    "FILLER",
    "SUBTLE",
    "Confusion",
    "DriftReport",
    "choose_threshold",
    "cohen_kappa",
    "confusion",
    "drift_check",
    "human_label",
    "kappa_by_threshold",
    "mean_score",
    "padding_effect",
    "pearson",
    "position_flip_rate",
    "self_preference_gap",
    "split_half",
]
