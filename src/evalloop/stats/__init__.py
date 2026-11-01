"""Statistics for evals: uncertainty, comparisons, planning.

An eval score is an estimate from a finite task sample.
This package puts intervals around it, tests whether one
config beats another on the same tasks, corrects for many
comparisons, and sizes the eval before it is run.
"""

from evalloop.stats.intervals import (
    Interval,
    bootstrap_interval,
    difference_interval,
    paired_bootstrap,
    paired_differences,
    wilson_interval,
    z_score,
)
from evalloop.stats.planning import (
    min_detectable_effect,
    paired_sample_size,
    power_unpaired,
    sample_size,
)
from evalloop.stats.testing import (
    Comparison,
    McNemar,
    cohens_h,
    compare,
    holm,
    mcnemar,
    normal_p_value,
    sign_test,
    two_proportion_test,
)

__all__ = [
    "Comparison",
    "Interval",
    "McNemar",
    "bootstrap_interval",
    "cohens_h",
    "compare",
    "difference_interval",
    "holm",
    "mcnemar",
    "min_detectable_effect",
    "normal_p_value",
    "paired_bootstrap",
    "paired_differences",
    "paired_sample_size",
    "power_unpaired",
    "sample_size",
    "sign_test",
    "two_proportion_test",
    "wilson_interval",
    "z_score",
]
