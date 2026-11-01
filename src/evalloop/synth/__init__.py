"""Synthetic eval tasks (Chapter 5).

Templates, personas and perturbations generate seeded
tasks whose expectations come from the world. Validity
checks and dedup filter them, and shift metrics compare the
result with a real-like set before anyone trusts it.
"""

from evalloop.synth.generate import (
    ASKS,
    PERSONAS,
    PERTURB_RATE,
    QUESTIONS,
    TEMPLATES,
    Persona,
    Template,
    apply_persona,
    from_template,
    make_synthetic,
    perturb,
)
from evalloop.synth.perturb import (
    PERTURBATIONS,
    inject,
    make_ambiguous,
    paraphrase,
    typos,
    verbose,
)
from evalloop.synth.quality import (
    Shift,
    distribution,
    problems,
    quality_filter,
    shift,
    tvd,
)

__all__ = [
    "ASKS",
    "PERSONAS",
    "PERTURBATIONS",
    "PERTURB_RATE",
    "QUESTIONS",
    "TEMPLATES",
    "Persona",
    "Shift",
    "Template",
    "apply_persona",
    "distribution",
    "from_template",
    "inject",
    "make_ambiguous",
    "make_synthetic",
    "paraphrase",
    "perturb",
    "problems",
    "quality_filter",
    "shift",
    "tvd",
    "typos",
    "verbose",
]
