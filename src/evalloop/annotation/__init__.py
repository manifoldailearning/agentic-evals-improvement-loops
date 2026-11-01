"""Annotation operations for human labels at scale.

Chapter 9. Guidelines as data, a rater pool with skill
levels, overlapping assignment, Fleiss' kappa, adjudication,
gold questions and throughput and cost accounting.
"""

from evalloop.annotation.ops import (
    CostReport,
    Labels,
    adjudicate,
    assign,
    collect,
    contested,
    cost_report,
    flagged,
    fleiss_kappa,
    gold_scores,
)
from evalloop.annotation.raters import (
    EXPERT,
    GUIDE_V1,
    GUIDE_V2,
    POOL,
    RULE_TEXT,
    Annotator,
    Guideline,
    annotate,
    coverage,
)

__all__ = [
    "EXPERT",
    "GUIDE_V1",
    "GUIDE_V2",
    "POOL",
    "RULE_TEXT",
    "Annotator",
    "CostReport",
    "Guideline",
    "Labels",
    "adjudicate",
    "annotate",
    "assign",
    "collect",
    "contested",
    "cost_report",
    "coverage",
    "flagged",
    "fleiss_kappa",
    "gold_scores",
]
