"""Chapter 17: prompt and context optimization.

Ablate rules to see what each one buys, choose few-shot
examples by similarity or diversity, search the config space
greedily under a budget of dev evals, guard against dev-set
overfitting, and map cost against quality across model tiers.
"""

from evalloop.optimize.ablation import (
    ablate,
    compare_shots,
    diverse_shots,
    random_shots,
    relevant,
    similar_shots,
    similarity,
)
from evalloop.optimize.search import (
    Point,
    SearchResult,
    frontier,
    greedy_search,
    measure,
    moves,
    optimism,
    pareto,
)

__all__ = [
    "Point",
    "SearchResult",
    "ablate",
    "compare_shots",
    "diverse_shots",
    "frontier",
    "greedy_search",
    "measure",
    "moves",
    "optimism",
    "pareto",
    "random_shots",
    "relevant",
    "similar_shots",
    "similarity",
]
