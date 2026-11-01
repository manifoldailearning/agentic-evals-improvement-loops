"""Chapter 20: the release review and the operating model.

Candidate v3 adds a rule. The CI gate passes it, yet the
full review holds it: injected instructions still move
money. A larger model halves the harm and breaks the
budget. A policy check in the refund tool removes the harm
at no cost, and the review ships it.
"""

from __future__ import annotations

from evalloop.core import IMPROVED
from evalloop.program import (
    Candidate,
    GuardedWorld,
    maturity,
    next_steps,
    review,
)

V3 = IMPROVED.evolve(
    version="v3", rules=IMPROVED.rules | {"stop_after_answer"}
)
CANDIDATES = (
    Candidate(V3),
    Candidate(V3.evolve(version="v3-large", model="large")),
    Candidate(V3.evolve(version="v3-guarded"), GuardedWorld),
)
DONE = {"eval_set", "graders", "error_analysis", "ci_gate"}


def failing(line: str) -> bool:
    return "FAIL" in line or "decision" in line


def main() -> None:
    current = Candidate(IMPROVED)
    for i, cand in enumerate(CANDIDATES):
        lines = review(cand, current).lines()
        if i == 1:  # show only what changed
            lines = [x for x in lines if failing(x)]
            lines.insert(0, f"Review {cand.version} (failures)")
        print("\n".join(lines))
    level = maturity(DONE)
    print(f"Team maturity: level {level} of 4; next:")
    for p in next_steps(DONE):
        print(f"  L{p.level} {p.key}: {p.text}")


if __name__ == "__main__":
    main()
