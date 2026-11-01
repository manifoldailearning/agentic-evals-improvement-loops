"""Guidelines and annotators: where human labels come from.

Human labels are a product with inputs: a written guideline
and people of varying skill. The guideline is data, a map
from failure type to instruction; where it is silent, each
rater falls back on a private opinion, and disagreement
follows. Raters are simulated from ground-truth faults.
"""

from __future__ import annotations

from dataclasses import dataclass

from evalloop.core import MODES, Trajectory, rng_for

SUBTLE = {"hallucinated_fact"}  # harder to notice


@dataclass(frozen=True)
class Guideline:
    version: str
    rules: dict[str, str]  # failure mode -> instruction

    def covers(self, mode: str) -> bool:
        return mode in self.rules


RULE_TEXT = {
    "wrong_tool": "Fail if the agent used the wrong tool.",
    "missing_arg": "Fail if a tool call lacks a needed value.",
    "hallucinated_fact": "Fail any fact no tool result shows.",
    "policy_violation": "Fail refunds over 200 or 30 days.",
    "no_citation": "Fail help-center facts with no [kb:] cite.",
    "loop": "Fail if the agent never answers.",
    "premature_answer": "Fail if it answers before acting.",
    "ignored_ambiguity": "Fail guesses; it must ask for the id.",
    "bad_tone": "Fail replies that dismiss an upset customer.",
}
SILENT_IN_V1 = ("no_citation", "premature_answer", "bad_tone")
GUIDE_V1 = Guideline(
    "v1",
    {m: t for m, t in RULE_TEXT.items() if m not in SILENT_IN_V1},
)
GUIDE_V2 = Guideline("v2", dict(RULE_TEXT))


@dataclass(frozen=True)
class Annotator:
    id: str
    skill: float  # chance of noticing a covered fault
    items_per_hour: float
    hourly_rate: float
    strictness: float  # chance an uncovered fault counts

    def counts_as_failure(self, mode: str, guide: Guideline):
        """Covered modes always count; others by opinion."""
        if guide.covers(mode):
            return True
        opinion = rng_for("opinion", self.id, mode).random()
        return opinion < self.strictness


POOL = (
    Annotator("rater-1", 0.95, 25, 45.0, 0.6),
    Annotator("rater-2", 0.90, 30, 35.0, 0.3),
    Annotator("rater-3", 0.88, 30, 35.0, 0.8),
    Annotator("rater-4", 0.85, 40, 25.0, 0.5),
    Annotator("rater-5", 0.65, 60, 18.0, 0.2),
)
EXPERT = Annotator("expert", 0.98, 15, 90.0, 0.5)


def annotate(
    rater: Annotator, traj: Trajectory, guide: Guideline
) -> bool:
    """A simulated pass (True) or fail label for one run."""
    rng = rng_for(
        "label",
        rater.id,
        traj.task_id,
        traj.config_key,
        guide.version,
    )
    if not traj.faults:
        return rng.random() >= (1 - rater.skill) / 4
    for mode in traj.faults:
        notice = rater.skill * (0.75 if mode in SUBTLE else 1.0)
        seen = rng.random() < notice
        if seen and rater.counts_as_failure(mode, guide):
            return False
    return True


def coverage(guide: Guideline) -> str:
    return f"{len(guide.rules)} of {len(MODES)} failure modes"
