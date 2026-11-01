"""Simulated model judges: rubric, pairwise and reference-based.

A judge reads evidence well but not perfectly, and its habits
are documented, tunable biases: ``leniency`` lifts scores,
``length_bias`` rewards long replies, ``self_preference``
favours its own model tier, ``position_bias`` favours slot A
and ``noise`` makes repeat calls disagree.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

from evalloop.core import (
    KB,
    Task,
    Trajectory,
    rng_for,
    stable_hash,
)
from evalloop.judge.rubric import RUBRIC, find_issues

SHARPNESS = {"small": 0.6, "medium": 0.8, "large": 0.92}
PIVOT = 60  # reply length (chars) with no length effect


@dataclass(frozen=True)
class JudgeConfig:
    name: str = "judge-v1"
    model: str = "medium"  # sets how often issues are noticed
    leniency: float = 0.4  # points added to every score
    length_bias: float = 1.0  # points per 100 chars over PIVOT
    self_preference: float = 0.5  # bonus for same-tier authors
    position_bias: float = 0.6  # pairwise bonus for slot A
    noise: float = 0.5  # sd of per-call score noise
    pass_at: int = 4  # scores at or above this pass

    def key(self) -> str:
        return stable_hash(*asdict(self).values())[:12]


@dataclass(frozen=True)
class JudgeResult:
    score: int  # 1..5
    passed: bool
    rationale: str


def run_id(traj: Trajectory) -> str:
    return f"{traj.task_id}/{traj.config_key}/{traj.seed}"


def length_term(answer: str, bias: float) -> float:
    return bias * (len(answer) - PIVOT) / 100


@dataclass(frozen=True)
class RubricJudge:
    """5 minus noticed issue weights, plus biases and noise."""

    config: JudgeConfig = JudgeConfig()
    rubric: tuple = RUBRIC

    def noticed(self, task, traj, sample: int) -> list[str]:
        """Issues the judge spots; each is missed sometimes."""
        found, c = find_issues(task, traj), self.config
        notes = []
        for crit in self.rubric:
            for issue in found[crit.name][:1]:
                rng = rng_for(
                    c.key(), run_id(traj), crit.name, sample
                )
                if rng.random() < SHARPNESS[c.model]:
                    notes.append(f"{crit.name}: {issue}")
        return notes

    def quality(self, task, traj, sample=0, author=None) -> float:
        """Perceived quality before leniency and noise."""
        weights = {c.name: c.weight for c in self.rubric}
        notes = self.noticed(task, traj, sample)
        q = 5 - sum(weights[n.split(":")[0]] for n in notes)
        q += length_term(traj.answer, self.config.length_bias)
        if author == self.config.model:
            q += self.config.self_preference
        return q

    def grade(
        self,
        task: Task,
        traj: Trajectory,
        sample: int = 0,
        author: str | None = None,
    ) -> JudgeResult:
        """Score a run 1-5 against the rubric, with rationale."""
        c = self.config
        rng = rng_for(c.key(), run_id(traj), "noise", sample)
        raw = self.quality(task, traj, sample, author)
        raw += c.leniency + rng.gauss(0, c.noise)
        score = min(5, max(1, round(raw)))
        notes = self.noticed(task, traj, sample)
        why = "; ".join(notes) or "meets every criterion"
        return JudgeResult(score, score >= c.pass_at, why)


@dataclass(frozen=True)
class PairwiseJudge:
    """Which of two runs is better; ties within ``tie_band``."""

    config: JudgeConfig = JudgeConfig()
    tie_band: float = 0.5

    def compare(
        self, task: Task, a: Trajectory, b: Trajectory, sample=0
    ) -> str:
        """'A', 'B' or 'tie'; slot A gets ``position_bias``."""
        rubric = RubricJudge(self.config)
        c = self.config
        rng = rng_for(c.key(), run_id(a), run_id(b), sample)
        diff = rubric.quality(task, a, sample)
        diff -= rubric.quality(task, b, sample)
        diff += c.position_bias + rng.gauss(0, c.noise)
        if abs(diff) < self.tie_band:
            return "tie"
        return "A" if diff > 0 else "B"


def compare_both_orders(
    judge: PairwiseJudge, task: Task, a, b, sample: int = 0
) -> str:
    """Ask twice with slots swapped; disagreement is a tie."""
    first = judge.compare(task, a, b, sample)
    second = judge.compare(task, b, a, sample)
    unswapped = {"A": "B", "B": "A", "tie": "tie"}[second]
    return first if first == unswapped else "tie"


def majority_vote(
    judge: RubricJudge, task: Task, traj, samples: int = 5
) -> JudgeResult:
    """Median score over samples; pass if most samples pass."""
    results = [judge.grade(task, traj, s) for s in range(samples)]
    score = int(statistics.median_low(r.score for r in results))
    passes = sum(r.passed for r in results)
    middle = next(r for r in results if r.score == score)
    return JudgeResult(
        score, passes * 2 > samples, middle.rationale
    )


def length_corrected(
    score: float, answer: str, slope: float
) -> float:
    """Remove an estimated length effect (slope from ch08)."""
    return score - length_term(answer, slope)


def reference_answer(task: Task) -> list[str]:
    """Key facts a correct reply states: listed facts plus KB."""
    kb = task.expected["kb"]
    return task.expected["answer_contains"] + (
        [KB[kb]] if kb else []
    )


@dataclass(frozen=True)
class ReferenceJudge:
    """Scores overlap with a reference; blind to extra claims."""

    config: JudgeConfig = JudgeConfig()

    def grade(
        self, task: Task, traj: Trajectory, sample: int = 0
    ) -> JudgeResult:
        c, facts = self.config, reference_answer(task)
        text = traj.answer.lower()
        hits = sum(f.lower() in text for f in facts)
        share = hits / len(facts) if facts else 1.0
        rng = rng_for(c.key(), run_id(traj), "ref", sample)
        raw = 1 + 4 * share + c.leniency + rng.gauss(0, c.noise)
        score = min(5, max(1, round(raw)))
        why = f"{hits}/{len(facts)} reference facts present"
        return JudgeResult(score, score >= c.pass_at, why)
