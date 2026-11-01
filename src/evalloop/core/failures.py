"""Failure modes: how, and how often, the simulated agent errs.

Agents fail in recurring, nameable ways. Each mode applies
to some tasks, has a base rate scaled by difficulty and tags,
and is cut by specific config features: its rule removes
80%, a larger model scales every mode down, few-shot
examples reduce tool mistakes, and temperature adds noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evalloop.core.types import AgentConfig, Task, rng_for

MODEL_SCALE = {  # small-ft: small tuned on curated data
    "small": 1.0,
    "small-ft": 0.65,
    "medium": 0.5,
    "large": 0.3,
}
RULE_CUT = 0.2  # a rule keeps 20% of its failure mode
SHOT_DECAY = 0.8  # each few-shot example keeps 80%
SHOTS = {"wrong_tool", "missing_arg"}  # helped by examples
HOT = {"wrong_tool", "hallucinated_fact"}  # worse when hot
SCOPES = {  # which tasks a mode can affect
    "any": lambda t: bool(t.expected["tools"]),
    "order": lambda t: "lookup_order" in t.expected["tools"],
    "chain": lambda t: len(t.expected["tools"]) > 1,
    "kb": lambda t: bool(t.expected["kb"]),
    "over": lambda t: t.expected["eligible"] is False,
    "vague": lambda t: t.expected["clarify"],
    "angry": lambda t: "angry" in t.tags,
}


@dataclass(frozen=True)
class FailureMode:
    name: str
    base: float
    rule: str  # the prompt rule that fixes it, or ""
    scope: str  # key into SCOPES
    boost: dict[str, float] = field(default_factory=dict)


FM = FailureMode
FAILURE_MODES = (
    FM("wrong_tool", 0.05, "", "any"),
    FM("missing_arg", 0.06, "verify_order_id", "order",
       {"long_input": 2.0}),
    FM("hallucinated_fact", 0.05, "", "any", {"faq": 1.5}),
    FM("policy_violation", 0.5, "check_refund_policy", "over",
       {"adversarial": 1.4}),
    FM("no_citation", 0.2, "cite_kb", "kb"),
    FM("loop", 0.025, "stop_after_answer", "any",
       {"multi_step": 2.0, "long_input": 2.0}),
    FM("premature_answer", 0.06, "", "chain",
       {"adversarial": 1.5}),
    FM("ignored_ambiguity", 0.55, "ask_when_ambiguous", "vague"),
    FM("bad_tone", 0.22, "escalate_angry_customers", "angry"),
)  # fmt: skip
MODES = tuple(m.name for m in FAILURE_MODES)


def mode_prob(
    m: FailureMode, task: Task, c: AgentConfig
) -> float:
    """Probability that mode ``m`` occurs on ``task``."""
    if not SCOPES[m.scope](task):
        return 0.0
    p = m.base * (0.5 + task.difficulty)
    for key in {task.category, *task.tags}:
        p *= m.boost.get(key, 1.0)
    p *= MODEL_SCALE[c.model]
    if m.rule in c.rules:
        p *= RULE_CUT
    if m.name in SHOTS:
        p *= SHOT_DECAY**c.few_shots
    if m.name in HOT:
        p *= 1 + 0.5 * c.temperature
    return round(min(p, 0.95), 4)


def failure_probs(task: Task, c: AgentConfig) -> dict[str, float]:
    """Pure: each mode's probability for this task and config."""
    return {m.name: mode_prob(m, task, c) for m in FAILURE_MODES}


def draw(
    task_id: str, mode: str, seed: int, temp: float
) -> float:
    """A uniform draw shared by every config.

    At temperature 0 the seed is ignored, so reruns agree.
    Above 0, that share of draws is re-rolled per seed.
    """
    if rng_for(task_id, mode, seed, "pick").random() < temp:
        return rng_for(task_id, mode, seed, "roll").random()
    return rng_for(task_id, mode, "fixed").random()


def sample_faults(
    task: Task, c: AgentConfig, seed: int
) -> list[str]:
    """The modes that occur in this run, in MODES order."""
    probs = failure_probs(task, c)
    t = c.temperature
    return [
        m
        for m, p in probs.items()
        if draw(task.id, m, seed, t) < p
    ]
