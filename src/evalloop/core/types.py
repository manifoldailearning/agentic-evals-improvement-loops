"""Shared types: tasks, agent configs, and what a run produced.

An eval compares what an agent did against what it should
have done, under a known configuration. These types carry
exactly that: the task with its expectations, the config
that defines the agent, and the trajectory it produced.
All randomness flows through ``rng_for`` so runs replay.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field, replace
from typing import Any

CATEGORIES = (
    "order_status",
    "refund",
    "faq",
    "account",
    "escalation",
)
MODELS = ("small", "small-ft", "medium", "large")
RULES = (
    "verify_order_id",
    "cite_kb",
    "check_refund_policy",
    "ask_when_ambiguous",
    "stop_after_answer",
    "escalate_angry_customers",
)


def stable_hash(*parts: object) -> str:
    """Hex digest of the parts; identical across processes."""
    text = "|".join(str(p) for p in parts)
    return hashlib.sha256(text.encode()).hexdigest()


def rng_for(*parts: object) -> random.Random:
    """A random generator seeded only by the given parts."""
    return random.Random(int(stable_hash(*parts)[:16], 16))


@dataclass(frozen=True)
class Task:
    id: str
    input: str
    category: str
    expected: dict[str, Any]
    tags: frozenset[str] = frozenset()
    difficulty: float = 0.5  # 0 easy .. 1 hard

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "tags": sorted(self.tags)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Task:
        return cls(**{**d, "tags": frozenset(d["tags"])})


@dataclass(frozen=True)
class AgentConfig:
    version: str = "v0"
    model: str = "small"  # one of MODELS
    rules: frozenset[str] = frozenset()
    few_shots: int = 0
    max_steps: int = 6
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"unknown model {self.model!r}")
        unknown = set(self.rules) - set(RULES)
        if unknown:
            raise ValueError(f"unknown rules {sorted(unknown)}")

    def key(self) -> str:
        """Hash of behavior-relevant fields (not version)."""
        d = asdict(self)
        d.pop("version")
        d["rules"] = sorted(self.rules)
        return stable_hash(json.dumps(d, sort_keys=True))[:12]

    def evolve(self, **changes: Any) -> AgentConfig:
        """A copy with some fields changed."""
        return replace(self, **changes)


@dataclass(frozen=True)
class Step:
    kind: str  # tool | answer
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None  # dict for tools, text for answers


@dataclass
class Trajectory:
    task_id: str
    version: str
    config_key: str
    seed: int
    steps: list[Step]
    answer: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    cost: float
    faults: list[str] = field(default_factory=list)

    @property
    def tool_calls(self) -> list[Step]:
        return [s for s in self.steps if s.kind == "tool"]

    @property
    def ok(self) -> bool:
        """Ground truth: no fault. Graders must not read it."""
        return not self.faults

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trajectory:
        steps = [Step(**s) for s in d["steps"]]
        return cls(**{**d, "steps": steps})
