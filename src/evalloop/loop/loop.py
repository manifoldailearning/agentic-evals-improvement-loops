"""The improvement loop: triage, change, dev eval, test gate.

Changing an agent because a few runs looked better is how
regressions ship. Each iteration here triages dev failures,
tries the change that targets the most of them, keeps it
only if dev improves and the held-out test set confirms it,
and logs the decision with the config and dataset version.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from evalloop.core import AgentConfig, SimAgent, stable_hash
from evalloop.loop.triage import passed, triage


def apply_change(config: AgentConfig, change: str) -> AgentConfig:
    """Apply ``field:value``, e.g. ``rule:cite_kb``."""
    name, value = change.split(":")
    if name == "rule":
        return config.evolve(rules=config.rules | {value})
    if name == "model":
        return config.evolve(model=value)
    if name == "few_shots":
        return config.evolve(few_shots=int(value))
    if name == "temperature":
        return config.evolve(temperature=float(value))
    raise ValueError(f"unknown change {change!r}")


def evaluate(config: AgentConfig, tasks) -> tuple[float, list]:
    """Success rate under the local checker, plus the runs."""
    agent = SimAgent(config)
    runs = [agent.run(t) for t in tasks]
    wins = sum(map(passed, tasks, runs))
    return wins / len(tasks), runs


def split(tasks, dev_share: float = 0.4) -> tuple[list, list]:
    """A stable dev/test split by hashed task id."""
    dev, test = [], []
    for t in tasks:
        u = int(stable_hash("split", t.id)[:8], 16) / 16**8
        (dev if u < dev_share else test).append(t)
    return dev, test


def dataset_version(*task_sets) -> str:
    ids = [t.id for tasks in task_sets for t in tasks]
    return "ds-" + stable_hash(*ids)[:8]


@dataclass
class ExperimentRecord:
    iteration: int
    evidence: str  # the failure kind the change targets
    change: str
    version: str
    config_key: str
    dataset: str
    dev: tuple[float, float]  # before, after
    test: tuple[float, float]
    decision: str  # accept | reject
    reason: str


@dataclass
class ExperimentLog:
    records: list[ExperimentRecord] = field(default_factory=list)

    def add(self, record: ExperimentRecord) -> None:
        self.records.append(record)

    def accepted(self) -> list[ExperimentRecord]:
        return [r for r in self.records if r.decision == "accept"]

    def save(self, path: str | Path) -> None:
        rows = [json.dumps(asdict(r)) for r in self.records]
        Path(path).write_text("\n".join(rows) + "\n")


def before_after(old, new, tasks) -> tuple[float, float]:
    return evaluate(old, tasks)[0], evaluate(new, tasks)[0]


def gate(dev: tuple, test: tuple, min_gain: float) -> tuple:
    """Accept only a dev gain that the test set confirms."""
    if dev[1] - dev[0] < min_gain:
        return "reject", "no dev gain"
    if test[1] - test[0] < min_gain:
        return "reject", "dev gain not confirmed on test"
    return "accept", "dev and test improved"


@dataclass
class Loop:
    dev: list
    test: list
    min_gain: float = 0.005
    log: ExperimentLog = field(default_factory=ExperimentLog)
    tried: set[str] = field(default_factory=set)

    def trial(
        self, config: AgentConfig, change: str, evidence: str
    ) -> AgentConfig:
        """Eval a change on dev, gate it on test, log it."""
        self.tried.add(change)
        n = len(self.log.records) + 1
        cand = apply_change(config, change)
        cand = cand.evolve(version=f"v{n}")
        dev = before_after(config, cand, self.dev)
        test = before_after(config, cand, self.test)
        decision, why = gate(dev, test, self.min_gain)
        ds = dataset_version(self.dev, self.test)
        self.log.add(
            ExperimentRecord(
                n, evidence, change, cand.version, cand.key(),
                ds, dev, test, decision, why,
            )
        )  # fmt: skip
        return cand if decision == "accept" else config

    def step(self, config: AgentConfig) -> AgentConfig:
        """Triage dev failures and try the top hypothesis."""
        _, runs = evaluate(config, self.dev)
        ideas = triage(self.dev, runs, frozenset(self.tried))
        if not ideas:
            return config
        top = ideas[0]
        return self.trial(config, top.change, top.evidence)

    def sweep(
        self, config: AgentConfig, name: str, values
    ) -> AgentConfig:
        """Tune one knob on dev, then gate the winner."""
        changes = [f"{name}:{v}" for v in values]
        scores = [
            evaluate(apply_change(config, c), self.dev)[0]
            for c in changes
        ]
        best = changes[scores.index(max(scores))]
        return self.trial(config, best, f"sweep of {len(values)}")

    def run(self, config: AgentConfig, iterations: int):
        """Several triage-driven steps from ``config``."""
        for _ in range(iterations):
            config = self.step(config)
        return config
