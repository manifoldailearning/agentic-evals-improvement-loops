"""Manifests and a result store: make every eval run replayable.

A number without its inputs cannot be trusted or compared.
Each run gets a manifest naming the dataset, config, grader
and code versions; results are stored as JSONL beside it,
and comparing two runs reports both what changed in the
inputs and which tasks flipped between pass and fail.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import evalloop
from evalloop.core import AgentConfig, stable_hash
from evalloop.infra.runner import GRADER, Result


def code_version() -> str:
    """Package version plus a hash of the agent's source code."""
    src = Path(evalloop.__file__).parent / "core"
    texts = [p.read_text() for p in sorted(src.glob("*.py"))]
    return f"{evalloop.__version__}+{stable_hash(*texts)[:8]}"


def dataset_hash(tasks) -> str:
    rows = [
        json.dumps(t.to_dict(), sort_keys=True) for t in tasks
    ]
    return "ds-" + stable_hash(*rows)[:8]


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    dataset: str
    config: str  # config.key(): behavior, not the label
    version: str  # the config's human label
    grader: str
    code: str
    seeds: tuple[int, ...]

    def differs(self, other: RunManifest) -> list[str]:
        """Input fields that differ between two runs."""
        names = ("dataset", "config", "grader", "code", "seeds")
        mine, theirs = asdict(self), asdict(other)
        return [n for n in names if mine[n] != theirs[n]]


def make_manifest(config: AgentConfig, tasks, seeds):
    """Record every input of a run; the run id hashes them."""
    inputs = {
        "dataset": dataset_hash(tasks),
        "config": config.key(),
        "grader": GRADER,
        "code": code_version(),
        "seeds": tuple(seeds),
    }
    run_id = "run-" + stable_hash(*inputs.values())[:6]
    return RunManifest(run_id, version=config.version, **inputs)


@dataclass
class ResultStore:
    root: Path

    def save(self, manifest: RunManifest, results) -> Path:
        """One JSONL file per run: manifest first, then rows."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{manifest.run_id}.jsonl"
        rows = [asdict(manifest)] + [asdict(r) for r in results]
        path.write_text("\n".join(map(json.dumps, rows)) + "\n")
        return path

    def load(self, run_id: str) -> tuple[RunManifest, list]:
        lines = (self.root / f"{run_id}.jsonl").read_text()
        head, *rows = map(json.loads, lines.splitlines())
        head["seeds"] = tuple(head["seeds"])
        results = [
            Result(**{**r, "evidence": tuple(r["evidence"])})
            for r in rows
        ]
        return RunManifest(**head), results

    def runs(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.jsonl"))


@dataclass(frozen=True)
class Comparison:
    changed: list[str]  # manifest fields that differ
    before: float
    after: float
    fixed: list[str]  # (task, seed) that went fail -> pass
    broken: list[str]  # (task, seed) that went pass -> fail


def compare(a: tuple, b: tuple) -> Comparison:
    """Diff two stored runs on the (task, seed) pairs in both."""
    (ma, ra), (mb, rb) = a, b
    old = {(r.task_id, r.seed): r.passed for r in ra}
    new = {(r.task_id, r.seed): r.passed for r in rb}
    both = sorted(old.keys() & new.keys())
    if not both:
        raise ValueError("the runs share no (task, seed) pairs")
    flips = [
        (k, old[k], new[k]) for k in both if old[k] != new[k]
    ]
    return Comparison(
        ma.differs(mb),
        sum(old[k] for k in both) / len(both),
        sum(new[k] for k in both) / len(both),
        [f"{t}/{s}" for (t, s), _, now in flips if now],
        [f"{t}/{s}" for (t, s), _, now in flips if not now],
    )
