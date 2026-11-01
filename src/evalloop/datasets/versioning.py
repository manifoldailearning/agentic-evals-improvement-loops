"""Versioning: name every eval set by its content.

A score means nothing unless you know which tasks produced
it. A ``DatasetVersion`` carries a content hash that changes
when any task changes, a manifest describing what is inside,
and a diff against an earlier version.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from evalloop.core import (
    Task,
    load_tasks,
    save_tasks,
    stable_hash,
)


def task_line(task: Task) -> str:
    """Canonical JSON for one task (sorted keys)."""
    return json.dumps(task.to_dict(), sort_keys=True)


@dataclass(frozen=True)
class DatasetVersion:
    name: str
    tasks: tuple[Task, ...]
    notes: str = ""

    @property
    def lines(self) -> dict[str, str]:
        """Task id -> canonical JSON line."""
        return {t.id: task_line(t) for t in self.tasks}

    @property
    def content_hash(self) -> str:
        """Order-independent hash of every task's content."""
        return stable_hash(*sorted(self.lines.values()))[:12]

    def manifest(self) -> dict:
        """What a reader needs to trust a reported score."""
        cats = Counter(t.category for t in self.tasks)
        tags = Counter(g for t in self.tasks for g in t.tags)
        return {
            "name": self.name,
            "hash": self.content_hash,
            "size": len(self.tasks),
            "categories": dict(sorted(cats.items())),
            "tags": dict(sorted(tags.items())),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class DatasetDiff:
    added: list[str]
    removed: list[str]
    changed: list[str]

    def summary(self) -> str:
        return (
            f"+{len(self.added)} added, -{len(self.removed)}"
            f" removed, ~{len(self.changed)} changed"
        )


def diff(old: DatasetVersion, new: DatasetVersion) -> DatasetDiff:
    """Task ids added, removed, or edited in place."""
    a, b = old.lines, new.lines
    return DatasetDiff(
        added=sorted(b.keys() - a.keys()),
        removed=sorted(a.keys() - b.keys()),
        changed=sorted(
            k for k in a.keys() & b.keys() if a[k] != b[k]
        ),
    )


def write_version(version: DatasetVersion, folder: Path) -> None:
    """Save tasks.jsonl plus manifest.json side by side."""
    folder.mkdir(parents=True, exist_ok=True)
    save_tasks(version.tasks, folder / "tasks.jsonl")
    text = json.dumps(version.manifest(), indent=2)
    (folder / "manifest.json").write_text(text + "\n")


def read_version(folder: Path) -> DatasetVersion:
    """Load a saved version; fail loudly if it was edited."""
    manifest = json.loads((folder / "manifest.json").read_text())
    tasks = tuple(load_tasks(folder / "tasks.jsonl"))
    version = DatasetVersion(
        manifest["name"], tasks, manifest["notes"]
    )
    if version.content_hash != manifest["hash"]:
        raise ValueError(f"{folder}: tasks do not match manifest")
    return version
