"""Coverage: what an eval set tests, and how to pick from it.

A set of easy order lookups says nothing about refunds under
attack. A coverage matrix counts tasks per category and tag,
exposes thin cells, and drives stratified sampling, named
slices and a small, hard golden set.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable

from evalloop.core import CATEGORIES, Task, rng_for

PLAIN = "plain"  # the tag column for untagged tasks
TAGS = (
    PLAIN,
    "ambiguous",
    "adversarial",
    "angry",
    "long_input",
    "multi_step",
)
Cell = tuple[str, str]


def cells(task: Task) -> list[Cell]:
    """The (category, tag) cells a task falls in."""
    tags = sorted(task.tags) or [PLAIN]
    return [(task.category, tag) for tag in tags]


def coverage(tasks: Iterable[Task]) -> Counter[Cell]:
    """Task count per (category, tag) cell."""
    return Counter(cell for t in tasks for cell in cells(t))


def thin_cells(
    matrix: Counter[Cell],
    minimum: int = 3,
    wanted: Iterable[Cell] | None = None,
) -> list[Cell]:
    """Cells under ``minimum`` tasks; default: covered cells."""
    pool = sorted(matrix) if wanted is None else wanted
    return [c for c in pool if matrix[c] < minimum]


def stratum(task: Task) -> str:
    """Category plus the full tag combination."""
    return "/".join([task.category, *sorted(task.tags)])


def allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Exactly ``n`` seats: one per stratum, rest by size.

    Leftover seats go to the largest remainders (Hamilton).
    """
    if not len(sizes) <= n <= sum(sizes.values()):
        raise ValueError(f"cannot fill {n} from {sizes}")
    spare = {k: v - 1 for k, v in sizes.items()}
    total = sum(spare.values()) or 1
    exact = {
        k: (n - len(sizes)) * v / total for k, v in spare.items()
    }
    quota = {k: 1 + int(x) for k, x in exact.items()}
    left = n - sum(quota.values())
    by_rest = sorted(
        exact, key=lambda k: (int(exact[k]) - exact[k], k)
    )
    for k in by_rest[:left]:
        quota[k] += 1
    return quota


def stratified_sample(
    tasks: list[Task],
    n: int,
    key: Callable[[Task], str] = stratum,
    seed: int = 0,
) -> list[Task]:
    """``n`` tasks with every stratum present."""
    strata: dict[str, list[Task]] = defaultdict(list)
    for t in tasks:
        strata[key(t)].append(t)
    sizes = {k: len(v) for k, v in strata.items()}
    picked = []
    for k, quota in sorted(allocate(sizes, n).items()):
        rng = rng_for("sample", seed, k)
        picked += rng.sample(strata[k], quota)
    return sorted(picked, key=lambda t: t.id)


def random_sample(tasks: list[Task], n: int, seed=0) -> list:
    """A plain random sample, for comparison."""
    picked = rng_for("random-sample", seed).sample(tasks, n)
    return sorted(picked, key=lambda t: t.id)


def slices(tasks: Iterable[Task]) -> dict[str, list[Task]]:
    """Named subsets: one per category and one per tag."""
    named: dict[str, list[Task]] = defaultdict(list)
    for t in tasks:
        named[f"category={t.category}"].append(t)
        for tag in sorted(t.tags):
            named[f"tag={tag}"].append(t)
    return dict(sorted(named.items()))


def slice_scores(
    named: dict[str, list[Task]], score: dict[str, float]
) -> dict[str, float]:
    """Mean per-task score inside each slice."""
    return {
        name: sum(score[t.id] for t in ts) / len(ts)
        for name, ts in named.items()
    }


def golden_set(tasks: list[Task], per_cell: int = 1) -> list:
    """The hardest ``per_cell`` tasks from every covered cell."""
    chosen: dict[str, Task] = {}
    for cell in sorted(coverage(tasks)):
        pool = [
            t
            for t in tasks
            if cell in cells(t) and t.id not in chosen
        ]
        pool.sort(key=lambda t: (-t.difficulty, t.id))
        chosen.update((t.id, t) for t in pool[:per_cell])
    return sorted(chosen.values(), key=lambda t: t.id)


def all_cells(tags=TAGS) -> list[Cell]:
    """Every category x tag combination."""
    return [(c, tag) for c in CATEGORIES for tag in tags]
