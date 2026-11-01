"""Dataset design for agent evals (Chapter 3).

Coverage matrices, stratified sampling, slices and golden
sets decide what an eval set tests; dedup and grouped splits
keep it honest; content-hashed versions make scores
traceable to the exact tasks that produced them.
"""

from evalloop.datasets.coverage import (
    PLAIN,
    TAGS,
    all_cells,
    allocate,
    cells,
    coverage,
    golden_set,
    random_sample,
    slice_scores,
    slices,
    stratified_sample,
    stratum,
    thin_cells,
)
from evalloop.datasets.hygiene import (
    dedup,
    duplicate_groups,
    find_twin,
    grouped_split,
    jaccard,
    leaks,
    normalize,
    random_split,
    token_set,
)
from evalloop.datasets.versioning import (
    DatasetDiff,
    DatasetVersion,
    diff,
    read_version,
    write_version,
)

__all__ = [
    "PLAIN",
    "TAGS",
    "DatasetDiff",
    "DatasetVersion",
    "all_cells",
    "allocate",
    "cells",
    "coverage",
    "dedup",
    "diff",
    "duplicate_groups",
    "find_twin",
    "golden_set",
    "grouped_split",
    "jaccard",
    "leaks",
    "normalize",
    "random_sample",
    "random_split",
    "read_version",
    "slice_scores",
    "slices",
    "stratified_sample",
    "stratum",
    "thin_cells",
    "token_set",
    "write_version",
]
