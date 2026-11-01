"""Error analysis for agent trajectories (Chapter 4).

Open coding writes free-text notes from trajectory
evidence; axial coding groups them into a taxonomy; counts,
Pareto and saturation say where to look and when to stop;
each category links to a candidate fix. Agreement with
ground-truth faults validates the whole process.
"""

from evalloop.analysis.annotate import (
    NOTICE,
    PHRASES,
    Note,
    check_amounts,
    check_conduct,
    check_quotes,
    check_status,
    check_tools,
    noticed,
    observe,
    open_code,
    review,
    write_note,
)
from evalloop.analysis.taxonomy import (
    CODEBOOK,
    FIXES,
    UNCODED,
    apply_fix,
    axial_code,
    category_counts,
    code_notes,
    describe_fix,
    pareto,
    saturation,
)
from evalloop.analysis.validate import (
    CATEGORY_MODE,
    Agreement,
    agreement,
    category_agreement,
    detection_agreement,
)

__all__ = [
    "CATEGORY_MODE",
    "CODEBOOK",
    "FIXES",
    "NOTICE",
    "PHRASES",
    "UNCODED",
    "Agreement",
    "Note",
    "agreement",
    "apply_fix",
    "axial_code",
    "category_agreement",
    "category_counts",
    "check_amounts",
    "check_conduct",
    "check_quotes",
    "check_status",
    "check_tools",
    "code_notes",
    "describe_fix",
    "detection_agreement",
    "noticed",
    "observe",
    "open_code",
    "pareto",
    "review",
    "saturation",
    "write_note",
]
