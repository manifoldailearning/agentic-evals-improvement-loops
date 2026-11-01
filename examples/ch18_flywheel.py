"""Chapter 18: from production logs to a training set to trust.

Logged runs are harvested (passing runs as-is, failures
corrected by a teacher config), filtered through a funnel
and described by a dataset card. The tuning step is
SIMULATED: core's small-ft model stands in for the result.
"""

from __future__ import annotations

from collections import Counter

from evalloop.core import (
    BASELINE,
    IMPROVED,
    SimAgent,
    make_tasks,
    run_suite,
)
from evalloop.flywheel import (
    curate,
    finetune,
    harvest,
    make_card,
    production_logs,
)
from evalloop.loop import dataset_version, evaluate

EVAL_SET = make_tasks(200, seed=7)
EVAL_ID = dataset_version(EVAL_SET)
CAVEATS = (
    "the n-gram check misses paraphrased eval items",
    "faq and account are below the cap of 30 per category",
)
WHY = {
    "quality": "teacher could not fix",
    "dedup": "repeats and retries",
    "decontaminate": "copies of eval items",
    "balance": "cap 30 per category",
}


def show_harvest(logs, examples) -> None:
    src = Counter(e.source for e in examples)
    fixed = sum(
        e.source == "corrected" and e.passed for e in examples
    )
    print(f"Logged {len(logs)} runs of the baseline config")
    print(
        f"  {src['success']} passed as-is; {src['corrected']}"
        f" failed, teacher corrected {fixed}"
    )


def show_funnel(funnel) -> None:
    print("Curation funnel")
    before = funnel.steps[0][1]
    for step, n in funnel.steps:
        note = (
            f"-{before - n:<4} {WHY[step]}" if step in WHY else ""
        )
        print(f"  {step:<14}{n:>4}  {note}".rstrip())
        before = n


def main() -> None:
    logs = production_logs(400)
    runs = run_suite(SimAgent(BASELINE), logs)
    examples = harvest(logs, runs, teacher=IMPROVED)
    show_harvest(logs, examples)
    data, funnel = curate(examples, EVAL_SET)
    show_funnel(funnel)
    card = make_card(
        "support-ft-v1", data, funnel, EVAL_ID, CAVEATS
    )
    print("\n".join(card.lines()))
    tuned = finetune(BASELINE, data)
    before = evaluate(BASELINE, EVAL_SET)[0]
    after = evaluate(tuned, EVAL_SET)[0]
    print("Tuning is SIMULATED: small-ft is a fixed core model")
    print("  (failure rates x0.65); it never reads this data.")
    print(
        f"  eval set: small {before:.1%} -> small-ft {after:.1%}"
    )


if __name__ == "__main__":
    main()
