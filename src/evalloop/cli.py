"""Command line: run, compare, gate and review agent configs.

``evalloop gate`` is the step CI runs: it grades the
candidate on the eval set, checks it against the committed
baseline file and exits non-zero when the gate fails, which
blocks the merge. Every number comes from the same graders.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from evalloop.core import BASELINE, IMPROVED, run_suite
from evalloop.gates import load_baseline, save_baseline
from evalloop.graders import DEFAULT_SUITE
from evalloop.program import (
    LADDER,
    Candidate,
    GuardedWorld,
    ReleasePolicy,
    as_baseline,
    gate,
    gather,
    maturity,
    next_steps,
    quality,
    review,
    unknown,
    versus,
)

V3 = IMPROVED.evolve(
    version="v3", rules=IMPROVED.rules | {"stop_after_answer"}
)
CANDIDATES = {
    "baseline": Candidate(BASELINE),
    "improved": Candidate(IMPROVED),  # the current release
    "v3": Candidate(V3),
    "v3-guarded": Candidate(
        V3.evolve(version="v3-guarded"), GuardedWorld
    ),
}
BASELINE_FILE = "evals/baseline.json"


def policy_for(args: argparse.Namespace) -> ReleasePolicy:
    return ReleasePolicy(tasks=args.tasks)


def cmd_run(args: argparse.Namespace) -> int:
    policy, cand = policy_for(args), CANDIDATES[args.config]
    ev = gather(cand, policy)
    print(f"{cand.version}: {quality(ev, policy).detail}")
    tasks = {t.id: t for t in policy.task_set()}
    runs = run_suite(cand.agent(), tasks.values(), policy.seeds)
    failed: Counter = Counter()
    for run in runs:
        grade = DEFAULT_SUITE(tasks[run.task_id], run)
        failed.update(r.split(":")[0] for r in grade.reasons)
    for check, n in failed.most_common(5):
        print(f"  {n:>4}  {check}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    policy = policy_for(args)
    a = gather(CANDIDATES[args.a], policy)
    b = gather(CANDIDATES[args.b], policy)
    print(f"{args.b} vs {args.a}: {versus(b, a).detail}")
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    policy = policy_for(args)
    ev = gather(CANDIDATES[args.config], policy)
    save_baseline(args.out, as_baseline(ev, policy))
    print(f"wrote {args.out} from {args.config}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    try:
        base = load_baseline(args.baseline)
    except FileNotFoundError:
        print(f"no baseline at {args.baseline}; run baseline")
        return 2
    policy = policy_for(args)
    ev = gather(CANDIDATES[args.candidate], policy)
    signal = gate(ev, base, policy)
    verdict = "PASS" if signal.passed else "FAIL"
    print(f"gate {ev.version} vs {base.version}: {verdict}")
    print(f"  {signal.detail}")
    return 0 if signal.passed else 1


def cmd_review(args: argparse.Namespace) -> int:
    result = review(
        CANDIDATES[args.candidate],
        CANDIDATES[args.current],
        policy_for(args),
    )
    print("\n".join(result.lines()))
    return 0 if result.decision == "ship" else 1


def cmd_maturity(args: argparse.Namespace) -> int:
    done = set(args.practices)
    if bad := unknown(done):
        print("unknown practices: " + ", ".join(sorted(bad)))
        print("known: " + ", ".join(p.key for p in LADDER))
        return 2
    print(f"maturity level {maturity(done)} of 4; next:")
    for p in next_steps(done):
        print(f"  L{p.level} {p.key}: {p.text}")
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="evalloop")
    ap.add_argument("--tasks", type=int, default=120)
    sub = ap.add_subparsers(dest="cmd", required=True)
    names = sorted(CANDIDATES)
    run = sub.add_parser("run", help="grade one config")
    run.add_argument(
        "--config", choices=names, default="improved"
    )
    cmp_ = sub.add_parser("compare", help="paired comparison")
    cmp_.add_argument("a", choices=names)
    cmp_.add_argument("b", choices=names)
    base = sub.add_parser("baseline", help="write baseline")
    base.add_argument(
        "--config", choices=names, default="improved"
    )
    base.add_argument("--out", default=BASELINE_FILE)
    gate_ = sub.add_parser("gate", help="CI regression gate")
    gate_.add_argument(
        "--candidate", choices=names, default="v3-guarded"
    )
    gate_.add_argument("--baseline", default=BASELINE_FILE)
    rev = sub.add_parser("review", help="full release review")
    rev.add_argument(
        "--candidate", choices=names, default="v3-guarded"
    )
    rev.add_argument(
        "--current", choices=names, default="improved"
    )
    mat = sub.add_parser("maturity", help="score practices")
    mat.add_argument("practices", nargs="*")
    return ap


COMMANDS = {
    "run": cmd_run,
    "compare": cmd_compare,
    "baseline": cmd_baseline,
    "gate": cmd_gate,
    "review": cmd_review,
    "maturity": cmd_maturity,
}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return COMMANDS[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
