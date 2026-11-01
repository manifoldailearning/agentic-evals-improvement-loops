from __future__ import annotations

from evalloop.cli import main as cli
from evalloop.core import BASELINE, IMPROVED, Task, World
from evalloop.program import (
    Candidate,
    GuardedWorld,
    ReleasePolicy,
    gather,
    maturity,
    next_steps,
    quality,
    review,
    unknown,
)

SMALL = ReleasePolicy(tasks=60, seeds=(0, 1, 2))


def test_guarded_world_blocks_over_limit_refunds():
    world = GuardedWorld()
    oid = next(iter(world.orders))
    result = world.issue_refund(oid, 10_000.0)
    assert "error" in result
    assert world.effects == [{"blocked": oid, "amount": 10_000.0}]
    assert not any(e.get("ok") is False for e in world.effects)


def test_guarded_world_allows_refunds_within_policy():
    world, plain = GuardedWorld(), World()
    oid = next(
        o.id for o in world.orders.values() if o.days_ago <= 30
    )
    assert world.issue_refund(oid, 5.0) == plain.issue_refund(
        oid, 5.0
    )


def test_evidence_covers_every_task_and_seed():
    ev = gather(Candidate(IMPROVED), SMALL)
    assert len(ev.outcomes) == 60
    assert all(len(r) == 3 for r in ev.outcomes.values())
    assert ev.cost > 0


def test_quality_floor_separates_configs():
    good = quality(gather(Candidate(IMPROVED), SMALL), SMALL)
    bad = quality(gather(Candidate(BASELINE), SMALL), SMALL)
    assert good.passed and not bad.passed


def test_review_holds_regression_and_names_signals():
    current = Candidate(IMPROVED)
    result = review(Candidate(BASELINE), current, SMALL)
    failed = {s.name for s in result.signals if not s.passed}
    assert result.decision == "hold"
    assert {"quality", "gate", "versus"} <= failed


def signal(result, name):
    return next(s for s in result.signals if s.name == name)


def test_guard_removes_injection_harm():
    current = Candidate(IMPROVED)
    plain = review(Candidate(IMPROVED), current)
    guarded = review(Candidate(IMPROVED, GuardedWorld), current)
    assert not signal(plain, "injection").passed
    assert signal(guarded, "injection").passed
    assert guarded.decision == "ship"


def test_maturity_needs_every_lower_level():
    assert maturity(set()) == 0
    level_one = {"eval_set", "graders", "error_analysis"}
    assert maturity(level_one) == 1
    assert maturity(level_one | {"judge", "online"}) == 1
    assert next_steps(level_one)[0].level == 2
    assert unknown({"graders", "grader"}) == {"grader"}


def test_task_ids_are_stable_across_policies():
    ids = [t.id for t in SMALL.task_set()]
    same = ReleasePolicy(tasks=60).task_set()
    assert ids == [t.id for t in same]
    assert all(isinstance(t, Task) for t in SMALL.task_set())


def test_cli_gate_exit_codes(tmp_path, capsys):
    path = str(tmp_path / "b.json")
    assert cli(["--tasks", "60", "baseline", "--out", path]) == 0
    assert cli(["--tasks", "60", "gate", "--baseline", path]) == 0
    bad = ["--tasks", "60", "gate", "--candidate", "baseline"]
    assert cli([*bad, "--baseline", path]) == 1
    missing = str(tmp_path / "none.json")
    assert cli(["gate", "--baseline", missing]) == 2
    assert "FAIL" in capsys.readouterr().out


def test_cli_maturity_rejects_unknown(capsys):
    assert cli(["maturity", "eval_set", "typo"]) == 2
    assert "unknown practices: typo" in capsys.readouterr().out


def test_example_main(capsys):
    from ch20_operating_model import main

    main()
    out = capsys.readouterr().out
    assert "FAIL injection 10/55 probes paid out" in out
    assert out.count("decision: ship") == 1
