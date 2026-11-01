from dataclasses import replace

import ch04_error_analysis
from evalloop.analysis import (
    CODEBOOK,
    PHRASES,
    UNCODED,
    Note,
    agreement,
    apply_fix,
    axial_code,
    category_agreement,
    category_counts,
    code_notes,
    detection_agreement,
    observe,
    open_code,
    pareto,
    review,
    saturation,
)
from evalloop.core import (
    BASELINE,
    IMPROVED,
    SimAgent,
    make_tasks,
    run_suite,
)

TASKS = make_tasks(200, seed=7)
BY_ID = {t.id: t for t in TASKS}
RUNS = run_suite(SimAgent(BASELINE), TASKS)
CODED = code_notes(review(TASKS, RUNS))


def run_of(task_id):
    return next(r for r in RUNS if r.task_id == task_id)


def test_annotator_ignores_ground_truth_labels():
    run = run_of("T-0012")  # over-limit refund was issued
    task = BY_ID["T-0012"]
    hidden = replace(run, faults=[])
    assert observe(task, hidden) == observe(task, run)
    assert [code for code, _ in observe(task, run)] == ["policy"]


def test_clean_runs_get_no_notes():
    clean = [r for r in RUNS if not r.faults]
    assert all(
        open_code(BY_ID[r.task_id], r) == [] for r in clean
    )


def test_evidence_covers_loops_and_misquotes():
    looped = run_of("T-0009")
    assert ("loop", {"tool": "search_kb", "n": 6}) in observe(
        BY_ID["T-0009"], looped
    )
    codes = [
        c for c, _ in observe(BY_ID["T-0023"], run_of("T-0023"))
    ]
    assert codes == ["quote"]


def test_every_phrase_is_coded_into_a_category():
    details = {
        "got": "x", "want": "y", "tool": "t", "oid": "A-1",
        "said": "s", "true": "t", "kb": "refund-policy",
        "amount": 1.0, "n": 2, "phrase": "calm down",
    }  # fmt: skip
    names = {name for name, _ in CODEBOOK}
    coded = {
        code: {axial_code(t.format(**details)) for t in texts}
        for code, texts in PHRASES.items()
    }
    assert all(len(cats) == 1 for cats in coded.values())
    assert set().union(*coded.values()) == names
    assert axial_code("the answer was fine") == UNCODED


def test_review_is_deterministic_and_depends_on_annotator():
    again = code_notes(review(TASKS, RUNS))
    assert again == CODED
    other = code_notes(review(TASKS, RUNS, annotator="ann-2"))
    assert other != CODED


def test_pareto_is_sorted_and_ends_at_one():
    rows = pareto(category_counts(CODED))
    counts = [n for _, n, _ in rows]
    assert counts == sorted(counts, reverse=True)
    assert rows[0][0] == "missing citation"
    assert rows[-1][2] == 1.0


def test_saturation_finds_every_category_then_flattens():
    curve = saturation(CODED)
    assert sum(curve) == len(category_counts(CODED))
    assert curve[-5:] == [0] * 5
    assert saturation({}) == []


def test_agreement_against_ground_truth():
    found = detection_agreement(CODED, RUNS)
    assert found.precision == 1.0 and 0.9 < found.recall < 1.0
    per = category_agreement(CODED, RUNS)
    assert per["policy breach"].recall == 1.0
    assert per["stopped early"].precision < 0.5
    empty = agreement(set(), set())
    assert (empty.precision, empty.recall) == (1.0, 1.0)


def test_fixes_change_the_config():
    fixed = apply_fix(BASELINE, "missing citation")
    assert "cite_kb" in fixed.rules
    assert apply_fix(BASELINE, "wrong tool").few_shots == 3
    notes = [Note("T-1", "stuck calling lookup_order; no answer")]
    assert code_notes(notes) == {"T-1": {"no answer"}}
    better = run_suite(SimAgent(IMPROVED), TASKS)
    assert len(code_notes(review(TASKS, better))) < len(CODED)


def test_example_main_prints_taxonomy_and_check(capsys):
    ch04_error_analysis.main()
    out = capsys.readouterr().out
    assert "notes on 69 traces" in out
    assert "missing citation  14" in out
    assert "Top 3 fixes: success 63.5% -> 76.5%" in out
    lines = out.splitlines()
    assert len(lines) <= 25
    assert max(len(line) for line in lines) <= 64
