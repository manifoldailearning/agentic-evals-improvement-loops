from collections import Counter

import pytest

from evalloop.core import (
    BASELINE,
    IMPROVED,
    SimAgent,
    make_tasks,
    run_suite,
)
from evalloop.flywheel import (
    Example,
    balance,
    contaminated,
    curate,
    dedup,
    eval_index,
    finetune,
    harvest,
    make_card,
    ngrams,
    production_logs,
    quality,
    scrub,
    scrub_pii,
)

EVAL_SET = make_tasks(200, seed=7)
LOGS = production_logs(120)


def ex(i, text, category="faq", source="success"):
    return Example(f"E-{i}", category, text, ("search_kb",),
                   "ok [kb:returns]", source)  # fmt: skip


def test_logs_are_seeded_noisy_and_retried():
    assert production_logs(120) == LOGS
    assert any(t.id.endswith("r") for t in LOGS)
    assert any("@example.com" in t.input for t in LOGS)


def test_harvest_keeps_passes_and_corrects_failures():
    runs = run_suite(SimAgent(BASELINE), LOGS)
    got = harvest(LOGS, runs, teacher=IMPROVED)
    assert len(got) == len(LOGS)
    by_source = Counter(e.source for e in got)
    assert by_source["success"] and by_source["corrected"]
    assert all(e.passed for e in got if e.source == "success")


def test_quality_penalizes_failed_or_empty_runs():
    assert quality(ex(1, "q")) == 1.0
    bad = Example("E-2", "faq", "q", (), "", "corrected", False)
    assert quality(bad) == 0.25  # only the length check holds


def test_dedup_normalizes_case_and_punctuation():
    rows = [ex(1, "Where is A-1001?"), ex(2, "where is a-1001")]
    assert [e.id for e in dedup(rows)] == ["E-1"]


def test_scrub_masks_email_and_phone():
    text = "Mail a.b@example.com or call +1 415 555 0134."
    assert scrub(text) == "Mail [email] or call [phone]."
    rows = scrub_pii([ex(1, text), ex(2, "plain")])
    assert "@" not in rows[0].input and rows[1].input == "plain"


def test_contamination_catches_copies_not_paraphrases():
    index = eval_index(EVAL_SET)
    item = EVAL_SET[2].input  # an faq question
    assert contaminated(ex(1, "Hi, " + item + " Thanks!"), index)
    para = "How many days until my parcel shows up, roughly?"
    assert not contaminated(ex(2, para), index)
    assert ngrams("a b", 8) == {("a", "b")}


def test_balance_caps_each_category():
    rows = [ex(i, f"q{i}", "faq") for i in range(5)]
    rows += [ex(9, "q9", "account")]
    kept = balance(rows, cap=2)
    assert Counter(e.category for e in kept) == {
        "faq": 2,
        "account": 1,
    }


def test_curate_funnel_and_card():
    runs = run_suite(SimAgent(BASELINE), LOGS)
    data, funnel = curate(harvest(LOGS, runs, IMPROVED), EVAL_SET)
    sizes = [n for _, n in funnel.steps]
    assert sizes == sorted(sizes, reverse=True)
    assert not any("@" in e.input for e in data)
    card = make_card("t", data, funnel, "ds-x", ["c"])
    assert card.size == len(data)
    assert card.version.startswith("sha-")
    assert all(len(line) <= 64 for line in card.lines())


def test_finetune_is_a_simulated_model_swap():
    data = [ex(i, f"q{i}") for i in range(100)]
    tuned = finetune(BASELINE, data)
    assert tuned.model == "small-ft"
    assert tuned.rules == BASELINE.rules
    with pytest.raises(ValueError):
        finetune(BASELINE, data[:5])
    with pytest.raises(ValueError):
        finetune(IMPROVED, data)


def test_example_main(capsys):
    from ch18_flywheel import main

    main()
    out = capsys.readouterr().out
    assert "Tuning is SIMULATED" in out
    assert "small 63.5% -> small-ft 74.0%" in out
    lines = out.splitlines()
    assert len(lines) <= 25 and max(map(len, lines)) <= 64
