from dataclasses import replace

import pytest

import ch05_synthetic
from evalloop.core import (
    BASELINE,
    IMPROVED,
    SimAgent,
    World,
    make_tasks,
    rng_for,
    run_suite,
)
from evalloop.synth import (
    PERSONAS,
    TEMPLATES,
    apply_persona,
    from_template,
    inject,
    make_ambiguous,
    make_synthetic,
    paraphrase,
    problems,
    quality_filter,
    shift,
    tvd,
    typos,
    verbose,
)

SYNTH = make_synthetic(300, seed=11)
KEPT, REJECTED = quality_filter(SYNTH)
REAL = make_tasks(200, seed=7)
STATUS = next(
    t for t in TEMPLATES if t.category == "order_status"
)


def base_task(oid="A-1002"):
    return replace(from_template(STATUS, oid, World()), id="X-1")


def test_generation_is_seeded_and_ids_are_sequential():
    assert make_synthetic(20, seed=11) == SYNTH[:20]
    assert make_synthetic(20, seed=12) != SYNTH[:20]
    assert [t.id for t in SYNTH[:2]] == ["S-0001", "S-0002"]


def test_expectations_come_from_the_world():
    task = base_task()
    status = World().lookup_order("A-1002")["status"]
    assert task.expected["answer_contains"] == [status]
    assert task.expected["args"] == {"order_id": "A-1002"}
    assert problems(task) == []


def test_persona_changes_voice_not_expectations():
    task = base_task()
    angry = apply_persona(task, PERSONAS[2])
    assert angry.input.startswith("This is unacceptable.")
    assert angry.expected == task.expected


def test_text_perturbations_add_tags():
    task = base_task()
    rng = rng_for("test")
    assert "long_input" in verbose(task, rng).tags
    hit = inject(task, rng)
    assert "adversarial" in hit.tags and "Ignore" in hit.input
    assert typos(task, rng, rate=1.0).input != task.input
    wordy = replace(task, input="Order help: I want a check")
    said = paraphrase(wordy, rng_for("p"))
    assert "paraphrased" in said.tags
    assert said.input != wordy.input
    assert said.input[0].isupper()


def test_ambiguity_rewrites_the_answer_key():
    vague = make_ambiguous(base_task(), rng_for("v"))
    assert "A-1002" not in vague.input
    assert vague.expected["clarify"] is True
    assert vague.expected["tools"] == []
    assert problems(vague) == []
    faq = next(t for t in SYNTH if t.category == "faq")
    assert make_ambiguous(faq, rng_for("v")) == faq


def test_validity_catches_broken_answer_keys():
    assert problems(base_task("A-1063")) == ["unknown order"]
    typo = replace(base_task(), input="Has order A1-002 shipped?")
    assert problems(typo) == ["order id not in input"]
    wrong = base_task()
    wrong.expected["answer_contains"] = ["teleported"]
    assert problems(wrong) == ["wrong status"]


def test_quality_filter_keeps_only_valid_unique_tasks():
    assert len(KEPT) + len(REJECTED) == len(SYNTH)
    assert all(problems(t) == [] for t in KEPT)
    assert set(REJECTED.values()) == {
        "duplicate",
        "unknown order",
        "order id not in input",
    }
    run_suite(SimAgent(BASELINE), KEPT, seeds=(0, 1))


def test_tvd_and_shift():
    assert tvd({"a": 1.0}, {"a": 1.0}) == 0.0
    assert tvd({"a": 1.0}, {"b": 1.0}) == 1.0
    same = shift(REAL, REAL)
    assert same.category_tvd == 0.0 and same.vocab_overlap == 1.0
    s = shift(REAL, KEPT)
    assert 0.0 < s.category_tvd < s.tag_tvd
    assert s.length_ratio == pytest.approx(1.38, abs=0.01)


def test_synthetic_set_ranks_configs_like_the_real_set():
    def rate(config, tasks):
        runs = run_suite(SimAgent(config), tasks)
        return sum(r.ok for r in runs) / len(runs)

    assert rate(IMPROVED, KEPT) > rate(BASELINE, KEPT) + 0.2
    assert rate(IMPROVED, REAL) > rate(BASELINE, REAL) + 0.2


def test_example_main_reports_filter_and_shift(capsys):
    ch05_synthetic.main()
    out = capsys.readouterr().out
    assert "Quality filter: 300 -> 198 kept" in out
    assert "Baseline success: real 63.5%" in out
    lines = out.splitlines()
    assert len(lines) <= 25
    assert max(len(line) for line in lines) <= 64
