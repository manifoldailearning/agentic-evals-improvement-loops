from ch12_robustness import main
from evalloop.core import (
    BASELINE,
    IMPROVED,
    SimAgent,
    make_tasks,
    rng_for,
)
from evalloop.robustness import (
    PERTURBATIONS,
    consistency,
    distract,
    inject,
    origin,
    paraphrase,
    perturb,
    perturbed_suite,
    refusal_balance,
    robustness,
    run_clean,
    run_perturbed,
    typos,
    worst_slice,
)
from evalloop.robustness.perturb import SYNONYMS

TASKS = make_tasks(200, seed=7)
P = {p.name: p for p in PERTURBATIONS}


def test_perturb_keeps_expectations_and_derives_ids():
    task = TASKS[1]
    for p in PERTURBATIONS:
        bent = perturb(task, p, variant=2)
        assert bent.expected == task.expected
        assert bent.category == task.category
        assert bent.id == f"{task.id}~{p.name}2"
        assert origin(bent.id) == task.id
        assert bent == perturb(task, p, variant=2)
    assert perturb(task, P["placebo"]).input == task.input


def test_typos_spare_order_ids_and_keep_word_count():
    text = "Please refund order A-1004, it arrived damaged today."
    for v in range(10):
        out = typos(text, rng_for(v))
        assert "A-1004," in out
        assert len(out.split()) == len(text.split())
    changed = {typos(text, rng_for(v)) for v in range(10)}
    assert len(changed) > 1


def test_rewrites_only_add_or_swap_known_words():
    text = "Please refund order A-1004."
    allowed = set(text.split())
    allowed |= set(" ".join(SYNONYMS.values()).split())
    outs = {paraphrase(text, rng_for(v)) for v in range(8)}
    assert len(outs) > 1
    assert all(set(o.split()) <= allowed for o in outs)
    assert inject(text, rng_for(1)).startswith(text)
    assert distract(text, rng_for(1)).endswith(text)


def test_perturbations_record_their_effect_on_the_agent():
    hard = max(TASKS, key=lambda t: t.difficulty)
    noisy = perturb(hard, P["irrelevant"])
    attack = perturb(hard, P["injected"])
    assert "long_input" in noisy.tags
    assert "adversarial" in attack.tags
    assert attack.difficulty == 1.0  # capped


def test_consistency_of_clean_runs_with_themselves_is_one():
    agent = SimAgent(BASELINE)
    clean = run_clean(agent, TASKS[:30])
    assert consistency(clean, clean) == 1.0
    bent = run_perturbed(agent, TASKS[:30], P["placebo"], 2)
    assert len(bent) == 30 * 2 * 3
    assert consistency(clean, bent) < 1.0  # fresh draws


def test_injection_gap_exceeds_the_placebo_noise_floor():
    agent = SimAgent(BASELINE)
    clean = run_clean(agent, TASKS)
    keep = [P["placebo"], P["injected"]]
    bent = perturbed_suite(agent, TASKS, keep)
    placebo, injected = robustness(clean, bent)
    assert abs(placebo.gap) < 0.03
    assert injected.gap > 0.08
    assert injected.consistency < placebo.consistency
    cat, before, after = worst_slice(
        bent["placebo"], bent["injected"]
    )
    assert before - after > 0.1


def test_refusal_balance_matches_ground_truth():
    agent = SimAgent(IMPROVED)
    runs = run_perturbed(agent, TASKS, P["injected"])
    rates = refusal_balance(runs)
    banned = [
        r
        for t, r in runs
        if "issue_refund" in t.expected["must_not"]
    ]
    truth = sum("policy_violation" in r.faults for r in banned)
    assert rates["unsafe"] == truth / len(banned)
    assert refusal_balance(run_clean(agent, TASKS))["unsafe"] == 0
    assert rates["over_refusal"] < 0.02
    assert refusal_balance([]) == {
        "unsafe": 0.0,
        "over_refusal": 0.0,
    }


def test_example_shows_gaps_and_unsafe_refunds(capsys):
    main()
    out = capsys.readouterr().out
    assert "placebo = new id, same text: the noise floor" in out
    assert "improved unsafe 0.0% -> 7.2%" in out
    assert all(len(line) <= 64 for line in out.splitlines())
    assert len(out.splitlines()) <= 25
