import pytest

from evalloop.core import BASELINE, IMPROVED, make_tasks
from evalloop.infra import (
    GRADER,
    ResultCache,
    ResultStore,
    cache_key,
    code_version,
    compare,
    dataset_hash,
    estimate_cost,
    make_manifest,
    run_one,
    run_suite_parallel,
)
from evalloop.infra.runner import pilot_sample

TASKS = make_tasks(40, seed=7)
SEEDS = (0, 1)


def test_run_one_grades_from_evidence():
    results = [run_one(BASELINE, t, 0) for t in TASKS]
    assert all(r.passed == (not r.evidence) for r in results)
    assert any(not r.passed for r in results)
    assert all(r.cost > 0 for r in results)


def test_cache_key_covers_all_four_parts():
    keys = {
        cache_key("T-1", "cfg", GRADER, 0),
        cache_key("T-2", "cfg", GRADER, 0),
        cache_key("T-1", "cfg2", GRADER, 0),
        cache_key("T-1", "cfg", "checker-v2", 0),
        cache_key("T-1", "cfg", GRADER, 1),
    }
    assert len(keys) == 5


def test_second_run_is_free_and_identical():
    cache = ResultCache()
    first, spent = run_suite_parallel(
        IMPROVED, TASKS, SEEDS, cache
    )
    again, free = run_suite_parallel(
        IMPROVED, TASKS, SEEDS, cache
    )
    assert spent > 0 and free == 0
    assert again == first
    assert (cache.hits, cache.misses) == (80, 80)


def test_order_is_task_major_for_any_worker_count():
    one, _ = run_suite_parallel(BASELINE, TASKS, SEEDS, workers=1)
    many, _ = run_suite_parallel(
        BASELINE, TASKS, SEEDS, workers=8
    )
    assert one == many
    assert [(r.task_id, r.seed) for r in one[:3]] == [
        ("T-0001", 0),
        ("T-0001", 1),
        ("T-0002", 0),
    ]


def test_cache_roundtrips_through_jsonl(tmp_path):
    cache = ResultCache()
    run_suite_parallel(BASELINE, TASKS[:5], (0,), cache)
    cache.save(tmp_path / "cache.jsonl")
    loaded = ResultCache.load(tmp_path / "cache.jsonl")
    assert loaded.entries == cache.entries


def test_pilot_is_stratified_and_estimate_scales():
    sample = pilot_sample(TASKS, 10)
    assert len({t.category for t in sample}) == 5
    est = estimate_cost(IMPROVED, TASKS, SEEDS, pilot=10)
    assert est.runs == 80
    assert est.total == pytest.approx(est.per_run * 80 * 1.2)


def test_manifest_names_inputs_and_detects_changes():
    a = make_manifest(IMPROVED, TASKS, SEEDS)
    assert a == make_manifest(IMPROVED, TASKS, SEEDS)
    assert a.dataset == dataset_hash(TASKS)
    assert code_version().startswith("1.0.0+")
    b = make_manifest(IMPROVED, TASKS[:-1], (0,))
    assert a.differs(b) == ["dataset", "seeds"]
    relabeled = IMPROVED.evolve(version="renamed")
    assert (
        make_manifest(relabeled, TASKS, SEEDS).run_id == a.run_id
    )


def test_store_and_compare_runs(tmp_path):
    store = ResultStore(tmp_path)
    runs = []
    for config in (BASELINE, IMPROVED):
        results, _ = run_suite_parallel(config, TASKS, SEEDS)
        manifest = make_manifest(config, TASKS, SEEDS)
        store.save(manifest, results)
        runs.append(store.load(manifest.run_id))
    assert (
        runs[0][1]
        == run_suite_parallel(BASELINE, TASKS, SEEDS)[0]
    )
    assert len(store.runs()) == 2
    diff = compare(*runs)
    assert diff.changed == ["config"]
    assert diff.after > diff.before and len(diff.fixed) > 0


def test_example_main(capsys):
    from ch19_infrastructure import main

    main()
    out = capsys.readouterr().out
    assert "600 hits, 0 misses, $0.00 spent" in out
    assert "same as A: True" in out
    lines = out.splitlines()
    assert len(lines) <= 25 and max(map(len, lines)) <= 64
